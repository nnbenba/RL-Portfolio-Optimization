import numpy as np
import pandas as pd
from reward import DifferentialSharpeRatio

# ── Dimensions ───────────────────────────────────────────────────────────────
N_SECTORS = 9
N_ASSETS  = N_SECTORS + 1     # 9 sectors + cash
T         = 60                # lookback window in days
LOOKBACK  = T
N_REGIMES = 3                 # HMM regimes: bear / neutral / bull

# NSMDP flattened state:
#   S_t = [ vec(R_t) || w_t || b_t || H(b_t) || z(vol20) || z(vol20/vol60) || z(VIX) ]
# vec(R_t) = the T real log-returns r_{t-T}..r_{t-1}  PLUS  N_APPEND synthetic
# next-return draws r_hat_{t+1} from the regime-mixture Student-t (nonstationary.py).
# The 3 macro features are kept RAW here but expanding-z-scored inside the env
# (leak-free); they sit alongside the HMM belief so the agent gets both the
# un-compressed signal and the regime summary.
N_APPEND  = 1                                        # synthetic r_{t+1} appended
MACRO_DIM = 3                                        # z(vol20), z(vol20/vol60), z(VIX)
RET_BLOCK = N_SECTORS * (LOOKBACK + N_APPEND)        # (60+1) x 9 = 549
STATE_DIM = RET_BLOCK + N_ASSETS + N_REGIMES + 1 + MACRO_DIM   # 549+10+3+1+3 = 566


def _expand_std(s: pd.Series) -> np.ndarray:
    """Expanding-window z-score (no look-ahead): mean/std use only data up to t.
    Paper Section 5.1. NaNs (feature warm-up) -> 0 so the state never carries NaN."""
    mu    = s.expanding().mean()
    sigma = s.expanding().std().clip(lower=1e-8)
    return np.nan_to_num(((s - mu) / sigma).values, nan=0.0, posinf=0.0, neginf=0.0)


def _rebalance(
    weights: np.ndarray,
    prices_t: np.ndarray,
    port_val: float,
    n_sectors: int,
) -> tuple[np.ndarray, float]:
    """Target weights → whole share counts + remaining cash."""
    target_vals = weights[:n_sectors] * port_val
    shares      = np.floor(target_vals / np.where(prices_t > 0, prices_t, 1e-8))
    cash        = port_val - shares @ prices_t
    return shares, float(cash)


def _observed_weights(
    shares: np.ndarray,
    prices: np.ndarray,
    cash: float,
    port_val: float,
    n_assets: int,
) -> np.ndarray:
    """Back-compute actual weight vector after a price move."""
    w = np.zeros(n_assets)
    if port_val < 1e-8:
        w[-1] = 1.0
        return w
    w[:-1] = shares * prices / port_val
    w[-1]  = cash / port_val
    return w


class PortfolioEnv:
    """
    Single-agent market-replay environment (NSMDP variant).

        state = env.reset(start_idx)
        while not done:
            weights = policy(state)              # (N_ASSETS,), sums to 1
            state, reward, done, info = env.step(weights)

    Flattened state vector, length STATE_DIM = 566:
        S_t = [ vec(R_t) || w_t || b_t || H(b_t) || z(vol20) || z(vol20/vol60) || z(VIX) ]
          vec(R_t) : 9 sectors x (60+1) log-returns, r_{t-60}..r_{t-1} + r_hat_{t+1}
          w_t      : current portfolio weights (10)
          b_t      : HMM regime belief P(regime_t | x_1..x_t)  (3, filtered)
          H(b_t)   : normalized Shannon entropy of the belief   (1, in [0,1])
          z(macro) : expanding-z-scored vol20, vol20/vol60, VIX  (3, leak-free)

    The 3 macro features sit ALONGSIDE the HMM belief (not replaced): the agent
    gets both the un-compressed signal and the regime summary. They are z-scored
    with an expanding window inside the env so VIX (~10-80) does not swamp the
    return-scale inputs.

    vec(R_t) is augmented with a synthetic next-return r_hat_{t+1} drawn from a
    per-window regime-mixture Student-t (`ns_model`, nonstationary.py). This is a
    forecast SAMPLE conditioned on b_t, not the realized return — no look-ahead.
    If ns_model is None the appended slot is zeros (keeps STATE_DIM fixed).

    Reward: Differential Sharpe Ratio D_t. No transaction costs.
    """

    def __init__(
        self,
        sector_prices: pd.DataFrame,
        log_returns:   pd.DataFrame,
        vix:           pd.Series,       # raw CBOE VIX level
        vol20:         pd.Series,       # 20-day rolling vol of S&P
        vol60:         pd.Series,       # 60-day rolling vol of S&P
        belief:        np.ndarray,      # (D, N_REGIMES) filtered regime posterior
        start_cash:    float = 100_000,
        lookback:      int   = LOOKBACK,
        eta:           float = 1 / 252,
        ns_model=None,                  # NonStationaryReturnModel or None
        seed:          int | None = None,
        turnover_coef: float = 0.0,     # lambda in  R_net = R_t - lambda * Sum|dw|
    ):
        idx            = log_returns.index
        self.log_ret   = log_returns.reindex(idx).values.astype(np.float32)   # (D, n_sectors)
        self.prices    = sector_prices.reindex(idx).values.astype(np.float32) # (D, n_sectors)
        self.belief    = np.asarray(belief, dtype=np.float32)                  # (D, N_REGIMES)
        # Expanding-window z-scores of the macro features (leak-free)
        self.vol20_z   = _expand_std(vol20.reindex(idx)).astype(np.float32)
        self.vratio_z  = _expand_std((vol20 / vol60).reindex(idx)).astype(np.float32)
        self.vix_z     = _expand_std(vix.reindex(idx)).astype(np.float32)
        self.dates     = idx
        self.n_days    = len(idx)

        self.ns_model  = ns_model
        self.rng       = np.random.default_rng(seed)
        self.turnover_coef = turnover_coef

        self.n_sectors = log_returns.shape[1]
        self.n_assets  = self.n_sectors + 1
        self.T         = lookback
        self.start_cash = start_cash
        self.eta       = eta

        assert self.belief.shape == (self.n_days, N_REGIMES), \
            f"belief shape {self.belief.shape} != {(self.n_days, N_REGIMES)}"

        self.t = self.shares = self.cash = self.weights = self.port_val = self.dsr = None

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, start_idx: int | None = None) -> np.ndarray:
        if start_idx is None:
            start_idx = self.T
        if start_idx < self.T:
            raise ValueError(f"start_idx ({start_idx}) must be >= T ({self.T})")

        self.t           = start_idx
        self.shares      = np.zeros(self.n_sectors, dtype=np.float32)
        self.cash        = float(self.start_cash)
        self.weights     = np.zeros(self.n_assets, dtype=np.float32)
        self.weights[-1] = 1.0
        self.port_val    = float(self.start_cash)
        self.dsr         = DifferentialSharpeRatio(eta=self.eta)
        return self._build_state()

    def step(self, weights: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        assert self.t is not None, "Call reset() before step()."

        # Turnover c_t = Sum_i|w_target - w_held|, measured BEFORE overwriting
        # self.weights (which holds the drifted, currently-held weights).
        turnover = float(np.abs(weights - self.weights).sum())

        # 1. Rebalance at today's closing prices (frictionless, immediate)
        self.shares, self.cash = _rebalance(
            weights, self.prices[self.t], self.port_val, self.n_sectors
        )
        self.weights = weights.copy()

        # 2. Advance one trading day
        self.t += 1
        done = self.t >= self.n_days - 1

        # 3. Portfolio value after price move
        prices_new = self.prices[self.t]
        new_val    = float(self.shares @ prices_new) + self.cash
        R_t        = (new_val - self.port_val) / max(self.port_val, 1e-8)

        # 4. Reward = DSR on the NET return R~_t = R_t - lambda*c_t. The cost is
        #    subtracted from the return that enters the DSR (Moody et al.), so it
        #    flows through the numerator via dA_t = R~_t - A_{t-1} and
        #    dB_t = R~_t^2 - B_{t-1}. turnover_coef=0 = plain frictionless DSR.
        R_net  = R_t - self.turnover_coef * turnover
        reward = self.dsr.step(R_net)

        # 5. Update state
        self.port_val = new_val
        self.weights  = _observed_weights(
            self.shares, prices_new, self.cash, new_val, self.n_assets
        )

        info = {
            "date":     self.dates[self.t],
            "port_val": new_val,
            "R_t":      R_t,
            "turnover": turnover,
        }
        return self._build_state(), reward, done, info

    def date_to_idx(self, date) -> int:
        return self.dates.get_loc(pd.Timestamp(date))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_state(self) -> np.ndarray:
        """S_t = [ vec(R_t) || w_t || b_t || H(b_t) || z(macro) ], length STATE_DIM.

        vec(R_t) = the T real returns r_{t-T}..r_{t-1} with N_APPEND synthetic
        next-return draws r_hat_{t+1} ~ regime-mixture Student-t(b_t) appended.
        z(macro) = expanding-z-scored vol20, vol20/vol60, VIX at t (leak-free)."""
        t = self.t
        b = self.belief[t]                            # (N_REGIMES,)

        # Real lookback window (strictly before t, no look-ahead)
        R = self.log_ret[t - self.T : t]              # (T, n_sectors)

        # Append synthetic r_hat_{t+1} sample(s) conditioned on the belief b_t
        if self.ns_model is not None:
            r_next = self.ns_model.sample_next(b, self.rng, n=N_APPEND)  # (N_APPEND, N) or (N,)
            r_next = np.atleast_2d(r_next).astype(np.float32)
        else:
            r_next = np.zeros((N_APPEND, self.n_sectors), dtype=np.float32)
        R_aug = np.vstack([R, r_next])                # (T + N_APPEND, n_sectors)

        H = float(-(b * np.log(b + 1e-12)).sum() / np.log(N_REGIMES))
        macro = np.float32([self.vol20_z[t], self.vratio_z[t], self.vix_z[t]])   # z-scored
        return np.concatenate([
            R_aug.reshape(-1),                        # vec(R_t)  ((T+N_APPEND)*n_sectors,)
            self.weights.astype(np.float32),          # w_t       (n_assets,)
            b,                                        # b_t       (N_REGIMES,)
            np.float32([H]),                          # H(b_t)    (1,)
            macro,                                    # z(macro)  (MACRO_DIM,)
        ]).astype(np.float32)


# ── Quick sanity-check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yfinance as yf

    TICKERS = ["XLB", "XLI", "XLY", "XLP", "XLV", "XLF", "XLK", "XLU", "XLE", "^VIX"]
    raw    = yf.download(TICKERS, start="2006-01-01", end="2021-12-31",
                         auto_adjust=True, progress=False)
    prices = raw["Close"][TICKERS]
    vix           = prices["^VIX"]
    sector_prices = prices.drop(columns=["^VIX"])
    log_returns   = np.log(sector_prices / sector_prices.shift(1)).dropna()

    sp500  = yf.download("^GSPC", start="2006-01-01", end="2021-12-31",
                         auto_adjust=True, progress=False)["Close"].squeeze()
    sp_lr  = np.log(sp500 / sp500.shift(1)).dropna()
    vol20  = sp_lr.rolling(20).std()
    vol60  = sp_lr.rolling(60).std()

    D = len(log_returns)
    belief = np.full((D, N_REGIMES), 1.0 / N_REGIMES, dtype=np.float32)  # uniform stub

    env   = PortfolioEnv(sector_prices, log_returns, vix, vol20, vol60, belief)
    state = env.reset()
    print(f"State shape : {state.shape}   (expected ({STATE_DIM},))")
    print(f"  vec(R_t): {RET_BLOCK} | w_t: {N_ASSETS} | b_t: {N_REGIMES} | H: 1 | macro: {MACRO_DIM}")

    ew = np.full(N_ASSETS, 1 / N_ASSETS)
    for step in range(3):
        state, reward, done, info = env.step(ew)
        print(f"  step {step+1} | {info['date'].date()} | "
              f"port=${info['port_val']:,.2f} | R_t={info['R_t']:+.5f} | D_t={reward:+.4f}")
    print(f"tail (b_t,H,z-macro): {state[-(N_REGIMES+1+MACRO_DIM):].round(3)}")
