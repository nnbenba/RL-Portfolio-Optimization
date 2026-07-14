import numpy as np
import pandas as pd
from reward import DifferentialSharpeRatio

N_SECTORS = 9
N_ASSETS  = N_SECTORS + 1
T         = 60


def _expand_std(s: pd.Series) -> np.ndarray:
    """Expanding-window z-score (no look-ahead). Paper Section 5.1."""
    mu    = s.expanding().mean()
    sigma = s.expanding().std().clip(lower=1e-8)
    return ((s - mu) / sigma).values


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
    w[:-1]  = shares * prices / port_val
    w[-1]   = cash / port_val
    return w


class PortfolioEnv:
    """
    Single-agent market-replay environment for portfolio optimization.

    The agent interacts with it as a standard gym-style loop:
        state = env.reset(start_idx)
        while not done:
            weights = policy(state)          # (N_ASSETS,), sums to 1
            state, reward, done, info = env.step(weights)

    State matrix shape: (N_ASSETS, T) = (10, 60)
      Rows 0–8  : [w_i, r_{t-T+2}, …, r_t]   sector log-returns (oldest → newest)
      Row  9    : [w_c, vol20*, vratio*, vix*, 0, …]   * = expanding z-score

    Reward: Differential Sharpe Ratio D_t (paper Section 4.3).
    No transaction costs (paper assumption).
    """

    def __init__(
        self,
        sector_prices: pd.DataFrame,
        log_returns:   pd.DataFrame,
        vix:           pd.Series,
        vol20:         pd.Series,
        vol60:         pd.Series,
        start_cash:    float = 100_000,
        lookback:      int   = T,
        eta:           float = 1 / 252,
    ):
        # Align everything to log_returns index (log_returns already dropped NaN row 0)
        idx               = log_returns.index
        self.log_ret      = log_returns.reindex(idx).values.astype(np.float32)   # (D, n_sectors)
        self.prices       = sector_prices.reindex(idx).values.astype(np.float32) # (D, n_sectors)
        self.vol20_z      = _expand_std(vol20.reindex(idx)).astype(np.float32)
        self.vratio_z     = _expand_std((vol20 / vol60).reindex(idx)).astype(np.float32)
        self.vix_z        = _expand_std(vix.reindex(idx)).astype(np.float32)
        self.dates        = idx
        self.n_days       = len(idx)

        self.n_sectors    = log_returns.shape[1]
        self.n_assets     = self.n_sectors + 1
        self.T            = lookback
        self.start_cash   = start_cash
        self.eta          = eta

        # State will be set by reset()
        self.t            = None
        self.shares       = None
        self.cash         = None
        self.weights      = None
        self.port_val     = None
        self.dsr          = None

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, start_idx: int | None = None) -> np.ndarray:
        """
        Reset to an all-cash portfolio.
        start_idx: position in dates array; must be >= T (default: T).
        Returns initial state matrix (N_ASSETS, T).
        """
        if start_idx is None:
            start_idx = self.T
        if start_idx < self.T:
            raise ValueError(f"start_idx ({start_idx}) must be >= T ({self.T})")

        self.t        = start_idx
        self.shares   = np.zeros(self.n_sectors, dtype=np.float32)
        self.cash     = float(self.start_cash)
        self.weights  = np.zeros(self.n_assets, dtype=np.float32)
        self.weights[-1] = 1.0
        self.port_val = float(self.start_cash)
        self.dsr      = DifferentialSharpeRatio(eta=self.eta)

        return self._build_state()

    def step(self, weights: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """
        Apply weight vector, advance one day, return (next_state, reward, done, info).

        weights: (N_ASSETS,) array, must sum to 1 and be >= 0.
        """
        assert self.t is not None, "Call reset() before step()."

        # Turnover Sum|w_target - w_held|, measured BEFORE overwriting self.weights
        # (self.weights currently holds the drifted, currently-held weights). This is
        # the actual trade a transaction cost charges — NOT the post-move drift.
        turnover = float(np.abs(weights - self.weights).sum())

        # 1. Rebalance at today's closing prices (frictionless, immediate — paper Sec. 4)
        self.shares, self.cash = _rebalance(
            weights, self.prices[self.t], self.port_val, self.n_sectors
        )
        self.weights = weights.copy()

        # 2. Advance to next trading day
        self.t += 1
        done = self.t >= self.n_days - 1

        # 3. Portfolio value after price move
        prices_new  = self.prices[self.t]
        new_val     = float(self.shares @ prices_new) + self.cash
        R_t         = (new_val - self.port_val) / max(self.port_val, 1e-8)

        # 4. Differential Sharpe Ratio reward
        reward = self.dsr.step(R_t)

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
        """Convert a date string / Timestamp to an integer index."""
        return self.dates.get_loc(pd.Timestamp(date))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_state(self) -> np.ndarray:
        """Construct (N_ASSETS, T) state matrix at current time index self.t."""
        t     = self.t
        state = np.zeros((self.n_assets, self.T), dtype=np.float32)

        # Sector rows: weight + T-1 log-returns (oldest first → most recent last).
        # Newest return is r_{t-1}, matching the paper's observation vector
        # [r_{t-1}, …, r_{t-T+1}] (Section 4.2): the agent conditions on returns
        # realized strictly before the bar it rebalances into.
        recent = self.log_ret[t - self.T + 1 : t]   # (T-1, n_sectors), newest = r_{t-1}
        for i in range(self.n_sectors):
            state[i, 0]  = self.weights[i]
            state[i, 1:] = recent[:, i]

        # Cash row: weight + 3 standardised macro features
        state[-1, 0] = self.weights[-1]
        state[-1, 1] = self.vol20_z[t]
        state[-1, 2] = self.vratio_z[t]
        state[-1, 3] = self.vix_z[t]

        return state


# ── Quick sanity-check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yfinance as yf

    TICKERS = ["XLB","XLI","XLY","XLP","XLV","XLF","XLK","XLU","XLE","^VIX"]

    raw    = yf.download(TICKERS, start="2006-01-01", end="2021-12-31", auto_adjust=True, progress=False)
    prices = raw["Close"]
    prices.columns = TICKERS

    vix           = prices["^VIX"]
    sector_prices = prices.drop(columns=["^VIX"])
    log_returns   = np.log(sector_prices / sector_prices.shift(1)).dropna()

    sp500            = yf.download("^GSPC", start="2006-01-01", end="2021-12-31",
                                   auto_adjust=True, progress=False)["Close"].squeeze()
    sp500_lr         = np.log(sp500 / sp500.shift(1)).dropna()
    vol20            = sp500_lr.rolling(20).std()
    vol60            = sp500_lr.rolling(60).std()

    env   = PortfolioEnv(sector_prices, log_returns, vix, vol20, vol60)
    state = env.reset()

    print(f"State shape : {state.shape}   (expected (10, 60))")
    print(f"Date        : {env.dates[env.t].date()}")
    print(f"Port value  : ${env.port_val:,.2f}")
    print(f"Weights     : {env.weights.round(4)}")

    # Run 5 steps with equal-weight portfolio
    ew = np.full(N_ASSETS, 1 / N_ASSETS)
    for step in range(5):
        next_state, reward, done, info = env.step(ew)
        print(f"  step {step+1} | date={info['date'].date()} | "
              f"port_val=${info['port_val']:>10,.2f} | "
              f"R_t={info['R_t']:+.5f} | D_t={reward:+.4f}")

    print(f"\nNext state shape: {next_state.shape}")
    print(f"State min/max: {next_state.min():.4f} / {next_state.max():.4f}")
