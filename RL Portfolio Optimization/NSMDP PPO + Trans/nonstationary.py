"""
Non-stationary return generator for the NSMDP state update.

Given the regime belief b_t (from regime.py) we synthesize the next sector-return
vector r_{t+1} from a regime-mixture multivariate Student-t whose parameters are
belief-weighted blends of per-regime parameters:

    mu_k        regime-specific mean return vector (belief-weighted average)
    Sigma_k     Ledoit-Wolf-shrunk covariance of regime k's returns
    v_k         Student-t dof of regime k,  v_k = 6 / exkurt_k + 4
                (invert the t excess-kurtosis identity exkurt = 6/(v-4), v>4)

Belief-weighted mixture at time t (b = b_t, sums to 1):

    mu_bar_z    = Σ_k b[k] mu_k                     (N,)   regime-conditioned mean
    Sigma_bar_z = Σ_k b[k] Sigma_k                  (N,N)
    L_bar       = chol(Sigma_bar_z)                 lower-triangular
    v_bar       = Σ_k b[k] v_k                      scalar dof

Draw (epsilon ~ N(0, I_N),  W ~ chi^2_{v_bar}):

    r_{t+1} = mu_bar + (L_bar @ epsilon) / sqrt(W / v_bar)

which makes r_{t+1} a multivariate Student-t with location mu_bar, scale
Sigma_bar and v_bar degrees of freedom. This r_{t+1} is what gets appended to
vec(R_t) for the non-stationary state update.

All parameters are fit PER WINDOW on that window's train returns + belief
(leak-free), mirroring how regime.py fits the HMM.
"""

import numpy as np
from scipy.stats import kurtosis
from sklearn.covariance import LedoitWolf

N_REGIMES = 3

# Student-t dof clamps: v>4 needed for finite (excess) kurtosis; large v ≈ Gaussian.
NU_MIN = 4.5
NU_MAX = 250.0


class NonStationaryReturnModel:
    """Regime-mixture Student-t generator of r_{t+1}, fit per window."""

    def __init__(self, n_assets: int, n_regimes: int = N_REGIMES):
        self.n_assets  = n_assets
        self.n_regimes = n_regimes
        self.mu_k      = None     # (K, N)  regime mean returns
        self.cov_k     = None     # (K, N, N) Ledoit-Wolf regime covariances
        self.nu_k      = None     # (K,)    regime Student-t dof
        self.exkurt_k  = None     # (K,)    regime excess kurtosis (diagnostic)

    # ── Fitting (train span only) ────────────────────────────────────────────
    def fit(self, returns: np.ndarray, belief: np.ndarray):
        """
        returns : (Ttr, N) sector log-returns over the window's train span
        belief  : (Ttr, K) filtered regime posteriors over the same span
        """
        R = np.asarray(returns, dtype=np.float64)
        B = np.asarray(belief,  dtype=np.float64)
        K, N = self.n_regimes, self.n_assets

        # 1) mu_k : regime-specific mean return (belief-weighted average of returns)
        wsum      = B.sum(0)                                   # (K,)
        self.mu_k = (B.T @ R) / np.maximum(wsum[:, None], 1e-8)   # (K, N)

        # 2) Sigma_k : Ledoit-Wolf covariance of each regime's returns (hard label)
        labels     = B.argmax(1)
        self.cov_k = np.zeros((K, N, N))
        for k in range(K):
            Xk = R[labels == k]
            if len(Xk) > N:                                   # enough rows to shrink
                self.cov_k[k] = LedoitWolf().fit(Xk).covariance_
            else:                                             # fall back to pooled cov
                self.cov_k[k] = np.cov(R, rowvar=False)

        # 3) v_k : from each regime's excess kurtosis,  v = 6/exkurt + 4
        self.exkurt_k = np.zeros(K)
        self.nu_k     = np.zeros(K)
        for k in range(K):
            Xk = R[labels == k]
            exk = float(np.mean(kurtosis(Xk, axis=0, fisher=True, bias=False))) \
                  if len(Xk) > 4 else 0.0
            self.exkurt_k[k] = exk
            if exk <= 6.0 / (NU_MAX - 4.0):                   # thin/near-Gaussian tails
                self.nu_k[k] = NU_MAX
            else:
                self.nu_k[k] = np.clip(6.0 / exk + 4.0, NU_MIN, NU_MAX)
        return self

    # ── Belief-weighted mixture parameters at time t ─────────────────────────
    def mixture(self, b: np.ndarray):
        """b : (K,) belief b_t → (mu_bar, Sigma_bar, L_bar, v_bar)."""
        b = np.asarray(b, dtype=np.float64)
        mu_bar    = b @ self.mu_k                             # (N,)
        Sigma_bar = np.einsum("k,kij->ij", b, self.cov_k)     # (N, N)
        v_bar     = float(b @ self.nu_k)                      # scalar
        # jitter guards the Cholesky against tiny numerical non-PSD-ness
        jit = 1e-12 * np.trace(Sigma_bar) / self.n_assets
        L_bar = np.linalg.cholesky(Sigma_bar + jit * np.eye(self.n_assets))
        return mu_bar, Sigma_bar, L_bar, v_bar

    # ── Sample r_{t+1} ~ multivariate Student-t(mu_bar, Sigma_bar, v_bar) ────
    def sample_next(self, b: np.ndarray, rng: np.random.Generator | None = None,
                    n: int = 1) -> np.ndarray:
        """r_{t+1} = mu_bar + (L_bar @ eps) / sqrt(W / v_bar). Shape (N,) if n==1
        else (n, N)."""
        rng = rng or np.random.default_rng()
        mu_bar, _, L_bar, v_bar = self.mixture(b)
        eps = rng.standard_normal((n, self.n_assets))         # (n, N)
        W   = rng.chisquare(v_bar, size=(n, 1))               # (n, 1)
        r   = mu_bar[None, :] + (eps @ L_bar.T) / np.sqrt(W / v_bar)
        return r[0] if n == 1 else r


# ── Quick sanity-check (uses the real per-window belief) ────────────────────────
if __name__ == "__main__":
    from train import load_market_data, _make_env, _date_idx, WINDOWS
    from regime import regime_belief

    data   = load_market_data()
    ref    = _make_env(data)
    w0     = WINDOWS[0]
    tr0    = max(60, _date_idx(ref, w0["train_start"]))
    tr1    = _date_idx(ref, w0["train_end"])
    belief, _ = regime_belief(data, tr0, tr1)

    R = data[1].values.astype(np.float64)[tr0:tr1]            # train sector returns
    B = belief[tr0:tr1]

    model = NonStationaryReturnModel(n_assets=R.shape[1]).fit(R, B)
    print("regime mean (bps/day):\n", (model.mu_k * 1e4).round(2))
    print("regime excess kurtosis:", model.exkurt_k.round(2))
    print("regime dof v_k        :", model.nu_k.round(1))

    b_t = belief[tr1]                                         # belief just after train
    mu_bar, Sig_bar, L_bar, v_bar = model.mixture(b_t)
    print(f"\nb_t = {b_t.round(3)}   v_bar = {v_bar:.1f}")
    print("mu_bar (bps/day):", (mu_bar * 1e4).round(2))
    rng = np.random.default_rng(0)
    draws = model.sample_next(b_t, rng, n=100000)
    print("sample mean (bps):", (draws.mean(0) * 1e4).round(2))
    print("sample vs model std ratio:",
          (draws.std(0) / np.sqrt(np.diag(Sig_bar) * v_bar / (v_bar - 2))).round(3))
