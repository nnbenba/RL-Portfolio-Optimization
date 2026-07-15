"""
Per-window Gaussian-HMM regime belief for the NSMDP state (leak-free).

For a given sliding window we fit a 3-state Gaussian HMM on ONLY that window's
train span (features [ret, vol20, vol20/vol60, VIX], standardized with train-only
stats), then run a causal forward filter from train_start onward to produce, for
every day t >= train_start, the belief

    b_t = P(regime_t | x_1..x_t)        (filtered, no look-ahead)

States are VIX-anchored (label 0=bear=highest-VIX ... 2=bull=lowest-VIX) so the
belief columns mean the same thing in every window. This is the same model used
in the standalone ../regimes.py, packaged to hand a (D,3) belief array to the env.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.special import logsumexp
from hmmlearn.hmm import GaussianHMM

N_REGIMES   = 3
SEED        = 42
VIX_FEATURE = 3   # column index of VIX in the feature vector

_FEAT_CACHE: dict = {}


def _features(data) -> np.ndarray:
    """[ret, vol20, vol20/vol60, VIX] aligned to the env date index (log_returns
    index). Cached per date-index so we download the S&P once per process."""
    log_returns, vix = data[1], data[2]
    idx = log_returns.index
    key = (idx[0], idx[-1], len(idx))
    if key in _FEAT_CACHE:
        return _FEAT_CACHE[key]

    sp = yf.download("^GSPC", start=str(idx[0].date()),
                     end=str((idx[-1] + pd.Timedelta(days=1)).date()),
                     auto_adjust=True, progress=False)["Close"].squeeze()
    r     = np.log(sp / sp.shift(1))
    vol20 = r.rolling(20).std()
    vol60 = r.rolling(60).std()
    feat  = pd.DataFrame({
        "ret":    r,
        "vol20":  vol20,
        "vratio": vol20 / vol60,
        "vix":    vix,
    }).reindex(idx).ffill().bfill()

    arr = feat.values.astype(np.float64)
    _FEAT_CACHE[key] = arr
    return arr


def _filtered(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Normalized forward algorithm -> filtered posteriors P(s_t | x_1..x_t)."""
    framell   = model._compute_log_likelihood(X)
    log_start = np.log(model.startprob_ + 1e-300)
    log_trans = np.log(model.transmat_ + 1e-300)
    T, K = framell.shape
    logf = np.empty((T, K))
    logf[0] = log_start + framell[0]
    logf[0] -= logsumexp(logf[0])
    for t in range(1, T):
        logf[t] = framell[t] + logsumexp(logf[t - 1][:, None] + log_trans, axis=0)
        logf[t] -= logsumexp(logf[t])
    return np.exp(logf)


def regime_belief(data, train_start: int, train_end: int):
    """Fit HMM on [train_start, train_end); filter forward to the end of the data.

    Returns (belief, entropy):
        belief  : (D, 3) float32, rows sum to 1, columns [bear, neutral, bull].
                  Days before train_start are uniform (never observed by the agent).
        entropy : (D,)  float32, normalized Shannon entropy of the belief in [0, 1].
    """
    X = _features(data)
    D = len(X)

    Xtr    = X[train_start:train_end]
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd     = np.where(sd < 1e-12, 1.0, sd)

    model = GaussianHMM(n_components=N_REGIMES, covariance_type="full",
                        n_iter=500, tol=1e-4, random_state=SEED)
    model.fit((Xtr - mu) / sd)

    raw_vix = model.means_[:, VIX_FEATURE] * sd[VIX_FEATURE] + mu[VIX_FEATURE]
    order   = np.argsort(raw_vix)[::-1]                 # high VIX (bear) -> low (bull)

    belief = np.full((D, N_REGIMES), 1.0 / N_REGIMES, dtype=np.float32)
    pf = _filtered(model, (X[train_start:] - mu) / sd)[:, order]
    belief[train_start:] = pf.astype(np.float32)

    ent = -(belief * np.log(belief + 1e-12)).sum(1) / np.log(N_REGIMES)
    return belief, ent.astype(np.float32)
