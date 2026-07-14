"""
Mean-Variance Optimization baseline (paper Section 5.3 / 5.4) + full Table 2.

MVO: at each trading day use the trailing 60-day return window to estimate
  - expected returns = sample means
  - covariance = Ledoit-Wolf shrinkage, repaired to PSD (neg eigenvalues -> 0)
then solve the long-only max-Sharpe (tangency) portfolio and trade it in the
same market-replay environment. No training, no look-ahead (window is strictly
prior to the rebalance day). Metrics aggregated the paper's way and printed next
to the DRL agent and the paper's reported Table 2 numbers.
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
from scipy.stats import skew, kurtosis, linregress
from pypfopt import EfficientFrontier, risk_models, expected_returns

from networks import PolicyNetwork
from environment import T as LOOKBACK
from train import load_market_data, _make_env, _date_idx, WINDOWS, START_CASH, N_SEEDS

warnings.filterwarnings("ignore")

N_SEC = 9
LB    = 60   # MVO lookback (same as DRL, paper Sec 5.3)

data = load_market_data()
ref  = _make_env(data)
SECTORS = list(data[0].columns)   # 9 sector tickers, correctly labeled


# ── metrics (Pyfolio-style, daily, rf=0) ────────────────────────────────────────
def metrics(r):
    n = len(r); pv = START_CASH * np.cumprod(1 + r)
    tot  = pv[-1] / START_CASH - 1
    ann  = (pv[-1] / START_CASH) ** (252 / n) - 1
    vol  = r.std(ddof=1) * np.sqrt(252)
    shrp = r.mean() / r.std(ddof=1) * np.sqrt(252)
    peak = np.maximum.accumulate(pv); mdd = ((pv - peak) / peak).min()
    calmar = ann / abs(mdd) if mdd < 0 else np.nan
    cl = np.cumsum(np.log1p(r)); stab = linregress(np.arange(n), cl).rvalue ** 2
    d = r - 0.0; omega = d[d > 0].sum() / (-d[d < 0].sum())
    dd = np.sqrt(np.mean(np.minimum(r, 0) ** 2)); sortino = r.mean() / dd * np.sqrt(252)
    tail = abs(np.percentile(r, 95)) / abs(np.percentile(r, 5))
    var  = np.percentile(r, 5)
    return dict(ann_ret=ann, cum=tot, vol=vol, sharpe=shrp, calmar=calmar, stability=stab,
                mdd=mdd, omega=omega, sortino=sortino, skew=skew(r), kurt=kurtosis(r),
                tail=tail, var=var)


# ── DRL rollout ─────────────────────────────────────────────────────────────────
def run_drl(policy, s, e):
    env = _make_env(data); st = env.reset(s); r = []; done = False
    while env.t < e and not done:
        with torch.no_grad():
            w = policy(torch.tensor(st).unsqueeze(0)).squeeze(0).numpy()
        st, rew, done, info = env.step(w); r.append(info["R_t"])
    return np.array(r)


# ── MVO: long-only max-Sharpe via PyPortfolioOpt (paper Sec 5.3) ────────────────
def mvo_weights(price_window: pd.DataFrame) -> np.ndarray:
    """Sample-mean returns + Ledoit-Wolf shrunk covariance (PSD-repaired) ->
    long-only max-Sharpe weights, exactly the paper's PyPortfolioOpt pipeline."""
    mu = expected_returns.mean_historical_return(price_window)           # sample means
    S  = risk_models.CovarianceShrinkage(price_window).ledoit_wolf()     # Ledoit-Wolf
    S  = risk_models.fix_nonpositive_semidefinite(S, fix_method="spectral")
    try:
        ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
        ef.max_sharpe(risk_free_rate=0.0)
        w = ef.clean_weights()
    except Exception:                    # e.g. all expected returns <= 0 -> fall back
        ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
        ef.min_volatility()
        w = ef.clean_weights()
    return np.array([w[c] for c in price_window.columns], dtype=np.float32)


def run_mvo(s, e):
    env = _make_env(data); st = env.reset(s); r = []; tn = []; done = False
    while env.t < e and not done:
        t = env.t
        pw = pd.DataFrame(env.prices[t - LB:t], columns=SECTORS, index=env.dates[t - LB:t])
        w = np.zeros(10, dtype=np.float32); w[:N_SEC] = mvo_weights(pw)
        st, rew, done, info = env.step(w); r.append(info["R_t"]); tn.append(info["turnover"])
    return np.array(r), np.array(tn)


def aggregate(per_window):
    keys = ["ann_ret", "cum", "vol", "sharpe", "calmar", "stability", "mdd",
            "omega", "sortino", "skew", "kurt", "tail", "var"]
    agg = {k: float(np.nanmean([w[k] for w in per_window])) for k in keys}
    agg["mdd"] = min(w["mdd"] for w in per_window)   # worst period (paper caption)
    return agg


def main():
    drl_pw, mvo_pw = [], []
    mvo_daily = []                               # stitched daily MVO returns (for compare.py)
    for i, w in enumerate(WINDOWS):
        s = max(LOOKBACK, _date_idx(ref, w["test_start"])); e = _date_idx(ref, w["test_end"])
        # DRL: average 5 seeds
        seedm = []
        for sd in range(N_SEEDS):
            p = f"checkpoints/window_{i:02d}_seed_{sd}.pt"
            if not os.path.exists(p):
                continue
            pol = PolicyNetwork(); pol.load_state_dict(torch.load(p, map_location="cpu")["state_dict"]); pol.eval()
            seedm.append(metrics(run_drl(pol, s, e)))
        drl_pw.append({k: np.mean([m[k] for m in seedm]) for k in seedm[0]})
        # MVO: single deterministic run
        r_mvo, t_mvo = run_mvo(s, e)
        mvo_pw.append(metrics(r_mvo))
        mvo_daily.append(pd.DataFrame({"R_t": r_mvo, "turnover": t_mvo,
                                       "window": w["test_start"][:4]}))
        print(f"  window {i:02d} ({w['test_start'][:4]}) done")

    # Export the daily OOS MVO returns, tagged by test window, for compare.py
    pd.concat(mvo_daily, ignore_index=True).to_csv("mvo_returns.csv", index=False)

    drl, mvo = aggregate(drl_pw), aggregate(mvo_pw)
    paper_drl = dict(ann_ret=.1211, cum=.1195, vol=.1249, sharpe=1.1662, calmar=2.3133,
                     stability=.6234, mdd=-.3296, omega=1.2360, sortino=1.7208,
                     skew=-.4063, kurt=2.7054, tail=1.0423, var=-.0152)
    paper_mvo = dict(ann_ret=.0653, cum=.0650, vol=.1460, sharpe=.6776, calmar=1.1608,
                     stability=.4841, mdd=-.3303, omega=1.1315, sortino=1.0060,
                     skew=-.3328, kurt=2.6801, tail=.9448, var=-.0181)
    names = dict(ann_ret="Annual return", cum="Cumulative returns", vol="Annual volatility",
                 sharpe="Sharpe ratio", calmar="Calmar ratio", stability="Stability",
                 mdd="Max drawdown", omega="Omega ratio", sortino="Sortino ratio",
                 skew="Skew", kurt="Kurtosis", tail="Tail ratio", var="Daily value at risk")
    keys = list(names)
    print("\n" + "=" * 74)
    print(f"{'Metric':<22}{'DRL (ours)':>13}{'DRL (paper)':>13}{'MVO (ours)':>13}{'MVO (paper)':>13}")
    print("-" * 74)
    for k in keys:
        print(f"{names[k]:<22}{drl[k]:>13.4f}{paper_drl[k]:>13.4f}{mvo[k]:>13.4f}{paper_mvo[k]:>13.4f}")


if __name__ == "__main__":
    main()
