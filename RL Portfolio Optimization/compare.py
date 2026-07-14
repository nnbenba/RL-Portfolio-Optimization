"""
Unified, apples-to-apples comparison of the strategies — GROSS and NET of cost.

Each backtest exports daily OOS returns (2012->2021) tagged by test window, plus a
per-day `turnover` column:
    Baseline/baseline_ppo_returns.csv   Baseline PPO   (5-agent daily ensemble)
    Updated/updated_ppo_returns.csv     Updated PPO    (ensemble, mean over eval seeds)
    Updated v2/updated_ppo_returns.csv  Updated v2     (turnover-penalized reward)
    Baseline/mvo_returns.csv            MVO

compare.py applies ONE metric function to every strategy under BOTH lenses
(per-window-averaged = paper Table 2; stitched-continuous = whole-decade curve),
and for each it reports GROSS and NET-of-cost, where

    R_net = R_t - c * turnover,   c = C_EVAL   (proportional transaction cost)

A cost-UNAWARE agent (trained frictionless) churns freely, so its NET collapses;
a cost-AWARE agent (trained with the turnover penalty) should hold up on NET.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, linregress

HERE       = os.path.dirname(os.path.abspath(__file__))
START_CASH = 100_000
C_EVAL     = 0.0005   # transaction cost c per unit turnover (= 5 bps)

SOURCES = [
    ("Paper",             os.path.join(HERE, "Paper", "baseline_ppo_returns.csv")),
    ("NSMDP PPO",         os.path.join(HERE, "NSMDP PPO", "updated_ppo_returns.csv")),
    ("NSMDP PPO + Trans", os.path.join(HERE, "NSMDP PPO + Trans", "updated_ppo_returns.csv")),
    ("MVO",               os.path.join(HERE, "Paper", "mvo_returns.csv")),
]

NAMES = dict(ann_ret="Annual return", vol="Annual volatility", sharpe="Sharpe ratio",
             sortino="Sortino ratio", calmar="Calmar ratio", mdd="Max drawdown",
             turn="Turnover/day")
PAPER_DRL = dict(ann_ret=.1211, vol=.1249, sharpe=1.1662, sortino=1.7208, calmar=2.3133, mdd=-.3296)
PAPER_MVO = dict(ann_ret=.0653, vol=.1460, sharpe=.6776, sortino=1.0060, calmar=1.1608, mdd=-.3303)


def metrics(r: np.ndarray, turn: np.ndarray | None = None) -> dict:
    r = np.asarray(r, dtype=float)
    n = len(r); pv = START_CASH * np.cumprod(1 + r)
    ann = (pv[-1] / START_CASH) ** (252 / n) - 1
    vol = r.std(ddof=1) * np.sqrt(252)
    shrp = r.mean() / r.std(ddof=1) * np.sqrt(252)
    peak = np.maximum.accumulate(pv); mdd = ((pv - peak) / peak).min()
    calmar = ann / abs(mdd) if mdd < 0 else np.nan
    dd = np.sqrt(np.mean(np.minimum(r, 0) ** 2)); sortino = r.mean() / dd * np.sqrt(252)
    return dict(ann_ret=ann, vol=vol, sharpe=shrp, sortino=sortino, calmar=calmar,
                mdd=mdd, turn=(float(np.mean(turn)) if turn is not None else np.nan))


def pw_avg(win, r, turn=None):
    """Per-window-averaged (paper Table 2): score each window, average; maxDD=worst."""
    win = np.asarray(win); r = np.asarray(r, float)
    mets = []
    for w in pd.unique(win):
        m = win == w
        mets.append(metrics(r[m], None if turn is None else np.asarray(turn)[m]))
    agg = {k: float(np.nanmean([mm[k] for mm in mets])) for k in NAMES}
    agg["mdd"] = float(min(mm["mdd"] for mm in mets))
    return agg


def load(path):
    df = pd.read_csv(path)
    r    = df["R_t"].values
    win  = df["window"].values
    turn = df["turnover"].values if "turnover" in df.columns else None
    r_net = r - C_EVAL * turn if turn is not None else None
    return dict(
        pw_gross=pw_avg(win, r, turn), st_gross=metrics(r, turn),
        pw_net=(pw_avg(win, r_net, turn) if turn is not None else None),
        st_net=(metrics(r_net, turn)     if turn is not None else None),
        has_net=turn is not None,
    )


def _cell(v):
    return f"{v:>11.4f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else f"{'—':>11}"


def print_table(title, cols):
    """cols: list of (name, dict_or_None)."""
    W = 13
    width = 20 + W * len(cols)
    print("\n" + "=" * width + f"\n{title}\n" + "=" * width)
    print(f"{'Metric':<20}" + "".join(f"{n:>{W}}" for n, _ in cols))
    print("-" * width)
    for k in NAMES:
        row = f"{NAMES[k]:<20}"
        for _, d in cols:
            row += f"{_cell(None if d is None else d.get(k))}"
        print(row)


def main():
    data = {}
    for title, path in SOURCES:
        if os.path.exists(path):
            data[title] = load(path)
        else:
            print(f"[pending] {os.path.relpath(path, HERE)}")
    if not data:
        print("\nNothing to compare — run the backtests to export the return CSVs first.")
        return
    order = [t for t, _ in SOURCES if t in data]

    # ── Per-window-averaged (paper methodology) ─────────────────────────────────
    cols = [(f"{t} (G)", data[t]["pw_gross"]) for t in order]
    cols += [("DRL paper", PAPER_DRL), ("MVO paper", PAPER_MVO)]
    print_table("PER-WINDOW-AVERAGED  GROSS  (paper Table 2 methodology)", cols)

    cols = []
    for t in order:
        cols.append((f"{t} (N)", data[t]["pw_net"]))
    print_table(f"PER-WINDOW-AVERAGED  NET @ {int(C_EVAL*1e4)}bp  (— = no turnover in CSV)", cols)

    # ── Stitched-continuous: gross vs net side by side ──────────────────────────
    cols = []
    for t in order:
        cols.append((f"{t} G", data[t]["st_gross"]))
        cols.append((f"{t} N", data[t]["st_net"]))
    print_table(f"STITCHED-CONTINUOUS 2012->2021   G=gross  N=net@{int(C_EVAL*1e4)}bp", cols)

    print(f"\nnote: NET = R_t - {int(C_EVAL*1e4)}bp * turnover. Cost-unaware agents (Baseline, Updated)")
    print("      churn ~1/day so their NET craters; a turnover-penalized agent (v2) should hold up.")


if __name__ == "__main__":
    main()
