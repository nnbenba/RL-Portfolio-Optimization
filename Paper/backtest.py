"""
Walk-forward out-of-sample backtest — one test year per window (paper Sec. 5.2).

Window i: train [2006+i, 2011+i), val [2011+i, 2012+i), TEST [2012+i, 2013+i).
For each window we load checkpoints/window_NN.pt (best seed by val reward),
run the policy deterministically over its 1-year test span, and report the
full metric set. The 10 test years stitch into a continuous 2012->2021
out-of-sample equity curve, benchmarked vs equal-weight sectors and B&H S&P 500.
"""

import os
import numpy as np
import pandas as pd
import torch
import yfinance as yf

from networks import PolicyNetwork, N_ASSETS
from environment import T as LOOKBACK
from train import (
    load_market_data, _make_env, _date_idx, WINDOWS, START_CASH, CHECKPOINT_DIR, N_SEEDS,
)

ANN = 1252  # trading days per year for annualization


# ── One pass over a test span, returns daily date / port_val / R_t / D_t ────────

def run_span(data, weight_fn, start_idx, end_idx, device):
    env   = _make_env(data)
    state = env.reset(start_idx)
    rows, total_D = [], 0.0
    done = False
    while env.t < end_idx and not done:
        w = weight_fn(state)
        state, reward, done, info = env.step(w)
        total_D += reward
        rows.append((info["date"], info["port_val"], info["R_t"], info["turnover"]))
    df = pd.DataFrame(rows, columns=["date", "port_val", "R_t", "turnover"]).set_index("date")
    return df, total_D


def policy_weight_fn(policy, device):
    def fn(state):
        st = torch.tensor(state, device=device).unsqueeze(0)
        with torch.no_grad():
            return policy(st).squeeze(0).cpu().numpy()
    return fn


def equal_weight_fn(state):
    w = np.zeros(N_ASSETS, dtype=np.float32)
    w[:-1] = 1.0 / (N_ASSETS - 1)   # 1/9 across sectors, fully invested
    return w


# ── Metrics from a daily return series ──────────────────────────────────────────

def metrics(df, total_D=None):
    r   = df["R_t"].values
    pv  = df["port_val"].values
    n   = len(r)
    tot = pv[-1] / START_CASH - 1.0
    cagr = (pv[-1] / START_CASH) ** (ANN / n) - 1.0 if n > 0 else 0.0
    vol  = r.std(ddof=1) * np.sqrt(ANN) if n > 1 else 0.0
    sharpe = (r.mean() / r.std(ddof=1) * np.sqrt(ANN)) if (n > 1 and r.std() > 0) else 0.0
    downside = r[r < 0]
    sortino = (r.mean() / downside.std(ddof=1) * np.sqrt(ANN)) if (len(downside) > 1 and downside.std() > 0) else np.nan
    peak = np.maximum.accumulate(pv)
    maxdd = float(((peak - pv) / peak).max()) if n > 0 else 0.0
    calmar = cagr / maxdd if maxdd > 0 else np.nan
    win = float((r > 0).mean()) if n > 0 else 0.0
    out = {
        "days": n,
        "total_return": tot,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": maxdd,
        "calmar": calmar,
        "win_rate": win,
        "final_val": pv[-1],
    }
    if total_D is not None:
        out["D_t_reward"] = total_D
    return out


def fmt_row(name, m):
    return (f"{name:<10} {m['days']:>4d}  {m['total_return']*100:>8.2f}%  "
            f"{m['cagr']*100:>8.2f}%  {m['ann_vol']*100:>7.2f}%  {m['sharpe']:>6.2f}  "
            f"{m['sortino'] if not np.isnan(m['sortino']) else float('nan'):>6.2f}  "
            f"{m['max_dd']*100:>7.2f}%  {m['win_rate']*100:>6.1f}%  ${m['final_val']:>11,.0f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\nLoading market data...")
    data = load_market_data()
    ref_env = _make_env(data)

    # S&P 500 for buy-and-hold benchmark, aligned to env trading days
    sp = yf.download("^GSPC", start="2006-01-01", end="2021-12-31",
                     auto_adjust=True, progress=False)["Close"].squeeze()
    sp = sp.reindex(ref_env.dates).ffill()

    strat_returns, ew_returns = [], []   # stitched daily R_t across all windows
    strat_turn = []                      # stitched daily ensemble turnover (for net grading)
    sp_pieces = []
    per_window = []

    hdr = (f"{'window/yr':<10} {'days':>4}  {'tot_ret':>8}  {'cagr':>8}  "
           f"{'vol':>7}  {'shrp':>6}  {'sort':>6}  {'maxDD':>7}  {'win':>6}  {'final_val':>12}")

    for i, w in enumerate(WINDOWS):
        s_idx = max(LOOKBACK, _date_idx(ref_env, w["test_start"]))
        e_idx = _date_idx(ref_env, w["test_end"])
        test_year = w["test_start"][:4]

        # Load all seed agents for this window (fall back to single best if absent)
        seed_paths = [os.path.join(CHECKPOINT_DIR, f"window_{i:02d}_seed_{sd}.pt")
                      for sd in range(N_SEEDS)]
        seed_paths = [p for p in seed_paths if os.path.exists(p)]
        if not seed_paths:
            single = os.path.join(CHECKPOINT_DIR, f"window_{i:02d}.pt")
            if not os.path.exists(single):
                print(f"!! no checkpoints for window {i}, skipping")
                continue
            seed_paths = [single]

        # Backtest each agent; average their per-year metrics (paper Sec. 6),
        # and equal-blend their daily returns into one ensemble curve.
        seed_metrics, seed_daily, seed_turn = [], [], []
        for p in seed_paths:
            ck = torch.load(p, map_location=device)
            policy = PolicyNetwork().to(device)
            policy.load_state_dict(ck["state_dict"])
            policy.eval()
            sdf, sD = run_span(data, policy_weight_fn(policy, device), s_idx, e_idx, device)
            seed_metrics.append(metrics(sdf, sD))
            seed_daily.append(sdf["R_t"].rename(None))
            seed_turn.append(sdf["turnover"].rename(None))

        sm = {k: float(np.nanmean([m[k] for m in seed_metrics])) for k in seed_metrics[0]}
        sm["days"] = int(round(sm["days"]))
        ens      = pd.concat(seed_daily, axis=1).mean(axis=1)   # 5-agent ensemble returns
        ens_turn = pd.concat(seed_turn,  axis=1).mean(axis=1)   # 5-agent ensemble turnover

        edf, _ = run_span(data, equal_weight_fn, s_idx, e_idx, device)
        em = metrics(edf)

        sp_slice = sp.loc[ens.index]
        sp_val = START_CASH * (sp_slice / sp_slice.iloc[0])

        per_window.append((test_year, sm, em, float(sp_val.iloc[-1])))
        strat_returns.append(ens); ew_returns.append(edf["R_t"]); strat_turn.append(ens_turn)
        sp_pieces.append(sp_slice.pct_change().fillna(0.0))

        print(f"\n=== Window {i:02d}  TEST {test_year} "
              f"[{w['test_start']} -> {w['test_end']}]  ({len(seed_paths)} agents averaged) ===")
        print(hdr)
        print(fmt_row(f"RL {test_year}", sm))
        print(fmt_row("EqualWt", em))

    # ── Stitched continuous 2012->2021 out-of-sample portfolio ──────────────────
    sr = pd.concat(strat_returns); er = pd.concat(ew_returns); spr = pd.concat(sp_pieces)
    # Export daily ensemble OOS returns + turnover, tagged by test window, for compare.py
    pd.concat([pd.DataFrame({"R_t": ens.values, "turnover": tn.values, "window": yr}, index=ens.index)
               for (yr, *_), ens, tn in zip(per_window, strat_returns, strat_turn)]
              ).to_csv("baseline_ppo_returns.csv")
    strat_eq = START_CASH * (1 + sr).cumprod()
    ew_eq    = START_CASH * (1 + er).cumprod()
    sp_eq    = START_CASH * (1 + spr).cumprod()

    full_strat = metrics(pd.DataFrame({"port_val": strat_eq.values, "R_t": sr.values}, index=sr.index))
    full_ew    = metrics(pd.DataFrame({"port_val": ew_eq.values,    "R_t": er.values},  index=er.index))
    full_sp    = metrics(pd.DataFrame({"port_val": sp_eq.values,    "R_t": spr.values}, index=spr.index))

    print("\n" + "=" * 100)
    print("FULL WALK-FORWARD BACKTEST  2012 -> 2021  (continuous, compounded from $100k)")
    print("=" * 100)
    print(hdr)
    print(fmt_row("RL ens", full_strat))
    print(fmt_row("EqualWt", full_ew))
    print(fmt_row("S&P500 B&H", full_sp))

    # Paper-style summary: average each metric across the 10 per-window results
    # (each already averaged across the 5 seeds).  Matches Table 2 methodology.
    keys = ["total_return", "cagr", "ann_vol", "sharpe", "sortino", "max_dd", "win_rate"]
    avg = {k: float(np.nanmean([sm[k] for _, sm, _, _ in per_window])) for k in keys}
    print("\nPaper-style (mean across 10 windows of the 5-seed-averaged metrics):")
    print(f"  annual return={avg['cagr']*100:+.2f}%   ann_vol={avg['ann_vol']*100:.2f}%   "
          f"Sharpe={avg['sharpe']:.3f}   Sortino={avg['sortino']:.3f}   "
          f"maxDD={avg['max_dd']*100:.1f}%   (paper DRL: ret 12.1%, Sharpe 1.17)")

    # ── Per-window summary CSV ──────────────────────────────────────────────────
    rows = []
    for yr, sm, em, sp_final in per_window:
        rows.append({
            "test_year": yr,
            "rl_total_return": sm["total_return"], "rl_sharpe": sm["sharpe"],
            "rl_max_dd": sm["max_dd"], "rl_final": sm["final_val"], "rl_D_reward": sm.get("D_t_reward"),
            "ew_total_return": em["total_return"], "ew_sharpe": em["sharpe"], "ew_final": em["final_val"],
            "sp_final": sp_final,
        })
    pd.DataFrame(rows).to_csv("backtest_results.csv", index=False)
    print("\nSaved per-window metrics -> backtest_results.csv")

    # ── Plot ────────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(strat_eq.index, strat_eq.values, label="RL ensemble (5 agents)", lw=1.8, color="C0")
    ax.plot(ew_eq.index,    ew_eq.values,    label="Equal-weight sectors", lw=1.3, color="C1", alpha=0.85)
    ax.plot(sp_eq.index,    sp_eq.values,    label="S&P 500 (buy & hold)", lw=1.3, color="C2", alpha=0.85)
    for yr, *_ in per_window:
        ax.axvline(pd.Timestamp(f"{yr}-01-01"), color="0.85", lw=0.6, zorder=0)
    ax.set_title("Walk-forward out-of-sample portfolio value (2012–2021)")
    ax.set_ylabel("Portfolio value ($)"); ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("backtest_portfolio_value.png", dpi=130)
    print("Saved plot -> backtest_portfolio_value.png")


if __name__ == "__main__":
    main()
