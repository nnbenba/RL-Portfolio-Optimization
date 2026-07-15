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
from environment import LOOKBACK
from regime import regime_belief
from nonstationary import NonStationaryReturnModel
from train import (
    load_market_data, _make_env, _date_idx, WINDOWS, START_CASH, CHECKPOINT_DIR, N_SEEDS,
)

ANN    = 252     # trading days per year for annualization
C_EVAL = 0.0005  # transaction cost c per unit turnover (= lambda = 5 bps) for NET grading


# ── One pass over a test span, returns daily date / port_val / R_t / D_t ────────

def run_span(data, weight_fn, start_idx, end_idx, device, belief=None, ns_model=None, seed=0):
    env   = _make_env(data, belief, ns_model, seed=seed)   # seed fixes the r_hat draws
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

    hdr = (f"{'window/yr':<10} {'days':>4}  {'tot_ret':>8}  {'cagr':>8}  "
           f"{'vol':>7}  {'shrp':>6}  {'sort':>6}  {'maxDD':>7}  {'win':>6}  {'final_val':>12}")

    # ── Precompute per-window context (seed-independent): belief, ns_model, loaded
    #    agents, test span, and the equal-weight / S&P slices (both ignore r_hat) ─
    ctx = []
    for i, w in enumerate(WINDOWS):
        s_idx = max(LOOKBACK, _date_idx(ref_env, w["test_start"]))
        e_idx = _date_idx(ref_env, w["test_end"])
        test_year = w["test_start"][:4]

        tr0 = max(LOOKBACK, _date_idx(ref_env, w["train_start"]))
        tr1 = _date_idx(ref_env, w["train_end"])
        belief, _ = regime_belief(data, tr0, tr1)          # leak-free, per-window
        R_tr = data[1].values.astype(np.float64)[tr0:tr1]
        ns_model = NonStationaryReturnModel(n_assets=R_tr.shape[1]).fit(R_tr, belief[tr0:tr1])

        seed_paths = [os.path.join(CHECKPOINT_DIR, f"window_{i:02d}_seed_{sd}.pt")
                      for sd in range(N_SEEDS)]
        seed_paths = [p for p in seed_paths if os.path.exists(p)]
        if not seed_paths:
            single = os.path.join(CHECKPOINT_DIR, f"window_{i:02d}.pt")
            if not os.path.exists(single):
                print(f"!! no checkpoints for window {i}, skipping")
                continue
            seed_paths = [single]
        policies = []
        for p in seed_paths:
            ck = torch.load(p, map_location=device)
            pol = PolicyNetwork().to(device)
            pol.load_state_dict(ck["state_dict"]); pol.eval()
            policies.append(pol)

        edf, _   = run_span(data, equal_weight_fn, s_idx, e_idx, device)   # r_hat-independent
        em       = metrics(edf)
        sp_slice = sp.loc[edf.index]
        ctx.append(dict(year=test_year, s_idx=s_idx, e_idx=e_idx, belief=belief,
                        ns_model=ns_model, policies=policies, edf=edf, em=em,
                        sp_slice=sp_slice))
        print(f"  prepared window {i:02d} ({test_year}): {len(policies)} agents")

    # ── Multi-seed evaluation: repeat the whole walk-forward once per eval seed.
    #    Each seed = one realization of the appended r_hat_{t+1} draws. The market
    #    P&L in step() is identical; only the observed forecast feature changes. ─
    EVAL_SEEDS       = [0, 1, 2, 3, 4]
    full_by_seed     = []         # full-period GROSS metrics, one per eval seed
    full_by_seed_net = []         # full-period NET-of-cost metrics (c = C_EVAL)
    per_window_s0    = []         # per-window GROSS metrics for the first eval seed
    seed_returns     = {}         # stitched daily OOS gross returns per seed (for compare.py)
    seed_turnover    = {}         # stitched daily ensemble turnover per seed (for net grading)
    win_tag_parts    = []         # per-row test-window label (for per-window-averaged metrics)
    stitch0          = None
    for si, es in enumerate(EVAL_SEEDS):
        strat_returns, strat_returns_net, strat_turn = [], [], []
        for c in ctx:
            seed_daily, seed_daily_net, seed_turn, seed_metrics = [], [], [], []
            for pol in c["policies"]:
                sdf, sD = run_span(data, policy_weight_fn(pol, device),
                                   c["s_idx"], c["e_idx"], device,
                                   c["belief"], c["ns_model"], seed=es)
                seed_daily.append(sdf["R_t"].rename(None))
                # net-of-cost daily return for THIS agent: R_t - c * turnover
                seed_daily_net.append((sdf["R_t"] - C_EVAL * sdf["turnover"]).rename(None))
                seed_turn.append(sdf["turnover"].rename(None))
                seed_metrics.append(metrics(sdf, sD))
            ens      = pd.concat(seed_daily, axis=1).mean(axis=1)      # agent ensemble (gross)
            ens_net  = pd.concat(seed_daily_net, axis=1).mean(axis=1)  # agent ensemble (net)
            ens_turn = pd.concat(seed_turn, axis=1).mean(axis=1)       # agent ensemble (turnover)
            strat_returns.append(ens)
            strat_returns_net.append(ens_net)
            strat_turn.append(ens_turn)
            if si == 0:
                sm = {k: float(np.nanmean([m[k] for m in seed_metrics])) for k in seed_metrics[0]}
                sm["days"] = int(round(sm["days"]))
                sp_final = float((START_CASH * (c["sp_slice"] / c["sp_slice"].iloc[0])).iloc[-1])
                per_window_s0.append((c["year"], sm, c["em"], sp_final))
                win_tag_parts.append(pd.Series(c["year"], index=ens.index))

        sr     = pd.concat(strat_returns)
        sr_net = pd.concat(strat_returns_net)
        seed_returns[f"seed{es}"]  = sr
        seed_turnover[f"seed{es}"] = pd.concat(strat_turn)
        strat_eq = START_CASH * (1 + sr).cumprod()
        net_eq   = START_CASH * (1 + sr_net).cumprod()
        fm     = metrics(pd.DataFrame({"port_val": strat_eq.values, "R_t": sr.values},     index=sr.index))
        fm_net = metrics(pd.DataFrame({"port_val": net_eq.values,   "R_t": sr_net.values}, index=sr_net.index))
        full_by_seed.append(fm)
        full_by_seed_net.append(fm_net)
        if si == 0:
            stitch0 = strat_eq
        print(f"  eval seed {es}:  GROSS Sharpe={fm['sharpe']:.3f} cagr={fm['cagr']*100:+.2f}%   |   "
              f"NET@{C_EVAL*1e4:.0f}bp Sharpe={fm_net['sharpe']:.3f} cagr={fm_net['cagr']*100:+.2f}% "
              f"maxDD={fm_net['max_dd']*100:.1f}%")

    # ── Per-window table (first eval seed) ──────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"PER-WINDOW OUT-OF-SAMPLE  (RL = 5-agent ensemble, eval seed {EVAL_SEEDS[0]})")
    print("=" * 100 + "\n" + hdr)
    for yr, sm, em, _ in per_window_s0:
        print(fmt_row(f"RL {yr}", sm))
        print(fmt_row(f"EW {yr}", em))

    # ── Stitched continuous curve (RL seed 0; EW & S&P are r_hat-independent) ────
    er  = pd.concat([c["edf"]["R_t"] for c in ctx])
    spr = pd.concat([c["sp_slice"].pct_change().fillna(0.0) for c in ctx])
    ew_eq = START_CASH * (1 + er).cumprod()
    sp_eq = START_CASH * (1 + spr).cumprod()
    full_ew = metrics(pd.DataFrame({"port_val": ew_eq.values, "R_t": er.values}, index=er.index))
    full_sp = metrics(pd.DataFrame({"port_val": sp_eq.values, "R_t": spr.values}, index=spr.index))

    print("\n" + "=" * 100)
    print("FULL WALK-FORWARD BACKTEST  2012 -> 2021  (continuous, compounded from $100k)")
    print("=" * 100 + "\n" + hdr)
    print(fmt_row(f"RL seed{EVAL_SEEDS[0]}", full_by_seed[0]))
    print(fmt_row("EqualWt", full_ew))
    print(fmt_row("S&P500 B&H", full_sp))

    # ── Distribution of full-period RL metrics across the eval seeds (mean ± std)
    #    i.e. how much the score wobbles with the r_hat_{t+1} sampling. ──────────
    dist_keys = [("total_return", "tot_ret", 100, "%"), ("cagr", "cagr", 100, "%"),
                 ("ann_vol", "vol", 100, "%"), ("sharpe", "Sharpe", 1, ""),
                 ("sortino", "Sortino", 1, ""), ("max_dd", "maxDD", 100, "%")]
    print(f"\nFull-period 2012->2021 across {len(EVAL_SEEDS)} eval seeds "
          f"(r_hat_{{t+1}} sampling), mean ± std:")
    print(f"  {'metric':<9}{'GROSS':>18}{'NET @ ' + str(int(C_EVAL*1e4)) + 'bp':>18}")
    for k, lbl, sc, u in dist_keys:
        g = np.array([fm[k]  for fm in full_by_seed])     * sc
        n = np.array([fm[k]  for fm in full_by_seed_net]) * sc
        print(f"  {lbl:<9}{g.mean():>10.3f}±{g.std():<6.3f}{n.mean():>10.3f}±{n.std():<6.3f}  {u}")
    print(f"\n  (NET subtracts c={C_EVAL*1e4:.0f}bp * turnover per step. A cost-UNAWARE agent's"
          f"\n   net collapses vs its gross — that gap is the cost of ignoring turnover.)")

    # ── Per-window summary CSV (first eval seed) ────────────────────────────────
    rows = []
    for yr, sm, em, sp_final in per_window_s0:
        rows.append({
            "test_year": yr,
            "rl_total_return": sm["total_return"], "rl_sharpe": sm["sharpe"],
            "rl_max_dd": sm["max_dd"], "rl_final": sm["final_val"], "rl_D_reward": sm.get("D_t_reward"),
            "ew_total_return": em["total_return"], "ew_sharpe": em["sharpe"], "ew_final": em["final_val"],
            "sp_final": sp_final,
        })
    pd.DataFrame(rows).to_csv("backtest_results.csv", index=False)

    # Full-period metric distribution across eval seeds -> CSV
    pd.DataFrame([{"eval_seed": es, **{k: full_by_seed[i][k] for k in
                   ["total_return", "cagr", "ann_vol", "sharpe", "sortino", "max_dd"]}}
                  for i, es in enumerate(EVAL_SEEDS)]).to_csv("backtest_eval_seeds.csv", index=False)
    # Unified export for compare.py: [window, R_t, turnover], each averaged over the
    # eval seeds (R_t = mean gross ensemble return, turnover = mean ensemble turnover).
    _up = pd.DataFrame({
        "window":   pd.concat(win_tag_parts),
        "R_t":      pd.DataFrame(seed_returns).mean(axis=1),
        "turnover": pd.DataFrame(seed_turnover).mean(axis=1),
    })
    _up.to_csv("updated_ppo_returns.csv")
    print("\nSaved per-window metrics -> backtest_results.csv")
    print("Saved eval-seed distribution -> backtest_eval_seeds.csv")
    print("Saved daily returns per eval seed -> updated_ppo_returns.csv")

    # ── Plot (RL seed 0 curve) ──────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stitch0.index, stitch0.values, label=f"RL ensemble (seed {EVAL_SEEDS[0]})", lw=1.8, color="C0")
    ax.plot(ew_eq.index,   ew_eq.values,   label="Equal-weight sectors", lw=1.3, color="C1", alpha=0.85)
    ax.plot(sp_eq.index,   sp_eq.values,   label="S&P 500 (buy & hold)", lw=1.3, color="C2", alpha=0.85)
    for yr, *_ in per_window_s0:
        ax.axvline(pd.Timestamp(f"{yr}-01-01"), color="0.85", lw=0.6, zorder=0)
    ax.set_title("Walk-forward out-of-sample portfolio value (2012–2021)")
    ax.set_ylabel("Portfolio value ($)"); ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("backtest_portfolio_value.png", dpi=130)
    print("Saved plot -> backtest_portfolio_value.png")


if __name__ == "__main__":
    main()
