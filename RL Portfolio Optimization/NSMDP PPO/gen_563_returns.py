"""Regenerate the finished NSMDP-563 daily OOS returns + turnover (+ window) into
updated_ppo_returns.csv, in the unified compare.py format. The 563 env was
overwritten (now 566), so we rebuild the 563 state inline (eval seed 0)."""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from train import load_market_data, WINDOWS, N_SEEDS
from environment import _rebalance, _observed_weights, N_ASSETS, N_SECTORS
from regime import regime_belief
from nonstationary import NonStationaryReturnModel

LOOKBACK = 60
CKPT = "checkpoints_563_nsmdp"

data = load_market_data()
sector_prices, log_returns, vix, vol20, vol60 = data
dates   = log_returns.index
prices  = sector_prices.reindex(dates).values.astype(np.float32)
log_ret = log_returns.reindex(dates).values.astype(np.float32)
R_all   = log_returns.values.astype(np.float64)


def didx(ds): return int(dates.searchsorted(pd.Timestamp(ds)))


def load_net(sd, indim):
    net = nn.Sequential(nn.Linear(indim, 64), nn.Tanh(),
                        nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, N_ASSETS))
    net.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("net.")})
    net.eval(); return net


def build_state(t, weights, belief, ns, rng):
    R = log_ret[t - LOOKBACK:t]
    rhat = np.atleast_2d(ns.sample_next(belief[t], rng)).astype(np.float32)
    Raug = np.vstack([R, rhat]); b = belief[t]
    H = float(-(b * np.log(b + 1e-12)).sum() / np.log(len(b)))
    return np.concatenate([Raug.reshape(-1), weights.astype(np.float32), b, np.float32([H])])


pieces = []
for i, w in enumerate(WINDOWS):
    tr0 = max(LOOKBACK, didx(w["train_start"])); tr1 = didx(w["train_end"])
    s   = max(LOOKBACK, didx(w["test_start"]));  e = min(didx(w["test_end"]), len(dates) - 1)
    belief, _ = regime_belief(data, tr0, tr1)
    ns = NonStationaryReturnModel(n_assets=N_SECTORS).fit(R_all[tr0:tr1], belief[tr0:tr1])

    agentR, agentT, idx_dates = [], [], None
    for sd in range(N_SEEDS):
        p = os.path.join(CKPT, f"window_{i:02d}_seed_{sd}.pt")
        if not os.path.exists(p):
            continue
        net = load_net(torch.load(p, map_location="cpu")["state_dict"], 563)
        rng = np.random.default_rng(0)
        weights = np.zeros(N_ASSETS, dtype=np.float32); weights[-1] = 1.0
        port = 100_000.0; t = s; Rs, Ts, ds = [], [], []
        while t < e:
            st = build_state(t, weights, belief, ns, rng)
            with torch.no_grad():
                wv = torch.softmax(net(torch.tensor(st).unsqueeze(0)), -1).squeeze(0).numpy()
            Ts.append(float(np.abs(wv - weights).sum()))
            shares, cash = _rebalance(wv, prices[t], port, N_SECTORS)
            t += 1
            new = float(shares @ prices[t]) + cash
            Rs.append((new - port) / port); port = new
            weights = _observed_weights(shares, prices[t], cash, new, N_ASSETS).astype(np.float32)
            ds.append(dates[t])
        agentR.append(Rs); agentT.append(Ts); idx_dates = ds

    ensR = np.mean(agentR, axis=0); ensT = np.mean(agentT, axis=0)
    pieces.append(pd.DataFrame({"R_t": ensR, "turnover": ensT, "window": w["test_start"][:4]},
                               index=pd.DatetimeIndex(idx_dates)))
    print(f"  window {i:02d} ({w['test_start'][:4]}) done")

out = pd.concat(pieces); out.index.name = "date"
out.to_csv("updated_ppo_returns.csv")
print(f"wrote {len(out)} days -> updated_ppo_returns.csv (turnover/day mean {out['turnover'].mean():.3f})")
