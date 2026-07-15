"""Measure the finished NSMDP (563-dim) daily turnover Sum_i|w_target - w_held|,
same definition as Baseline/turnover.py. The 563 env was overwritten (now 566), so
we rebuild the 563 state inline and reuse the unchanged rebalance/drift helpers."""
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
CKPT_DIR = "checkpoints_563_nsmdp"

data = load_market_data()
sector_prices, log_returns, vix, vol20, vol60 = data
dates   = log_returns.index
prices  = sector_prices.reindex(dates).values.astype(np.float32)
log_ret = log_returns.reindex(dates).values.astype(np.float32)
R_all   = log_returns.values.astype(np.float64)


def didx(ds): return int(dates.searchsorted(pd.Timestamp(ds)))


def load_net(state_dict, in_dim):
    """Rebuild the [in_dim,64,64,10] tanh policy body and load its weights."""
    net = nn.Sequential(nn.Linear(in_dim, 64), nn.Tanh(),
                        nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, N_ASSETS))
    net.load_state_dict({k[4:]: v for k, v in state_dict.items() if k.startswith("net.")})
    net.eval(); return net


def build_state_563(t, weights, belief, ns, rng):
    R = log_ret[t - LOOKBACK:t]                                  # (60, 9)
    r_hat = np.atleast_2d(ns.sample_next(belief[t], rng)).astype(np.float32)
    R_aug = np.vstack([R, r_hat])                                # (61, 9)
    b = belief[t]; H = float(-(b * np.log(b + 1e-12)).sum() / np.log(len(b)))
    return np.concatenate([R_aug.reshape(-1), weights.astype(np.float32),
                           b, np.float32([H])])                  # 549+10+3+1 = 563


per_seed = []
for i, w in enumerate(WINDOWS):
    tr0 = max(LOOKBACK, didx(w["train_start"])); tr1 = didx(w["train_end"])
    s   = max(LOOKBACK, didx(w["test_start"]))
    e   = min(didx(w["test_end"]), len(dates) - 1)   # cap like the env (t stops at n_days-1)
    belief, _ = regime_belief(data, tr0, tr1)
    ns = NonStationaryReturnModel(n_assets=N_SECTORS).fit(R_all[tr0:tr1], belief[tr0:tr1])
    for sd in range(N_SEEDS):
        p = os.path.join(CKPT_DIR, f"window_{i:02d}_seed_{sd}.pt")
        if not os.path.exists(p):
            continue
        net = load_net(torch.load(p, map_location="cpu")["state_dict"], 563)
        rng = np.random.default_rng(0)                           # match backtest eval seed
        weights = np.zeros(N_ASSETS, dtype=np.float32); weights[-1] = 1.0
        port = 100_000.0; t = s; deltas = []
        while t < e:
            state = build_state_563(t, weights, belief, ns, rng)
            with torch.no_grad():
                wv = torch.softmax(net(torch.tensor(state).unsqueeze(0)), -1).squeeze(0).numpy()
            deltas.append(float(np.abs(wv - weights).sum()))     # target vs currently-held
            shares, cash = _rebalance(wv, prices[t], port, N_SECTORS)
            t += 1
            port = float(shares @ prices[t]) + cash
            weights = _observed_weights(shares, prices[t], cash, port, N_ASSETS).astype(np.float32)
        per_seed.append(np.mean(deltas[1:]))

per_seed = np.array(per_seed)
print(f"NSMDP-563     daily turnover Sum|dw|:  mean={per_seed.mean():.4f}  "
      f"std={per_seed.std():.4f}  (n={len(per_seed)} seed-windows)")
print(f"  annualized turnover (x252): {per_seed.mean()*252:.1f}x")
