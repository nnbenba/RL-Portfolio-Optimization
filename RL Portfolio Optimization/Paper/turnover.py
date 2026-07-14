"""Measure Baseline PPO daily turnover Sum_i|w_target - w_held| over the OOS test
spans, averaged across all seeds/windows. Cost-relevant: how much the policy moves
weights each day (what a transaction cost would charge)."""
import os
import numpy as np
import torch
from networks import PolicyNetwork
from environment import T as LOOKBACK
from train import load_market_data, _make_env, _date_idx, WINDOWS, CHECKPOINT_DIR, N_SEEDS

data = load_market_data()
ref  = _make_env(data)
per_seed = []
for i, w in enumerate(WINDOWS):
    s = max(LOOKBACK, _date_idx(ref, w["test_start"])); e = _date_idx(ref, w["test_end"])
    for sd in range(N_SEEDS):
        p = os.path.join(CHECKPOINT_DIR, f"window_{i:02d}_seed_{sd}.pt")
        if not os.path.exists(p):
            continue
        pol = PolicyNetwork(); pol.load_state_dict(torch.load(p, map_location="cpu")["state_dict"]); pol.eval()
        env = _make_env(data); st = env.reset(s); deltas = []
        while env.t < e:
            with torch.no_grad():
                wv = pol(torch.tensor(st).unsqueeze(0)).squeeze(0).numpy()
            deltas.append(float(np.abs(wv - env.weights).sum()))   # target vs currently-held
            st, _, done, _ = env.step(wv)
            if done:
                break
        per_seed.append(np.mean(deltas[1:]))   # drop day-1 (all-cash -> first target)

per_seed = np.array(per_seed)
print(f"BASELINE PPO  daily turnover Sum|dw|:  mean={per_seed.mean():.4f}  "
      f"std={per_seed.std():.4f}  (n={len(per_seed)} seed-windows)")
print(f"  annualized turnover (x252): {per_seed.mean()*252:.1f}x")
