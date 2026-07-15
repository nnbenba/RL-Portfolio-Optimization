"""
Smoke test: exercise the whole NSMDP pipeline end-to-end at tiny scale.
Not a training run — validates shapes, wiring, and that one PPO update + a short
eval execute without error. Run from inside Updated/:  python smoke_test.py
"""
import numpy as np
import torch

import train
from train import (load_market_data, _make_env, _date_idx, WINDOWS,
                   collect_rollouts, compute_gae, ppo_update, evaluate)
from networks import PolicyNetwork, ValueNetwork
from environment import STATE_DIM, N_ASSETS, LOOKBACK, N_SECTORS
from regime import regime_belief
from nonstationary import NonStationaryReturnModel

torch.manual_seed(0); np.random.seed(0)
device = torch.device("cpu")

# Shrink the PPO rollout for a fast smoke (module-level globals read by the funcs)
train.N_STEPS   = 8
train.BATCH_SIZE = 8
train.N_EPOCHS  = 2
N_ENVS = 3

print("1) load market data ...")
data = load_market_data()
ref  = _make_env(data)
print(f"   env trading days: {len(ref.dates)}  ({ref.dates[0].date()} -> {ref.dates[-1].date()})")

w0  = WINDOWS[0]
tr0 = max(LOOKBACK, _date_idx(ref, w0["train_start"]))
tr1 = _date_idx(ref, w0["train_end"])
va0 = max(LOOKBACK, _date_idx(ref, w0["val_start"]))
va1 = _date_idx(ref, w0["val_end"])
print(f"   window0 train[{tr0}:{tr1}] val[{va0}:{va1}]")

print("2) regime belief (HMM, leak-free) ...")
belief, ent = regime_belief(data, tr0, tr1)
assert belief.shape == (len(ref.dates), 3)
assert np.allclose(belief[tr0:tr1].sum(1), 1.0, atol=1e-4)
print(f"   belief OK {belief.shape}, train-mean b_t={belief[tr0:tr1].mean(0).round(3)}")

print("3) non-stationary Student-t generator ...")
R_tr = data[1].values.astype(np.float64)[tr0:tr1]
ns   = NonStationaryReturnModel(n_assets=R_tr.shape[1]).fit(R_tr, belief[tr0:tr1])
print(f"   mu_k shape {ns.mu_k.shape}, cov_k {ns.cov_k.shape}, v_k={ns.nu_k.round(1)}")
r_hat = ns.sample_next(belief[tr1], np.random.default_rng(0))
assert r_hat.shape == (N_SECTORS,)
print(f"   r_hat sample {r_hat.shape}: {(r_hat*1e4).round(1)} bps")

print("4) env state shape + stochastic append ...")
env = _make_env(data, belief, ns, seed=0)
s   = env.reset(tr0)
assert s.shape == (STATE_DIM,), s.shape
env.rng = np.random.default_rng(1); s1 = env._build_state()
env.rng = np.random.default_rng(2); s2 = env._build_state()
diff = not np.allclose(s1, s2)
print(f"   state {s.shape} == STATE_DIM({STATE_DIM}) OK | r_hat resampled per build: {diff}")
assert diff, "appended r_hat should differ across RNG draws"

print("5) one PPO iteration (collect -> GAE -> update) ...")
policy = PolicyNetwork().to(device); critic = ValueNetwork().to(device)
opt = torch.optim.Adam(list(policy.parameters()) + list(critic.parameters()), lr=3e-4)
envs = [_make_env(data, belief, ns, seed=i) for i in range(N_ENVS)]
env_states = [e.reset(np.random.randint(tr0, tr1)) for e in envs]
roll = collect_rollouts(envs, env_states, policy, critic, device, tr0, tr1)
assert roll["states"].shape == (train.N_STEPS, N_ENVS, STATE_DIM), roll["states"].shape
adv, ret = compute_gae(roll)
losses = ppo_update(policy, critic, opt, roll, adv, ret, device, 0.0)
print(f"   rollout states {roll['states'].shape} | losses "
      f"{ {k: round(v,4) for k,v in losses.items()} }")

print("6) short deterministic eval ...")
vr, pv = evaluate(data, policy, va0, min(va0 + 40, va1), device, belief, ns)
print(f"   eval reward={vr:+.3f}  final_port=${pv:,.0f}")

print("\nSMOKE TEST PASSED")
