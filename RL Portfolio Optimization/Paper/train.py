"""
PPO training loop — J.P. Morgan paper Section 5.2.

Sliding-window schedule (10 windows):
  5yr train / 1yr val / 1yr test, offset by 1yr each (2006-2021).
  5 seeds per window, 7.5M timesteps each, 10 parallel envs.
  Best seed (by val cumulative D_t reward) warm-starts the next window.
"""

import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yfinance as yf

from networks import PolicyNetwork, ValueNetwork, N_ASSETS, T as LOOKBACK
from environment import PortfolioEnv

# ── Hyperparameters (paper Table 1 / Section 5.2) ─────────────────────────────
N_ENVS          = 10
N_STEPS         = 756         # steps per env per rollout → 7560 total transitions
BATCH_SIZE      = 1260        # mini-batch size; 7560 / 1260 = 6 mini-batches/epoch
N_EPOCHS        = 16
CLIP_EPS        = 0.25
GAMMA           = 0.9
GAE_LAMBDA      = 0.9
LR_START        = 3e-4
LR_END          = 1e-5
TOTAL_TIMESTEPS = 7_500_000
N_SEEDS         = 5
ENTROPY_COEF    = 0.0     # SB3 default; paper's Table 1 omits it (i.e. uses 0.0)
VALUE_COEF      = 0.5
MAX_GRAD_NORM   = 0.5
START_CASH      = 100_000
CHECKPOINT_DIR  = "checkpoints"

# ── 10 sliding windows ─────────────────────────────────────────────────────────
# Window i: train [2006+i, 2011+i), val [2011+i, 2012+i), test [2012+i, 2013+i)
WINDOWS = [
    {
        "train_start": f"{2006 + i}-01-01",
        "train_end":   f"{2011 + i}-01-01",
        "val_start":   f"{2011 + i}-01-01",
        "val_end":     f"{2012 + i}-01-01",
        "test_start":  f"{2012 + i}-01-01",
        "test_end":    f"{2013 + i}-01-01",
    }
    for i in range(10)
]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_market_data():
    """Download and preprocess all market data. Called once before training."""
    tickers = ["XLB", "XLI", "XLY", "XLP", "XLV", "XLF", "XLK", "XLU", "XLE", "^VIX"]
    raw     = yf.download(tickers, start="2006-01-01", end="2021-12-31",
                          auto_adjust=True, progress=False)
    # yfinance returns Close columns sorted alphabetically, NOT in `tickers` order.
    # Select by NAME (never positionally) so sectors keep their correct identities.
    prices        = raw["Close"][tickers]

    vix           = prices["^VIX"]
    sector_prices = prices.drop(columns=["^VIX"])
    log_returns   = np.log(sector_prices / sector_prices.shift(1)).dropna()

    sp500    = yf.download("^GSPC", start="2006-01-01", end="2021-12-31",
                           auto_adjust=True, progress=False)["Close"].squeeze()
    sp500_lr = np.log(sp500 / sp500.shift(1)).dropna()
    vol20    = sp500_lr.rolling(20).std()
    vol60    = sp500_lr.rolling(60).std()

    print(f"Loaded {len(log_returns)} trading days, {sector_prices.shape[1]} sectors.")
    return sector_prices, log_returns, vix, vol20, vol60


def _make_env(data):
    sector_prices, log_returns, vix, vol20, vol60 = data
    return PortfolioEnv(sector_prices, log_returns, vix, vol20, vol60, START_CASH)


def _date_idx(env, date_str):
    """First trading-day index in env.dates that is >= date_str."""
    return int(env.dates.searchsorted(pd.Timestamp(date_str)))


# ── Rollout collection ─────────────────────────────────────────────────────────

def collect_rollouts(envs, env_states, policy, critic, device, train_start, train_end):
    """
    Run each env for N_STEPS steps. Resets to random training position on episode end.
    env_states is mutated in-place so state persists across consecutive rollout calls.
    Returns raw numpy transition arrays.
    """
    n = len(envs)
    states_b  = np.zeros((N_STEPS, n, N_ASSETS, LOOKBACK), dtype=np.float32)  # (756,10,10,60)
    actions_b = np.zeros((N_STEPS, n, N_ASSETS),           dtype=np.float32)
    logprob_b = np.zeros((N_STEPS, n),                      dtype=np.float32)
    values_b  = np.zeros((N_STEPS, n),                      dtype=np.float32)
    rewards_b = np.zeros((N_STEPS, n),                      dtype=np.float32)
    dones_b   = np.zeros((N_STEPS, n),                      dtype=np.float32)

    policy.eval()
    critic.eval()

    for step in range(N_STEPS):
        batch = torch.tensor(np.stack(env_states), device=device)   # (n, N_ASSETS, T)

        with torch.no_grad():
            dist    = policy.distribution(batch)
            acts    = dist.sample()                   # (n, N_ASSETS) pre-softmax
            lp      = dist.log_prob(acts).sum(-1)     # (n,)
            vals    = critic(batch)                    # (n,)
            weights = F.softmax(acts, dim=-1).cpu().numpy()

        states_b[step]  = np.stack(env_states)
        actions_b[step] = acts.cpu().numpy()
        logprob_b[step] = lp.cpu().numpy()
        values_b[step]  = vals.cpu().numpy()

        for e, env in enumerate(envs):
            nxt, rew, done, _ = env.step(weights[e])
            # Bound the episode to the TRAIN window. Without this the env keeps
            # stepping past train_end straight through the validation/test/future
            # bars, and the agent trains on out-of-sample data — a severe look-ahead
            # leak that lets it "memorize" its own backtest year.
            done = bool(done) or (env.t >= train_end)
            rewards_b[step, e] = rew
            dones_b[step, e]   = float(done)
            if done:
                s = np.random.randint(train_start, max(train_start + 1, train_end))
                env_states[e] = env.reset(s)
            else:
                env_states[e] = nxt

    # Bootstrap value for the last observed state
    with torch.no_grad():
        next_v = critic(torch.tensor(np.stack(env_states), device=device)).cpu().numpy()

    return dict(states=states_b, actions=actions_b, logprobs=logprob_b,
                values=values_b, rewards=rewards_b, dones=dones_b, next_values=next_v)


# ── GAE ───────────────────────────────────────────────────────────────────────

def compute_gae(rollout):
    """
    Generalized Advantage Estimation (γ=0.9, λ=0.9).
    Returns flat (N_STEPS * N_ENVS,) arrays: advantages and value targets.
    """
    rewards, values, dones, next_v = (
        rollout["rewards"], rollout["values"], rollout["dones"], rollout["next_values"]
    )
    n_steps, n_envs = rewards.shape
    advantages = np.zeros_like(rewards)
    gae        = np.zeros(n_envs, dtype=np.float32)

    for t in reversed(range(n_steps)):
        next_val = values[t + 1] if t < n_steps - 1 else next_v
        mask     = 1.0 - dones[t]
        delta    = rewards[t] + GAMMA * next_val * mask - values[t]
        gae      = delta + GAMMA * GAE_LAMBDA * mask * gae
        advantages[t] = gae

    return advantages.reshape(-1), (advantages + values).reshape(-1)


# ── PPO update ────────────────────────────────────────────────────────────────

def ppo_update(policy, critic, optimizer, rollout, advantages, returns, device, lr_frac):
    """
    N_EPOCHS passes over the 7560-transition rollout buffer in mini-batches of
    BATCH_SIZE=1260 (6 mini-batches per epoch, 96 gradient steps total per rollout).
    lr_frac in [0,1] controls linear LR annealing from LR_START to LR_END.
    """
    states  = torch.tensor(rollout["states"].reshape(-1, N_ASSETS, LOOKBACK), device=device)
    actions = torch.tensor(rollout["actions"].reshape(-1, N_ASSETS),           device=device)
    old_lp  = torch.tensor(rollout["logprobs"].reshape(-1),                    device=device)
    adv     = torch.tensor(advantages, device=device)
    ret     = torch.tensor(returns,    device=device)

    # (advantages are normalized per mini-batch below, matching SB3)

    lr = LR_START + (LR_END - LR_START) * lr_frac
    for g in optimizer.param_groups:
        g["lr"] = lr

    pl_list, vl_list, ent_list = [], [], []
    n = len(states)

    policy.train()
    critic.train()

    for _ in range(N_EPOCHS):
        perm = torch.randperm(n, device=device)

        for start in range(0, n, BATCH_SIZE):
            mb = perm[start : start + BATCH_SIZE]
            s, a, olp      = states[mb], actions[mb], old_lp[mb]
            adv_mb, ret_mb = adv[mb], ret[mb]
            adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)   # per mini-batch (SB3)

            dist    = policy.distribution(s)
            new_lp  = dist.log_prob(a).sum(-1)
            entropy = dist.entropy().sum(-1).mean()
            vals    = critic(s)

            ratio  = (new_lp - olp).exp()
            surr   = torch.min(
                ratio * adv_mb,
                ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * adv_mb
            )
            p_loss = -surr.mean()
            v_loss = F.mse_loss(vals, ret_mb)
            loss   = p_loss + VALUE_COEF * v_loss - ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(critic.parameters()), MAX_GRAD_NORM
            )
            optimizer.step()

            pl_list.append(p_loss.item())
            vl_list.append(v_loss.item())
            ent_list.append(entropy.item())

    return {"policy_loss": np.mean(pl_list), "value_loss": np.mean(vl_list),
            "entropy": np.mean(ent_list), "lr": lr}


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(data, policy, start_idx, end_idx, device):
    """
    Run policy deterministically on [start_idx, end_idx).
    Returns (total_D_t_reward, final_portfolio_value).
    """
    env     = _make_env(data)
    state   = env.reset(start_idx)
    total_r = 0.0

    while env.t < end_idx:
        st = torch.tensor(state, device=device).unsqueeze(0)
        with torch.no_grad():
            w = policy(st).squeeze(0).cpu().numpy()   # deterministic softmax weights
        state, reward, done, _ = env.step(w)
        total_r += reward
        if done:
            break

    return total_r, env.port_val


# ── Window training ───────────────────────────────────────────────────────────

def train_window(w_idx, window, data, seed_policy_sd, device):
    """
    Train N_SEEDS agents on one sliding window.
    seed_policy_sd: PolicyNetwork state dict to warm-start from (None for random init).
    Returns (best_state_dict, best_val_reward).
    """
    ref_env     = _make_env(data)
    train_start = max(LOOKBACK, _date_idx(ref_env, window["train_start"]))
    train_end   = _date_idx(ref_env, window["train_end"])
    val_start   = max(LOOKBACK, _date_idx(ref_env, window["val_start"]))
    val_end     = _date_idx(ref_env, window["val_end"])

    best_val_r = -np.inf
    best_sd    = None

    for seed in range(N_SEEDS):
        torch.manual_seed(seed * 100 + w_idx)
        np.random.seed(seed * 100 + w_idx)

        policy = PolicyNetwork().to(device)
        critic = ValueNetwork().to(device)
        if seed_policy_sd is not None:
            policy.load_state_dict(seed_policy_sd)

        optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(critic.parameters()), lr=LR_START
        )

        # Parallel environments — each starts at a random training date
        envs       = [_make_env(data) for _ in range(N_ENVS)]
        env_states = []
        for env in envs:
            s = np.random.randint(train_start, max(train_start + 1, train_end))
            env_states.append(env.reset(s))

        total_steps = 0
        n_updates   = 0

        while total_steps < TOTAL_TIMESTEPS:
            rollout     = collect_rollouts(envs, env_states, policy, critic,
                                           device, train_start, train_end)
            adv, ret    = compute_gae(rollout)
            losses      = ppo_update(policy, critic, optimizer, rollout, adv, ret,
                                     device, total_steps / TOTAL_TIMESTEPS)
            total_steps += N_ENVS * N_STEPS
            n_updates   += 1

            if n_updates % 200 == 0:
                pct = 100 * total_steps / TOTAL_TIMESTEPS
                print(f"  W{w_idx:02d} S{seed} [{pct:5.1f}%]  "
                      f"π={losses['policy_loss']:+.4f}  "
                      f"v={losses['value_loss']:.4f}  "
                      f"H={losses['entropy']:.3f}  "
                      f"lr={losses['lr']:.1e}")

        val_r, val_pv = evaluate(data, policy, val_start, val_end, device)
        print(f"  W{w_idx:02d} S{seed}  val_reward={val_r:+.2f}  val_port=${val_pv:,.0f}")

        # Save EVERY seed: the paper reports metrics averaged across all 5 agents
        # per window (Sec. 6). The best-by-val agent still seeds the next window.
        torch.save({"state_dict": policy.state_dict(), "val_reward": val_r, "window": window},
                   os.path.join(CHECKPOINT_DIR, f"window_{w_idx:02d}_seed_{seed}.pt"))

        if val_r > best_val_r:
            best_val_r = val_r
            best_sd    = copy.deepcopy(policy.state_dict())

    return best_sd, best_val_r


# ── Entry point ────────────────────────────────────────────────────────────────

def main(resume_from: int = 0):
    """
    resume_from: window index to start from (0 = full run).
    If checkpoints exist for windows before resume_from, the previous
    window's best policy is loaded automatically as the warm-start.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("\nLoading market data...")
    data = load_market_data()

    # Load warm-start policy from the checkpoint just before resume_from
    seed_policy_sd = None
    if resume_from > 0:
        prev_ckpt = os.path.join(CHECKPOINT_DIR, f"window_{resume_from - 1:02d}.pt")
        if os.path.exists(prev_ckpt):
            seed_policy_sd = torch.load(prev_ckpt, map_location=device)["state_dict"]
            print(f"Resuming from window {resume_from}, "
                  f"warm-start loaded from {prev_ckpt}")
        else:
            print(f"Warning: {prev_ckpt} not found, starting with random init.")

    for w_idx, window in enumerate(WINDOWS):
        if w_idx < resume_from:
            continue

        print(f"\n{'=' * 64}")
        print(f"Window {w_idx:02d}  "
              f"train [{window['train_start']} → {window['train_end']}]  "
              f"val [{window['val_start']} → {window['val_end']}]")
        print(f"{'=' * 64}")

        best_sd, val_r = train_window(w_idx, window, data, seed_policy_sd, device)

        ckpt = os.path.join(CHECKPOINT_DIR, f"window_{w_idx:02d}.pt")
        torch.save({"state_dict": best_sd, "val_reward": val_r, "window": window}, ckpt)
        print(f"  Saved {ckpt}  (val_reward={val_r:+.4f})")

        seed_policy_sd = best_sd


if __name__ == "__main__":
    import sys
    resume = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(resume_from=resume)
