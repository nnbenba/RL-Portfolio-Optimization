import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Dimensions (single source of truth: environment.py) ──────────────────────
from environment import N_SECTORS, N_ASSETS, T, STATE_DIM

# NSMDP flattened state:  S_t = [ vec(R_t) || w_t || b_t || H(b_t) ], length 554
INPUT_DIM = STATE_DIM


def _sb3_orthogonal_init(net: nn.Sequential, output_gain: float) -> None:
    """SB3-style init: orthogonal, gain sqrt(2) on hidden layers, `output_gain`
    on the final head (0.01 for the policy, 1.0 for the value fn), biases 0."""
    linears = [m for m in net if isinstance(m, nn.Linear)]
    for m in linears[:-1]:
        nn.init.orthogonal_(m.weight, gain=2 ** 0.5)
        nn.init.zeros_(m.bias)
    nn.init.orthogonal_(linears[-1].weight, gain=output_gain)
    nn.init.zeros_(linears[-1].bias)


class PolicyNetwork(nn.Module):
    """
    Actor: S_t -> w  (portfolio weight vector, sums to 1, all >= 0).

    [64, 64] FC + tanh backbone (paper Section 5.2, Table 1).
    At inference: softmax enforces the simplex constraint on the output.
    At training:  actions are sampled from Normal(mean, exp(log_std))
                  in pre-softmax space; log_std is learnable, init = -1.
    """

    def __init__(self, log_std_init: float = -1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 64), nn.Tanh(),
            nn.Linear(64, 64),        nn.Tanh(),
            nn.Linear(64, N_ASSETS),
        )
        # Learnable per-asset log std, initialised to -1 (paper Table 1)
        self.log_std = nn.Parameter(torch.full((N_ASSETS,), log_std_init))
        _sb3_orthogonal_init(self.net, output_gain=0.01)   # small policy head (SB3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Deterministic pass -> softmax weights. Shape: (B, N_ASSETS)."""
        logits = self.net(state.flatten(start_dim=1))
        return F.softmax(logits, dim=-1)

    def distribution(self, state: torch.Tensor) -> torch.distributions.Normal:
        """Stochastic Normal distribution over pre-softmax actions (for training)."""
        mean = self.net(state.flatten(start_dim=1))
        std  = self.log_std.exp()                     # free log_std, as in SB3
        return torch.distributions.Normal(mean, std)


class ValueNetwork(nn.Module):
    """
    Critic: S_t -> V(s)  (scalar state-value estimate).

    Used to compute the GAE advantage:
        A_t = D_t + gamma * V(s_{t+1}) - V(s_t)
    where D_t is the Differential Sharpe Ratio reward (paper Section 4.3).
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 64), nn.Tanh(),
            nn.Linear(64, 64),        nn.Tanh(),
            nn.Linear(64, 1),
        )
        _sb3_orthogonal_init(self.net, output_gain=1.0)    # value head gain 1.0 (SB3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns scalar V(s) per sample. Shape: (B,)."""
        return self.net(state.flatten(start_dim=1)).squeeze(-1)


if __name__ == "__main__":
    policy = PolicyNetwork()
    critic = ValueNetwork()

    # Dummy batch: 4 flat NSMDP state vectors (length STATE_DIM)
    states = torch.zeros(4, INPUT_DIM)

    weights = policy(states)               # (4, 10)  -- softmax weights
    values  = critic(states)               # (4,)     -- value estimates
    dist    = policy.distribution(states)
    samples = F.softmax(dist.sample(), dim=-1)  # (4, 10) -- stochastic weights

    print("=== Policy Network ===")
    print(policy)
    print(f"\n  Output shape : {weights.shape}")
    print(f"  Weights[0]   : {[round(x, 4) for x in weights[0].tolist()]}")
    print(f"  Sum          : {weights[0].sum().item():.6f}")

    print("\n=== Value Network ===")
    print(critic)
    print(f"\n  Output shape    : {values.shape}")
    print(f"  Value estimates : {[round(x, 4) for x in values.tolist()]}")
