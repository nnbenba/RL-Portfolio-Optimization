import numpy as np


class DifferentialSharpeRatio:
    """
    Online Differential Sharpe Ratio reward (paper Section 4.3).

    Maintains exponential moving averages A_t (mean return) and B_t
    (mean squared return) and returns the instantaneous gradient D_t
    of the Sharpe ratio with respect to the adaptation rate η.

    Update rules:
        ΔA_t = R_t - A_{t-1}
        ΔB_t = R_t² - B_{t-1}
        A_t  = A_{t-1} + η · ΔA_t
        B_t  = B_{t-1} + η · ΔB_t

    Reward:
        D_t = (B_{t-1}·ΔA_t - ½·A_{t-1}·ΔB_t) / (B_{t-1} - A_{t-1}²)^(3/2)
    """

    def __init__(self, eta: float = 1 / 252, eps: float = 1e-8):
        self.eta = eta
        self.eps = eps
        self.A = 0.0  # EMA of returns
        self.B = 0.0  # EMA of squared returns

    def reset(self):
        self.A = 0.0
        self.B = 0.0

    def step(self, R_t: float) -> float:
        """
        Consume one portfolio return R_t, update state, return D_t.
        Returns 0 until enough variance has accumulated to avoid division instability.
        """
        dA = R_t - self.A
        dB = R_t ** 2 - self.B

        variance = self.B - self.A ** 2
        if variance < self.eps:
            # Not enough variance yet — update state but give no reward signal
            self.A += self.eta * dA
            self.B += self.eta * dB
            return 0.0

        numerator = self.B * dA - 0.5 * self.A * dB
        denominator = variance ** 1.5

        self.A += self.eta * dA
        self.B += self.eta * dB

        return numerator / denominator


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Simulate 500 days of N(0.0003, 0.01) daily returns
    returns = rng.normal(loc=0.0003, scale=0.01, size=500)

    dsr = DifferentialSharpeRatio()
    rewards = [dsr.step(r) for r in returns]

    print(f"A (mean return EMA) : {dsr.A:.6f}")
    print(f"B (mean sq-ret EMA) : {dsr.B:.6f}")
    print(f"Variance estimate   : {dsr.B - dsr.A**2:.6f}")
    print(f"Implied Sharpe      : {dsr.A / (dsr.B - dsr.A**2)**0.5 * 252**0.5:.4f}")
    print(f"\nFirst 10 rewards    : {[round(r, 6) for r in rewards[:10]]}")
    print(f"Last  10 rewards    : {[round(r, 6) for r in rewards[-10:]]}")
    print(f"Mean |D_t| (t>50)   : {np.abs(rewards[50:]).mean():.6f}")
