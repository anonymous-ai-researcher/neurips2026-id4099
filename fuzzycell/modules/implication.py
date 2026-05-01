"""Fuzzy implication operators for differentiable ontology reasoning.

Implements the Sigmoidal Reichenbach implication I_σ, which provably eliminates
the gradient degeneracy of standard fuzzy implications while converging to
classical logic in the limit (s → ∞).
"""

import torch
import torch.nn as nn
from typing import Optional


class SigmoidalReichenbach(nn.Module):
    """Sigmoidal Reichenbach implication I_σ(a, b) = σ(s · (I_R(a,b) - 1/2)).

    Properties (proven in Appendix A):
        1. Gradient non-degeneracy: ∇I_σ ≠ 0 on (0,1)² for all s > 0
        2. Semantic soundness: I_σ → classical implication as s → ∞
        3. Value centering: I_σ(1/2, 1/2) = 1/2 for all s > 0

    Args:
        s: Steepness parameter controlling sigmoid sharpness. Default: 10.0.
            s=1: nearly linear (weak discrimination)
            s=10: moderate (recommended)
            s=30: nearly step function (may cause optimization instability)
        learnable: Whether s is a learnable parameter. Default: False.
    """

    def __init__(self, s: float = 10.0, learnable: bool = False):
        super().__init__()
        if learnable:
            self.s = nn.Parameter(torch.tensor(s))
        else:
            self.register_buffer("s", torch.tensor(s))

    def reichenbach(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Standard Reichenbach implication I_R(a, b) = 1 - a + a·b."""
        return 1.0 - a + a * b

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute I_σ(a, b) = σ(s · (I_R(a, b) - 1/2)).

        Args:
            a: Antecedent membership degrees in [0, 1]. Shape: (*).
            b: Consequent membership degrees in [0, 1]. Shape: (*).

        Returns:
            Implication values in (0, 1). Shape: (*).
        """
        ir = self.reichenbach(a, b)
        return torch.sigmoid(self.s * (ir - 0.5))

    def extra_repr(self) -> str:
        s_val = self.s.item() if isinstance(self.s, (nn.Parameter, torch.Tensor)) else self.s
        return f"s={s_val:.1f}"


class ProductTNorm(nn.Module):
    """Product t-norm T_P(a, b) = a · b.

    The only t-norm whose symmetric configuration yields consistent
    improvements in differentiable learning (van Krieken et al., 2022).
    """

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b


class GodelImplication(nn.Module):
    """Gödel implication I_G(a, b) = 1 if a ≤ b, else b.

    Provided for comparison. Has zero gradient on {a ≤ b}, i.e., 50% of [0,1]².
    """

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.where(a <= b, torch.ones_like(a), b)


class ReichenbachImplication(nn.Module):
    """Standard Reichenbach implication I_R(a, b) = 1 - a + a·b.

    Provided for comparison (ablation: "w/o I_σ"). Suffers from implication bias:
    ∂I_R/∂a = b - 1 → 0 when b → 1.
    """

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return 1.0 - a + a * b
