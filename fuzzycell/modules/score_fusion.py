"""Score fusion: combines neural similarity and ontological consistency."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreFusion(nn.Module):
    """Fuse neural and ontological scores into a final ranking.

    s_final = alpha * s_neural + (1 - alpha) * s_onto

    Args:
        alpha: Fusion weight in [0, 1]. Default: 0.6.
            alpha=0: pure ontological scoring
            alpha=1: pure neural scoring (recovers baseline)
    """

    def __init__(self, alpha: float = 0.6):
        super().__init__()
        self.alpha = alpha

    def forward(
        self, s_neural: torch.Tensor, s_onto: torch.Tensor
    ) -> torch.Tensor:
        """Compute fused scores.

        Args:
            s_neural: (batch, k) rescaled cosine similarities in [0, 1].
            s_onto: (batch, k) ontological consistency scores in [0, 1].

        Returns:
            s_final: (batch, k) fused scores.
        """
        return self.alpha * s_neural + (1.0 - self.alpha) * s_onto

    def graded_membership(self, s_final: torch.Tensor) -> torch.Tensor:
        """Normalize s_final to a graded membership vector summing to 1.

        Args:
            s_final: (batch, k) fused scores.

        Returns:
            mu: (batch, k) graded membership vector.
        """
        return F.softmax(s_final, dim=-1)
