"""Context-aware type inference module.

Infers semantic type memberships tau^{I_f}(x) for each cell, used to
evaluate existential restrictions and subsumption axioms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TypeInference(nn.Module):
    """Infer semantic type memberships for each cell in the logic space.

    For each semantic type tau in Gamma, computes:
        tau^{I_f}(x) = sigma(z'_cell · a'_tau / theta)

    where a'_tau = pi_type(e_tau) is the projected semantic type embedding.

    Args:
        n_semantic_types: Number of semantic types |Gamma|. Default: 128.
        d_logic: Logic space dimension. Default: 256.
        theta_init: Initial temperature. Default: 0.1.
        theta_learnable: Whether temperature is learnable. Default: True.
    """

    def __init__(
        self,
        n_semantic_types: int = 128,
        d_logic: int = 256,
        theta_init: float = 0.1,
        theta_learnable: bool = True,
    ):
        super().__init__()
        self.n_semantic_types = n_semantic_types
        if theta_learnable:
            self.theta = nn.Parameter(torch.tensor(theta_init))
        else:
            self.register_buffer("theta", torch.tensor(theta_init))

    def forward(
        self,
        z_cell_proj: torch.Tensor,
        semantic_type_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute semantic type memberships.

        Args:
            z_cell_proj: (batch_size, d_logic) projected cell embeddings.
            semantic_type_embeddings: (|Gamma|, d_logic) projected semantic type embeddings.

        Returns:
            memberships: (batch_size, |Gamma|) type membership degrees in [0, 1].
        """
        # (batch_size, |Gamma|)
        logits = torch.matmul(z_cell_proj, semantic_type_embeddings.T) / self.theta
        return torch.sigmoid(logits)
