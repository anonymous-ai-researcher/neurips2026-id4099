"""Dual projection heads mapping cell and type embeddings into a shared logic space."""

import torch
import torch.nn as nn


class DualProjection(nn.Module):
    """Project cell and type embeddings into a shared logic space R^{d_L}.

    Architecture: Linear(d_in, d_L) + LayerNorm + GELU + Linear(d_L, d_L)

    Args:
        d_cell: Cell embedding dimension (from TFM backbone).
        d_type: Type embedding dimension (from PubMedBERT, typically 768).
        d_logic: Logic space dimension. Default: 256.
    """

    def __init__(self, d_cell: int, d_type: int = 768, d_logic: int = 256):
        super().__init__()
        self.pi_cell = nn.Sequential(
            nn.Linear(d_cell, d_logic),
            nn.LayerNorm(d_logic),
            nn.GELU(),
            nn.Linear(d_logic, d_logic),
        )
        self.pi_type = nn.Sequential(
            nn.Linear(d_type, d_logic),
            nn.LayerNorm(d_logic),
            nn.GELU(),
            nn.Linear(d_logic, d_logic),
        )

    def forward(self, z_cell: torch.Tensor, e_type: torch.Tensor):
        """Project both embeddings into the logic space.

        Args:
            z_cell: (batch_size, d_cell) cell embeddings.
            e_type: (n_candidates, d_type) type embeddings.

        Returns:
            z_cell_proj: (batch_size, d_logic) projected cell embeddings.
            e_type_proj: (n_candidates, d_logic) projected type embeddings.
        """
        return self.pi_cell(z_cell), self.pi_type(e_type)
