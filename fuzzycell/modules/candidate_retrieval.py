"""Stage 1: Candidate retrieval via FAISS nearest-neighbor search."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

try:
    import faiss
except ImportError:
    faiss = None


class CandidateRetrieval(nn.Module):
    """Retrieve top-k candidate cell types from CL using FAISS.

    Given a cell embedding z_cell and precomputed type embeddings {e_C},
    retrieves the k types with highest cosine similarity.

    Args:
        k: Number of candidates to retrieve. Default: 64.
        metric: Distance metric for FAISS. Default: 'cosine'.
    """

    def __init__(self, k: int = 64, metric: str = "cosine"):
        super().__init__()
        self.k = k
        self.metric = metric
        self.index = None
        self.type_ids = None

    def build_index(self, type_embeddings: torch.Tensor, type_ids: list):
        """Build FAISS index from precomputed type embeddings.

        Args:
            type_embeddings: (n_types, d') type embedding matrix.
            type_ids: List of CL concept IDs corresponding to each row.
        """
        assert faiss is not None, "FAISS is required. Install via: pip install faiss-gpu"
        embeddings_np = type_embeddings.detach().cpu().numpy().astype(np.float32)
        if self.metric == "cosine":
            faiss.normalize_L2(embeddings_np)
        d = embeddings_np.shape[1]
        self.index = faiss.IndexFlatIP(d)  # Inner product (cosine after normalization)
        self.index.add(embeddings_np)
        self.type_ids = type_ids

    def forward(
        self, z_cell: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """Retrieve top-k candidates for each cell.

        Args:
            z_cell: (batch_size, d) cell embeddings.

        Returns:
            similarities: (batch_size, k) cosine similarities.
            indices: (batch_size, k) FAISS indices.
            candidate_ids: List of lists of CL concept IDs.
        """
        queries = z_cell.detach().cpu().numpy().astype(np.float32)
        if self.metric == "cosine":
            faiss.normalize_L2(queries)
        similarities, indices = self.index.search(queries, self.k)
        candidate_ids = [
            [self.type_ids[idx] for idx in row] for row in indices
        ]
        return (
            torch.tensor(similarities, device=z_cell.device),
            torch.tensor(indices, device=z_cell.device),
            candidate_ids,
        )
