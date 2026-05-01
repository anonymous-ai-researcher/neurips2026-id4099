"""FuzzyCell: main model integrating retrieval and reasoning stages."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from fuzzycell.modules import (
    CandidateRetrieval, DualProjection, TypeInference,
    AxiomScorer, ScoreFusion,
)


class FuzzyCell(nn.Module):
    """FuzzyCell: Differentiable Cell Ontology Reasoning for Graded Cell Type Annotation.

    Two-stage pipeline:
        Stage 1 (Retrieve): FAISS top-k candidate retrieval from CL
        Stage 2 (Reason): Fuzzy EL_bot axiom scoring with Sigmoidal Reichenbach implication

    Args:
        backbone: Pretrained TFM backbone (scGPT, Geneformer, etc.)
        d_cell: Cell embedding dimension from backbone.
        d_type: Type embedding dimension (PubMedBERT). Default: 768.
        d_logic: Logic space dimension. Default: 256.
        k: Number of candidates. Default: 64.
        n_semantic_types: Number of semantic types |Gamma|. Default: 128.
        s: Sigmoidal steepness. Default: 10.0.
        alpha: Fusion weight. Default: 0.6.
        theta_init: Initial temperature. Default: 0.1.
    """

    def __init__(
        self,
        backbone: nn.Module,
        d_cell: int,
        d_type: int = 768,
        d_logic: int = 256,
        k: int = 64,
        n_semantic_types: int = 128,
        s: float = 10.0,
        alpha: float = 0.6,
        theta_init: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.retrieval = CandidateRetrieval(k=k)
        self.projection = DualProjection(d_cell=d_cell, d_type=d_type, d_logic=d_logic)
        self.type_inference = TypeInference(
            n_semantic_types=n_semantic_types, d_logic=d_logic, theta_init=theta_init
        )
        self.axiom_scorer = AxiomScorer(s=s)
        self.fusion = ScoreFusion(alpha=alpha)

    def forward(
        self,
        x: torch.Tensor,
        type_embeddings: torch.Tensor,
        candidate_indices: torch.Tensor,
        candidate_type_memberships: torch.Tensor,
        semantic_type_embeddings: torch.Tensor,
        tissue_labels: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            x: (batch, n_genes) gene expression matrix.
            type_embeddings: (k, d_type) candidate type embeddings.
            candidate_indices: (k,) indices into N_C.
            candidate_type_memberships: (k, |Gamma|) binary memberships.
            semantic_type_embeddings: (|Gamma|, d_type) semantic type embeddings.
            tissue_labels: Optional tissue context.

        Returns:
            Dict with keys: s_final, s_neural, s_onto, mu, hard_label
        """
        # Stage 1: Encode
        z_cell = self.backbone(x)  # (batch, d_cell)

        # Stage 2: Project
        z_cell_proj, e_type_proj = self.projection(z_cell, type_embeddings)
        _, sem_type_proj = self.projection.pi_cell(z_cell), self.projection.pi_type(semantic_type_embeddings)

        # Neural score
        z_norm = F.normalize(z_cell_proj, dim=-1)
        e_norm = F.normalize(e_type_proj, dim=-1)
        cosine = torch.matmul(z_norm, e_norm.T)  # (batch, k)
        s_neural = 0.5 * (cosine + 1.0)  # rescale to [0, 1]

        # Type inference
        cell_type_memberships = self.type_inference(z_cell_proj, sem_type_proj)

        # Axiom scoring
        s_onto = self.axiom_scorer(
            cell_type_memberships, candidate_type_memberships,
            candidate_indices, tissue_labels
        )

        # Fusion
        s_final = self.fusion(s_neural, s_onto)
        mu = self.fusion.graded_membership(s_final)
        hard_label = candidate_indices[s_final.argmax(dim=-1)]

        return {
            "s_final": s_final,
            "s_neural": s_neural,
            "s_onto": s_onto,
            "mu": mu,
            "hard_label": hard_label,
            "type_memberships": cell_type_memberships,
        }

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, **kwargs) -> "FuzzyCell":
        """Load a pretrained FuzzyCell model."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("config", {})
        config.update(kwargs)
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
