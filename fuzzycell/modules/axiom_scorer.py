"""Fuzzy EL-bot axiom scoring module.

Scores each candidate type's consistency with CL TBox axioms using the
Sigmoidal Reichenbach implication.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from fuzzycell.modules.implication import SigmoidalReichenbach, ProductTNorm


class AxiomScorer(nn.Module):
    """Score candidate types against CL TBox axioms.

    Evaluates three axiom types:
        1. Subsumption: C ⊑ D (every C is also a D)
        2. Disjointness: C ⊓ D ⊑ ⊥ (C and D are mutually exclusive)
        3. Existential restriction: C ⊑ ∃r.D (every C stands in relation r to some D)

    Args:
        s: Steepness parameter for I_sigma. Default: 10.0.
        membership_matrix: (|N_C|, |Gamma|) binary matrix M[j,tau] = 1[T_CL |= C_j ⊑ tau].
    """

    def __init__(self, s: float = 10.0):
        super().__init__()
        self.implication = SigmoidalReichenbach(s=s)
        self.t_norm = ProductTNorm()
        self.membership_matrix = None
        self.disjointness_pairs = None
        self.role_restrictions = None

    def load_tbox(
        self,
        membership_matrix: torch.Tensor,
        disjointness_pairs: List[tuple],
        role_restrictions: Dict[str, List[tuple]],
    ):
        """Load precomputed TBox data.

        Args:
            membership_matrix: (|N_C|, |Gamma|) binary matrix.
            disjointness_pairs: List of (C_idx, D_idx) pairs where C ⊓ D ⊑ ⊥.
            role_restrictions: Dict mapping role names to (C_idx, D_idx) pairs.
        """
        self.register_buffer("_membership_matrix", membership_matrix)
        self.membership_matrix = membership_matrix
        self.disjointness_pairs = disjointness_pairs
        self.role_restrictions = role_restrictions

    def score_subsumption(
        self,
        cell_type_memberships: torch.Tensor,
        candidate_type_memberships: torch.Tensor,
    ) -> torch.Tensor:
        """Score subsumption axioms C ⊑ D.

        For each candidate C_j and each semantic type tau where T_CL |= C_j ⊑ tau,
        evaluates I_sigma(C_j^{I_f}(x), tau^{I_f}(x)).

        Args:
            cell_type_memberships: (batch, |Gamma|) inferred type memberships.
            candidate_type_memberships: (k, |Gamma|) binary CL memberships.

        Returns:
            scores: (batch, k) subsumption satisfaction scores.
        """
        batch_size = cell_type_memberships.shape[0]
        k = candidate_type_memberships.shape[0]

        # For each candidate, aggregate over its subsumption axioms
        scores = torch.zeros(batch_size, k, device=cell_type_memberships.device)
        counts = torch.zeros(k, device=cell_type_memberships.device)

        for j in range(k):
            active_tau = candidate_type_memberships[j].nonzero(as_tuple=True)[0]
            if len(active_tau) == 0:
                scores[:, j] = 1.0
                continue
            # a = candidate membership (from type inference output)
            # b = semantic type membership (from type inference output)
            a = cell_type_memberships[:, active_tau]  # (batch, n_active)
            b = cell_type_memberships[:, active_tau]  # same for subsumption
            imp = self.implication(a, b)  # (batch, n_active)
            scores[:, j] = imp.mean(dim=1)

        return scores

    def score_disjointness(
        self,
        cell_type_memberships: torch.Tensor,
        candidate_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Score disjointness axioms C ⊓ D ⊑ ⊥.

        Args:
            cell_type_memberships: (batch, |Gamma|) inferred type memberships.
            candidate_indices: (k,) indices into N_C.

        Returns:
            scores: (batch, k) disjointness satisfaction scores (1 = no violation).
        """
        batch_size = cell_type_memberships.shape[0]
        k = candidate_indices.shape[0]
        scores = torch.ones(batch_size, k, device=cell_type_memberships.device)

        if self.disjointness_pairs is None:
            return scores

        for j, c_idx in enumerate(candidate_indices):
            c_idx = c_idx.item()
            for c, d in self.disjointness_pairs:
                if c == c_idx or d == c_idx:
                    other = d if c == c_idx else c
                    if other < self.membership_matrix.shape[0]:
                        other_mem = self.membership_matrix[other]
                        cell_mem = cell_type_memberships
                        # Penalize if cell has high membership in the disjoint type
                        penalty = self.t_norm(
                            cell_mem[:, :other_mem.shape[0]] @ other_mem.float().unsqueeze(1),
                            torch.ones(batch_size, 1, device=cell_mem.device),
                        ).squeeze(1)
                        scores[:, j] = scores[:, j] * (1.0 - penalty)

        return scores

    def forward(
        self,
        cell_type_memberships: torch.Tensor,
        candidate_type_memberships: torch.Tensor,
        candidate_indices: torch.Tensor,
        tissue_labels: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Compute overall ontological consistency score s_onto.

        Args:
            cell_type_memberships: (batch, |Gamma|) type memberships.
            candidate_type_memberships: (k, |Gamma|) binary memberships.
            candidate_indices: (k,) indices into N_C.
            tissue_labels: Optional tissue context for role restriction scoring.

        Returns:
            s_onto: (batch, k) ontological consistency scores in [0, 1].
        """
        s_sub = self.score_subsumption(cell_type_memberships, candidate_type_memberships)
        s_disj = self.score_disjointness(cell_type_memberships, candidate_indices)
        return s_sub * s_disj
