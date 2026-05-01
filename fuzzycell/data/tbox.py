"""TBox loader for Cell Ontology EL-bot axioms."""
import json, torch
from pathlib import Path
from typing import List

class TBoxLoader:
    def __init__(self, tbox_path: str):
        self.tbox_path = Path(tbox_path)
        with open(self.tbox_path) as f:
            data = json.load(f)
        self.concept_names = data["concept_names"]
        self.role_names = data.get("role_names", {})
        self.subsumptions = data.get("subsumptions", [])
        self.disjointness = data.get("disjointness", [])
        self.existential_restrictions = data.get("existential_restrictions", {})

    @property
    def n_concepts(self): return len(self.concept_names)

    def build_membership_matrix(self, semantic_types: List[str]) -> torch.Tensor:
        n_c = len(self.concept_names)
        n_gamma = len(semantic_types)
        M = torch.zeros(n_c, n_gamma)
        concept_list = list(self.concept_names.keys())
        tau_to_idx = {tau: i for i, tau in enumerate(semantic_types)}
        for c, d in self.subsumptions:
            if c in concept_list and d in tau_to_idx:
                M[concept_list.index(c), tau_to_idx[d]] = 1.0
        return M
