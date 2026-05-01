"""Tests for axiom scoring module."""
import torch
from fuzzycell.modules.axiom_scorer import AxiomScorer

def test_scorer_creation():
    scorer = AxiomScorer(s=10.0)
    assert scorer.implication.s.item() == 10.0

if __name__ == "__main__":
    test_scorer_creation()
    print("All tests passed!")
