"""Tests for evaluation metrics."""
from fuzzycell.metrics.hdf1 import hierarchy_distance_f1

def test_perfect_predictions():
    preds = ["A", "B", "C"]
    gts = ["A", "B", "C"]
    result = hierarchy_distance_f1(preds, gts, {}, d_max=18)
    assert result["hdf1"] == 1.0

if __name__ == "__main__":
    test_perfect_predictions()
    print("All tests passed!")
