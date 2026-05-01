"""Hierarchy-Distance-Weighted F1."""
import numpy as np
from typing import Dict, List

def hierarchy_distance_f1(predictions, ground_truths, distance_matrix, d_max=18):
    penalties = []
    for pred, gt in zip(predictions, ground_truths):
        if pred == gt:
            penalties.append(0.0)
        else:
            d = distance_matrix.get((pred, gt), distance_matrix.get((gt, pred), d_max))
            penalties.append(d / d_max)
    return {"hdf1": float(1.0 - np.mean(penalties))}
