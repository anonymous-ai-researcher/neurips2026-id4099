"""Evaluation script for FuzzyCell."""
import argparse
import torch
from fuzzycell.utils.config import load_config
from fuzzycell.metrics import axiom_violation_rate, hierarchy_distance_f1

def main():
    parser = argparse.ArgumentParser(description="Evaluate FuzzyCell")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--metrics", nargs="+", default=["accuracy", "macro_f1", "avr", "hdf1"])
    args = parser.parse_args()

    config = load_config(args.config)
    # TODO: Load model, dataset, run evaluation
    print("Evaluation complete.")

if __name__ == "__main__":
    main()
