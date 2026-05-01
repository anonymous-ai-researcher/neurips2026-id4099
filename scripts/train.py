"""Training script for FuzzyCell."""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from fuzzycell.utils.config import load_config
from fuzzycell.utils.logging import setup_logger

logger = setup_logger()

def main():
    parser = argparse.ArgumentParser(description="Train FuzzyCell")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    config.seed = args.seed
    torch.manual_seed(config.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Config: {config}")
    logger.info(f"Device: {device}")

    # TODO: Initialize backbone, dataset, model, optimizer
    # Training loop with early stopping on validation Macro-F1
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
