"""Configuration management."""
import yaml
from dataclasses import dataclass, field
from typing import List

@dataclass
class FuzzyCellConfig:
    backbone: str = "scgpt"
    d_cell: int = 512
    d_type: int = 768
    d_logic: int = 256
    k: int = 64
    n_semantic_types: int = 128
    s: float = 10.0
    alpha: float = 0.6
    theta_init: float = 0.1
    optimizer: str = "adamw"
    lr: float = 5e-5
    lr_backbone: float = 5e-6
    weight_decay: float = 1e-2
    batch_size: int = 256
    max_epochs: int = 50
    early_stopping_patience: int = 5
    margin: float = 0.2
    lambda_type: float = 1.0
    lambda_onto: float = 0.5
    dataset: str = "tabula_sapiens"
    n_hvg: int = 2000
    n_runs: int = 5
    seed: int = 42

def load_config(path: str) -> FuzzyCellConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return FuzzyCellConfig(**{k: v for k, v in raw.items() if k in FuzzyCellConfig.__dataclass_fields__})
