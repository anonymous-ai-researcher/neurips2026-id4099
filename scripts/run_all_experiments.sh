#!/bin/bash
# Run all experiments: 4 backbones x 3 datasets x 5 seeds
set -e

SEEDS=(42 123 456 789 1024)
CONFIGS=(
    configs/fuzzycell_scgpt_ts.yaml
    configs/fuzzycell_geneformer_ts.yaml
    configs/fuzzycell_scgpt_hlca.yaml
    configs/fuzzycell_scgpt_aida.yaml
)

for config in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "Running $config with seed $seed"
        python scripts/train.py --config "$config" --seed "$seed"
    done
done

echo "All experiments complete."
