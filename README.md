<p align="center">
  <h1 align="center">🧬 FuzzyCell</h1>
  <p align="center">
    <strong>Differentiable Cell Ontology Reasoning for Graded Cell Type Annotation</strong>
  </p>
  <p align="center">
    <a href="#installation">Installation</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#method-overview">Method</a> •
    <a href="#reproducing-results">Reproducing Results</a> •
    <a href="#project-structure">Structure</a>
  </p>
</p>

---

## Highlights

- **Ontology-constrained annotation**: The first framework to integrate differentiable fuzzy EL⊥ reasoning into single-cell cell type annotation
- **Sigmoidal Reichenbach implication**: A new fuzzy implication operator that provably eliminates the gradient degeneracy of standard fuzzy implications while converging to classical logic in the limit
- **Graded membership**: Outputs continuous membership vectors (not just hard labels), capturing the biological reality that differentiating cells occupy transitional states
- **Backbone-agnostic plug-in**: Works with any pretrained transformer foundation model (scGPT, Geneformer, CellFM, scCello) at <4ms overhead per cell
- **70–85% axiom violation reduction**: Cuts biologically impossible predictions while improving Macro-F1 by 4–8 points on rare and zero-shot types

## Method Overview

FuzzyCell recasts cell type annotation as a two-stage pipeline:

```
Input (cell + tissue) → Stage 1: Retrieve → Stage 2: Reason → Graded Membership μ(x)
```

**Stage 1 (Retrieve):** A pretrained TFM backbone encodes each cell into a dense vector. FAISS retrieves the top-*k* candidate types from the Cell Ontology based on cosine similarity with PubMedBERT type embeddings.

**Stage 2 (Reason):** Dual projection maps cell and candidate embeddings into a shared logic space. A differentiable fuzzy EL⊥ reasoner scores each candidate's consistency with CL axioms (subsumption, disjointness, existential restrictions) using the Sigmoidal Reichenbach implication *I*<sub>σ</sub>. The output is a graded membership vector μ(**x**) ∈ [0,1]<sup>k</sup>.

### Key Innovation: Sigmoidal Reichenbach Implication

The standard Reichenbach implication *I*<sub>R</sub>(*a*,*b*) = 1 − *a* + *a*·*b* suffers from **implication bias**: when *b* ≈ 1, the gradient ∂*I*<sub>R</sub>/∂*a* = *b* − 1 ≈ 0, making the loss insensitive to the antecedent. Our sigmoidal variant:

*I*<sub>σ</sub>(*a*,*b*) = σ(*s* · (*I*<sub>R</sub>(*a*,*b*) − 1/2))

provably maintains nonzero gradients everywhere on (0,1)² and converges to classical logic as *s* → ∞.

## Installation

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- CUDA ≥ 11.8 (for GPU training)

### Setup

```bash
# Clone the repository
git clone https://github.com/anonymous/FuzzyCell.git
cd FuzzyCell

# Create conda environment
conda create -n fuzzycell python=3.10 -y
conda activate fuzzycell

# Install dependencies
pip install -r requirements.txt

# Install FuzzyCell in development mode
pip install -e .
```

## Quick Start

### 1. Prepare the Cell Ontology TBox

```bash
# Download CL (release 2024-04-05) and Uberon, then extract the TBox
python scripts/build_tbox.py \
    --cl-owl data/ontology/cl.owl \
    --uberon-owl data/ontology/uberon.owl \
    --output data/tbox/cl_tbox.json
```

### 2. Precompute Type Embeddings

```bash
# Encode all CL concept names with PubMedBERT
python scripts/precompute_type_embeddings.py \
    --tbox data/tbox/cl_tbox.json \
    --model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
    --output data/embeddings/type_embeddings.pt
```

### 3. Train FuzzyCell

```bash
# Train with scGPT backbone on Tabula Sapiens
python scripts/train.py \
    --config configs/fuzzycell_scgpt_ts.yaml \
    --seed 42
```

### 4. Evaluate

```bash
# Evaluate on test set
python scripts/evaluate.py \
    --config configs/fuzzycell_scgpt_ts.yaml \
    --checkpoint checkpoints/best_model.pt \
    --metrics accuracy macro_f1 weighted_f1 avr hdf1
```

### 5. Inference on New Data

```python
from fuzzycell import FuzzyCell

model = FuzzyCell.from_pretrained("checkpoints/best_model.pt")
# x: (n_cells, n_genes) gene expression matrix
# tissue: list of tissue labels
memberships = model.predict(x, tissue=tissue)
# memberships: dict with keys 'hard_label', 'graded_membership', 'scores'
```

## Reproducing Results

### Data Preparation

| Dataset | Cells | Types | Source |
|---------|-------|-------|--------|
| Tabula Sapiens | 483K | 476 | [CZI CELLxGENE](https://cellxgene.cziscience.com/) |
| HLCA | 580K | 61 | [CZI CELLxGENE](https://cellxgene.cziscience.com/) |
| AIDA v2 | 212K | 85 | [CZI CELLxGENE](https://cellxgene.cziscience.com/) |

```bash
# Download and preprocess all datasets
bash scripts/download_data.sh
python scripts/preprocess_datasets.py --config configs/data_config.yaml
```

### Training All Configurations

```bash
# Train FuzzyCell with all 4 backbones on all 3 datasets (12 configs × 5 seeds)
bash scripts/run_all_experiments.sh
```

### Reproducing Tables and Figures

```bash
# Table 1: Main results on Tabula Sapiens
python scripts/evaluate.py --config configs/fuzzycell_scgpt_ts.yaml --all-baselines

# Figure 2: Violation breakdown + rare type analysis
python scripts/plot_figure2.py --results-dir results/tabula_sapiens/

# Figure 3: Case studies
python scripts/plot_figure3.py --results-dir results/tabula_sapiens/

# Ablation study
bash scripts/run_ablation.sh
```

## Configuration

All experiments are configured via YAML files in `configs/`. Key hyperparameters:

```yaml
# Model
k: 64                    # Number of candidate types
d_L: 256                 # Logic space dimension
gamma_size: 128          # Number of semantic types |Γ|
s: 10.0                  # Sigmoidal steepness
alpha: 0.6               # Fusion weight (neural vs ontological)

# Training
optimizer: adamw
lr: 5e-5
lr_backbone: 5e-6
batch_size: 256
max_epochs: 50
early_stopping_patience: 5
margin: 0.2               # Ranking loss margin γ
lambda_type: 1.0           # Type inference loss weight
lambda_onto: 0.5           # Ontology consistency loss weight

# Evaluation
metrics: [accuracy, macro_f1, weighted_f1, avr, hdf1]
n_runs: 5
```

## Project Structure

```
FuzzyCell/
├── configs/                          # Experiment configurations
│   ├── fuzzycell_scgpt_ts.yaml       # scGPT + Tabula Sapiens
│   ├── fuzzycell_geneformer_ts.yaml  # Geneformer + Tabula Sapiens
│   ├── fuzzycell_cellfm_ts.yaml      # CellFM + Tabula Sapiens
│   ├── fuzzycell_sccello_ts.yaml     # scCello + Tabula Sapiens
│   ├── fuzzycell_scgpt_hlca.yaml     # scGPT + HLCA
│   ├── fuzzycell_scgpt_aida.yaml     # scGPT + AIDA v2
│   └── data_config.yaml              # Data preprocessing config
├── fuzzycell/                        # Core library
│   ├── __init__.py
│   ├── model.py                      # FuzzyCell main model
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── candidate_retrieval.py    # Stage 1: FAISS retrieval
│   │   ├── dual_projection.py       # Dual projection into logic space
│   │   ├── type_inference.py         # Context-aware type inference
│   │   ├── axiom_scorer.py           # Fuzzy EL⊥ axiom scoring
│   │   ├── implication.py            # Sigmoidal Reichenbach I_σ
│   │   └── score_fusion.py           # Neural + ontological fusion
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                # PyTorch dataset classes
│   │   ├── tbox.py                   # TBox loading and reasoning
│   │   └── preprocessing.py          # Gene expression preprocessing
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── avr.py                    # Axiom Violation Rate
│   │   └── hdf1.py                   # Hierarchy-Distance-Weighted F1
│   └── utils/
│       ├── __init__.py
│       ├── config.py                 # Configuration management
│       └── logging.py                # Logging utilities
├── scripts/
│   ├── build_tbox.py                 # Extract TBox from OWL files
│   ├── precompute_type_embeddings.py # PubMedBERT type encoding
│   ├── preprocess_datasets.py        # Data preprocessing
│   ├── download_data.sh              # Dataset download script
│   ├── train.py                      # Training script
│   ├── evaluate.py                   # Evaluation script
│   ├── run_all_experiments.sh        # Full experiment pipeline
│   ├── run_ablation.sh               # Ablation experiments
│   ├── plot_figure2.py               # Violation + rare type figure
│   └── plot_figure3.py               # Case study figure
├── tests/
│   ├── test_implication.py           # Tests for I_σ properties
│   ├── test_axiom_scorer.py          # Tests for axiom scoring
│   └── test_metrics.py               # Tests for AVR and HD-F1
├── requirements.txt
├── setup.py
├── LICENSE                           # MIT License
└── README.md
```

## Evaluation Metrics

### Axiom Violation Rate (AVR)

Fraction of predictions violating at least one CL axiom (disjointness, role restriction, or subsumption transitivity). Lower is better. AVR = 0 means all predictions are ontologically consistent.

### Hierarchy-Distance-Weighted F1 (HD-F1)

F1 metric that penalizes misclassifications proportionally to their distance in the CL hierarchy. Confusing a T cell with a B cell (siblings, distance 2) costs less than confusing it with a neuron (distance ≥ 12).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

### Third-Party Licenses

| Asset | License |
|-------|---------|
| Cell Ontology | CC-BY 4.0 |
| Uberon | CC-BY 4.0 |
| Tabula Sapiens | CC-BY 4.0 |
| HLCA | CC-BY 4.0 |
| PubMedBERT | MIT |
| scGPT | BSD-3-Clause |
| Geneformer | Apache-2.0 |
