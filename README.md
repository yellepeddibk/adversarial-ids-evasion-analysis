# Adversarial Robustness Analysis of Neural Network-Based Intrusion Detection Systems Using FGSM and PGD Attacks

> Adversarial robustness evaluation of an MLP-based intrusion detection model under FGSM/PGD white-box attacks, with adversarial training defense and ε-sweep robustness curves.

**Authors:** Conrad Miller, Prathik Bengaluru Prabhakara, Bhargav Yellepeddi

# Adversarial Robustness Analysis of Neural Network-Based Intrusion Detection Systems

> A PyTorch-based cybersecurity project that measures how white-box adversarial attacks degrade intrusion detection performance and how adversarial training restores robustness.

**Authors:** Conrad Miller, Prathik Bengaluru Prabhakara, Bhargav Yellepeddi

## Overview

This repository studies adversarial evasion against a neural network intrusion detection system trained on the NSL-KDD dataset. The project uses a Multi-Layer Perceptron (MLP) to classify network traffic as normal or attack, then evaluates the model under Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) attacks across multiple perturbation strengths.

To improve robustness, the project also trains a second model with adversarial training and compares both models on clean and adversarial examples. The end result is a compact, reproducible case study of the accuracy-versus-robustness trade-off in machine learning security.

## Key Results

- The baseline model reaches 99.53% clean test accuracy.
- The adversarially trained model preserves clean accuracy at 99.56%.
- At epsilon = 0.30, baseline accuracy drops to 65.14% under FGSM and 56.44% under PGD.
- At the same epsilon, adversarial training improves robust accuracy to 96.87% under FGSM and 95.63% under PGD.
- PGD is consistently stronger than FGSM, while adversarial training delivers large robustness gains with almost no clean-accuracy penalty.

## Method Summary

1. Preprocess NSL-KDD with one-hot encoding for categorical features and standardization for continuous features.
2. Train a two-hidden-layer MLP for binary normal-versus-attack classification.
3. Generate white-box adversarial examples with FGSM and PGD under an L-infinity constraint.
4. Restrict perturbations to continuous features and clip them to valid training ranges.
5. Train a second model with FGSM-based adversarial training and compare clean and robust accuracy across an epsilon sweep.

## Selected Results

| Epsilon | Baseline FGSM | Adv. Trained FGSM | Baseline PGD | Adv. Trained PGD |
|---|---:|---:|---:|---:|
| 0.00 | 99.53% | 99.56% | 99.53% | 99.56% |
| 0.10 | 96.78% | 99.20% | 96.50% | 99.13% |
| 0.30 | 65.14% | 96.87% | 56.44% | 95.63% |

## Repository Layout

```text
adversarial-ids-evasion-analysis/
├── configs/          # Experiment configuration
├── data/             # Raw and processed datasets
├── docs/             # Public-facing project documents and visuals
├── notebooks/        # Exploration and analysis notebooks
├── results/          # Figures, tables, and saved model checkpoints
├── scripts/          # Training and evaluation entry points
├── src/              # Core project code
├── tests/            # Test scaffolding
├── ProjectCode.ipynb # Consolidated notebook workflow
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.9+
- pip
- Git

### Installation

```bash
git clone https://github.com/<org>/adversarial-ids-evasion-analysis.git
cd adversarial-ids-evasion-analysis

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Dataset

Download the NSL-KDD dataset from the [official dataset page](https://www.unb.ca/cic/datasets/nsl.html) or a trusted mirror and place the files in `data/raw/`.

Expected inputs:

- `data/raw/KDDTrain+.csv`
- `data/raw/KDDTest+.csv`

Raw data is excluded from version control.

### Run Experiments

```bash
python scripts/train_baseline.py --config configs/default.yaml
python scripts/train_adversarial.py --config configs/default.yaml
python scripts/evaluate.py --model results/models/baseline.pt --config configs/default.yaml
python scripts/evaluate.py --model results/models/adversarial.pt --config configs/default.yaml
python scripts/run_epsilon_sweep.py --config configs/default.yaml
```

## Documentation

- [docs/README.md](docs/README.md) for project documents, visuals, and report assets.
- [docs/reports/final_project_report.tex](docs/reports/final_project_report.tex) for the final report source.
- [docs/reports/final_project_report.pdf](docs/reports/final_project_report.pdf) for the rendered report.
- [docs/scripts/generate_presentation_visuals.py](docs/scripts/generate_presentation_visuals.py) for the matplotlib slide visual generator.

## Limitations

- The evaluation uses a single dataset: NSL-KDD.
- Perturbations are restricted to continuous features, which preserves feature validity but may understate worst-case vulnerability.
- The experiments focus on one architecture, an MLP, rather than a broader model family.
- PGD attacks and adversarial training increase runtime cost compared with clean-only evaluation.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and workflow expectations.

## License

This project is licensed under the terms of the [MIT License](LICENSE).
| `results/figures/` | Saved plots (accuracy vs. ε curves, confusion matrices, etc.) |
