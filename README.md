# Adversarial Robustness Analysis of Neural Network-Based Intrusion Detection Systems Using FGSM and PGD Attacks

> Completed PyTorch cybersecurity project analyzing adversarial evasion in intrusion detection and the effectiveness of adversarial training as a defense.

**Authors:** Conrad Miller, Prathik Bengaluru Prabhakara, Bhargav Yellepeddi

## Project Status

- Completed in May 2026.
- Final report source and PDF are available in `docs/reports/`.
- Presentation visuals and supporting public-facing assets are available in `docs/`.
- Core training, attack, evaluation, and epsilon-sweep workflows are implemented in the repository.

## Overview

This repository studies adversarial robustness for a neural network-based intrusion detection system trained on the NSL-KDD dataset. The project uses a Multi-Layer Perceptron (MLP) to classify network traffic as normal or attack, then evaluates that model under two white-box attacks: Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD).

To improve robustness, the project also trains a second model using adversarial training and compares both models on clean and adversarial examples. The final result is a compact, reproducible case study of the trade-off between clean accuracy and adversarial robustness in a cybersecurity setting.

## What Was Completed

1. Built a preprocessing pipeline for NSL-KDD with one-hot encoding for categorical features and standardization for continuous features.
2. Trained a baseline MLP for binary normal-versus-attack classification.
3. Implemented FGSM and PGD white-box attacks under an L-infinity perturbation budget.
4. Restricted perturbations to continuous features and clipped perturbed samples to valid training ranges.
5. Trained an adversarially robust MLP using FGSM-based adversarial training.
6. Measured clean and robust accuracy across an epsilon sweep and documented the final findings in the report.

## Final Results

- Baseline clean test accuracy: 99.53%
- Adversarially trained clean test accuracy: 99.56%
- At epsilon = 0.30, baseline accuracy drops to 65.14% under FGSM and 56.44% under PGD
- At the same epsilon, adversarial training improves robust accuracy to 96.87% under FGSM and 95.63% under PGD
- PGD is consistently stronger than FGSM, while adversarial training provides large robustness gains with negligible clean-accuracy cost

## Results Snapshot

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
├── docs/             # Final report, presentation visuals, and public assets
├── notebooks/        # Exploration and analysis notebooks
├── results/          # Figures, tables, and saved model checkpoints
├── scripts/          # Training and evaluation entry points
├── src/              # Core project code
├── tests/            # Test scaffolding
├── ProjectCode.ipynb # Consolidated notebook workflow
└── README.md
```

## Reproducing the Work

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

## Final Artifacts

- `docs/reports/final_project_report.tex` — final report source
- `docs/reports/final_project_report.pdf` — rendered final report
- `docs/scripts/generate_presentation_visuals.py` — slide visual generator
- `docs/visuals/` — rendered presentation graphics

## Limitations

- The evaluation uses a single dataset: NSL-KDD.
- Perturbations are restricted to continuous features, which preserves feature validity but may understate worst-case vulnerability.
- The experiments focus on one architecture, an MLP, rather than a broader model family.
- PGD attacks and adversarial training increase runtime cost compared with clean-only evaluation.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and workflow expectations.

## License

This project is licensed under the terms of the [MIT License](LICENSE).
