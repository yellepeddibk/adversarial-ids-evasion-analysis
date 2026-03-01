# Adversarial Robustness Analysis of Neural Network-Based Intrusion Detection Systems Using FGSM and PGD Attacks

> Adversarial robustness evaluation of an MLP-based intrusion detection model under FGSM/PGD white-box attacks, with adversarial training defense and ε-sweep robustness curves.

**Authors:** Conrad Miller, Prathik Bengaluru Prabhakara, Bhargav Yellepeddi

---

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Dataset](#dataset)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Architecture](#model-architecture)
  - [Training Procedure](#training-procedure)
  - [Adversarial Attacks](#adversarial-attacks)
  - [Adversarial Training (Defense)](#adversarial-training-defense)
  - [Robustness Evaluation & ε-Sweep](#robustness-evaluation--ε-sweep)
  - [Evaluation Metrics](#evaluation-metrics)
- [Usage](#usage)
  - [Train the Baseline Model](#train-the-baseline-model)
  - [Train the Adversarially Robust Model](#train-the-adversarially-robust-model)
  - [Run Adversarial Evaluation](#run-adversarial-evaluation)
  - [Run ε-Sweep Experiments](#run-ε-sweep-experiments)
- [Expected Results](#expected-results)
- [Tools & Computation](#tools--computation)
- [Literature & References](#literature--references)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

Machine learning has become widely used in cybersecurity for detecting network intrusions and malicious activity. However, neural networks are vulnerable to **adversarial attacks**, where small, carefully crafted perturbations to input features can cause misclassification. In a cybersecurity setting, this could allow attackers to evade detection systems.

This project develops a machine learning-based **intrusion detection system (IDS)** using a Multi-Layer Perceptron (MLP) trained on the **NSL-KDD dataset** to classify network traffic as *normal* or *attack*. We evaluate the model's vulnerability to white-box adversarial attacks — specifically **FGSM** and **PGD** — and analyze its robustness under different perturbation levels (ε).

To improve security, we implement **adversarial training** as a defense mechanism and compare the performance of a baseline model with an adversarially trained model. The project aims to demonstrate the trade-off between accuracy and robustness and assess how adversarial defenses strengthen machine learning systems in cybersecurity applications.

---

## Project Structure

```
adversarial-ids-evasion-analysis/
│
├── .github/
│   └── pull_request_template.md  # PR template for contributions
│
├── configs/
│   └── default.yaml              # Hyperparameters & experiment configuration
│
├── data/
│   ├── raw/                      # Original NSL-KDD dataset files
│   │   └── .gitkeep
│   └── processed/                # Preprocessed / encoded data
│       └── .gitkeep
│
├── notebooks/                    # Jupyter notebooks for exploration & analysis
│   └── .gitkeep
│
├── results/
│   ├── figures/                  # Generated plots & visualizations
│   │   └── .gitkeep
│   ├── tables/                   # Performance comparison tables
│   │   └── .gitkeep
│   └── models/                   # Saved model checkpoints (.pt files)
│       └── .gitkeep
│
├── scripts/
│   ├── train_baseline.py         # Train the baseline MLP classifier
│   ├── train_adversarial.py      # Train the adversarially robust MLP
│   ├── evaluate.py               # Evaluate model(s) on clean & adversarial data
│   └── run_epsilon_sweep.py      # Run ε-sweep robustness experiments
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py             # Dataset loading utilities
│   │   └── preprocessing.py      # Feature encoding, scaling, splitting
│   ├── models/
│   │   ├── __init__.py
│   │   └── mlp.py                # MLP architecture definition
│   ├── attacks/
│   │   ├── __init__.py
│   │   ├── fgsm.py               # FGSM attack implementation
│   │   └── pgd.py                # PGD attack implementation
│   ├── defenses/
│   │   ├── __init__.py
│   │   └── adversarial_training.py  # Adversarial training logic
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py            # Accuracy, robust accuracy metrics
│   │   └── epsilon_sweep.py      # ε-sweep evaluation loop
│   └── utils/
│       ├── __init__.py
│       └── visualization.py      # Plotting & figure-generation utilities
│
├── tests/                        # Unit tests
│   └── .gitkeep
│
├── .gitignore                    # Git ignore rules
├── CONTRIBUTING.md               # Contribution guidelines
├── LICENSE                       # Project license
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

### Key Directories

| Directory | Purpose |
|---|---|
| `configs/` | YAML configuration files for hyperparameters (learning rate, epochs, ε values, etc.) |
| `data/raw/` | Store the original NSL-KDD `.csv` / `.arff` files (not committed to Git) |
| `data/processed/` | Cached preprocessed tensors / DataFrames |
| `notebooks/` | Jupyter notebooks for EDA, result visualization, and experimentation |
| `results/figures/` | Saved plots (accuracy vs. ε curves, confusion matrices, etc.) |
| `results/tables/` | Markdown / CSV performance tables |
| `results/models/` | Serialized PyTorch model checkpoints (`.pt`) |
| `scripts/` | Top-level runnable scripts for training, evaluation, and sweeps |
| `src/` | Core library code — data loading, model, attacks, defenses, evaluation, utils |
| `tests/` | Unit and integration tests |

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip (or conda)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/<org>/adversarial-ids-evasion-analysis.git
cd adversarial-ids-evasion-analysis

# Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset

This project uses the **NSL-KDD** dataset. Download the dataset files and place them in `data/raw/`:

1. Obtain the dataset from the [official NSL-KDD page](https://www.unb.ca/cic/datasets/nsl.html) or [Kaggle mirror](https://www.kaggle.com/datasets/hassan06/nslkdd).
2. Place `KDDTrain+.csv` and `KDDTest+.csv` (or equivalent) into `data/raw/`.

> **Note:** Raw data files are excluded from version control via `.gitignore`.

---

## Methodology

### Data Preprocessing

The NSL-KDD dataset contains both **continuous** features (e.g., duration, source bytes, destination bytes) and **categorical** features (e.g., protocol type, service type).

- **Continuous features** are standardized to zero mean and unit variance so that adversarial perturbations bounded by ε are applied proportionally across features.
- **Categorical features** are converted using **one-hot encoding**. During adversarial attacks, only continuous features are perturbed; one-hot encoded features remain fixed to preserve valid network configurations.
- After adversarial sample generation, continuous features are **clipped** to a valid range (training min/max) to maintain realistic inputs.
- The classification task is formulated as **binary**: all attack categories → label `0` (attack), normal traffic → label `1` (normal).

### Model Architecture

We implement a **Multi-Layer Perceptron (MLP)**:

| Layer | Details |
|---|---|
| Input | Number of processed features (after encoding) |
| Hidden 1 | 64 neurons, ReLU activation |
| Hidden 2 | 32 neurons, ReLU activation |
| Output | 1 neuron, Sigmoid activation (binary classification) |

The sigmoid output produces a probability representing the likelihood of normal behavior.

### Training Procedure

| Component | Choice |
|---|---|
| **Loss Function** | Binary Cross-Entropy (BCE) |
| **Optimizer** | Adam (adaptive learning rate) |
| **Training Split** | ~70% train / ~15% validation / ~15% test |
| **Early Stopping** | Optional, based on validation loss |

- The **training set** is used to learn model parameters.
- The **validation set** is used for hyperparameter tuning and early stopping.
- The **test set** is reserved strictly for final evaluation and adversarial robustness analysis.

### Adversarial Attacks

Two white-box gradient-based attacks under an **L∞ norm** constraint:

| Attack | Type | Description |
|---|---|---|
| **FGSM** | Single-step | One gradient-sign update scaled by ε |
| **PGD** | Iterative | Multiple smaller steps with projection back into the ε-ball |

Both attacks perturb input features in the direction that **maximizes the model's loss**. No individual feature can be modified beyond ±ε.

### Adversarial Training (Defense)

A second MLP is trained using **adversarial examples generated via FGSM** during training. The training set is augmented with adversarially perturbed samples to encourage the model to learn more robust decision boundaries.

**Models compared:**

| Model | Training Data |
|---|---|
| Baseline MLP | Clean training data only |
| Adversarially Trained MLP | Clean + FGSM-perturbed training data |

### Robustness Evaluation & ε-Sweep

Robustness is analyzed across multiple perturbation strengths:

$$\varepsilon \in \{\varepsilon_1, \varepsilon_2, \varepsilon_3, \ldots\}$$

For each ε value:
1. Generate FGSM adversarial examples
2. Generate PGD adversarial examples
3. Evaluate both the baseline and adversarially trained models

### Evaluation Metrics

| Metric | Description |
|---|---|
| **Clean Accuracy** | Accuracy on unperturbed test data |
| **Robust Accuracy** | Accuracy on adversarially perturbed test data |
| **Accuracy vs. ε Curves** | Plots showing degradation as ε increases |

Results will be presented in tables and plots for both FGSM and PGD across both models.

---

## Usage

All runnable scripts are in the `scripts/` directory. Configuration is loaded from `configs/default.yaml`.

### Train the Baseline Model

```bash
python scripts/train_baseline.py --config configs/default.yaml
```

### Train the Adversarially Robust Model

```bash
python scripts/train_adversarial.py --config configs/default.yaml
```

### Run Adversarial Evaluation

```bash
python scripts/evaluate.py --model results/models/baseline.pt --config configs/default.yaml
python scripts/evaluate.py --model results/models/adversarial.pt --config configs/default.yaml
```

### Run ε-Sweep Experiments

```bash
python scripts/run_epsilon_sweep.py --config configs/default.yaml
```

> Plots and tables are saved to `results/figures/` and `results/tables/`, respectively.

---

## Expected Results

- The **baseline model** is expected to show significant accuracy degradation as ε increases under both FGSM and PGD attacks.
- The **adversarially trained model** is expected to exhibit improved robust accuracy at the cost of slightly reduced clean accuracy.
- Final deliverables include **performance tables** and **visualizations** illustrating robustness trends across ε values, along with an analysis of the **accuracy–robustness trade-off** in intrusion detection systems.

---

## Tools & Computation

| Category | Tools |
|---|---|
| **Data Processing** | pandas, NumPy, scikit-learn |
| **Deep Learning** | PyTorch |
| **Adversarial Attacks** | Custom FGSM/PGD implementations (PyTorch) |
| **Visualization** | Matplotlib, seaborn |
| **Configuration** | PyYAML |
| **Environment** | Python 3.9+, CPU (GPU optional) |

Experiments will be conducted on standard personal computing environments using CPU, with optional GPU acceleration for faster training and attack generation.

---

## Literature & References

1. **Goodfellow, I. J., Shlens, J., & Szegedy, C.** (2015). *Explaining and Harnessing Adversarial Examples.* ICLR 2015. [arXiv:1412.6572](https://arxiv.org/abs/1412.6572)
2. **Cinà, A. E., et al.** (2023). *Wild Patterns Reloaded: A Survey of Machine Learning Security against Training Data Poisoning.* [arXiv:2205.01992](https://arxiv.org/abs/2205.01992)
3. **Costa, V. G., et al.** (2024). *How Deep Learning Sees the World: A Survey on Adversarial Attacks & Defenses.*
4. **Li, Y., et al.** (2022). *A Review of Adversarial Attack and Defense for Classification Methods.*
5. **Li, et al.** *Adversarial Defense in Modulation Recognition via Diffusion and Segment-Wise Classification (FlowSlicer).*
6. **NSL-KDD Dataset** — Canadian Institute for Cybersecurity, University of New Brunswick. [https://www.unb.ca/cic/datasets/nsl.html](https://www.unb.ca/cic/datasets/nsl.html)

---

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project, including branching conventions, code style, and the pull request process.

---

## License

This project is licensed under the terms of the [MIT License](LICENSE).
