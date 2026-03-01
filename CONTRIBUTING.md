# Contributing to Adversarial IDS Evasion Analysis

Thank you for your interest in contributing! This document outlines our conventions for branching, code style, commit messages, and the pull request process.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
  - [Branching Strategy](#branching-strategy)
  - [Making Changes](#making-changes)
  - [Commit Messages](#commit-messages)
- [Code Style & Standards](#code-style--standards)
- [Project Layout Conventions](#project-layout-conventions)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

---

## Getting Started

1. **Fork** (or clone) the repository.
2. Create a Python virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate

   pip install -r requirements.txt
   ```
3. Make sure the project runs and any existing tests pass before starting work.

---

## Development Workflow

### Branching Strategy

We follow a simple **feature-branch** workflow off of the `main` branch.

| Branch Pattern | Purpose |
|---|---|
| `main` | Stable, reviewed code only — never commit directly |
| `feature/<description>` | New features or experiments |
| `fix/<description>` | Bug fixes |
| `docs/<description>` | Documentation-only changes |
| `refactor/<description>` | Code restructuring without behavior change |

**Examples:**
```
feature/fgsm-attack-implementation
fix/preprocessing-scaling-bug
docs/update-readme-results
refactor/reorganize-evaluation-module
```

### Making Changes

1. Pull the latest `main`:
   ```bash
   git checkout main
   git pull origin main
   ```
2. Create your feature branch:
   ```bash
   git checkout -b feature/my-feature
   ```
3. Implement your changes with clear, incremental commits.
4. Push your branch and open a Pull Request against `main`.

### Commit Messages

Use clear, imperative-style commit messages:

```
<type>: <short summary>

<optional body explaining why / what changed>
```

**Types:** `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

**Examples:**
```
feat: implement FGSM attack module
fix: correct feature scaling in preprocessing pipeline
docs: add ε-sweep results to README
test: add unit tests for PGD attack
chore: update requirements.txt with matplotlib
```

---

## Code Style & Standards

- **Python version**: 3.9+
- **Formatter**: We recommend [Black](https://black.readthedocs.io/) with default settings (line length 88).
- **Linter**: [Flake8](https://flake8.pycqa.org/) or [Ruff](https://docs.astral.sh/ruff/) for style checks.
- **Type hints**: Use type annotations for function signatures where practical.
- **Docstrings**: Use Google-style or NumPy-style docstrings for all public functions and classes.
- **Imports**: Group imports in order — standard library, third-party, local — separated by blank lines. Use `isort` if available.

**Example function:**
```python
def fgsm_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    loss_fn: torch.nn.Module,
) -> torch.Tensor:
    """Generate adversarial examples using the FGSM attack.

    Args:
        model: Target neural network.
        x: Clean input tensor.
        y: True labels.
        epsilon: Maximum perturbation magnitude (L-inf).
        loss_fn: Loss function for computing gradients.

    Returns:
        Adversarial examples as a tensor.
    """
    ...
```

---

## Project Layout Conventions

- **Source code** goes in `src/` under the appropriate subpackage (`data`, `models`, `attacks`, `defenses`, `evaluation`, `utils`).
- **Runnable scripts** go in `scripts/`. Each script should be executable from the project root, e.g., `python scripts/train_baseline.py`.
- **Configuration** goes in `configs/` as YAML files. Avoid hard-coding hyperparameters in source code.
- **Notebooks** go in `notebooks/` and should be numbered for ordering (e.g., `01_data_exploration.ipynb`).
- **Results** (plots, tables, saved models) go in the appropriate `results/` subdirectory.
- **Raw data** goes in `data/raw/` and is **not** committed to Git (listed in `.gitignore`).
- **Processed data** goes in `data/processed/` and is also excluded from Git.

---

## Testing

- Place tests in the `tests/` directory.
- Use [pytest](https://docs.pytest.org/) as the test runner.
- Name test files `test_<module>.py` (e.g., `test_fgsm.py`).
- Run the test suite before opening a PR:
  ```bash
  pytest tests/
  ```

---

## Pull Request Process

1. **Ensure your branch is up to date** with `main`:
   ```bash
   git fetch origin
   git rebase origin/main
   ```
2. **Run tests** and verify nothing is broken.
3. **Push** your branch and open a **Pull Request** on GitHub.
4. Fill out the **PR template** completely — include a description, what was changed, how to test, and any relevant notes.
5. Request a review from at least **one team member**.
6. Address all review comments before merging.
7. After approval, the PR will be **squash-merged** into `main`.

### PR Checklist

Before submitting, confirm:

- [ ] Code follows the project style guidelines
- [ ] New code includes docstrings and type hints
- [ ] All existing tests pass (`pytest tests/`)
- [ ] New functionality includes tests (if applicable)
- [ ] Configuration changes are reflected in `configs/default.yaml`
- [ ] README or docs are updated if behavior changes

---

## Reporting Issues

If you find a bug or have a feature request:

1. Check existing [Issues](../../issues) to avoid duplicates.
2. Open a new issue with a clear title and description.
3. Include steps to reproduce (for bugs) or a rationale (for features).
4. Label appropriately: `bug`, `enhancement`, `question`, etc.

---

Thank you for contributing!
