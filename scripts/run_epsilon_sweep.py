"""Run epsilon-sweep robustness experiments across multiple perturbation budgets.

Usage:
    python scripts/run_epsilon_sweep.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from src.data.loader import add_binary_label_column, load_nsl_kdd_csv, load_nsl_kdd_hf
from src.data.preprocessing import build_loaders
from src.evaluation.epsilon_sweep import evaluate_epsilon_sweep
from src.models.mlp import MLP


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_dataset_pair(raw_dir: str):
    raw_path = Path(raw_dir)
    if (raw_path / "KDDTrain+.csv").exists() and (raw_path / "KDDTest+.csv").exists():
        return load_nsl_kdd_csv(raw_path)
    return load_nsl_kdd_hf()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run epsilon sweep experiments.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--attack", choices=["fgsm", "pgd"], default="fgsm")
    parser.add_argument("--model", default="results/models/baseline.pt")
    args = parser.parse_args()

    config = load_config(args.config)
    train_frame, test_frame = load_dataset_pair(config["dataset"]["raw_dir"])

    train_frame = add_binary_label_column(train_frame)
    test_frame = add_binary_label_column(test_frame)

    categorical_columns = config["preprocessing"]["categorical_features"]
    continuous_columns = [column for column in train_frame.columns if column not in categorical_columns + ["class", "label"]]

    x_train = train_frame.drop(columns=["class", "label"])
    y_train = train_frame["label"]
    x_test = test_frame.drop(columns=["class", "label"])
    y_test = test_frame["label"]

    loaders = build_loaders(
        x_train=x_train,
        x_val=x_test,
        x_test=x_test,
        y_train=y_train,
        y_val=y_test,
        y_test=y_test,
        categorical_columns=categorical_columns,
        continuous_columns=continuous_columns,
        batch_size=config["training"]["batch_size"],
    )

    input_dim = next(iter(loaders.train_loader))[0].shape[1]
    model = MLP(input_dim=input_dim, hidden_layers=tuple(config["model"]["hidden_layers"]))
    criterion = torch.nn.BCEWithLogitsLoss()

    model_path = Path(args.model)
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

    test_inputs, test_labels = next(iter([(torch.cat([batch[0] for batch in loaders.test_loader], dim=0), torch.cat([batch[1] for batch in loaders.test_loader], dim=0))]))
    epsilon_values = config["epsilon_sweep"]["values"]

    sweep_df = evaluate_epsilon_sweep(
        model=model,
        inputs=test_inputs,
        labels=test_labels,
        continuous_indices=loaders.continuous_indices,
        categorical_indices=loaders.categorical_indices,
        criterion=criterion,
        epsilons=epsilon_values,
        attack=args.attack,
        feature_bounds=(loaders.feature_min, loaders.feature_max),
    )

    print(sweep_df.to_string(index=False))


if __name__ == "__main__":
    main()
