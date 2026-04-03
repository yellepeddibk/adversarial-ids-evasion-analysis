"""Evaluate model(s) on clean and adversarially perturbed test data.

Usage:
    python scripts/evaluate.py --model results/models/baseline.pt --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.data.loader import add_binary_label_column, load_nsl_kdd_csv, load_nsl_kdd_hf
from src.data.preprocessing import build_loaders
from src.evaluation.metrics import evaluate_classification
from src.models.mlp import MLP


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_dataset_pair(raw_dir: str):
    raw_path = Path(raw_dir)
    if (raw_path / "KDDTrain+.csv").exists() and (raw_path / "KDDTest+.csv").exists():
        return load_nsl_kdd_csv(raw_path)
    return load_nsl_kdd_hf()


def evaluate_model_suite(config_path: str) -> pd.DataFrame:
    config = load_config(config_path)
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

    model_path = Path(config["paths"]["model_save_dir"]) / "baseline.pt"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

    clean_metrics = evaluate_classification(model, loaders.test_loader, criterion=criterion)

    test_inputs, test_labels = next(iter([(torch.cat([batch[0] for batch in loaders.test_loader], dim=0), torch.cat([batch[1] for batch in loaders.test_loader], dim=0))]))
    fgsm_inputs = fgsm_attack(
        model=model,
        inputs=test_inputs,
        labels=test_labels,
        epsilon=config["attacks"]["fgsm"]["epsilon"],
        continuous_indices=loaders.continuous_indices,
        categorical_indices=loaders.categorical_indices,
        criterion=criterion,
        feature_bounds=(loaders.feature_min, loaders.feature_max),
    )
    pgd_inputs = pgd_attack(
        model=model,
        inputs=test_inputs,
        labels=test_labels,
        epsilon=config["attacks"]["pgd"]["epsilon"],
        continuous_indices=loaders.continuous_indices,
        categorical_indices=loaders.categorical_indices,
        criterion=criterion,
        num_steps=config["attacks"]["pgd"]["num_steps"],
        step_size=config["attacks"]["pgd"]["step_size"],
        feature_bounds=(loaders.feature_min, loaders.feature_max),
    )

    fgsm_metrics = evaluate_classification(model, [(fgsm_inputs, test_labels)], criterion=criterion)
    pgd_metrics = evaluate_classification(model, [(pgd_inputs, test_labels)], criterion=criterion)

    return pd.DataFrame(
        [
            {"setting": "clean", **clean_metrics},
            {"setting": "fgsm", **fgsm_metrics},
            {"setting": "pgd", **pgd_metrics},
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the model.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    print(evaluate_model_suite(args.config).to_string(index=False))


if __name__ == "__main__":
    main()
