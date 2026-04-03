"""Train the baseline MLP classifier on the NSL-KDD dataset.

Usage:
    python scripts/train_baseline.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from sklearn.model_selection import train_test_split

from src.attacks.fgsm import fgsm_attack
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


def tensor_from_loader(data_loader):
    inputs = []
    labels = []
    for batch_inputs, batch_labels in data_loader:
        inputs.append(batch_inputs)
        labels.append(batch_labels)
    return torch.cat(inputs, dim=0), torch.cat(labels, dim=0)


def train_baseline(config_path: str) -> dict[str, float]:
    config = load_config(config_path)
    train_frame, test_frame = load_dataset_pair(config["dataset"]["raw_dir"])

    train_frame = add_binary_label_column(train_frame)
    test_frame = add_binary_label_column(test_frame)

    categorical_columns = config["preprocessing"]["categorical_features"]
    continuous_columns = [column for column in train_frame.columns if column not in categorical_columns + ["class", "label"]]

    # Split the training frame into train/validation so the existing report setup matches the notebook flow.
    x_train_full = train_frame.drop(columns=["class", "label"])
    y_train_full = train_frame["label"]
    x_test = test_frame.drop(columns=["class", "label"])
    y_test = test_frame["label"]

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=config["split"]["val"],
        stratify=y_train_full,
        random_state=config["split"]["random_seed"],
    )

    loaders = build_loaders(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        categorical_columns=categorical_columns,
        continuous_columns=continuous_columns,
        batch_size=config["training"]["batch_size"],
    )

    input_dim = next(iter(loaders.train_loader))[0].shape[1]
    model = MLP(input_dim=input_dim, hidden_layers=tuple(config["model"]["hidden_layers"]))
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    for _ in range(config["training"]["epochs"]):
        model.train()
        for inputs, labels in loaders.train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    save_path = Path(config["paths"]["model_save_dir"])
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / "baseline.pt")

    metrics = evaluate_classification(model, loaders.test_loader, criterion=criterion)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the baseline model.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    metrics = train_baseline(args.config)
    print(metrics)


if __name__ == "__main__":
    main()
