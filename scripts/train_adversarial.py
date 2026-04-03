"""Train an adversarially robust MLP using FGSM-augmented training data.

Usage:
    python scripts/train_adversarial.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from sklearn.model_selection import train_test_split

from src.data.loader import add_binary_label_column, load_nsl_kdd_csv, load_nsl_kdd_hf
from src.data.preprocessing import build_loaders
from src.defenses.adversarial_training import evaluate_model, train_with_fgsm_adversarial_examples
from src.models.mlp import MLP


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_dataset_pair(raw_dir: str):
    raw_path = Path(raw_dir)
    if (raw_path / "KDDTrain+.csv").exists() and (raw_path / "KDDTest+.csv").exists():
        return load_nsl_kdd_csv(raw_path)
    return load_nsl_kdd_hf()


def train_adversarial(config_path: str) -> dict[str, float]:
    config = load_config(config_path)
    train_frame, test_frame = load_dataset_pair(config["dataset"]["raw_dir"])

    train_frame = add_binary_label_column(train_frame)
    test_frame = add_binary_label_column(test_frame)

    categorical_columns = config["preprocessing"]["categorical_features"]
    continuous_columns = [column for column in train_frame.columns if column not in categorical_columns + ["class", "label"]]

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

    train_with_fgsm_adversarial_examples(
        model=model,
        train_loader=loaders.train_loader,
        val_loader=loaders.val_loader,
        optimizer=optimizer,
        criterion=criterion,
        continuous_indices=loaders.continuous_indices,
        categorical_indices=loaders.categorical_indices,
        epsilon=config["adversarial_training"]["epsilon"],
        epochs=config["training"]["epochs"],
        feature_bounds=(loaders.feature_min, loaders.feature_max),
    )

    save_path = Path(config["paths"]["model_save_dir"])
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / "adversarial.pt")

    metrics = evaluate_model(model, loaders.test_loader, criterion)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the adversarially robust model.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    metrics = train_adversarial(args.config)
    print(metrics)


if __name__ == "__main__":
    main()
