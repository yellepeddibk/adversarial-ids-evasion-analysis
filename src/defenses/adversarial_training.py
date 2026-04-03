"""Adversarial training defense — train with FGSM-augmented data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from src.attacks.fgsm import fgsm_attack


@dataclass
class TrainingHistory:
	train_loss: list[float]
	train_accuracy: list[float]
	val_accuracy: list[float]


def _accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
	predictions = (torch.sigmoid(logits) >= 0.5).float()
	return (predictions == labels).float().mean().item()


def train_with_fgsm_adversarial_examples(
	model: torch.nn.Module,
	train_loader,
	val_loader,
	optimizer: torch.optim.Optimizer,
	criterion: torch.nn.Module,
	continuous_indices: Sequence[int],
	categorical_indices: Sequence[int],
	epsilon: float,
	epochs: int = 10,
	feature_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> TrainingHistory:
	"""Train a model on clean and FGSM-augmented batches."""

	history = TrainingHistory(train_loss=[], train_accuracy=[], val_accuracy=[])

	for _ in range(epochs):
		model.train()
		epoch_loss = 0.0
		correct = 0.0
		total = 0.0

		for clean_inputs, labels in train_loader:
			adv_inputs = fgsm_attack(
				model=model,
				inputs=clean_inputs,
				labels=labels,
				epsilon=epsilon,
				continuous_indices=continuous_indices,
				categorical_indices=categorical_indices,
				criterion=criterion,
				feature_bounds=feature_bounds,
			)

			combined_inputs = torch.cat([clean_inputs, adv_inputs], dim=0)
			combined_labels = torch.cat([labels, labels], dim=0)

			optimizer.zero_grad(set_to_none=True)
			logits = model(combined_inputs)
			loss = criterion(logits, combined_labels)
			loss.backward()
			optimizer.step()

			epoch_loss += loss.item()
			correct += (torch.sigmoid(logits) >= 0.5).float().eq(combined_labels).sum().item()
			total += combined_labels.size(0)

		history.train_loss.append(epoch_loss / max(len(train_loader), 1))
		history.train_accuracy.append(correct / max(total, 1.0))
		history.val_accuracy.append(evaluate_model(model, val_loader, criterion)["accuracy"])

	return history


def evaluate_model(model: torch.nn.Module, data_loader, criterion: torch.nn.Module) -> dict[str, float]:
	"""Evaluate a model on a loader and return standard classification metrics."""

	model_was_training = model.training
	model.eval()

	total_loss = 0.0
	total_correct = 0.0
	total_examples = 0.0

	with torch.no_grad():
		for inputs, labels in data_loader:
			logits = model(inputs)
			loss = criterion(logits, labels)
			total_loss += loss.item()
			total_correct += (torch.sigmoid(logits) >= 0.5).float().eq(labels).sum().item()
			total_examples += labels.size(0)

	if model_was_training:
		model.train()

	return {
		"loss": total_loss / max(len(data_loader), 1),
		"accuracy": total_correct / max(total_examples, 1.0),
	}
