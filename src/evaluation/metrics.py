"""Classification accuracy and robust accuracy metrics."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_classification(model: torch.nn.Module, data_loader, criterion: torch.nn.Module | None = None) -> dict[str, float]:
	"""Evaluate a binary classifier on a data loader."""

	model_was_training = model.training
	model.eval()

	predictions: list[float] = []
	probabilities: list[float] = []
	labels: list[float] = []
	total_loss = 0.0

	with torch.no_grad():
		for inputs, batch_labels in data_loader:
			logits = model(inputs)
			probs = torch.sigmoid(logits)
			preds = (probs >= 0.5).float()

			if criterion is not None:
				total_loss += criterion(logits, batch_labels).item()

			predictions.extend(preds.cpu().numpy().ravel().tolist())
			probabilities.extend(probs.cpu().numpy().ravel().tolist())
			labels.extend(batch_labels.cpu().numpy().ravel().tolist())

	if model_was_training:
		model.train()

	labels_array = np.asarray(labels)
	predictions_array = np.asarray(predictions)
	probabilities_array = np.asarray(probabilities)

	metrics = {
		"accuracy": float((predictions_array == labels_array).mean()),
		"precision": float(precision_score(labels_array, predictions_array, zero_division=0)),
		"recall": float(recall_score(labels_array, predictions_array, zero_division=0)),
		"f1": float(f1_score(labels_array, predictions_array, zero_division=0)),
		"roc_auc": float(roc_auc_score(labels_array, probabilities_array)),
		"pr_auc": float(average_precision_score(labels_array, probabilities_array)),
	}

	if criterion is not None:
		metrics["loss"] = total_loss / max(len(data_loader), 1)

	return metrics


def compare_model_outputs(clean_metrics: dict[str, float], adversarial_metrics: dict[str, float]) -> dict[str, float]:
	"""Compute the deltas between two metric dictionaries."""

	return {
		key: adversarial_metrics[key] - clean_metrics[key]
		for key in clean_metrics
		if key in adversarial_metrics
	}
