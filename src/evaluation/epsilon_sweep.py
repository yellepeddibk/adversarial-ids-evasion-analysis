"""Epsilon-sweep robustness evaluation loop."""

from __future__ import annotations

from typing import Callable, Sequence

import pandas as pd
import torch

from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.evaluation.metrics import evaluate_classification


def evaluate_epsilon_sweep(
	model: torch.nn.Module,
	inputs: torch.Tensor,
	labels: torch.Tensor,
	continuous_indices: Sequence[int],
	categorical_indices: Sequence[int],
	criterion: torch.nn.Module,
	epsilons: Sequence[float],
	attack: str = "fgsm",
	feature_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> pd.DataFrame:
	"""Evaluate a model under FGSM or PGD across multiple epsilon values."""

	rows: list[dict[str, float]] = []
	clean_accuracy = evaluate_classification(model, [(inputs, labels)], criterion=criterion)["accuracy"]

	for epsilon in epsilons:
		if attack == "fgsm":
			adversarial_inputs = fgsm_attack(
				model=model,
				inputs=inputs,
				labels=labels,
				epsilon=epsilon,
				continuous_indices=continuous_indices,
				categorical_indices=categorical_indices,
				criterion=criterion,
				feature_bounds=feature_bounds,
			)
		elif attack == "pgd":
			adversarial_inputs = pgd_attack(
				model=model,
				inputs=inputs,
				labels=labels,
				epsilon=epsilon,
				continuous_indices=continuous_indices,
				categorical_indices=categorical_indices,
				criterion=criterion,
				feature_bounds=feature_bounds,
			)
		else:
			raise ValueError(f"Unsupported attack type: {attack}")

		with torch.no_grad():
			adversarial_metrics = evaluate_classification(model, [(adversarial_inputs, labels)], criterion=criterion)

		rows.append(
			{
				"epsilon": float(epsilon),
				"clean_accuracy": clean_accuracy,
				f"{attack}_accuracy": adversarial_metrics["accuracy"],
				"accuracy_drop": clean_accuracy - adversarial_metrics["accuracy"],
			}
		)

	return pd.DataFrame(rows)
