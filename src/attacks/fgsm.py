"""Fast Gradient Sign Method (FGSM) attack implementation."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch


def _normalize_feature_indices(feature_indices: Sequence[int] | None, width: int) -> list[int]:
	if feature_indices is None:
		return list(range(width))
	return list(feature_indices)


def fgsm_attack(
	model: torch.nn.Module,
	inputs: torch.Tensor,
	labels: torch.Tensor,
	epsilon: float,
	continuous_indices: Sequence[int],
	categorical_indices: Sequence[int] | None = None,
	criterion: torch.nn.Module | None = None,
	feature_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
	"""Generate FGSM adversarial examples.

	Only continuous features are perturbed. Categorical / one-hot features are
	restored from the clean input after the attack step.
	"""

	if criterion is None:
		criterion = torch.nn.BCEWithLogitsLoss()

	model_was_training = model.training
	model.eval()

	adv_inputs = inputs.detach().clone().requires_grad_(True)
	logits = model(adv_inputs)
	loss = criterion(logits, labels)

	grad = torch.autograd.grad(loss, adv_inputs, retain_graph=False, create_graph=False)[0]
	perturbation = torch.zeros_like(adv_inputs)
	perturbation[:, list(continuous_indices)] = epsilon * grad[:, list(continuous_indices)].sign()

	adv_inputs = inputs.detach() + perturbation

	if feature_bounds is not None:
		feature_min, feature_max = feature_bounds
		adv_inputs[:, list(continuous_indices)] = torch.clamp(
			adv_inputs[:, list(continuous_indices)],
			min=feature_min[list(continuous_indices)],
			max=feature_max[list(continuous_indices)],
		)

	if categorical_indices is not None:
		adv_inputs[:, list(categorical_indices)] = inputs[:, list(categorical_indices)]

	if model_was_training:
		model.train()

	return adv_inputs.detach()
