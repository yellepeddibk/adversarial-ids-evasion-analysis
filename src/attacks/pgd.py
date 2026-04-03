"""Projected Gradient Descent (PGD) attack implementation."""

from __future__ import annotations

from typing import Sequence

import torch


def pgd_attack(
	model: torch.nn.Module,
	inputs: torch.Tensor,
	labels: torch.Tensor,
	epsilon: float,
	continuous_indices: Sequence[int],
	categorical_indices: Sequence[int] | None = None,
	criterion: torch.nn.Module | None = None,
	num_steps: int = 10,
	step_size: float | None = None,
	feature_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
	"""Generate projected gradient descent adversarial examples."""

	if criterion is None:
		criterion = torch.nn.BCEWithLogitsLoss()
	if step_size is None:
		step_size = epsilon / 4 if epsilon > 0 else 0.0

	model_was_training = model.training
	model.eval()

	continuous_indices = list(continuous_indices)
	adv_inputs = inputs.detach().clone()

	if epsilon > 0:
		random_noise = torch.zeros_like(adv_inputs)
		random_noise[:, continuous_indices] = torch.empty(
			adv_inputs.size(0), len(continuous_indices), device=adv_inputs.device
		).uniform_(-epsilon, epsilon)
		adv_inputs = adv_inputs + random_noise

	original_inputs = inputs.detach().clone()

	for _ in range(num_steps):
		adv_inputs = adv_inputs.detach().clone().requires_grad_(True)
		logits = model(adv_inputs)
		loss = criterion(logits, labels)
		grad = torch.autograd.grad(loss, adv_inputs, retain_graph=False, create_graph=False)[0]

		perturbation = torch.zeros_like(adv_inputs)
		perturbation[:, continuous_indices] = step_size * grad[:, continuous_indices].sign()
		adv_inputs = adv_inputs.detach() + perturbation

		delta = adv_inputs - original_inputs
		delta[:, continuous_indices] = torch.clamp(delta[:, continuous_indices], min=-epsilon, max=epsilon)
		adv_inputs = original_inputs + delta

		if feature_bounds is not None:
			feature_min, feature_max = feature_bounds
			adv_inputs[:, continuous_indices] = torch.clamp(
				adv_inputs[:, continuous_indices],
				min=feature_min[continuous_indices],
				max=feature_max[continuous_indices],
			)

		if categorical_indices is not None:
			adv_inputs[:, list(categorical_indices)] = original_inputs[:, list(categorical_indices)]

	if model_was_training:
		model.train()

	return adv_inputs.detach()
