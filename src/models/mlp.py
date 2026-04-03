"""Multi-Layer Perceptron (MLP) architecture for binary intrusion detection."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch.nn as nn


class MLP(nn.Module):
	"""Simple feedforward network for binary tabular classification."""

	def __init__(self, input_dim: int, hidden_layers: Sequence[int] = (64, 32)) -> None:
		super().__init__()

		layers: list[nn.Module] = []
		current_dim = input_dim
		for hidden_dim in hidden_layers:
			layers.append(nn.Linear(current_dim, hidden_dim))
			layers.append(nn.ReLU())
			current_dim = hidden_dim
		layers.append(nn.Linear(current_dim, 1))

		self.network = nn.Sequential(*layers)

	def forward(self, inputs):
		return self.network(inputs)
