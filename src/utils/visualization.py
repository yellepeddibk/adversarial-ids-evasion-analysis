"""Plotting and figure-generation utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_epsilon_curve(frame: pd.DataFrame, value_column: str, title: str, output_path: str | Path | None = None) -> None:
	"""Plot an epsilon-accuracy curve from a results DataFrame."""

	plt.figure(figsize=(8, 5))
	plt.plot(frame["epsilon"], frame[value_column], marker="o")
	plt.xlabel("Epsilon (ε)")
	plt.ylabel("Accuracy")
	plt.title(title)
	plt.grid(True)
	plt.tight_layout()

	if output_path is not None:
		Path(output_path).parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(output_path, dpi=200, bbox_inches="tight")

	plt.show()


def plot_comparison_curves(frame: pd.DataFrame, columns: list[str], title: str, output_path: str | Path | None = None) -> None:
	"""Plot multiple epsilon curves on the same axes."""

	plt.figure(figsize=(10, 6))
	for column in columns:
		plt.plot(frame["epsilon"], frame[column], marker="o", label=column)

	plt.xlabel("Epsilon (ε)")
	plt.ylabel("Accuracy")
	plt.title(title)
	plt.legend()
	plt.grid(True)
	plt.tight_layout()

	if output_path is not None:
		Path(output_path).parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(output_path, dpi=200, bbox_inches="tight")

	plt.show()
