"""Dataset loading utilities for the NSL-KDD dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd


def load_nsl_kdd_csv(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Load the standard NSL-KDD train and test CSV files."""

	base_path = Path(data_dir)
	train_path = base_path / "KDDTrain+.csv"
	test_path = base_path / "KDDTest+.csv"

	train_frame = pd.read_csv(train_path)
	test_frame = pd.read_csv(test_path)
	return train_frame, test_frame


def load_nsl_kdd_hf(dataset_name: str = "Mireu-Lab/NSL-KDD") -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Load the NSL-KDD dataset from the Hugging Face dataset hub."""

	from datasets import load_dataset

	dataset = load_dataset(dataset_name)
	return dataset["train"].to_pandas(), dataset["test"].to_pandas()


def add_binary_label_column(frame: pd.DataFrame, class_column: str = "class") -> pd.DataFrame:
	"""Map normal traffic to 1 and attacks to 0."""

	output = frame.copy()
	output["label"] = output[class_column].apply(lambda value: 1 if value == "normal" else 0)
	return output
