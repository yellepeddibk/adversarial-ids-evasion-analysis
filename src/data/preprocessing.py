"""Feature encoding, scaling, and train/val/test splitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class PreprocessedSplits:
	train_loader: DataLoader
	val_loader: DataLoader
	test_loader: DataLoader
	continuous_indices: list[int]
	categorical_indices: list[int]
	feature_min: torch.Tensor
	feature_max: torch.Tensor
	encoder: OneHotEncoder
	scaler: StandardScaler


def binary_label_split(
	frame: pd.DataFrame,
	label_column: str = "label",
	test_size: float = 0.2,
	val_size: float = 0.25,
	random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
	"""Split a labeled DataFrame into train, validation, and test sets."""

	features = frame.drop(columns=[label_column])
	labels = frame[label_column]

	x_temp, x_test, y_temp, y_test = train_test_split(
		features,
		labels,
		test_size=test_size,
		stratify=labels,
		random_state=random_state,
	)

	x_train, x_val, y_train, y_val = train_test_split(
		x_temp,
		y_temp,
		test_size=val_size,
		stratify=y_temp,
		random_state=random_state,
	)

	return x_train, x_val, x_test, y_train, y_val, y_test


def fit_preprocessor(
	x_train: pd.DataFrame,
	categorical_columns: Sequence[str],
	continuous_columns: Sequence[str],
) -> tuple[OneHotEncoder, StandardScaler, np.ndarray, torch.Tensor, torch.Tensor]:
	"""Fit the encoder/scaler pair and return processed train arrays plus bounds."""

	encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
	scaler = StandardScaler()

	train_cat = np.asarray(encoder.fit_transform(x_train[list(categorical_columns)]))
	train_cont = np.asarray(scaler.fit_transform(x_train[list(continuous_columns)]))
	train_processed = np.hstack([train_cont, train_cat])

	feature_min = torch.tensor(train_processed.min(axis=0), dtype=torch.float32)
	feature_max = torch.tensor(train_processed.max(axis=0), dtype=torch.float32)
	return encoder, scaler, train_processed, feature_min, feature_max


def transform_features(
	frame: pd.DataFrame,
	encoder: OneHotEncoder,
	scaler: StandardScaler,
	categorical_columns: Sequence[str],
	continuous_columns: Sequence[str],
) -> np.ndarray:
	"""Transform a feature frame into the processed numeric matrix."""

	categorical = np.asarray(encoder.transform(frame[list(categorical_columns)]))
	continuous = np.asarray(scaler.transform(frame[list(continuous_columns)]))
	return np.hstack([continuous, categorical])


def build_loaders(
	x_train: pd.DataFrame,
	x_val: pd.DataFrame,
	x_test: pd.DataFrame,
	y_train: pd.Series,
	y_val: pd.Series,
	y_test: pd.Series,
	categorical_columns: Sequence[str],
	continuous_columns: Sequence[str],
	batch_size: int = 64,
) -> PreprocessedSplits:
	"""Fit preprocessing on the training set and create PyTorch data loaders."""

	encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
	scaler = StandardScaler()

	train_cat = np.asarray(encoder.fit_transform(x_train[list(categorical_columns)]))
	val_cat = np.asarray(encoder.transform(x_val[list(categorical_columns)]))
	test_cat = np.asarray(encoder.transform(x_test[list(categorical_columns)]))

	train_cont = np.asarray(scaler.fit_transform(x_train[list(continuous_columns)]))
	val_cont = np.asarray(scaler.transform(x_val[list(continuous_columns)]))
	test_cont = np.asarray(scaler.transform(x_test[list(continuous_columns)]))

	train_processed = np.hstack([train_cont, train_cat])
	val_processed = np.hstack([val_cont, val_cat])
	test_processed = np.hstack([test_cont, test_cat])

	y_train_tensor = torch.tensor(y_train.to_numpy(dtype=np.float32)).view(-1, 1)
	y_val_tensor = torch.tensor(y_val.to_numpy(dtype=np.float32)).view(-1, 1)
	y_test_tensor = torch.tensor(y_test.to_numpy(dtype=np.float32)).view(-1, 1)

	train_tensor = torch.tensor(train_processed, dtype=torch.float32)
	val_tensor = torch.tensor(val_processed, dtype=torch.float32)
	test_tensor = torch.tensor(test_processed, dtype=torch.float32)

	train_loader = DataLoader(TensorDataset(train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(TensorDataset(val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
	test_loader = DataLoader(TensorDataset(test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

	continuous_indices = list(range(len(continuous_columns)))
	categorical_indices = list(range(len(continuous_columns), train_tensor.shape[1]))

	feature_min = torch.tensor(train_processed.min(axis=0), dtype=torch.float32)
	feature_max = torch.tensor(train_processed.max(axis=0), dtype=torch.float32)

	return PreprocessedSplits(
		train_loader=train_loader,
		val_loader=val_loader,
		test_loader=test_loader,
		continuous_indices=continuous_indices,
		categorical_indices=categorical_indices,
		feature_min=feature_min,
		feature_max=feature_max,
		encoder=encoder,
		scaler=scaler,
	)
