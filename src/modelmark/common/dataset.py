from __future__ import annotations

import torch
from torch.utils.data import Dataset

import copy
import numpy as np
import pandas as pd

from modelmark.common.utils import load_config
config = load_config()

class CausalDataset(Dataset):
	"""
	Sliding-window dataset for ETT.

	Returns:
		x: [seq_len, num_features]
		y: [pred_len, 1]
	"""

	def __init__(
		self,
		data_config: dict,
		seq_len: int,
		pred_len: int,
		split: str = "train",
		mean: np.ndarray | None = None,
		std: np.ndarray | None = None,
	):
		super().__init__()

		self.seq_len = seq_len
		self.pred_len = pred_len

		# Extract config
		data_path = config.data_path + data_config["path"]
		features = copy.deepcopy(data_config["input_features"])
		targets = data_config["output_features"]
		features.extend([item for item in targets if item not in features])
		train_ratio = data_config["train_ratio"]
		val_ratio = data_config["val_ratio"]

		# Load, extract, convert
		df = pd.read_csv(data_path)
		
		df = df[features]
		data = df.values.astype(np.float32)
		self.input_indices = torch.tensor(df.columns.get_indexer(data_config["input_features"]), dtype=torch.int)
		self.output_indices = torch.tensor(df.columns.get_indexer(data_config["output_features"]), dtype=torch.int)

		n = len(data)

		train_end = int(n * train_ratio)
		val_end = int(n * (train_ratio + val_ratio))

		if split == "train":
			start = 0
			end = train_end
		elif split == "val":
			start = train_end
			end = val_end
		elif split == "test":
			start = val_end
			end = n
		else:
			raise ValueError(f"Unknown split: {split}")

		# Save scaling information
		if split == "train":
			self.mean = data[start:train_end].mean(axis=0)
			self.std = data[start:train_end].std(axis=0)

			# Prevent division by zero
			self.std[self.std < 1e-8] = 1.0
		else:
			if mean is None or std is None:
				raise ValueError(
					"Validation/test datasets need training mean and std."
				)

			self.mean = mean
			self.std = std

		# Scale using TRAIN statistics
		data = (data - self.mean) / self.std

		# Keep only the split region
		self.data = data[start:end]

		# Number of valid windows
		self.length = len(self.data) - seq_len - pred_len + 1

		if self.length <= 0:
			raise ValueError(
				f"Split '{split}' is too short for "
				f"seq_len={seq_len}, pred_len={pred_len}"
			)

	def __len__(self):
		return self.length

	def __getitem__(self, idx):
		x_start = idx
		x_end = x_start + self.seq_len

		y_start = x_end
		y_end = y_start + self.pred_len

		x = self.data[x_start:x_end, self.input_indices]
		y = self.data[y_start:y_end, self.output_indices]

		return (
			torch.from_numpy(x),
			torch.from_numpy(y),
		)