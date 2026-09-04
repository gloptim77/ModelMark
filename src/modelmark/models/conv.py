import torch
import torch.nn as nn
import torch.nn.functional as F

from modelmark.common.utils import load_config

class ConvModel(nn.Module):
	def __init__(self, input_dim: int, output_dim: int, input_len: int, output_len: int):
		super().__init__()

		# To avoid circular import, now it lives here
		config = load_config()

		self.input_size = input_dim
		self.output_size = output_dim
		self.input_len = input_len
		self.output_len = output_len
		
		self.hidden_size = config.model_config["Conv"]["hidden_size"]
		self.num_layers = config.model_config["Conv"]["num_layers"]
		self.kernel_size = config.model_config["Conv"]["kernel_size"]
		
		layers = []

		in_channels = self.input_size

		for _ in range(self.num_layers):
			layers.append(
				nn.Conv1d(
					in_channels=in_channels,
					out_channels=self.hidden_size,
					kernel_size=self.kernel_size,
					padding=self.kernel_size // 2,
				)
			)
			layers.append(nn.ReLU())

			in_channels = self.hidden_size

		self.conv = nn.Sequential(*layers)

		# Collapse the sequence dimension to one feature vector
		self.pool = nn.AdaptiveAvgPool1d(1)

		# Produce exactly context_size * output_size values
		self.fc = nn.Linear(self.hidden_size, output_len  * self.output_size)

	def forward(self, x):
		"""
			input: [B, input_len, input_dim]
			output: [B, output_len, output_dim]			
		"""

		if x.ndim != 3:
			raise ValueError(f"Expected x to have shape [B, L, D], got {x.shape}")

		# [batch, channels, sequence_length]
		x = x.transpose(1, 2)
		# [batch, hidden_size, sequence_length]
		x = self.conv(x)
		# [batch, hidden_size, 1]
		x = self.pool(x)
		# [batch, hidden_size]
		x = x.squeeze(-1)
		# [batch, context_size * output_size]
		x = self.fc(x)
		# Project to output dim and len
		x = x.view(x.size(0), self.output_len, self.output_size)

		return x