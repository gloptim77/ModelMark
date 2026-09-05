import torch
import torch.nn as nn
import torch.nn.functional as F

from modelmark.common.utils import load_config

class GRUModel(nn.Module):
	def __init__(self, input_dim: int, output_dim: int, input_len: int, output_len: int):
		super().__init__()

		# To avoid circular import, now it lives here
		config = load_config()

		self.input_dim = input_dim
		self.output_dim = output_dim
		self.input_len = input_len
		self.output_len = output_len
		
		self.hidden_size = config.model_config["GRU"]["hidden_size"]
		self.num_layers = config.model_config["GRU"]["num_layers"]

		self.gru = nn.GRU(
			input_size = self.input_dim,
			hidden_size = self.hidden_size,
			num_layers = self.num_layers,
			batch_first = True,
		)

		self.out = nn.Linear(self.hidden_size, self.output_len * self.output_dim)

	def forward(self, x):
		"""
			input: [B, input_len, input_dim]
			output: [B, output_len, output_dim]			
		"""

		if x.ndim != 3:
			raise ValueError(f"Expected x to have shape [B, L, D], got {x.shape}")

		_, hn = self.gru(x)

		# Last layer's hidden state
		h = hn[-1]

		x = self.out(h)
		# Project to output dim and len
		x = x.view(x.size(0), self.output_len, self.output_dim)

		return x