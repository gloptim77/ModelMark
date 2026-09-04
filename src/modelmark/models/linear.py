""" 
Description:
	
	This is an example of the Linear model definition.
	You can write any other model, but the core components must remain intact:

		1) The model __init__ core args must be included:
			
			(input_dim : int, output_dim : int, input_len: int, output_len: int)

			For mode parameter tweaking use your config file, 
			or define params right in the __init__() method (not recommended, config is more convenient).

		2) The forward() method:

			Input - [B, input_len, input_dim]
			Output - [B, output_len, output_dim]

			Currently only the continuous time-series data forecasting is available.

		3) Differentiability:

			The model should be differentiable, we will optimize it with Adam(). 

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Switch to user's config when available
from modelmark.common.utils import load_config

class Linear(nn.Module):
	def __init__(self, input_dim: int, output_dim: int, input_len: int, output_len: int):
		super().__init__()

		# It lives here to avoid circular config
		config = load_config()

		self.input_size = input_dim
		self.output_size = output_dim
		self.input_len = input_len
		self.output_len = output_len

		self.hidden_size = config.model_config["Linear"]["hidden_size"]
		self.num_layers = config.model_config["Linear"]["num_layers"]

		layers = [
			nn.Linear(self.input_size, self.hidden_size),
			nn.GELU(),
		]

		for _ in range(self.num_layers - 1):
			layers.extend([
				nn.Linear(self.hidden_size, self.hidden_size),
				nn.GELU(),
			])

		self.net = nn.Sequential(*layers)
		self.out = nn.Linear(self.hidden_size, self.output_size * self.output_len)

	def forward(self, x):
		"""
			input: [B, input_len, input_dim]
			output: [B, output_len, output_dim]			
		"""

		if x.ndim != 3:
			raise ValueError(f"Expected x to have shape [B, L, D], got {x.shape}")

		x = self.net(x)

		# Reshape to output dim and len
		x = x.mean(1)
		x = x.view(x.size(0), -1)
		x = self.out(x)
		x = x.view(x.size(0), self.output_len, self.output_size)

		return x