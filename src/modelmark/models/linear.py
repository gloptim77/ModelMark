""" 
Description:
	
	This is an example of the Linear model definition.
	You can write any other model, but the core components must remain intact:

		1) The model __init__ core args must be included:
			
			(input_dim : int, output_dim : int, context_size: int)

			For mode parameter tweaking use your config file, 
			or define params right in the __init__() method (not recommended, config is more convenient).

		2) The forward() method:

			Input - [B, T, input_dim]
			Output - [B, T, output_dim]

			Currently only the continuous time-series data forecasting is available.

		3) Differentiability:

			The model should be differentiable, we will optimize it with Adam(). 

"""

import torch
import torch.nn as nn

# Switch to user's config when available
from modelmark.common.utils import load_config

class Linear(nn.Module):
	def __init__(self, input_dim : int, output_dim : int, context_size: int):
		super().__init__()

		# To avoid circular import, now it lives here
		config = load_config()

		self.input_size = input_dim
		self.output_size = output_dim
		self.context_size = context_size
		
		self.hidden_size = config.model_config["Linear"]["hidden_size"]
		self.num_layers = config.model_config["Linear"]["num_layers"]

		layers = [
			nn.Linear(self.input_size, self.hidden_size),
			nn.GELU()
		]

		for _ in range(self.num_layers - 1):
			layers.append(nn.Linear(self.hidden_size, self.hidden_size))
			layers.append(nn.GELU())

		self.net = nn.Sequential(*layers)

		self.fc = nn.Linear(self.hidden_size, self.output_size)

	def forward(self, x):

		x = self.net(x)
		x = self.fc(x)

		return x