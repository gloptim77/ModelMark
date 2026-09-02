""" 
Put your final model's class definition here.
As an example, here's the Linear model definition.
"""

import torch
import torch.nn as nn

import config

class Model(nn.Module):
	def __init__(self, input_dim : int, output_dim : int, context_size: int):
		super().__init__()

		self.input_size = input_dim
		self.output_size = output_dim
		self.context_size = context_size
		
		self.hidden_size = config.model_config[config.test_subj_model_name]["hidden_size"]
		self.num_layers = config.model_config[config.test_subj_model_name]["num_layers"]

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