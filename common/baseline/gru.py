import torch
import torch.nn as nn

import config

class GRUModel(nn.Module):
	def __init__(self, input_dim : int, output_dim : int, context_size: int):
		super().__init__()
	
		self.input_size = input_dim
		self.output_size = output_dim
		self.context_size = context_size
		
		self.hidden_size = config.model_config["GRU"]["hidden_size"]
		self.num_layers = config.model_config["GRU"]["num_layers"]

		self.gru = nn.GRU(
			input_size = self.input_size,
			hidden_size = self.hidden_size,
			num_layers = self.num_layers,
			batch_first = True,
		)

		self.fc = nn.Linear(
			self.hidden_size,
			context_size * self.output_size,
		)

	def forward(self, x):
		_, hn = self.gru(x)

		# Last layer's hidden state
		h = hn[-1]

		out = self.fc(h)
		out = out.view(
			x.size(0),
			self.context_size,
			self.output_size,
		)

		return out