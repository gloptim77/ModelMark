import torch
import torch.nn as nn

import config

class ConvModel(nn.Module):
	def __init__(self, input_dim : int, output_dim : int, context_size : int):
		super().__init__()

		self.input_size = input_dim
		self.output_size = output_dim
		self.context_size = context_size
		
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
		self.fc = nn.Linear(
			self.hidden_size,
			context_size * self.output_size,
		)

	def forward(self, x):
		# x: [batch, sequence_length, input_size]

		# Conv1d expects:
		# [batch, channels, sequence_length]
		x = x.transpose(1, 2)

		# [batch, hidden_size, sequence_length]
		x = self.conv(x)

		# [batch, hidden_size, 1]
		x = self.pool(x)

		# [batch, hidden_size]
		x = x.squeeze(-1)

		# [batch, context_size * output_size]
		out = self.fc(x)

		# [batch, context_size, output_size]
		out = out.view(
			x.size(0),
			self.context_size,
			self.output_size,
		)

		return out