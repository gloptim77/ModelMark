import torch
import torch.nn as nn

import config

class TransformerModel(nn.Module):
	def __init__(self, input_dim : int, output_dim : int, context_size: int):
		super().__init__()
	
		self.input_size = input_dim
		self.output_size = output_dim
		self.context_size = context_size
		self.max_seq_len = self.context_size
		
		self.hidden_size = config.model_config["Transformer"]["hidden_size"]
		self.num_layers = config.model_config["Transformer"]["num_layers"]
		self.num_heads = config.model_config["Transformer"]["num_heads"]
		self.dropout = config.model_config["Transformer"]["dropout"]

		# Input projection
		self.input_proj = nn.Linear(self.input_size, self.hidden_size,)

		# Learnable positional embedding
		self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_seq_len, self.hidden_size))

		# Transformer decoder layer
		decoder_layer = nn.TransformerDecoderLayer(
			d_model=self.hidden_size,
			nhead=self.num_heads,
			dim_feedforward=4 * self.hidden_size,
			dropout=self.dropout,
			batch_first=True,
		)

		self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)

		# Map hidden representation to outputs
		self.fc = nn.Linear(self.hidden_size, self.output_size)

	def forward(self, x):
		# x:
		# [batch, sequence_length, input_size]

		batch_size, seq_len, _ = x.shape

		if seq_len > self.pos_embedding.size(1):
			raise ValueError(
				f"Sequence length {seq_len} exceeds "
				f"max_seq_len {self.pos_embedding.size(1)}"
			)

		# Project input to transformer dimension
		x = self.input_proj(x)
		# [batch, sequence_length, hidden_size]

		# Add positional information
		x = x + self.pos_embedding[:, :seq_len, :]

		# Causal mask:
		# position i can only attend to positions <= i
		mask = nn.Transformer.generate_square_subsequent_mask(
			seq_len,
			device=x.device,
		)

		# Self-attention through decoder layers
		x = self.decoder(
			tgt=x,
			memory=x,
			tgt_mask=mask,
			memory_mask=mask,
		)
		# [batch, sequence_length, hidden_size]

		# Take the final context_size positions
		x = x[:, -self.context_size:, :]
		# [batch, context_size, hidden_size]

		# Project to output
		out = self.fc(x)
		# [batch, context_size, output_size]

		return out