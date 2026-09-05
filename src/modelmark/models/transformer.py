import torch
import torch.nn as nn
import torch.nn.functional as F

from modelmark.common.utils import load_config

class TransformerModel(nn.Module):
	def __init__(self, input_dim: int, output_dim: int, input_len: int, output_len: int):
		super().__init__()

		# To avoid circular import, now it lives here
		config = load_config()

		self.input_dim = input_dim
		self.output_dim = output_dim
		self.input_len = input_len
		self.output_len = output_len
		self.max_seq_len = input_len
		
		self.hidden_size = config.model_config["Transformer"]["hidden_size"]
		self.num_layers = config.model_config["Transformer"]["num_layers"]
		self.num_heads = config.model_config["Transformer"]["num_heads"]
		self.dropout = config.model_config["Transformer"]["dropout"]

		# Input projection
		self.input_proj = nn.Linear(self.input_dim, self.hidden_size,)

		# Learnable positional embedding
		self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_seq_len, self.hidden_size))

		# Transformer decoder layer
		decoder_layer = nn.TransformerDecoderLayer(
			d_model 		= self.hidden_size,
			nhead 			= self.num_heads,
			dim_feedforward = 4 * self.hidden_size,
			dropout 		= self.dropout,
			batch_first 	= True,
		)

		self.decoder = nn.TransformerDecoder(decoder_layer, num_layers = self.num_layers)

		# Map hidden representation to outputs
		self.out = nn.Linear(self.hidden_size, self.output_dim * self.output_len)

	def forward(self, x):
		"""
			input: [B, input_len, input_dim]
			output: [B, output_len, output_dim]			
		"""

		if x.ndim != 3:
			raise ValueError(f"Expected x to have shape [B, L, D], got {x.shape}")

		_, seq_len, _ = x.shape
		
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
		mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)

		# Self-attention through decoder layers
		x = self.decoder(tgt=x, memory=x, tgt_mask=mask, memory_mask=mask)
		# [batch, sequence_length, hidden_size]

		# Take the final input_len positions
		x = x[:, -self.input_len:, :]
		# [batch, context_size, hidden_size]
		
		# Reshape to output dim and len
		x = x.mean(1)
		x = x.view(x.size(0), -1)
		x = self.out(x)
		x = x.view(x.size(0), self.output_len, self.output_dim)

		return x