from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from common.loader import Loader

import time
import config
import logging
logger = logging.getLogger(__name__)

class Tester:
	"""This class runs a benchmark of the current model and loader"""

	def __init__(self, model : nn.Module, loader : Loader):

		self.model = model
		self.train_loader = loader.train_loader
		self.val_loader = loader.val_loader
		self.test_loader = loader.test_loader

		self.optimizer = optim.Adam(model.parameters(), lr = config.learning_rate)
		self.criterion = config.test_criterion
		self.metric1 = config.test_metric1
		self.metric2 = config.test_metric2
				
		self.device = config.device

	def test(self):
		"""
		Train, validate, return to the best model weight state according to the validation loss, test and return.
		"""

		best_val_loss = float("inf")
		best_state = None
		train_time, train_gflops, train_mem = None, None, None

		for epoch in range(config.num_epochs):

			results = self.train(self.train_loader, measure_time=True, measure_flops=True, measure_memory=True)

			train_loss = results[0]
			train_time = results[1]
			train_gflops = results[2]
			train_mem = results[3]

			val_loss = self.evaluate(self.val_loader, self.criterion)
		
			logger.debug(f"Epoch {epoch:03d} | train MSE: {train_loss:.4f} | val MSE: {val_loss:.4f}")
		
			# Save best model according to validation performance
			if val_loss < best_val_loss:
				best_val_loss = val_loss
		
				best_state = {
					k: v.detach().cpu().clone()
					for k, v in self.model.state_dict().items()
				}
		
		# Restore best checkpoint
		self.model.load_state_dict(best_state)

		# Get final test metric values
		test_metric1 = self.evaluate(self.test_loader, self.metric1)
		test_metric2 = self.evaluate(self.test_loader, self.metric2)
		
		return test_metric1, test_metric2, train_time, train_gflops, train_mem

	def train(self, loader: DataLoader, measure_time: bool = False,
			  measure_flops: bool = False, measure_memory: bool = False):
		"""Train the model for one epoch."""

		self.model.train()

		total_loss = 0.0
		total_samples = 0
		total_flops = 0.0

		is_cuda = self.device.type == "cuda"

		if measure_memory and is_cuda:
			torch.cuda.reset_peak_memory_stats(self.device)

		if measure_time:
			if is_cuda:
				torch.cuda.synchronize()
			start_time = time.perf_counter()

		for i, (x, y) in enumerate(loader):

			x = x.to(self.device)
			y = y.to(self.device)

			if measure_flops and i == 0:
				from thop import profile
				with torch.no_grad():
					flops_per_batch, _ = profile(self.model, inputs=(x,), verbose=False)
				flops_per_batch *= 3

			self.optimizer.zero_grad(set_to_none=True)
			pred = self.model(x)
			loss = self.criterion(pred.squeeze(), y.squeeze())
			loss.backward()
			clip_grad_norm_(self.model.parameters(), max_norm=1.0)
			self.optimizer.step()

			batch_size = x.size(0)
			total_loss += loss.item() * batch_size
			total_samples += batch_size

			if measure_flops:
				total_flops += flops_per_batch

		avg_loss = total_loss / total_samples

		results = [avg_loss]

		if measure_time:
			if is_cuda:
				torch.cuda.synchronize()
			epoch_time = time.perf_counter() - start_time
			results.append(epoch_time)

		if measure_flops:
			results.append(total_flops / 1e9)

		if measure_memory:
			if is_cuda:
				peak_mem_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
			else:
				peak_mem_gb = 0.0  # or float('nan') to signal "not applicable"
			results.append(peak_mem_gb)

		return tuple(results)

	@torch.no_grad()
	def evaluate(self, loader: DataLoader, criterion):
		"""Evaluate the model on provided data"""
	
		self.model.eval()
	
		total_loss = 0.0
		total_samples = 0
	
		for x, y in loader:
			x = x.to(self.device)
			y = y.to(self.device)
	
			pred = self.model(x)
			loss = criterion(pred.squeeze(), y.squeeze())
	
			batch_size = x.size(0)
			total_loss += loss.item() * batch_size
			total_samples += batch_size
	
		return total_loss / total_samples
	