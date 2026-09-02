from __future__ import annotations

from common.dataset import CausalDataset
from torch.utils.data import DataLoader

import config
import logging
logger = logging.getLogger(__name__)

class Loader:

	def __init__(self, file_config : dict, context_size : int):

		# Get the dataloaders
		self.train_loader, self.val_loader, self.test_loader = self.load_data(file_config, context_size) 

		logger.debug(f"Dataset stats: train={len(self.train_loader)} val={len(self.val_loader)} test={len(self.test_loader)}")

	def load_data(self, file_config : dict, context_size : int) -> tuple[DataLoader, DataLoader, DataLoader]:
		"""Load the data, pack to datasets, create the loaders and return them"""
	
		train_ds = CausalDataset(data_config = file_config,
								seq_len = context_size, 
								pred_len = context_size, 
								split = "train")

		val_ds 	= CausalDataset(data_config = file_config, 
								seq_len = context_size, 
								pred_len = context_size, 
								split = "val", 
								mean = train_ds.mean, 
								std = train_ds.std)

		test_ds = CausalDataset(data_config = file_config, 
								seq_len = context_size, 
								pred_len = context_size, 
								split = "test", 
								mean = train_ds.mean, 
								std = train_ds.std)
		
		train_loader = DataLoader(
			train_ds,
			batch_size = config.batch_size,
			shuffle = True,
			num_workers = 0,
			pin_memory = True,
		)

		val_loader = DataLoader(
			val_ds,
			batch_size = config.batch_size,
			shuffle = False,
			num_workers = 0,
			pin_memory = True,
		)
	
		test_loader = DataLoader(
			test_ds,
			batch_size = config.batch_size,
			shuffle = False,
			num_workers = 0,
			pin_memory = True,
		)
	
		return train_loader, test_loader, val_loader