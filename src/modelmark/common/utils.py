from __future__ import annotations


import torch
import torch.nn as nn

import os
import sys
import time
import random
import numpy as np

import importlib.util
import modelmark.constants as constants

import logging
logger = logging.getLogger(__name__)

def set_seed(s : int = 0):
	# Seed all the generators for reproducible results
	torch.random.manual_seed(s)
	torch.cuda.manual_seed(s)
	torch.cuda.manual_seed_all(s)
	torch.use_deterministic_algorithms(True)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(s)
	random.seed(s)

def count_parameters(module: nn.Module) -> int:
	"""Count the total number of elements/parameters in any PyTorch module."""
	return sum(p.numel() for p in module.parameters()) / 1000

# Setup non-blocking key input based on Operating System
if os.name == 'nt':
	import msvcrt
	def get_keypress():
		if msvcrt.kbhit():
			# Return decoded string character
			return msvcrt.getch().decode('utf-8', errors='ignore')
		return None
else:
	import select
	import termios
	import tty
	def get_keypress():
		# Check if stdin has data waiting
		if select.select([sys.stdin], [], [], 0)[0]:
			fd = sys.stdin.fileno()
			old_settings = termios.tcgetattr(fd)
			try:
				tty.setraw(sys.stdin.fileno())
				ch = sys.stdin.read(1)
			finally:
				termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
			return ch
		return None

def start_timer(duration_seconds):
	start_time = time.time()
	#print("Timer started. Press any key to interrupt...")
	
	while True:
		elapsed = time.time() - start_time
		remaining = max(0, duration_seconds - elapsed)
		
		sys.stdout.write(f"\rStart in: {remaining:.1f}s")
		sys.stdout.flush()
		
		# Check for user interruption
		pressed_key = get_keypress()
		if pressed_key is not None:
			print("\n")
			return pressed_key
			
		if remaining <= 0:
			break
			
		time.sleep(0.05) # Lower sleep window for snappier key detection

	print("\n")
	#print("\n\nTimer finished naturally!")
	return None

def load_config():
	"""Load the config (user or pkg)"""

	user_config_path = constants.USER_CONFIG_PATH

	if user_config_path.exists():
	
		# Add the user working dir to sys.path
		user_dir = str(constants.USER_DIR)
		if user_dir not in sys.path:
			sys.path.insert(0, user_dir)

		spec = importlib.util.spec_from_file_location("config", user_config_path)
		user_config = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(user_config)
		return user_config

	else:
	
		from modelmark import config
		return config