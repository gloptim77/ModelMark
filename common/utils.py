from __future__ import annotations

import argparse
import torch
import torch.nn as nn

import os
import sys
import time
import random
import numpy as np

import config

def parse():
	"""Create parser and parse the arguments"""

	# Create parser
	parser = argparse.ArgumentParser(
		description="A custom CLI tool written in Python."
	)

	# Add arguments

	# Logger
	parser.add_argument(
		"-v", "--verbose", 
		type = str, 
		default = "info", 
		help = "Logger verbose level (info, debug, off)"
	)

	# Parse argument values
	args = parser.parse_args()

	# Update the config
	config.verbose = args.verbose

def set_seed(s : int = 0):
	# Seed all the generators for reproducible results
	torch.random.manual_seed(s)
	torch.cuda.manual_seed(s)
	torch.cuda.manual_seed_all(s)
	torch.use_deterministic_algorithms(True)
	torch.backends.cudnn.deterministic = True
	#torch.backends.cudnn.benchmark = False
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