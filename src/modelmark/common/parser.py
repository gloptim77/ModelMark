import argparse
import sys
import shutil
import pandas as pd

from modelmark.common.downloader import download_dataset
import modelmark.constants as constants

import logging
logger = logging.getLogger(__name__)
from rich.console import Console
console = Console()

class Parser:

	def __init__(self):
		"""Parse the arguments"""

		# Create parser
		self.parser = argparse.ArgumentParser(
			formatter_class=argparse.RawTextHelpFormatter,
			description = constants.PARSER_DESC
		)

		# Create a subparser container
		subparsers = self.parser.add_subparsers(dest="command", required=True)
	
		# Define your action words (subcommands)
		subparsers.add_parser("init", 	help = "Test init, create the modelmark_files/, customizable config.py file and models/ with examples.")
		subparsers.add_parser("load",	help = "Dataset download, create the data/ folder and download ETT dataset there.")				
		subparsers.add_parser("run", 	help = "Run the test.")
		subparsers.add_parser("clear", 	help = "Remove the modelmark_files folder.")
		subparsers.add_parser("reset", 	help = "Runs clean and init.")
		subparsers.add_parser("help", 	help = "Print the info above.")
	
		self.args = self.parser.parse_args()
	
		# Run without args / help request - print help
		if len(sys.argv) == 1:
			self.parser.print_help()
			sys.exit(1)

		# Setup other
		pd.set_option('display.colheader_justify', 'center')	

	def run(self):
		"""Select the task and run"""

		# Config initialization #
		if self.args.command == "init":
			self.init_config()
			return 0

		# Dataset download #
		if self.args.command == "load":
			download_dataset("ett")
			return 0

		# Run the test #
		if self.args.command == "run":
			return None

		# Clean the config folder #
		if self.args.command == "clear":
			self.clear_config()
			return 0

		# Reset the modelmark config #
		if self.args.command == "reset":
			self.reset_config()
			return 0

		# Print the modelmark usage #
		if self.args.command == "help":
			self.parser.print_help()
			return 0
		
		# Check if task is correct #
		if self.args.command != "run":
			logger.error(f"Unknown command: {self.args.command}, run 'modelmark help' for more info")
			console.print(f"Unknown command: {self.args.command}, run 'modelmark help' for more info", style="red")
			return 1

		# Otherwise return error
		return 1

	def init_config(self):
		"Copy the config example file to the user dir"

		user_config_dir = constants.USER_CONFIG_DIR
		user_models_dir = constants.USER_MODELS_DIR

		package_config = constants.PACKAGE_CONFIG_PATH
		user_config = constants.USER_CONFIG_PATH
		package_model = constants.PACKAGE_MODEL_PATH
		user_model = constants.USER_MODEL_PATH

		# Make the dirs
		user_config_dir.mkdir(parents=True, exist_ok=True)
		user_models_dir.mkdir(parents=True, exist_ok=True)

		if not user_config.exists():
			shutil.copy(package_config, user_config)

			logger.info(f"Customizable config created at: {user_config}")
			console.print(f"Customizable config created at: {user_config}")			

			shutil.copy(package_model, user_model)

			logger.info(f"Model example file created at: {user_model}")
			console.print(f"Model example file created at: {user_model}")
						
			console.print(f"Initialization complete.", style="green")
		else:

			console.print(f"config.py already exists", style="yellow")
			self.parser.print_help()

	def clear_config(self):
		"""Remove the user config dir"""
		user_config_path = constants.USER_CONFIG_DIR
		user_example_model_path = constants.USER_MODEL_PATH

		if user_config_path.exists():
			shutil.rmtree(user_config_path)
			logger.info(f"Folder {user_config_path} removed.")
			console.print(f"Folder {user_config_path} removed.")
		if user_example_model_path.exists():
			user_example_model_path.unlink(missing_ok=True)
			logger.info(f"File {user_example_model_path} removed.")
			console.print(f"File {user_example_model_path} removed.")
				
	def reset_config(self):
		self.clear_config()
		self.init_config()
		