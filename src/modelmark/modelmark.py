"""
ModelMark

	This tool will help you to test NN models against each other,
	and form a detailed report that is easy to embed to a website.   

	It can download dataset with
		modelmark -t load
	It can run the test with 
		modelmark -t run

	Test consists of F * O * M * S steps, where:
		F - number of dataset files in the config (e.g. ["ETTh1" : ..., "Weather" : ...] means F = 2)
		O - number of input/output sizes (e.g. [32, 64, 128] means O = 3)
		M - number of models (e.g. ["Linear" : ..., "LSTM" : ...] means M = 2)
		S - number of seeds (e.g. [42, 43, 44] means S = 2)

	At each testing run iteration, modelmark:
		
		1) Selects next dataset, input/output len, model and seed 
		2) Seeds the generators for reproducibility
		3) Creates the loader, model and tester objects
		4) Trains the model for E epochs, restores the state with the least validation loss
		5) Tracks the GFLOPs (AVG over one batch), Memory (AVG peak usage during full training), Time (AVG per epoch)
		5) Evaluates the model on metrics from configuration file (config.test_metric1&2)
		6) Stores the mean result over S runs 

		That way, the more seeds you run, the more "fair" the results are.
	
	Finally, modelmark will form the report with all the testing results, training stats and your machine metadata.

	About training stats:
		
		Time - average time per epoch 
		Params - total number of model params
		GFLOPs - average per batch
		Peak Memory - max per training iteration

	---	
	Example of the model: "modelmark/models/linear.py" 
	Example of the config: "modelmark/config.py"
"""

import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import custom functions
from modelmark.common.utils import  set_seed, count_parameters, start_timer
from modelmark.common.logger import setup_logging
# Import custom classes
from modelmark.common.report import Report
from modelmark.common.parser import Parser
from modelmark.common.loader import Loader
from modelmark.common.tester import Tester

import logging
logger = logging.getLogger(__name__)
from rich.console import Console
console = Console()
from modelmark.common.utils import load_config
config = load_config()

def _check_model(name = "Linear", device=config.test_config["optim"]["device"]):
	"""Check the testing model on shapes compatibility."""
	current_config = next(iter(config.data_config.values()))

	input_dim = len(current_config["input_features"])
	output_dim = len(current_config["output_features"])

	input_len = config.test_config["input_len"][0]
	output_len = config.test_config["output_len"][0]

	model = config.model_config[name]["class"](input_dim, output_dim, input_len, output_len).to(device)

	dummy_input = torch.zeros([1, input_len, input_dim]).to(device)
	dummy_target = torch.zeros([1, output_len, output_dim]).to(device)

	generated = model(dummy_input)
	if generated.shape == dummy_target.shape:
		console.print(f"Testing model {name} is ready", style="green")
		return None
	else:
		console.print(f"Testing model output expected shape {dummy_target.shape}, got {generated.shape}.\nCheck your definition of {name} model.", style="red")
		return 1

def main() -> int:

	# Parse the arguments
	parser = Parser()
	code = parser.run()
	if code is not None:
		return code

	# Setup logger
	setup_logging()
	logger.info("Logger started")
		
	# Check the models
	for k, _ in config.model_config.items():
		if _check_model(k) is not None:
			return 1
		
	# Display test information 
	logger.info("Display test info.")
	console.print("-" * 128, style = "cyan")
	console.print("Info\n", style = "magenta")
	console.print(f"This test runs multiple training/validation/testing passes of all configured model architectures: {[k for k, _ in config.model_config.items()]}", style = "white")
	console.print(f"To provide you with the most fair comparision, we run multiple tests over seeds, and average the results.", style = "white")
	console.print(f"There is no parallel execution at the moment, so the test may take a while to finish (it depends on your config and hardware).", style = "white")
	console.print("-" * 128, style="cyan")

	console.print("Start the test? [y/n] (Will start automatically in 30s.)", style = "white")
	k = start_timer(30)
	if (k == 'n') or (k == 'N'):
		logger.error("Test terminated by user.")
		console.print("Terminated", style="red")
		return 1

	# Start the test
	logger.info("Starting the test.")
	console.print("Starting the test.", style = "green")

	# Create test buffers
	report_data = []
	report_dataset_names = list(config.data_config)
	report_model_names = list(config.model_config)
	report_metric_names = list(config.test_config["metrics"])
	report_stats = np.zeros((len(report_model_names), 3))
	report_params = []

	device = config.test_config["optim"]["device"]
	c_current, c_total = 1, len(report_dataset_names) * len(report_model_names) * len(config.test_config["seeds"]) * len(config.test_config["output_len"])
	pbar = tqdm(total=c_total, desc="Progress")
	for file_name, file_config in config.data_config.items():

		for input_len, output_len in zip(config.test_config["input_len"], config.test_config["output_len"]):

			report_line = []
			for model_name, model_config in config.model_config.items():

				total_test_metrics = np.zeros(len(config.test_config["metrics"]))
				for test_seed in config.test_config["seeds"]:
					
					# (Re-)Seed
					set_seed(test_seed)
					tqdm.write(f"Running test ({c_current}/{c_total}) | File: {file_name} | Output len: {output_len} | Model: {model_name} | Seed: {test_seed}")
					logger.debug(f"Test ({c_current}/{c_total})| F={file_name} I/O={input_len}/{output_len} M={model_name} S={test_seed}")

					# Get in/out dims
					input_dim = len(file_config["input_features"])
					output_dim = len(file_config["output_features"])

					# Create loader
					loader = Loader(file_config = file_config, input_len = input_len, output_len = output_len)
					# Create model
					model = model_config["class"](input_dim, output_dim, input_len, output_len).to(device)
					# Create tester
					tester = Tester(model = model, loader = loader)

					# Run the test
					test_metrics, train_time, train_gflops, train_mem = tester.test()
					logger.debug(f"test_metrics={test_metrics} time={train_time} gflops={train_gflops} mem={train_mem}")
					
					# Accomulate
					total_test_metrics += test_metrics
					model_id = report_model_names.index(model_name)
					report_stats[model_id] += np.array([train_time, train_gflops, train_mem])

					# Update the progress bar
					c_current += 1
					pbar.update(1)

				# Calculate the stats (average over seeds)
				test_metrics_mean = total_test_metrics / len(config.test_config["seeds"])
				if len(report_params) < len(report_model_names):
					report_params.append(count_parameters(model))
				# Add to the current "line"
				for m in test_metrics_mean:
					report_line.append(f"{m:.3f}")
			# Add line to the final report
			report_data.append(report_line)
	pbar.close()

	# Calculate the  report stats
	report_stats = report_stats.T
	report_stats = report_stats / c_total

	report_time 	= [f"{v:.3f}" for v in report_stats[0]]
	report_gflops 	= [f"{v:.2f}" for v in report_stats[1]]
	report_mem 		= [f"{v:.2f}" for v in report_stats[2]]
	report_params 	= [f"{v:.2f}" for v in report_params  ]
	# Obtain the columns&rows names
	
	report_columns = pd.MultiIndex.from_product([report_model_names, report_metric_names], names=["Model", "Metric"])
	report_rows = pd.MultiIndex.from_product([report_dataset_names, config.test_config["output_len"]])
	# Pack to the dataframe
	df = pd.DataFrame(data = report_data, index = report_rows, columns = report_columns)

	model_stats={
		"Time (s/epoch)": report_time,
		"Params count (K)": report_params,
		"GFLOPs (f/batch)": report_gflops,
		"Peak Memory (Gb)": report_mem
	}

	report = Report()
	# Form and save the report
	report.report(df, config.title, config.subtitle, model_stats)

	return 0

def run():
	"""Entry point"""
	return sys.exit(main())