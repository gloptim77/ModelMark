"""
ModelMark

	This tool will help you to test your NN model against other popular architectures,
	and form a detailed report that is easy to embed to a website.   

	In the "model/model.py" and "config.py" files are complete
	examples of setup for testing a Linear model against Conv, GRU and Transformer with ETT datasets. 

	How to test your model with custom dataset?
	Put your dataset .csv file to the "data/%dataset_name%" folder, currently only causal time-series forecasting is supported. 
	Put your model definition to the "model/model.py" file, make sure to have the same args as the example model.
	You have to configure "config.py" file according to your testing tasks.
	When you are ready, just run this file "modelmark.py". 

"""

import sys
import torch
import numpy as np
import pandas as pd


from common.paper import Paper
from common.utils import parse,  set_seed, count_parameters, start_timer
from common.logger import setup_logging
from common.loader import Loader
from common.tester import Tester

import config
import logging
logger = logging.getLogger(__name__)

from tqdm import tqdm

def _check_model():
	"""Check the testing model on shapes compatibility."""
	first_config = next(iter(config.data_config.values()))
	input_dim = len(first_config["input_features"])
	output_dim = len(first_config["output_features"])

	model = config.model_config[config.test_subj_model_name]["class"](input_dim, output_dim, context_size = 10).to(config.device)
	dummy_input = torch.zeros([4, 10, input_dim]).to(config.device)
	dummy_target = torch.zeros([4, 10, output_dim]).to(config.device)
	generated = model(dummy_input)
	if generated.shape == dummy_target.shape:
		logger.info("Testing model is ready")
		return 0
	else:
		logger.error(f"Testing model output expected shape {dummy_target.shape}, got {generated.shape}.\nCheck your definition at model/model.py")
		return 1
		
def run() -> int:

	# Parse the arguments
	parse()
	# Setup logger
	setup_logging(level = config.verbose)
	logger.info("Logger started")
	# Setup other
	pd.set_option('display.colheader_justify', 'center')

	# Check the model
	if _check_model():
		return 1

	# Display test information 
	logger.info("-" * 128)
	logger.info("Info:")
	logger.info("This test runs multiple training/validation/testing passes of several model architectures (including your model)")
	logger.info("To provide you with the most fair comparision, we use multiple seeds and runs under the same conditions.")
	logger.info("There is no parallel execution at the moment, so the test may take a while to finish (it depends on your config and hardware).")
	logger.info("At the end you will get the files with testing evaluation results and relative comparisons of your model's architecture VS others.")
	logger.info("-" * 128)

	logger.info("Start the test? [y/n] (Will start automatically in 30s.)")
	k = start_timer(30)
	if (k == 'n') or (k == 'N'):
		logger.error("Terminated.")
		return 1

	# Start the test
	logger.info("Starting the test.")

	report_data = []
	report_dataset_names = list(config.data_config)
	report_model_names = list(config.model_config)
	report_stats = np.zeros((len(report_model_names), 3))
	report_params = []

	c_current, c_total = 1, len(report_dataset_names) * len(report_model_names) * len(config.test_seeds) * len(config.test_contexts)
	pbar = tqdm(total=c_total, desc="Progress")
	for file_name, file_config in config.data_config.items():

		for test_context in config.test_contexts:

			report_line = []
			for model_name, model_config in config.model_config.items():

				total_test_metric1 = 0
				total_test_metric2 = 0
				for test_seed in config.test_seeds:
					
					# (Re-)Seed
					set_seed(test_seed)
					tqdm.write(f"Running test ({c_current}/{c_total}) | File: {file_name} | Context: {test_context} | Model: {model_name} | Seed: {test_seed}")
					input_dim = len(file_config["input_features"])
					output_dim = len(file_config["output_features"])
					model = model_config["class"](input_dim, output_dim, context_size = test_context).to(config.device)
									
					# Create loader
					loader = Loader(file_config = file_config, context_size = test_context)
					# Create model
					input_dim = len(file_config["input_features"])
					output_dim = len(file_config["output_features"])
					model = model_config["class"](input_dim, output_dim, context_size = test_context).to(config.device)
					# Create tester
					tester = Tester(model = model, loader = loader)

					# Run the test
					test_metric1, test_metric2, train_time, train_gflops, train_mem = tester.test()
					logger.debug(f"test_metric1={test_metric1:.4f} test_metric2={test_metric2:.4f}")
					
					# Capture to history
					total_test_metric1 += test_metric1
					total_test_metric2 += test_metric2
					model_id = report_model_names.index(model_name)
					report_stats[model_id] += np.array([train_time, train_gflops, train_mem])

					# Update the progress bar
					c_current += 1
					pbar.update(1)

				# Calculate the stats
				test_metric1_mean 	= total_test_metric1 / len(config.test_seeds)
				test_metric2_mean 	= total_test_metric2 / len(config.test_seeds)
				if len(report_params) < len(report_model_names):
					report_params.append(count_parameters(model))

				report_line.append(f"{test_metric1_mean:.3f}")
				report_line.append(f"{test_metric2_mean:.3f}")

			report_data.append(report_line)
	pbar.close()

	# Calculate the  report stats
	report_stats = report_stats.T
	report_stats = report_stats / c_total

	report_time = [f"{v:.3f}" for v in report_stats[0]]
	report_gflops = [f"{v:.2f}" for v in report_stats[1]]
	report_mem = [f"{v:.2f}" for v in report_stats[2]]
	report_params = [f"{v:.2f}" for v in report_params]
	# Obtain the columns&rows names
	report_columns = pd.MultiIndex.from_product([report_model_names, config.test_metric_names], names=["Model", "Metric"])
	report_rows = pd.MultiIndex.from_product([report_dataset_names, config.test_contexts])
	# Pack to the dataframe
	df = pd.DataFrame(data = report_data, index = report_rows, columns = report_columns)

	title = f"{config.test_subj_model_name} evaluation report."
	subtitle=f"""Mean over {len(config.test_seeds)} runs."""

	model_stats={
		"Time (s/epoch)": report_time,
		"Params count (K)": report_params,
		"GFLOPs": report_gflops,
		"Peak memory usage (Gb)": report_mem
	}

	paper = Paper()
	# Form and save the report
	paper.report(df, title, subtitle, model_stats)

	return 0
	
if __name__ == "__main__":
	sys.exit(run())