# ModelMark

ModelMark is a CLI benchmarking tool for comparing neural-network models. It runs your models on one or more datasets, records performance and efficiency statistics, and generates an easy-to-embed HTML/PNG report.

![Evaluation result](https://raw.githubusercontent.com/gloptim77/ModelMark/refs/heads/main/result.png)

## Table of Contents

- [About](#about)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [How the Benchmark Works](#how-the-benchmark-works)
- [Reported Metrics](#reported-metrics)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## About

For each benchmark run, ModelMark:

1. Selects the next combination of dataset, input/output size, model, and seed.
2. Seeds the random generators for reproducibility.
3. Creates the data loader, model, and tester objects.
4. Trains the model for the configured number of epochs and restores the checkpoint with the lowest validation loss.
5. Records runtime and efficiency statistics.
6. Evaluates the model on the dataset using the metrics from the configuration file.

After all runs finish, ModelMark aggregates the results and generates a report containing testing results, training statistics, and your machine metadata.

## Requirements

- OS: Windows or Linux
- Python: 3.12 or newer
- Git: required for installation from GitHub

## Installation

Install ModelMark from PyPI:

```bash
pip install modelmark
```

## Quick Start

1. Run initialization in an empty folder:

   ```bash
   modelmark init
   ```

   This creates two folders:
   - `modelmark_files/` — configuration and log files
   - `models/` — your model files

2. Edit the configuration file:

   `modelmark_files/config.py`

   It contains three main configuration blocks:
   - `model_config` — model hyperparameters such as number of layers, hidden dimension, kernel size, etc.
   - `data_config` — dataset parameters such as file path, input/output features, train/val ratios, etc.
   - `test_config` — testing options such as optimizer, loss criterion, metrics, learning rate, etc.

3. Put the testing models to the `models/` folder.
	
	Make sure to import model's class definitions to the `config.py`.
	
	Configure `config.model_config` according to your task. 
 
5. Download the dataset files (ETT by default):

   ```bash
   modelmark load
   ```

6. Run the benchmark:

   ```bash
   modelmark run
   ```

7. View the generated report:
   - `result.html`
   - `result.png`

## Configuration

You can adjust the configuration and add your own model files in the `models/` directory to match your benchmarking needs.

Make sure your model implementation is compatible with the keys and settings used in your configuration.

A detailed configuration example is available at:
[src/modelmark/config.py](https://github.com/gloptim77/ModelMark/blob/main/src/modelmark/config.py)

A detailed example model is available at:
[src/modelmark/models/linear.py](https://github.com/gloptim77/ModelMark/blob/main/src/modelmark/models/linear.py)

## How the Benchmark Works

The benchmark consists of:

```text
F × O × M × S
```

where:

| Symbol | Meaning | Example |
|---|---|---|
| `F` | Number of dataset files in the configuration | `{"ETTh1": ..., "Weather": ...}` → `F = 2` |
| `O` | Number of input/output size configurations | `[32, 64, 128]` → `O = 3` |
| `M` | Number of models | `{"Linear": ..., "LSTM": ...}` → `M = 2` |
| `S` | Number of seeds | `[42, 43, 44]` → `S = 3` |

ModelMark repeats the training/evaluation process for each combination and stores the mean result over `S` runs. Using more seeds generally makes the comparison more fair and statistically stable.

## Reported Metrics

The report includes statistics such as:

| Metric | Description |
|---|---|
| Time | Average training time per epoch |
| Params | Total number of model parameters |
| GFLOPs | Average GFLOPs per batch |
| Peak Memory | Maximum memory observed during a training iteration |

The report also includes the evaluation metrics configured in `test_config` and your machine metadata.

## Examples

Run `modelmark init`, it will create config at `modelmark_files/config.py` and model's example folder at `models/` with Linear model file inside:

- Configuration example: [src/modelmark/config.py](https://github.com/gloptim77/ModelMark/blob/main/src/modelmark/config.py)
- Linear model example: [src/modelmark/models/linear.py](https://github.com/gloptim77/ModelMark/blob/main/src/modelmark/models/linear.py)

## Troubleshooting

If something does not work:

1. Check the application log:
   `modelmark_files/modelmark.log`
2. Try restarting ModelMark:

   ```bash
   modelmark restart
   ```

3. If the issue persists, delete the `modelmark_files` folder and run:

   ```bash
   modelmark init
   ```

When opening an issue on GitHub, please include the relevant part of your log file.

If you have questions or want to inspect the source code, see the
[ModelMark GitHub repository](https://github.com/gloptim77/ModelMark/tree/main).
