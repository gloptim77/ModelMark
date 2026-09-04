"""This is the testing config file example, customize as you need."""

import torch
import torch.nn as nn

# ---- Model import secion ---- #
# Put your model class definitions here
# Either built-in:
from modelmark.models.gru import GRUModel
from modelmark.models.conv import ConvModel
from modelmark.models.lstm import LSTMModel
from modelmark.models.linear import Linear
# Either custom (uncomment)
#from models.linear import Linear

# ---- Model configuration section ----
model_config = {
	"Linear" : {
		"num_layers": 3,
		"hidden_size": 64,
		"class": Linear
	},
	"Conv" : {	
		"num_layers": 3,
		"hidden_size": 64,
		"kernel_size": 4,
		"class": ConvModel,
	},
	"GRU" : {	
		"num_layers": 2,
		"hidden_size": 64,
		"class": GRUModel,
	},
	"LSTM" : {	
		"num_layers": 2,
		"hidden_size": 64,
		"class": LSTMModel,
	},
}

# ---- Data configuration section ----
# Your datasets folder path
data_path = "data/"
# Configuration of your datasets
data_config = {
	"ETTh1" : {
		"path": "ett/ETTh1.csv",  
		"input_features": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"], 
		"output_features": ["OT"],
		"train_ratio": 0.6,
		"val_ratio": 0.2, 
	},
	"ETTh2" : {
		"path": "ett/ETTh2.csv",  
		"input_features": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"], 
		"output_features": ["OT"],
		"train_ratio": 0.6,
		"val_ratio": 0.2,
	},
}

# ---- Testing configuration section ----

# You can define your own metric
import torch
import torch.nn.functional as F
def rmse(x, y):
	return torch.sqrt(F.mse_loss(x, y))
from torch.optim import Adam

test_config = {
	"optim": {
		"name" : "Adam",
		"class": Adam,
		"max_norm": 1.0,
		"criterion": nn.MSELoss(),
		"lr": 1e-3,
		"batch_size": 512,
        "num_epochs": 5,
		"device": torch.device("cuda"),
	},
	"metrics":
	{
		"RMSE": rmse,
		"MAE": nn.L1Loss(),
	},
	"seeds": [1, 2, 3],
	"contexts": [96, 192, 336, 720],
	}

# Other
_data_split = ", ".join([f"{k} ({v["train_ratio"]:.2f}/{v["val_ratio"]:.2f}/{(1 - (v["train_ratio"] + v["val_ratio"])):.2f})" for k, v in data_config.items()])
# --- Report configuration section ---
title = f"Models evaluation report."
subtitle=f"""Models were trained with {test_config["optim"]["name"]} optimizer, batch size = {test_config["optim"]["batch_size"]}, LR = {test_config["optim"]["lr"]}, epochs = {test_config["optim"]["num_epochs"]}
Dataset split (train/val/test): {_data_split}
Results are mean values over {len(test_config["seeds"])} runs, seeds used: {", ".join(map(str, test_config["seeds"]))}"""