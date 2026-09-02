"""This is the testing config file, customize as you need."""

import torch
import torch.nn as nn

from common.baseline.gru import GRUModel
from common.baseline.conv import ConvModel
from common.baseline.lstm import LSTMModel
from model.model import Model

# --- Data configuration ---
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

# --- Model configuration ---
# Subject testing model name
test_subj_model_name = "Linear"
# Configuration of baseline models
model_config = {
	test_subj_model_name : {
		"num_layers": 3,
		"hidden_size": 64,
		"class": Model
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

# --- Training params ---
device = torch.device("cuda")
batch_size = 512
learning_rate = 1e-3
num_epochs = 5

# --- Testing params ---
test_seeds = [1, 2, 3]
test_contexts = [96, 192, 336, 720]

### Custom metric definition ###
import torch
import torch.nn.functional as F
def rmse(x, y):
	return torch.sqrt(F.mse_loss(x, y))
################################################################

test_criterion = nn.MSELoss()
test_metric1 = rmse
test_metric2 = nn.L1Loss()
test_metric_names = ["RMSE", "MAE"]

# Logger verbose level
verbose = "info"

