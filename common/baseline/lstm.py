import torch
import torch.nn as nn

import config

class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, context_size: int):
        super().__init__()
        self.input_size = input_dim
        self.output_size = output_dim
        self.context_size = context_size
        self.hidden_size = config.model_config["LSTM"]["hidden_size"]
        self.num_layers = config.model_config["LSTM"]["num_layers"]
        
        # Change nn.GRU to nn.LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(
            self.hidden_size,
            context_size * self.output_size,
        )

    def forward(self, x):

        _, (hn, cn) = self.lstm(x)
        
        # Take the last layer's hidden state
        h = hn[-1]
        out = self.fc(h)
        out = out.view(
            x.size(0),
            self.context_size,
            self.output_size,
        )
        return out