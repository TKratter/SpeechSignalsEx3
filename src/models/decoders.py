from torch import nn
from torch.nn import functional as F


class MFCCEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                            batch_first=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x
