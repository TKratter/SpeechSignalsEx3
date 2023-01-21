import numpy as np
import torch
import torch.nn as nn


class MFCCEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, feature_extractor_depth=1, lstm_layers=1):
        super(MFCCEncoder, self).__init__()
        self.feature_extractor_depth = feature_extractor_depth
        self.feature_extractor = CNNFeatureExtractor(feature_extractor_depth)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True,
                            num_layers=lstm_layers)
        self.fc = nn.Linear(hidden_dim * lstm_layers, output_dim)

    def forward(self, x, input_lengths):
        x = self.feature_extractor(x)
        # x = nn.utils.rnn.pack_padded_sequence(x, input_lengths // (2 * self.feature_extractor_depth), batch_first=True,
        #                                       enforce_sorted=False)
        # pack = True
        # if x.data.shape[0] == 1:
        #     x = x.data.squeeze(0)
        #     pack = False
        x, _ = self.lstm(x)
        # if pack:
        #     x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.fc(x)
        return x


class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x.unsqueeze(1))
        x = self.bn(x)
        x = self.relu(x)
        return x.squeeze()


class CNNFeatureExtractor(nn.Module):
    def __init__(self, depth=1):
        super(CNNFeatureExtractor, self).__init__()
        self.depth = depth
        self.cbr_list = nn.ModuleList(
            [CBR(1, 1, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)) for _ in range(depth)])

    def forward(self, x):
        for i in range(self.depth):
            x = self.cbr_list[i](x)
        return x
