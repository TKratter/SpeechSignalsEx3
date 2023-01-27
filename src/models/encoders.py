import numpy as np
import torch
import torch.nn as nn

from torchaudio.pipelines import Wav2Vec2Bundle, WAV2VEC2_BASE

from src.datasets.wav_dataset import char_map, WavDataset


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

    def forward(self, x):
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        x = self.feature_extractor(x)

        x, _ = self.lstm(x)

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
            if len(x.shape) < 3:
                x = x.unsqueeze(0)
        return x


class WAV2VECEncoder(nn.Module):
    def __init__(self):
        super(WAV2VECEncoder, self).__init__()
        self.wav2vec = WAV2VEC2_BASE.get_model()
        self.wav2vec.eval()

        self.lstm = nn.LSTM(768, 50, batch_first=True, bidirectional=True,
                            num_layers=1)

        self.fc = nn.Linear(in_features=100, out_features=len(char_map))

    def forward(self, x):
        with torch.no_grad():
            x, _ = self.wav2vec(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    num_classes = len(char_map)
    train_dataset = WavDataset(path='/home/tomk42/PycharmProjects/SpeechSignalsEx3/train', preprocess=False)

    print(WAV2VECEncoder()(train_dataset[0]['audio'].unsqueeze(0))[0])
