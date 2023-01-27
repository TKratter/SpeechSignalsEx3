import pandas as pd
import torch
from tqdm import tqdm

from src.datasets.wav_dataset import WavDataset, char_map, letter_to_word_dict
from src.models.asr_model import ASRModel
from src.models.decoders import GreedyCTCDecoder, BeamSearchDecoder
from src.models.encoders import MFCCEncoder

num_classes = len(char_map)

encoder = MFCCEncoder(input_dim=20, hidden_dim=256, output_dim=num_classes, feature_extractor_depth=4,
                      lstm_layers=2)
encoder.load_state_dict(
    torch.load('/home/tomk42/PycharmProjects/SpeechSignalsEx3/src/scripts/asr_model_best1.pth'))

test_set = WavDataset('/home/tomk42/PycharmProjects/SpeechSignalsEx3/test', mode='test')

decoder = BeamSearchDecoder(beam_size=50)

asr_model = ASRModel(encoder, decoder)

lines = []

with open('output.txt', 'w') as f:

    for i in tqdm(range(len(test_set))):
        file = test_set.files[i]
        name = file.name

        transcript = asr_model.decode_wav_file(wav_path=file)

        lines.append(f'{name} - {transcript}\n')


    f.writelines(lines)