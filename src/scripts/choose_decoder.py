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

val_set = WavDataset('/home/tomk42/PycharmProjects/SpeechSignalsEx3/train/children/boy/bb')

beam_size_list = [1, 50, 500]

decoders_dict = dict()

decoders_dict['greedy_decoder'] = GreedyCTCDecoder(labels=list(char_map.keys()))

results_dict = dict()

for beam in beam_size_list:
    decoders_dict[f'ctc+lm_beam-{beam}'] = BeamSearchDecoder(beam_size=beam)
    decoders_dict[f'ctc_beam-{beam}'] = BeamSearchDecoder(use_lm=False, beam_size=beam)

for decoder_name, decoder in tqdm(decoders_dict.items()):

    asr_model = ASRModel(encoder, decoder)

    wer_list, cer_list = [], []

    for i in range(len(val_set)):
        file = val_set.files[i]
        letters = file.name[:-5]
        actual_transcript = ' '.join([letter_to_word_dict[letter] for letter in letters])
        (wer, cer) = asr_model.decode_sample(val_set[i], actual_transcript)
        wer_list.append(wer)
        cer_list.append(cer)

    results_dict[decoder_name] = {'WER': float(torch.Tensor(wer_list).mean()),
                                  'CER': float(torch.Tensor(cer_list).mean())}


df = pd.DataFrame(results_dict).T

df.to_csv('results_df.csv')

print(df)