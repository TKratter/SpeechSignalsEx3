import torch
from torch import nn
import torchaudio
from src.datasets.wav_dataset import WavDataset, char_map
from src.models.decoders import GreedyCTCDecoder
from src.models.encoders import MFCCEncoder


class ASRModel:
    def __init__(self, encoder: MFCCEncoder, decoder: nn.Module):
        self.encoder = encoder
        self.decoder = decoder

    def decode_wav_file(self, wav_path: str, actual_transcript: str = None):
        dataset = WavDataset(wav_path)
        for sample in dataset:
            (audio, labels, input_lengths, target_lengths) = sample.values()
            emission = self.encoder(audio, input_lengths)
            result = self.decoder(emission[0])
            transcript = " ".join(result)

            print("Predicted Transcript: ", transcript)
            if actual_transcript:
                wer = torchaudio.functional.edit_distance(actual_transcript, result) / len(actual_transcript)
                print("WER: ", wer)


if __name__ == '__main__':
    path = "/home/tomk42/PycharmProjects/SpeechSignalsEx3/train/adults/man/ae/1z88153a.wav"
    transcript = "one zero eight eight one five three"
    num_classes = len(char_map)
    encoder = MFCCEncoder(input_dim=20, hidden_dim=256, output_dim=num_classes, feature_extractor_depth=3,
                          lstm_layers=2)
    encoder.load_state_dict(
        torch.load('/home/tomk42/PycharmProjects/SpeechSignalsEx3/src/scripts/asr_model.pth'))

    decoder = GreedyCTCDecoder(labels=list(char_map.keys()))

    asr = ASRModel(encoder=encoder, decoder=decoder)

    asr.decode_wav_file(wav_path=path, actual_transcript=transcript)
