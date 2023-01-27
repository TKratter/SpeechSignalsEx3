import torch
import torch.nn.functional as F
import torchaudio
from src.datasets.wav_dataset import WavDataset, char_map
from src.models.decoders import GreedyCTCDecoder, BeamSearchDecoder
from src.models.encoders import MFCCEncoder, WAV2VECEncoder
import torchmetrics


class ASRModel:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        self.wer = torchmetrics.WordErrorRate()
        self.cer = torchmetrics.CharErrorRate()

    def decode_wav_file(self, wav_path: str, actual_transcript: str = None):
        dataset = WavDataset(wav_path, mode='test')
        for sample in dataset:
            (audio, labels, input_lengths, target_lengths) = sample.values()
            emission = F.log_softmax(self.encoder(audio.unsqueeze(0)), dim=2)
            result = self.decoder(emission[0])
            if isinstance(self.decoder, GreedyCTCDecoder):
                transcript = " ".join(result)
            else:
                transcript = result

            if actual_transcript:

                return self.wer(transcript, actual_transcript)

            else:
                return transcript

    def decode_sample(self, sample: dict, actual_transcript: str):
        (audio, labels, input_lengths, target_lengths) = sample.values()
        emission = F.log_softmax(self.encoder(audio.unsqueeze(0)), dim=2)
        result = self.decoder(emission[0])
        if isinstance(self.decoder, GreedyCTCDecoder):
            transcript = " ".join(result)
        else:
            transcript = result

        return self.wer(transcript, actual_transcript), self.cer(transcript, actual_transcript)




if __name__ == '__main__':
    path = "/home/tomk42/PycharmProjects/SpeechSignalsEx3/test/test_0.wav"
    transcript = "one"
    num_classes = len(char_map)
    # encoder = WAV2VECEncoder()
    encoder = MFCCEncoder(input_dim=20, hidden_dim=256, output_dim=num_classes, feature_extractor_depth=4,
                          lstm_layers=2)
    encoder.load_state_dict(
        torch.load('/home/tomk42/PycharmProjects/SpeechSignalsEx3/src/scripts/asr_model_best1.pth'))

    decoder = GreedyCTCDecoder(labels=list(char_map.keys()))
    # decoder = BeamSearchDecoder()

    asr = ASRModel(encoder=encoder, decoder=decoder)

    print(asr.decode_wav_file(wav_path=path, actual_transcript=transcript))
