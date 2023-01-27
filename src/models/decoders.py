import torch
from torch import nn
from torch.nn import functional as F
from typing import List

from torchaudio.models.decoder import CTCDecoderLM, CTCDecoderLMState, ctc_decoder

from src.datasets.wav_dataset import char_map, letter_to_word_dict


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> List[str]:
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = "".join([self.labels[i] for i in indices])
        return joined.strip().split()


class BeamSearchDecoder:
    def __init__(self, use_lm=True, beam_size=5):
        if use_lm:
            self.lm_path = '/home/tomk42/PycharmProjects/SpeechSignalsEx3/6gram.arpa'
        else:
            self.lm_path = None
        self.tokens = list(char_map.keys())
        self.tokens[1] = '|'
        self.decoder = ctc_decoder(
            lexicon='/home/tomk42/PycharmProjects/SpeechSignalsEx3/lexicon.txt',
            tokens=self.tokens,
            lm=self.lm_path,
            beam_size=beam_size
        )

    def __call__(self, emission):
        letters_list = self.decoder(emission.unsqueeze(0))[0][0].words
        words = ' '.join([letter_to_word_dict[l] for l in letters_list])
        return words
