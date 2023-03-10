import librosa
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from src.datasets.transforms import MFCC, Pad
from torchvision.transforms import Compose, ToTensor
import torch
from tqdm import tqdm

letter_to_word_dict = {
    'z': 'zero',
    'o': 'oh',
    '1': 'one',
    '2': 'two',
    '3': 'three',
    '4': 'four',
    '5': 'five',
    '6': 'six',
    '7': 'seven',
    '8': 'eight',
    '9': 'nine'
}

char_map = {c: i for i, c in enumerate(list("- abcdefghijklmnopqrstuvwxyz"))}


class WavDataset(Dataset):
    def __init__(self, path: str, preprocess: bool = True, max_length=800, max_characters=50, sr: int = 16000,
                 n_mfcc=20, n_fft=512, hop_length=160, max_raw_length=130000, device='cpu', mode='train'):
        self.max_length = max_length
        self.max_raw_length = max_raw_length
        self.max_characters = max_characters
        self.path = path
        if Path(self.path).is_dir():
            self.files = list(Path(self.path).rglob('*.wav'))
        else:
            self.files = [Path(path)]
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.device = device
        self.mode = mode
        if preprocess:
            self.preprocess = Compose(
                [MFCC(sr=self.sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length),
                 Pad(self.max_length),
                 ToTensor()]
            )
        else:
            self.preprocess = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        letters = file.name[:-5]

        if self.mode == 'train':
            words = [letter_to_word_dict[letter] for letter in letters]
            labels = self.text_to_labels(' '.join(words))
        else:
            labels = None
        audio = self.load_audio(file)
        input_lengths = torch.tensor(np.ceil(len(audio) / self.hop_length), dtype=torch.int)
        if self.preprocess:
            audio = self.preprocess(audio).squeeze(0)
        else:
            audio = np.pad(audio, (0, self.max_raw_length - audio.size), 'constant', constant_values=0)
            audio = torch.Tensor(audio)

        if self.mode == 'train':
            target_lengths = torch.tensor([len(labels)], dtype=torch.long)
            labels = self.pad_labels_to_max_chars(labels).to(self.device)
        else:
            target_lengths = None

        sample = {'audio': audio.T.to(self.device), 'labels': labels,
                  'input_lengths': input_lengths, 'target_lengths': target_lengths}
        return sample

    @staticmethod
    def letter_to_word(letter: str) -> str:
        return letter_to_word_dict[letter]

    def load_audio(self, file: Path) -> np.ndarray:
        return librosa.load(file, sr=self.sr)[0]

    @staticmethod
    def text_to_labels(text: str) -> torch.Tensor:
        labels = [char_map[c] for c in text]
        return torch.IntTensor(labels)

    def pad_labels_to_max_chars(self, labels: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(labels, (0, self.max_characters - len(labels)), value=0)


if __name__ == '__main__':
    dataset = WavDataset('/home/tomk42/PycharmProjects/SpeechSignalsEx3/train', preprocess=False)

    max_length = 0
    for i in tqdm(range(len(dataset))):

        (audio, labels, input_lengths, target_lengths) = dataset[i].values()
        if len(audio) > max_length:
            max_length = len(audio)
            print(max_length)
