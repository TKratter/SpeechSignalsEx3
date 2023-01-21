import librosa
import numpy as np


class MFCC(object):
    def __init__(self, n_mfcc=20, n_fft=512, hop_length=160, sr=16000):
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr

    def __call__(self, audio):
        return librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft,
                                    hop_length=self.hop_length)


class Pad(object):
    def __init__(self, max_length):
        self.max_length = max_length

    def __call__(self, audio: np.ndarray):
        return np.pad(audio, ((0, 0), (0, self.max_length - audio.shape[1])), 'constant', constant_values=0)
