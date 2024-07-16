import numpy as np
import pandas as pd
import os
import math

import torch, torchaudio
from torchvision.transforms import Compose, Resize
import torchaudio.transforms as T
from torch.utils.data import Dataset
import torch.nn as nn

from features.transforms import normalize_tensor, cut_and_pad, cut_and_pad_lr
from utils.utils import load_data, center_timestamp
from copy import deepcopy

class SpectrogramsGenerator(object):
    '''
    Class to generate spectrograms.
    '''

    def __init__(self,
            window_size: int,
            n_fft: int,
            win_length: int,
            hop_length: int,
            temporal_points: int,
            sampling_rate,
            downsampling_rate,
            transforms,
            resizes: tuple[int, int],
            overlap_in_s=None,
        ):
        super(SpectrogramsGenerator, self).__init__()
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.downsampling_rate = downsampling_rate
        self.temporal_points = temporal_points

        self.transforms = torch.nn.Sequential(
            T.Spectrogram(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                center=True,
                pad_mode="reflect",
                power=2.0,
                normalized=True
                ),
            Resize(resizes)
        )

        if self.downsampling_rate is not None:
            tmp = deepcopy(self.transforms)
            self.transforms = nn.Sequential(
                T.Resample(self.sampling_rate, self.downsampling_rate),
            )
            [self.transforms.append(module) for module in tmp]

        if transforms:
            self.transforms.append(transforms)

    def __call__(self, x: torch.Tensor):
        resample_ratio =  self.temporal_points * self.window_size / x.shape[1]
        resample_target = int(self.sampling_rate * resample_ratio)
        waveform = T.Resample(self.sampling_rate, resample_target)(x)
        spectrogram = self.transforms(waveform)
        spectrogram_db = T.AmplitudeToDB()(spectrogram)
        return spectrogram_db
        