import numpy as np
import pandas as pd
import os
import math

import torch, torchaudio
from torchvision.transforms import Compose, Resize, Normalize
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
            mean=None,
            std=None,
            overlap_in_s=None,
            is_train=False,
            freqm=48,
            timem=60,
            dtype=torch.float32,
            device=None
        ):
        super(SpectrogramsGenerator, self).__init__()
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.downsampling_rate = downsampling_rate
        self.temporal_points = temporal_points
        resample_ratio =  self.temporal_points * self.window_size / (self.window_size * self.sampling_rate)
        resample_target = int(self.sampling_rate * resample_ratio)

        self.transforms = torch.nn.Sequential(
            T.Resample(self.sampling_rate, resample_target),
            T.Spectrogram(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                # center=True,
                # pad_mode="reflect",
                # power=2.0,
                # normalized=True
                ),
            Resize(resizes),
        )
        self.freqm = T.FrequencyMasking(freqm)
        self.timem = T.TimeMasking(timem)
        self.is_train = is_train

        # if self.downsampling_rate is not None:
        #     tmp = deepcopy(self.transforms)
        #     self.transforms = nn.Sequential(
        #         T.Resample(self.sampling_rate, self.downsampling_rate),
        #     )
        #     [self.transforms.append(module) for module in tmp]

        if transforms:
            self.transforms.append(transforms)
            
        if is_train:
            self.transforms.append(self.freqm)
            self.transforms.append(self.timem)
        if mean is not None and std is not None:
            self.normalize = Normalize(mean, std)
        else:
            self.normalize = None
        self.transforms.append(T.AmplitudeToDB())

        if device is not None:
            self.transforms.to(device)

    def __call__(self, x: torch.Tensor):
        spectrogram_db = self.transforms(x)
        # TODO: remove the following line for normalization on amplitude
        if self.normalize:
            spectrogram_db = self.normalize(spectrogram_db)
        return spectrogram_db

