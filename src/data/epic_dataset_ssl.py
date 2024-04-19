import numpy as np
import pandas as pd
import os
import math

import torch, torchaudio
from torchvision.transforms import Compose
import torchaudio.transforms as T
from torch.utils.data import Dataset

from features.transforms import normalize_tensor, cut_and_pad
from utils.utils import load_data, center_timestamp

'''

TODO: The purpose of this class is to create a self-supervised learning dataset for the EPIC-KITCHENS dataset.
To do so, it is needed to keep track of the last index scanned for each file in the cache. This is necessary
to avoid loading the same file multiple times.

The cache is a dictionary that stores the data for each video. The key is the video ID and the value is a dictionary
with the following keys:
    • accl: The accelerometer data.
    • gyro: The gyroscope data.
    • last_idx: The last index scanned.

Eventually, it is possible to set the use_cache attribute to True to store the data in the cache.

# TODO (Optional): Implement an overlapping window strategy to increase the number of samples.

'''


class EpicDatasetSSL(Dataset):
    def __init__(self,
            src_dir: os.PathLike,
            annotations: os.PathLike,
            window_size,
            cache_size=math.inf,
            n_fft=128,
            hop_length=4,
            sampling_rate_accl=200,
            sampling_rate_gyro=400,
            downsampling_rate_accl=100,
            downsampling_rate_gyro=200,
            use_cache=False
            ):
        self.src_dir = src_dir
        self.annotations = self.__load_annotations(annotations)
        self.window_size = window_size
        self.cache_size = cache_size
        self.cache = {}  # Initialize an empty cache
        self.cache_index = 0  # Index to keep track of the current cache position

        self.transforms_accl = torch.nn.Sequential(
            T.Resample(sampling_rate_accl, downsampling_rate_accl),
            T.Spectrogram(
                n_fft=n_fft,
                hop_length=hop_length,
                center=True,
                pad_mode="reflect",
                power=2.0,
                normalized=True
                ),
            T.AmplitudeToDB()
        )

        self.transforms_gyro = torch.nn.Sequential(
            T.Resample(sampling_rate_gyro, downsampling_rate_gyro),
            T.Spectrogram(
                n_fft=n_fft,
                hop_length=hop_length,
                center=True,
                pad_mode="reflect",
                power=2.0,
                normalized=True
                ),
            T.AmplitudeToDB()
        )

        
    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, index):
        action = self.annotations.iloc[index, :]
        pid, vid = action['participant_id'], action['video_id']
        start, stop = action['start_timestamp'], action['stop_timestamp']

        x = None

        if not self.cache[vid]:
            self.cache[vid] = self.__load_data__(vid)
        else:
            x = self.cache[vid]
        
        accl = self.__get_windowed_data__(start, stop, x['accl'])
        gyro = self.__get_windowed_data__(start, stop, x['gyro'])

        accl = self.transforms_accl(accl)
        gyro = self.transforms_gyro(gyro)

        accl = normalize_tensor(accl)
        gyro = normalize_tensor(gyro)

        x = torch.cat((accl, gyro), dim=0)

        return x


    def set_use_cache(self, use_cache):
        if use_cache:
            self.cached_data = torch.stack(self.cached_data)
        else:
            self.cached_data = []
        self.use_cache = use_cache


    def __load_annotations__(self, annotations: os.PathLike):
        assert os.path.isfile(annotations), f'The file {annotations} does not exist'

        data = pd.read_pickle(self.annotations)
        data = data.dropna()
        data = data.reset_index(drop=True)
        return data

    def __load_data__(self, pid:str, vid: str):
        subdir = os.path.join(pid, 'meta_data')
        accl, gyro = load_data(src_dir=os.path.join(self.src_dir, subdir), video_id=vid, is_csv=False)
        #[ ] Check if the data is loaded correctly
        accl = torch.tensor(accl.values, dtype=torch.float32)
        gyro = torch.tensor(gyro.values, dtype=torch.float32)
        return {'accl': accl, 'gyro': gyro, 'last_idx': 0}
    
    def __get_windowed_data__(self, start, stop, data):
        center = center_timestamp(start, stop)
        start = center - self.window_size // 2
        stop = center + self.window_size // 2
        # [ ] Check if the shape is correctly returned
        window = data[start:stop, :]
        return window