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
            filename: str,
            window_size=2.5,
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
        self.annotations = self.__load_annotations__(os.path.join(annotations, filename))
        self.window_size = window_size
        self.cache_size = cache_size
        self.cache = {}  # Initialize an empty cache
        self.cache_index = 0  # Index to keep track of the current cache position
        self.sampling_rate_accl = sampling_rate_accl
        self.sampling_rate_gyro = sampling_rate_gyro
        self.downsampling_rate_accl = downsampling_rate_accl
        self.downsampling_rate_gyro = downsampling_rate_gyro
        self.use_cache = use_cache

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

        if vid not in self.cache:
            self.cache[vid] = self.__load_data__(pid, vid)
        
        '''
        Idxs:
        0: seconds
        1: acclX
        2: acclY
        3: acclZ
        '''
        accl = self.__get_windowed_data__(start, stop, self.cache[vid]['accl'])
        gyro = self.__get_windowed_data__(start, stop, self.cache[vid]['gyro'], is_accl=False)

        accl = self.transforms_accl(accl)
        gyro = self.transforms_gyro(gyro)

        # FIXME: Actual shape is (3, 65, 63)
        accl = normalize_tensor(accl)
        # FIXME: Actual shape is (3, 65, 126)
        gyro = normalize_tensor(gyro)

        # x = torch.cat((accl, gyro), dim=0)

        return (accl, gyro)


    def set_use_cache(self, use_cache):
        if use_cache:
            self.cached_data = torch.stack(self.cached_data)
        else:
            self.cached_data = []
        self.use_cache = use_cache


    def __load_annotations__(self, annotations: os.PathLike):
        assert os.path.isfile(annotations), f'The file {annotations} does not exist'

        data = pd.read_pickle(annotations)
        data = data.dropna()
        data = data.reset_index(drop=True)
        return data

    def __load_data__(self, pid:str, vid: str):
        subdir = os.path.join(pid, 'meta_data')
        accl, gyro = load_data(src_dir=os.path.join(self.src_dir, subdir), video_id=vid, is_csv=False)
        #[ ] Check if the data are loaded correctly
        accl = torch.tensor(accl.T.values, dtype=torch.float32)
        gyro = torch.tensor(gyro.T.values, dtype=torch.float32)
        return {'accl': accl, 'gyro': gyro, 'last_idx': 0}
    
    def __get_windowed_data__(self, start, stop, data, is_accl=True):
        center = center_timestamp(start, stop)
        wsize = (self.window_size * 1000.0)
        start = center - wsize // 2
        stop = center + wsize // 2
        # [ ] Check if the shape is correctly returned
        geq, leq = data[0] >= start, data[0] <= stop

        signal = data[1:, torch.where(geq & leq)[0]]
        signal = cut_and_pad(
            signal=signal,
            sampling_rate=self.sampling_rate_accl if is_accl else self.sampling_rate_gyro,
            seconds=self.window_size
            )
        return signal