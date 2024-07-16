import numpy as np
import pandas as pd
import os
import math

import torch, torchaudio
from torchvision.transforms import Compose, Resize
import torchaudio.transforms as T
from torch.utils.data import Dataset

from features.transforms import normalize_tensor, cut_and_pad
from utils.utils import load_data, center_timestamp
from features.imu_preprocessing import SpectrogramsGenerator


class WearDatasetSSL(Dataset):
    def __init__(self,
            src_dir: os.PathLike,
            annotations: os.PathLike,
            filename: str,
            window_size=10,
            overlap_in_s=None,
            cache_size=math.inf,
            n_fft=128,
            hop_length=4,
            sampling_rate=50,
            downsampling_rate=25,
            use_cache=False,
            transforms=None,
            resizes=(64, 64)
            ):
        self.src_dir = src_dir
        self.window_size = window_size
        self.cache_size = cache_size
        self.sampling_rate = sampling_rate
        self.downsampling_rate = downsampling_rate
        self.overlap_in_s = window_size / 2 if overlap_in_s is None else overlap_in_s
        self.cache = {}  # Initialize an empty cache
        self.annotations = self.__load_annotations__(os.path.join(annotations, filename))
        self.use_cache = use_cache

        self.transforms = SpectrogramsGenerator(
            window_size=window_size,
            overlap_in_s=overlap_in_s,
            n_fft=n_fft,
            hop_length=hop_length,
            sampling_rate=sampling_rate,
            downsampling_rate=downsampling_rate,
            transforms=transforms,
            resizes=resizes
        )
        
    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, index):
        action = self.annotations.iloc[index, :]
        subject = action['subject']
        start, stop = action['start_s'], action['stop_s']

        if subject not in self.cache:
            self.cache[subject] = self.__load_data__(subject=subject)
        
        x = self.cache[subject]
        
        accl = self.__get_windowed_data__(start, stop, x)

        accl = self.transforms(accl)

        return accl


    def set_use_cache(self, use_cache):
        if use_cache:
            self.cached_data = torch.stack(self.cached_data)
        else:
            self.cached_data = []
        self.use_cache = use_cache


    def __load_annotations__(self, annotations: os.PathLike):
        assert os.path.isfile(annotations), f'The file {annotations} does not exist'

        data = pd.read_pickle(annotations)
        data = data[['sbj_id', 'duration']].drop_duplicates()
        data = self.__create_annotations_windows__(data)
        return data


    def __load_data__(self, subject: str) -> dict:
        data = pd.read_csv(os.path.join(self.src_dir, 'raw', 'inertial', f'{subject}.csv'))
        
        right_arm = torch.tensor(data[['right_arm_acc_x', 'right_arm_acc_y', 'right_arm_acc_z']].values.T, dtype=torch.float32)
        left_arm = torch.tensor(data[['left_arm_acc_x', 'left_arm_acc_y', 'left_arm_acc_z']].values.T, dtype=torch.float32)
        right_leg = torch.tensor(data[['right_leg_acc_x', 'right_leg_acc_y', 'right_leg_acc_z']].values.T, dtype=torch.float32)
        left_leg = torch.tensor(data[['left_leg_acc_x', 'left_leg_acc_y', 'left_leg_acc_z']].values.T, dtype=torch.float32)

        accl = torch.cat([right_arm, left_arm, right_leg, left_leg], dim=0)
        # assert not torch.isnan(accl).any(), "ops, there is a NaN in the left arm data!"
        accl = torch.nan_to_num(accl, nan=0.0)
        return accl
    

    def __get_windowed_data__(self, start, stop, data):
        start = int(start * self.sampling_rate)
        stop = int(stop * self.sampling_rate)
        window = data[:, start:stop]
        
        assert not torch.isnan(window).any(), "ops, there is a NaN in the windowed data!"
        
        window = cut_and_pad(window, self.sampling_rate, self.window_size)
        return window
    

    def __create_annotations_windows__(self, data) -> pd.DataFrame:
        '''
        data: pd.DataFrame with columns:
        ['sbj_id', 'duration']
        '''

        df_ssl_wear = pd.DataFrame(columns=[
            'subject',
            'start_s',
            'stop_s'
        ])

        for i, row in data.iterrows():
            duration = row['duration']
            sbj_id = row['sbj_id']
            n_windows = int((duration * (self.window_size / self.overlap_in_s)) / self.window_size)

            for j in range(n_windows):
                start = j * self.overlap_in_s
                stop = start + self.window_size

                df_ssl_wear.loc[len(df_ssl_wear.index)] = [sbj_id, start, stop]
        return df_ssl_wear