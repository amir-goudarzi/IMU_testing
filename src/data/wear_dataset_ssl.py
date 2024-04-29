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

'''
TODO: At the moment it is not possible to load gyroscope data,
as the function load_data only returns accelerometer data

TODO: Check the implementation of this dataloader
'''
class WearDatasetSSL(Dataset):
    def __init__(self,
            src_dir: os.PathLike,
            annotations: os.PathLike,
            window_size: 10,
            overlap_in_s: 5,
            cache_size=math.inf,
            n_fft=128,
            hop_length=4,
            sampling_rate=50,
            downsampling_rate=25,
            use_cache=False,
            transforms=None
            ):
        self.src_dir = src_dir
        self.window_size = window_size
        self.cache_size = cache_size
        self.sampling_rate = sampling_rate
        self.downsampling_rate = downsampling_rate
        self.overlap_in_s = overlap_in_s
        self.cache = {}  # Initialize an empty cache
        self.cache_index = 0  # Index to keep track of the current cache position
        self.annotations = self.__load_annotations__(annotations)
        self.use_cache = use_cache

        self.transforms = torch.nn.Sequential(
            T.Resample(sampling_rate, downsampling_rate),
            T.Spectrogram(
                n_fft=n_fft,
                hop_length=hop_length,
                center=True,
                pad_mode="reflect",
                power=2.0,
                normalized=True
                ).float(),
            T.AmplitudeToDB(),
            Resize((64, 64))
        )
        if transforms:
            self.transforms.append(transforms)
        
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

        # accl = normalize_tensor(accl)

        return accl


    def set_use_cache(self, use_cache):
        if use_cache:
            self.cached_data = torch.stack(self.cached_data)
        else:
            self.cached_data = []
        self.use_cache = use_cache


    def __load_annotations__(self, annotations: os.PathLike):
        assert os.path.isfile(annotations), f'The file {annotations} does not exist'

        data = pd.read_json(annotations)
        data = self.__create_annotations_windows__(data)

        return data


    def __load_data__(self, subject: str) -> dict:
        data = pd.read_csv(os.path.join(self.src_dir, f'{subject}.csv'))
        #[ ] Check if the data are loaded correctly
        right_arm = torch.tensor(data[['right_arm_acc_x', 'right_arm_acc_y', 'right_arm_acc_z']].values.T, dtype=torch.float32)
        left_arm = torch.tensor(data[['left_arm_acc_x', 'left_arm_acc_y', 'left_arm_acc_z']].values.T, dtype=torch.float32)
        right_leg = torch.tensor(data[['right_leg_acc_x', 'right_leg_acc_y', 'right_leg_acc_z']].values.T, dtype=torch.float32)
        left_leg = torch.tensor(data[['left_leg_acc_x', 'left_leg_acc_y', 'left_leg_acc_z']].values.T, dtype=torch.float32)

        accl = torch.cat([right_arm, left_arm, right_leg, left_leg], dim=0)
        # assert not torch.isnan(accl).any(), "ops, there is a NaN in the left arm data!"
        accl = torch.nan_to_num(accl, nan=0.0)
        return accl
    

    def __get_windowed_data__(self, start, stop, data):
        start = start * self.sampling_rate
        stop = stop * self.sampling_rate
        # [ ] Check if the shape is correctly returned
        window = data[:, start:stop]
        
        assert not torch.isnan(window).any(), "ops, there is a NaN in the windowed data!"
        
        window = cut_and_pad(window, self.sampling_rate, self.window_size)
        return window
    

    def __create_annotations_windows__(self, data) -> pd.DataFrame:
        aux = {}

        for i, row in data.iterrows():
            
            if  i.startswith('sbj') and (i not in aux) and (row['database']['subset'] == 'Training'):
                
                aux[i] = {
                    'duration_s': row['database']['duration']
                }

        df_ssl_wear = pd.DataFrame(columns=[
            'subject',
            'start_s',
            'stop_s'
        ])

        for i, row in aux.items():
            duration = row['duration_s']
            n_windows = int((duration * (self.window_size / self.overlap_in_s)) / self.window_size)

            for j in range(n_windows):
                start = j * self.overlap_in_s
                stop = start + self.window_size

                df_ssl_wear.loc[len(df_ssl_wear.index)] = [i, start, stop]
        return df_ssl_wear