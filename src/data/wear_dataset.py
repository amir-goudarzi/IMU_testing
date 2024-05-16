import numpy as np
import pandas as pd
import os
import math

import torch, torchaudio
from torchvision.transforms import Compose, Resize
import torchaudio.transforms as T
from torch.utils.data import Dataset

from features.transforms import normalize_tensor, cut_and_pad, cut_and_pad_lr
from utils.utils import load_data, center_timestamp

'''
TODO: At the moment it is not possible to load gyroscope data,
as the function load_data only returns accelerometer data

TODO: Check the implementation of this dataloader
'''
class WearDataset(Dataset):
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
            num_classes=18
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
        self.num_classes = num_classes

        self.transforms = torch.nn.Sequential(
            T.Resample(sampling_rate, downsampling_rate),
            T.Spectrogram(
                n_fft=n_fft,
                hop_length=hop_length,
                center=True,
                pad_mode="reflect",
                power=2.0,
                normalized=True
                ),
            T.AmplitudeToDB(),
            Resize((64, 64))
        )

        if transforms:
            self.transforms.append(transforms)
        
    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, index):
        action = self.annotations.iloc[index, :]
        subject = action['sbj_id']
        start, stop = action['start_s'], action['stop_s']
        target = torch.tensor(action['label_id'], dtype=torch.uint8)

        if subject not in self.cache:
            self.cache[subject] = self.__load_data__(subject=subject)
        
        x = self.cache[subject]
        
        accl = self.__get_windowed_data__(start, stop, x)
        accl = self.transforms(accl)

        target = self.create_binary_array(target)

        return (accl, target)


    def set_use_cache(self, use_cache):
        if use_cache:
            self.cached_data = torch.stack(self.cached_data)
        else:
            self.cached_data = []
        self.use_cache = use_cache

    def create_binary_array(self, target_class):
        binary_array = torch.zeros(self.num_classes)  # Initialize array with zeros
        binary_array[int(target_class)] = 1  # Set the target class to 1
        assert binary_array.sum() == 1, f'Target class is not unique: {binary_array}'
        return binary_array

    def __load_annotations__(self, annotations: os.PathLike):
        assert os.path.isfile(annotations), f'The file {annotations} does not exist'
        data = pd.read_pickle(annotations)
        data['duration'] = data['stop_s'] - data['start_s']
        return data

    def __load_data__(self, subject: str) -> dict:
        data = pd.read_csv(os.path.join(self.src_dir, 'raw', 'inertial', f'{subject}.csv'))
        #[x] Check if the data are loaded correctly
        right_arm = torch.tensor(data[['right_arm_acc_x', 'right_arm_acc_y', 'right_arm_acc_z']].values.T, dtype=torch.float32)
        left_arm = torch.tensor(data[['left_arm_acc_x', 'left_arm_acc_y', 'left_arm_acc_z']].values.T, dtype=torch.float32)
        right_leg = torch.tensor(data[['right_leg_acc_x', 'right_leg_acc_y', 'right_leg_acc_z']].values.T, dtype=torch.float32)
        left_leg = torch.tensor(data[['left_leg_acc_x', 'left_leg_acc_y', 'left_leg_acc_z']].values.T, dtype=torch.float32)

        accl = torch.cat([right_arm, left_arm, right_leg, left_leg], dim=0)
        # assert not torch.isnan(accl).any(), "ops, there is a NaN in the left arm data!"
        accl = torch.nan_to_num(accl, nan=0.0)
        return accl
    
    def __get_windowed_data__(self, start, stop, data):
        center = (start + stop) / 2
        center_in_idx = int(center * self.sampling_rate)
        delta = self.window_size // 2 * self.sampling_rate
        new_start, new_stop = center_in_idx - delta, center_in_idx + delta
        
        padding = 'left'
        window = None

        if new_start < 0:
            new_start = 0
        elif new_stop > data.shape[-1]:
            new_stop = data.shape[-1]
            padding = 'right'
        else:
            window = cut_and_pad(
                signal=data[:, new_start:new_stop],
                sampling_rate=self.sampling_rate,
                seconds=self.window_size
            )
            assert not torch.isnan(window).any(), "ops, there is a NaN in the windowed data!"
            return window
        
        window = cut_and_pad_lr(
            signal=data[:, new_start:new_stop],
            sampling_rate=self.sampling_rate,
            seconds=self.window_size,
            side=padding
        )

        assert not torch.isnan(window).any(), "ops, there is a NaN in the windowed data!"
        
        return window