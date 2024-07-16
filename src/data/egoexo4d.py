import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T

import json
import os
import numpy as np
import pickle as pkl

from features.imu_preprocessing import SpectrogramsGenerator
from features.transforms import normalize_tensor, cut_and_pad

class EgoExo4D(Dataset):
    def __init__(
            self,
            data_dir: os.PathLike,
            takes_path: os.PathLike,
            stream_labels: list[str],
            window_size,
            # overlap_in_s,
            n_fft: int,
            win_length: int,
            hop_length: int,
            transforms,
            resizes: tuple[int, int],
            temporal_points: int,
            sampling_rate: int,
            downsampling_rate=None,
        ):
        assert os.path.isfile(takes_path), "Invalid path to EgoExo4D takes"
        assert os.path.exists(data_dir), "Invalid path to EgoExo4D data"
        assert stream_labels.__len__() <= 2, "Only two IMU streams are supported"
        assert all([stream_label == 'imu-left' or stream_label == 'imu-right' for stream_label in stream_labels]), "Invalid stream labels"

        self.data_dir = data_dir
        self.takes = pkl.load(open(takes_path, 'rb'))
        self.stream_labels = stream_labels
        self.sampling_rate = sampling_rate
        self.window_size = window_size

        self.transforms = SpectrogramsGenerator(
            window_size=window_size,
            # overlap_in_s=overlap_in_s,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            sampling_rate=sampling_rate,
            downsampling_rate=downsampling_rate,
            transforms=transforms,
            resizes=resizes,
            temporal_points=temporal_points
        )
        

    def __len__(self):
        return len(self.takes)


    def __getitem__(self, idx):
        # Get the file
        take = self.takes[idx]
        
        # Get the IMU streams
        imu = self.__get_imu__(take)

        # Get omnivore stream
        omnivore = self.__get_omnivore__(take)

        # Get the spectrogram
        spectrogram = self.transforms(imu)

        return spectrogram, omnivore


    def __get_imu__(self, take, positions=['imu_left', 'imu-right']) -> torch.Tensor:
        '''
        Get an IMU stream from .npy file.

        Args:
        - take: dict
        - positions: tuple[str, str]

        Returns:
        - torch.Tensor

        '''
        take_name = take['take_name']
        start_s = take['start_s']
        end_s = take['end_s']

        # Get IMU data
        left = self.__load_imu__(take_name, start_s, end_s, 'imu-left')
        right = self.__load_imu__(take_name, start_s, end_s, 'imu-right')

        # Cut the IMU data
        left = cut_and_pad(left, self.sampling_rate, self.window_size)
        right = cut_and_pad(right, self.sampling_rate, self.window_size)

        # Compute the mean of the IMU data
        return self.__compute_mean__(left, right)
    

    def __load_imu__(self, take_name, start_s, end_s, position='imu-right') -> torch.Tensor:
        '''
        Get an IMU stream from .npy file.

        Args:
        - take_name: str
        - start_s: float
        - end_s: float
        - position: str

        Returns:
        - torch.Tensor

        '''
        take_path = os.path.join(self.data_dir, 'features', take_name)
        start_ns, end_ns = start_s * 1e9, end_s * 1e9

        if position == 'imu-right':
            # 1 KHz
            imu_right = torch.from_numpy(np.load(os.path.join(take_path, f'right.npy'))).type(torch.float32)
            # Align to the first timestamp
            start_ns += imu_right[0, 0]
            end_ns += imu_right[0, 0]
            imu_right = imu_right[1:, (imu_right[0] >= start_ns) & (imu_right[0] <= end_ns)]
            return T.Resample(1000, self.sampling_rate)(imu_right)
        else:
            # 800 Hz
            imu_left = torch.from_numpy(np.load(os.path.join(take_path, f'left.npy'))).type(torch.float32)
            # Align to the first timestamp
            start_ns += imu_left[0, 0]
            end_ns += imu_left[0, 0]
            imu_left = imu_left[1:, (imu_left[0] >= start_ns) & (imu_left[0] <= end_ns)]
            return T.Resample(800, self.sampling_rate)(imu_left)
        
        
    def __compute_mean__(self, left, right) -> torch.Tensor:
        '''
        Compute the mean of the IMU data.

        Args:
        - left: torch.Tensor
        - right: torch.Tensor

        Returns:
        - torch.Tensor

        '''
        # Accl
        accl = torch.cat((left[:, :3], right[:, :3]), dim=1).mean(dim=0)
        # Gyro
        gyro = torch.cat((left[:, 3:], right[:, 3:]), dim=1).mean(dim=0)
        # IMU
        return torch.cat((accl, gyro), dim=0)
    

    def __get_omnivore__(self, take) -> torch.Tensor:
        '''
        Get an omnivore stream from .pt file.

        Args:
        - take: str

        Returns:
        - torch.Tensor

        '''
        take_name = take['take_name']
        omni_idx = take['omnivore_idx']
        features_path = os.path.join(self.data_dir, 'features', take_name)
        omnivore = torch.load(os.path.join(features_path, 'omnivore_video', f"{take['take_uid']}_aria01_rgb.pt"))

        # Get the exact omnivore feature. The file is a tensor of shape (#feat, 1, dim. feat).
        # For further information, please refer to the following link:
        # https://docs.ego-exo4d-data.org/data/features/
        return omnivore[omni_idx, :, :]
