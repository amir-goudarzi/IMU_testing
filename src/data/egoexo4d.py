import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T

import json
import pandas as pd
import os
import numpy as np
import pickle as pkl
import math
from time import time
from copy import deepcopy

from features.imu_preprocessing import SpectrogramsGenerator
from features.transforms import normalize_tensor, cut_and_pad
from data.dataset import register_dataset
from utils.utils import create_binary_array


@register_dataset("egoexo4d")
class EgoExo4D(Dataset):
    def __init__(
            self,
            data_dir: os.PathLike,
            annotations_file: os.PathLike,
            stream_labels: list[str],
            window_size,
            # overlap_in_s,
            n_fft: int,
            win_length: int,
            hop_length: int,
            resizes: tuple[int, int],
            temporal_points: int,
            sampling_rate: int,
            task_name: str,
            sensors: list[str],
            downsampling_rate=None,
            transforms=None,
            cache_len=math.inf,
            device='cuda:1',
            preload=False,
            tasks_file=None | os.PathLike,
            labels_file=None | os.PathLike
        ):
        assert os.path.isfile(annotations_file), "Invalid path to EgoExo4D annotations"
        assert os.path.exists(data_dir), "Invalid path to EgoExo4D data"
        assert stream_labels.__len__() <= 2, "Only two IMU streams are supported"
        assert all([stream_label == 'imu-left' or stream_label == 'imu-right' for stream_label in stream_labels]), "Invalid stream labels"


        self.data_dir = data_dir
        self.takes = pkl.load(open(annotations_file, 'rb'))
        self.stream_labels = stream_labels
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.task_name = task_name
        self.sensors = sensors
        self.preload = preload
        self.tasks_file = json.load(open(tasks_file, 'r')) if tasks_file is not None else tasks_file
        self.labels_file = pd.read_pickle(labels_file) if labels_file is not None else labels_file

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

        self.resample_left =  T.Resample(800, self.sampling_rate)
        self.resample_right = T.Resample(1000, self.sampling_rate)
        
        self.cache = {}
        self.cache_len = cache_len

        # Define the getitem function based on the task. 
        self.getitem = self.__define_get_function__()

        if self.preload:
            self.__preload__(self.labels_file is not None)

    def __len__(self):
        return len(self.takes)


    def __getitem__(self, idx):
        # Get the file
        take = self.takes[idx]
        
        #FIXME: Uncomment for classifying
        return self.getitem(take), take['label']
        # return self.getitem(take), 0
        # return self.getitem(take), create_binary_array(109, 0)


    def __define_get_function__(self):
        def get_imu(take):
            imu = self.__get_imu__(take)
            return self.transforms(imu)
        def get_combined(take):
            imu = self.__get_imu__(take)
            spectrogram = self.transforms(imu)
            omnivore = self.__get_omnivore__(take)
            return spectrogram, omnivore
        
        if self.task_name == "imu":
            return get_imu
        else:
            return get_combined

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
        start_s = take['start_sec']
        end_s = take['end_sec']

        # Get IMU data
        left, right = self.__load_imu__(take_name, start_s, end_s)

        # Cut the IMU data
        left = cut_and_pad(left, self.sampling_rate, self.window_size)
        right = cut_and_pad(right, self.sampling_rate, self.window_size)

        # Compute the mean of the IMU data
        return self.__compute_mean__(left, right)
    

    def __load_imu__(self, take_name, start_s, end_s, preload=False) -> torch.Tensor:
        '''
        Get an IMU stream from .npy file.

        Args:
        - take_name: str
        - start_s: float
        - end_s: float

        Returns:
        - torch.Tensor

        '''
        take_path = os.path.join(self.data_dir, 'features', 'imu_aria01', take_name)
        start_ns, end_ns = start_s * 1e9, end_s * 1e9
        imu_left = imu_right = None

        if take_name not in self.cache:
            if self.cache.__len__() >= self.cache_len:
                self.cache.popitem()
            channels = 6
            if "gyro" not in self.sensors:
                channels = 3
            # 1 KHz
            imu_right = torch.from_numpy(np.load(os.path.join(take_path, f'right.npy'))).type(torch.float32).T

            # 800 Hz
            imu_left = torch.from_numpy(np.load(os.path.join(take_path, f'left.npy'))).type(torch.float32).T
            imu_left = imu_left[:channels+1, :]
            imu_right = imu_right[:channels+1, :]
            self.cache[take_name] = (imu_left, imu_right)
        else:
            imu_left, imu_right = self.cache[take_name]

        if preload:
            return imu_left

        effective_start = imu_left[0,0] if imu_left[0,0] > imu_right[0,0] else imu_right[0,0]

        # Align to the first timestamp
        start_ns += effective_start
        end_ns += effective_start

        right = self.resample_right(imu_right[1:, (imu_right[0] >= start_ns) & (imu_right[0] < end_ns)])
        left = self.resample_left(imu_left[1:, (imu_left[0] >= start_ns) & (imu_left[0] < end_ns)])

        return left, right
        
        
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
        accl = torch.div(torch.add(left[:3, :], right[:3, :]), 2)
        if "gyro" not in self.sensors:
            return accl
        # Gyro
        gyro = torch.div(torch.add(left[:3, :], right[:3, :]), 2)
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
        features_path = os.path.join(self.data_dir, 'features')
        omnivore = torch.load(os.path.join(features_path, 'omnivore_video', f"{take['take_uid']}_aria01_rgb.pt"))

        # Get the exact omnivore feature. The file is a tensor of shape (#feat, 1, dim. feat).
        # For further information, please refer to the following link:
        # https://docs.ego-exo4d-data.org/data/features/
        return omnivore[omni_idx, :, :]


    def __preload__(self, is_classification=False):
        # tmp_takes = []
        for idx, take in enumerate(self.takes):
            # if is_classification:
            #     res = self.__get_label_by_entry__(take)
            #     if res is None:
            #         continue
            #     else:
            #         tmp_takes.append({**take, 'label': res })
            # else:
            #     tmp_takes.append({**take, 'label': None })
            self.__load_imu__(take["take_name"], start_s=0, end_s=0, preload=True)
        # self.takes = tmp_takes


    def __get_label_by_entry__(self, take):
        """
        Ref.: https://docs.ego-exo4d-data.org/annotations/keystep/
        """
        try:
            if take['take_uid'] not in self.tasks_file['annotations']:
                return None
            benchmark_take = pd.DataFrame(self.tasks_file['annotations'][take['take_uid']]['segments'])
            # start_s = take['start_sec'] + take['effective_start_sec']
            # end_s = take['end_sec'] + take['effective_start_sec']
            label = benchmark_take.where(
                (benchmark_take['start_time'] <= take['start_sec']) & (benchmark_take['end_time'] >= take['end_sec'])).dropna().iloc[-1]['step_unique_id']
            label = self.labels_file[self.labels_file['step_unique_id'] == str(int(label))]['verb_idx'].iloc[0]
            # print(f"Label: {label}")
            return label
        except Exception as e:
            return None
        
        