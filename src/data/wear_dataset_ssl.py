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
from data.dataset import register_dataset

@register_dataset("wear_ssl")
class WearDatasetSSL(Dataset):
    def __init__(self,
            src_dir: os.PathLike,
            annotations: os.PathLike,
            filename: str,
            window_size=2,
            win_length=5,
            overlap_in_s=None,
            cache_size=math.inf,
            n_fft=128,
            hop_length=4,
            sampling_rate=50,
            downsampling_rate=25,
            use_cache=False,
            transforms=None,
            temporal_points=None,
            resizes=(128, 320),
            is_train=False,
            mean_std_path=None,
            i3d=False,
            rgb=False
            ):
        self.src_dir = src_dir
        self.window_size = window_size
        self.cache_size = cache_size
        self.sampling_rate = sampling_rate
        self.downsampling_rate = downsampling_rate
        self.i3d = i3d
        self.rgb = rgb
        self.overlap_in_s = window_size / 2 if overlap_in_s is None else overlap_in_s
        self.cache = {}  # Initialize an empty cache
        self.annotations = self.__load_annotations__(os.path.join(annotations, filename))
        self.return_fn = self.get_return_fn()

        self.use_cache = use_cache

        if mean_std_path is not None:
            mean = torch.load(os.path.join(mean_std_path, 'accl_mean.pt'))
            std = torch.load(os.path.join(mean_std_path, 'accl_std.pt'))
            
        self.transforms = SpectrogramsGenerator(
            window_size=window_size,
            n_fft=n_fft,
            win_length=win_length,
            overlap_in_s=overlap_in_s,
            hop_length=hop_length,
            temporal_points=temporal_points,
            sampling_rate=sampling_rate,
            downsampling_rate=downsampling_rate,
            transforms=transforms,
            resizes=resizes,
            is_train=is_train,
            mean=mean,
            std=std
        )
        
    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, index):
        # action = self.annotations.iloc[index, :]
        # subject = action['subject']
        # start, stop = action['start_s'], action['stop_s']

        # if subject not in self.cache:
        #     # self.cache[subject] = self.__load_data__(subject=subject)
        #     self.cache[subject] = self.__load_combined_data__(subject=subject)
        subject, index = self.annotations[index]['subject'], self.annotations[index]['index']
        x = self.cache[subject]
        
        # accl = self.__get_windowed_data__(start, stop, x)
        accl, i3d = x['imu'][index], x['i3d'][index]
        accl = torch.tensor(accl, dtype=torch.float32)
        i3d = torch.tensor(i3d, dtype=torch.float32).squeeze(0)
        accl = self.transforms(accl)

        return self.return_fn((accl, i3d))


    def set_use_cache(self, use_cache):
        if use_cache:
            self.cached_data = torch.stack(self.cached_data)
        else:
            self.cached_data = []
        self.use_cache = use_cache

    def get_return_fn(self):
        def imu_return_fn(x):
            accl, _ = x
            return accl
        def i3d_return_fn(x):
            accl, i3d = x
            return accl, i3d
        def rgb_return_fn(x):
            accl, i3d = x
            return accl, i3d[:1024]
        
        if self.i3d:
            if self.rgb:
                return rgb_return_fn
            return i3d_return_fn
        return imu_return_fn

    def __load_annotations__(self, annotations: os.PathLike):
        # assert os.path.isfile(annotations), f'The file {annotations} does not exist'

        # data = pd.read_pickle(annotations)
        # data = data[['sbj_id', 'duration']].drop_duplicates()
        # data = self.__create_annotations_windows__(data)


        ########################
        ###   NEW CODE HERE  ###
        ########################
        annotations_split = pd.read_json(annotations)
        # Clan from labels
        annotations_split = annotations_split[~annotations_split['database'].isna()]
        # Filter only the training subset
        annotations_split_train = annotations_split['database'].apply(lambda x : x['subset'] == 'Training')
        subjects = annotations_split_train[annotations_split_train==True].index
        
        annotations_split_train = []
        for subject in subjects:
            self.cache[subject] = self.__load_combined_data__(subject)
            [annotations_split_train.append({'subject': subject, 'index': i}) for i in range(self.cache[subject]['imu'].shape[0])]
        return annotations_split_train


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

    def __load_combined_data__(self, subject: str) -> dict:
        frames = self.window_size * 60
        stride = frames // 2
        data = np.load(os.path.join(self.src_dir, 'processed', 'combined_features', 
                                        f'{frames}_frames_{stride}_stride', f'{subject}.npy'))
        imus = data[:, :self.sampling_rate * self.window_size * 12].reshape(-1, 12, self.sampling_rate * self.window_size)
        sample = {
            'imu': imus,
            'i3d': data[:, self.sampling_rate * self.window_size * 12: ]
        }

        assert sample['i3d'].shape[1] == 2048, f"ops, the i3d features have shape {sample['i3d'].shape}"

        return sample
    
    def __get_windowed_data__(self, start, stop, data):
        start = int(start * self.sampling_rate)
        stop = int(stop * self.sampling_rate)
        window = data[:, start:stop]
        
        assert not torch.isnan(window).any(), "ops, there is a NaN in the windowed data!"
        
        window = cut_and_pad(window, self.sampling_rate, self.window_size)
        return window
    
    def __get_combined_windowed_data__(self, start, stop, data):
        start = int(start * self.sampling_rate)
        stop = int(stop * self.sampling_rate)
        window = data['imu'][:, start:stop]
        
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