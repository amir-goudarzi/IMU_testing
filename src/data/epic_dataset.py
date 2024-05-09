import numpy as np
import pandas as pd
import os
import math
from datetime import timedelta

import torch, torchaudio
from torchvision.transforms import Compose, Resize, Normalize
import torchaudio.transforms as T
from torch.utils.data import Dataset

from features.transforms import normalize_tensor, cut_and_pad
from utils.utils import load_data, center_timestamp


class EpicDataset(Dataset):
    def __init__(self,
            src_dir: os.PathLike,
            annotations: os.PathLike,
            filename: str,
            window_size=2.5,
            cache_size=math.inf,
            n_fft=128,
            hop_length=4,
            sampling_rate_accl=200,
            sampling_rate_gyro=200,
            downsampling_rate_accl=100,
            downsampling_rate_gyro=100,
            use_cache=False
            ):
        self.src_dir = src_dir
        self.window_size = window_size
        self.cache_size = cache_size
        self.cache = {}  # Initialize an empty cache
        self.cache_index = 0  # Index to keep track of the current cache position
        self.sampling_rate_accl = sampling_rate_accl
        self.sampling_rate_gyro = sampling_rate_gyro
        self.downsampling_rate_accl = downsampling_rate_accl
        self.downsampling_rate_gyro = downsampling_rate_gyro
        self.use_cache = use_cache


        self.annotations = self.__create_annotation_windows__(
            self.__load_annotations__(os.path.join(annotations, filename))
            )
        

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
            T.AmplitudeToDB(),
            Resize((64, 64)),
            Normalize(
                mean=[-24.0039, -28.0563, -27.5905],
                std=[17.0473, 14.1716, 15.3116]
            )
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
            T.AmplitudeToDB(),
            Resize((64, 64)),
            Normalize(
                mean=[-42.7268, -42.6332, -43.2766],
                std=[13.3456, 12.9086, 11.9457]
            )
        )

        
    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, index):
        action = self.annotations.iloc[index, :]
        pid, vid = action['participant_id'], action['video_id']
        start_s, stop_s = action['start_timestamp'], action['stop_timestamp']
        target = torch.tensor(action['verb_class'], dtype=torch.int16)
        x = None

        if vid not in self.cache:
            x = self.cache[vid] = self.__load_data__(pid, vid)
        else:
            x = self.cache[vid]
        
        accl = self.__get_windowed_data__(start_s, stop_s, x['accl'])
        gyro = self.__get_windowed_data__(start_s, stop_s, x['gyro'])

        accl = self.transforms_accl(accl)
        gyro = self.transforms_gyro(gyro)

        x = torch.cat((accl, gyro), dim=0)

        return (x, target)


    def set_use_cache(self, use_cache: bool):
        if use_cache:
            self.cached_data = torch.stack(self.cached_data)
        else:
            self.cached_data = []
        self.use_cache = use_cache


    def __load_annotations__(self, annotations: os.PathLike) -> pd.DataFrame:
        assert os.path.isfile(annotations), f'The file {annotations} does not exist'

        data = pd.read_pickle(annotations)
        data = data.dropna()
        data = data.reset_index(drop=True)

        return data

    def __load_data__(self, pid:str, vid: str) -> dict[str, torch.Tensor]:
        subdir = os.path.join(pid, 'meta_data')
        accl, gyro = load_data(src_dir=os.path.join(self.src_dir, subdir), video_id=vid, is_csv=False)
        accl = torch.tensor(accl.values, dtype=torch.float32)
        gyro = torch.tensor(gyro.values, dtype=torch.float32)
        return {'accl': accl.T, 'gyro': gyro.T}
    
    
    def __get_windowed_data__(self, start_s, stop_s, data: torch.Tensor) -> torch.Tensor:
        center = center_timestamp(start_s, stop_s, mode='seconds')
        new_start = center - self.window_size * 1000 // 2
        new_stop  = center + self.window_size * 1000 // 2

        idx = torch.where(torch.logical_and(data[0, :] >= new_start, data[0, :] <= new_stop))[0]
        window = cut_and_pad(
            signal=data[1:, idx],
            sampling_rate=self.sampling_rate_accl,
            seconds=self.window_size
        )
        return window
    

    def __align_data__(self, accl: pd.DataFrame, gyro: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        idx_accl = np.where(pd.Series(accl['Milliseconds']).diff().round().to_numpy() >= (1 / self.sampling_rate_accl * 1000))[0]
        idx_gyro = np.where(pd.Series(gyro['Milliseconds']).diff().round().to_numpy() >= (1 / self.sampling_rate_gyro * 1000))[0]

        accl = accl.iloc[idx_accl, :]
        gyro = gyro.iloc[idx_gyro, :]

        last_ts = min(accl['Milliseconds'].iloc[-1], gyro['Milliseconds'].iloc[-1])
        first_ts = max(accl['Milliseconds'].iloc[0], gyro['Milliseconds'].iloc[0])

        accl = accl[(accl['Milliseconds'] >= first_ts) & (accl['Milliseconds'] <= last_ts)]
        gyro = gyro[(gyro['Milliseconds'] >= first_ts) & (gyro['Milliseconds'] <= last_ts)]
        
        return accl, gyro, first_ts / 1000.0, last_ts / 1000.0
    

    def __create_annotation_windows__(self, annotations: pd.DataFrame) -> pd.DataFrame:
        distinct_videos = annotations['video_id'].unique()
        annotations['start_timestamp'] = pd.to_timedelta(annotations['start_timestamp']).dt.total_seconds()
        annotations['stop_timestamp'] = pd.to_timedelta(annotations['stop_timestamp']).dt.total_seconds()

        for vid in distinct_videos:
            pid = vid.split('_')[0]
            filepath = os.path.join(self.src_dir, pid, 'meta_data')
            accl, gyro = load_data(filepath, video_id=vid, is_csv=True)
            accl, gyro, first_ts, last_ts = self.__align_data__(accl, gyro)

            accl = torch.tensor(accl.values, dtype=torch.float32)
            gyro = torch.tensor(gyro.values, dtype=torch.float32)

            self.cache[vid] = {
                'accl': accl.T,
                'gyro': gyro.T
            }
            
            annotations = annotations.drop(
                annotations[
                    (annotations.video_id==vid) & # bitwise AND
                    ((annotations['start_timestamp'] < first_ts) | # bitwise OR
                     (annotations['stop_timestamp'] > last_ts))].index)

        print(len(annotations))
        return annotations