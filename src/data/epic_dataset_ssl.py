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
            sampling_rate_gyro=200,
            downsampling_rate_accl=100,
            downsampling_rate_gyro=100,
            overlap_in_s=None,
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
        self.overlap_in_s = window_size / 2 if overlap_in_s is None else overlap_in_s

        self.annotations = self.__create_annotation_windows__(
            self.__load_annotations__(os.path.join(annotations, filename))
            )

        self.transforms_accl = torch.nn.Sequential(
            # T.Resample(sampling_rate_accl, downsampling_rate_accl),
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

        self.transforms_gyro = torch.nn.Sequential(
            # T.Resample(sampling_rate_gyro, downsampling_rate_gyro),
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

        
    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, index):
        action = self.annotations.iloc[index, :]
        vid, start_s, stop_s = action['video_id'], action['start_s'], action['stop_s']

        '''
        Idxs:
        0: seconds
        1: acclX
        2: acclY
        3: acclZ
        '''
        accl, gyro = self.__load_data__(vid, start_s, stop_s)

        accl = self.transforms_accl(accl)
        gyro = self.transforms_gyro(gyro)

        # # FIXME: Actual shape is (3, 65, 63)
        # accl = normalize_tensor(accl)
        # # FIXME: Actual shape is (3, 65, 126)
        # gyro = normalize_tensor(gyro)

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


    def __load_data__(self, vid: str, start_s: float, stop_s: float) -> tuple[torch.Tensor, torch.Tensor]:
        accl = self.cache[vid]['accl'].to_numpy().T
        gyro = self.cache[vid]['gyro'].to_numpy().T

        start_s, stop_s = start_s * 1000.0, stop_s * 1000.0
        idx_accl = np.where(np.logical_and(accl[0] >= start_s, accl[0] <= stop_s))[0]
        idx_gyro = np.where(np.logical_and(gyro[0] >= start_s, gyro[0] <= stop_s))[0]

        window_accl = cut_and_pad(
            torch.tensor(accl[1:, idx_accl], dtype=torch.float32), 
            self.sampling_rate_accl, self.window_size)
        window_gyro = cut_and_pad(
            torch.tensor(gyro[1:, idx_gyro], dtype=torch.float32), 
            self.sampling_rate_gyro, self.window_size)

        return window_accl, window_gyro
    

    def __align_data__(self, accl: pd.DataFrame, gyro: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        # TODO: Implement the alignment of the data

        idx_accl = np.where(pd.Series(accl['Milliseconds']).diff().round().to_numpy() >= (1 / self.sampling_rate_accl))[0]
        idx_gyro = np.where(pd.Series(gyro['Milliseconds']).diff().round().to_numpy() >= (1 / self.sampling_rate_gyro))[0]

        accl = accl.iloc[idx_accl, :]
        gyro = gyro.iloc[idx_gyro, :]

        last_ts = min(accl['Milliseconds'].iloc[-1], gyro['Milliseconds'].iloc[-1])
        first_ts = max(accl['Milliseconds'].iloc[0], gyro['Milliseconds'].iloc[0])

        accl = accl[(accl['Milliseconds'] >= first_ts) & (accl['Milliseconds'] <= last_ts)]
        gyro = gyro[(gyro['Milliseconds'] >= first_ts) & (gyro['Milliseconds'] <= last_ts)]
        
        return accl, gyro, first_ts / 1000.0, last_ts / 1000.0
    

    def __create_annotation_windows__(self, annotations: pd.DataFrame) -> pd.DataFrame:

        distinct_videos = annotations['video_id'].unique()

        df_ssl_epic = pd.DataFrame(columns=[
            'video_id',
            'start_s',
            'stop_s'
        ])

        for vid in distinct_videos:
            pid = vid.split('_')[0]
            filepath = os.path.join(self.src_dir, pid, 'meta_data')
            accl, gyro = load_data(filepath, video_id=vid, is_csv=True)
            accl, gyro, first_ts, last_ts = self.__align_data__(accl, gyro)
            
            self.cache[vid] = {
                'accl': accl,
                'gyro': gyro
            }

            duration = last_ts - first_ts
            n_windows = int((duration * (self.window_size / self.overlap_in_s)) / self.window_size)

            print(f'Video: {vid} - Duration: {duration} - Windows: {n_windows}')
            for j in range(n_windows):
                start = j * self.overlap_in_s
                stop = start + self.window_size

                df_ssl_epic.loc[len(df_ssl_epic.index)] = [vid, start, stop]

        print(len(df_ssl_epic))
        return df_ssl_epic
    