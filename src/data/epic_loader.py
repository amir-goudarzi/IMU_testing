import os
import numpy as np
import pandas as pd
from time import time

import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchaudio.transforms import MelSpectrogram, Spectrogram


class EpicKitchens100(Dataset):
    def __init__(
            self,
            root_dir,
            annotations_dir,
            seconds,
            filename='EPIC_100_train_clean.pkl',
            downsampling_rate=150,
            train=True,
            transform=None
        ):
        self.split = filename
        assert os.path.exists(root_dir), 'The directory does not exist'
        assert os.path.isfile(os.path.join(annotations_dir, self.split)), 'The file epic_kitchens_100_train.csv does not exist'

        self.seconds = seconds
        self.root_dir = root_dir
        self.transform = transform
        self.downsampling_rate = downsampling_rate

        self.data = pd.read_pickle(os.path.join(annotations_dir, self.split))
        self.data = self.data.dropna()
        self.data = self.data.reset_index(drop=True)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        participant_id = self.data.loc[idx, 'participant_id']
        video_id = self.data.loc[idx, 'video_id']
        clip_path = os.path.join(self.root_dir, participant_id, 'meta_data', video_id)
        
        seconds, signals, sampling_rate = self.load_data(clip_path)

        label = self.data.iloc[idx, -1]

        if self.transform:
            signals = self.transform(signals)

        return signals, label

    def load_data(self, filename):
        '''
        Load the WAV file and its label.
        Args :
            • filename: str. The path of a WAV file.
        Returns A tuple of two Pandas DataFrame objects:
            • signals: A DataFrame with the following columns:
                • seconds: The time in seconds.
                • AcclX: The acceleration along the x-axis.
                • AcclY: The acceleration along the y-axis.
                • AcclZ: The acceleration along the z-axis.
                • GyroX: The angular velocity along the x-axis.
                • GyroY: The angular velocity along the y-axis.
                • GyroZ: The angular velocity along the z-axis.
            • sampling_rate: The sampling rate of the WAV file.
        '''
        df_accl, df_gyro = pd.read_csv(filename + '-accl.csv'), pd.read_csv(filename + '-gyro.csv')
        seconds = df_accl['Milliseconds'] / 1000.0
        seconds.name = "seconds"
        sampling_rate = int(len(df_accl) / (seconds.iloc[-1] - seconds.iloc[0]))
        seconds = torch.tensor(seconds.values, dtype=torch.float32)
        accl, gyro = df_accl[["AcclX", "AcclY", "AcclZ"]], df_gyro[["GyroX", "GyroY", "GyroZ"]]
        
        signals = torch.tensor([accl.values, gyro.values], dtype=torch.float32)
        return seconds, signals, sampling_rate
