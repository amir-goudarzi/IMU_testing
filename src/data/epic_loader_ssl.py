import torch, torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from features.transforms import normalize_tensor, cut_and_pad
from utils.utils import load_data
import numpy as np

'''
TODO: At the moment it is not possible to load gyroscope data,
as the function load_data only returns accelerometer data

TODO: Check the implementation of this dataloader
'''
class EpicDatasetSSL(Dataset):
    def __init__(self, data_paths, window_size, cache_size=100):
        self.data_paths = data_paths  # Paths to accelerometer files
        self.window_size = window_size
        self.cache_size = cache_size
        self.cache = []  # Initialize an empty cache
        self.cache_index = 0  # Index to keep track of the current cache position

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # Check if data is in cache, if not, load from file and add to cache
        if idx >= len(self.cache) or len(self.cache) < self.cache_size:
            self._load_to_cache(idx)
        
        # Get data from cache
        sample = self.cache[idx % self.cache_size]

        # Implement sliding window
        start_idx = np.random.randint(0, len(sample) - self.window_size)
        end_idx = start_idx + self.window_size
        sample = sample[start_idx:end_idx]

        # Convert to PyTorch tensor if needed
        sample = torch.tensor(sample, dtype=torch.float32)

        return sample

    def _load_to_cache(self, idx):
        # Load data from file and add to cache
        path = self.data_paths[idx]
        # Assuming you have a function to load accelerometer data from a file
        data, _ = load_data(path) #TODO: Implement gyroscope data loading
        self.cache.append(data)

        # If cache is full, replace oldest entry
        if len(self.cache) > self.cache_size:
            self.cache_index += 1
            self.cache_index %= self.cache_size
            self.cache[self.cache_index] = data
