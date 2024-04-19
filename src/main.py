import os
from torch.utils.data import DataLoader
import torchaudio
import torch
from data.epic_loader import EpicKitchens100
from utils.clean_splits import clean_split

PREPROCESSING_ARGS = {
    'downsampling_rate': 16000,
    'frame_length_in_s': 0.016,
    'frame_step_in_s': 0.012,
    'num_mel_bins': 40,
    'lower_frequency': 20,
    'upper_frequency': 8000,
    'num_coefficients': 30
}
TRAINING_ARGS = {
    'batch_size': 20,
    'initial_learning_rate': 0.01,
    'end_learning_rate': 0,
    'epochs': 20
}


if __name__ == '__main__':

    #prova()
    root_dir = os.path.join('/data', 'EPIC-KITCHENS')
    annotations_dir = os.path.join('data', 'annotations')
    train = False
    clean_split(root_dir, annotations_dir, train)
        
