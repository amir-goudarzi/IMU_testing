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

def prova():

    # Dummy input parameters
    frequency = 198  # Frequency in Hz
    sample_rate = 44100
    num_samples = 10000
    num_channels = 3

    # Create time vector
    t = torch.arange(0, num_samples) / sample_rate

    # Create sinusoidal waveform for each channel
    waveform = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0).repeat(num_channels, 1)

    # Save the waveform as an audio file
    torchaudio.save("output.wav", waveform, sample_rate)



if __name__ == '__main__':

    #prova()
    root_dir = os.path.join('/data', 'EPIC-KITCHENS')
    annotations_dir = os.path.join('data', 'annotations')
    train = True
    filename = 'EPIC_100_train_clean.pkl'

    # Uncomment the following line to create the clean split
    # clean_split(root_dir, annotations_dir, filename, train=train)
    dataset = EpicKitchens100(
        root_dir=root_dir,
        annotations_dir=annotations_dir,
        seconds=1,
        filename=filename,
        train=train
    )

    train_loader = DataLoader(dataset, batch_size=1, shuffle=train)

    for i, (video, label) in enumerate(train_loader):
        prova_video = video
        print(video.shape)
        print(label)
        
