import math
from src.data.wear_dataset_ssl import WearDatasetSSL

import os

import torch
import torch.optim as optim
import scipy as sp
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as T

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

if __name__ == '__main__':
    root_dir = os.path.join('/data2', 'WEAR')
    annotations_dir = os.path.join(root_dir, 'annotations')
    inertial_dir = os.path.join(root_dir, 'raw', 'inertial')
    annotations_file = 'wear_split_1.json'

    dataset = WearDatasetSSL(
        src_dir=inertial_dir,
        annotations=os.path.join(annotations_dir, annotations_file),
        window_size=10,
        overlap_in_s=5,
        n_fft=128,
        hop_length=4,
        sampling_rate=50,
        downsampling_rate=25,
        use_cache=False,
        transforms=T.Normalize(
            mean=[-25.5149, -23.2097, -24.7798, -27.4930, -25.2731, -26.7750, -27.3466,
                -24.6720, -27.6342, -27.5520, -24.7439, -27.8204],
            std=[15.8735, 14.6558, 14.1539, 19.9334, 19.3062, 18.7667, 19.3694, 18.7194,
                17.1223, 19.3155, 18.7935, 17.1694]
        )
    )
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    print(len(dataloader))


    # psum    = torch.zeros(12).to(DEVICE)
    # psum_sq = torch.zeros(12).to(DEVICE)
    # count = torch.tensor(0).to(DEVICE)

    for i, data in enumerate(dataloader):
        if data.shape[0] == 0:
            continue
        data = data.to(DEVICE)
        # psum    += data.sum(dim = (0, 2, 3))
        # psum_sq += (data ** 2).sum(dim = (0, 2, 3))
        # count   += data.shape[0]
        print(data.shape)
    
    # psum.to('cpu')
    # psum_sq.to('cpu')
    # count.to('cpu')
    # div = count * 64 * 64
    # mean = psum / div
    # std  = torch.sqrt(psum_sq / div - mean ** 2)
    # print(mean, std)
