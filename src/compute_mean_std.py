import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import os
import numpy as np
import argparse

# from data.epic_dataset_ssl import EpicDatasetSSL
# from data.epic_dataset import EpicDataset
from data.dataset import make_dataset
# from data.wear_dataset_ssl import WearDatasetSSL
# from data.wear_dataset import WearDataset

from utils.os_utils import load_config


# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"
print(DEVICE)


'''
Mean and std for default values of class EpicDatasetSSL - training split:
accl_mean=[-24.0869, -28.0400, -27.4174]
accl_std=[17.0260, 14.2892, 15.4472]


gyro_mean=[-42.8106, -42.6817, -43.3577]
gyro_std=tensor[13.2689, 12.8669, 11.9387]

Mean and std for default values of class EpicDataset - training split:
accl_mean=[-16.9617, -23.7787, -21.7584]
accl_std=[16.9230, 15.2566, 16.0062]

gyro_mean=[-39.3884, -39.3591, -40.8935]
gyro_std=[15.7324, 15.0534, 14.3891]
'''

'''
Mean and std for default values of class WearDatasetSSL - training split:
mean=[-25.1615, -23.0049, -24.6689, -26.6896, -24.6670, -26.2407, -26.5644,
        -23.9310, -26.9497, -26.7723, -24.0176, -27.1356]
std=[15.8498, 14.6364, 14.0719, 19.2485, 18.4719, 17.9037, 19.2159, 18.5185,
        16.9717, 19.1566, 18.6452, 17.0249]

Mean and std for default values of class WearDataset - training split:
mean = [-24.1932, -21.6217, -23.2775, -25.6118, -23.3322, -24.7963, -23.0271,
        -20.5719, -23.9399, -23.1688, -20.6322, -24.1154]
std = [15.9730, 14.1860, 13.3603, 19.5357, 18.3608, 17.5932, 17.9725, 17.0493,
        15.6102, 17.7609, 16.9667, 15.5691]

Mean and std for default values of class EgoExo4D - training split:
accl_mean=tensor([-1.3760, -1.4863,  2.0684], dtype=torch.float16), accl_std=tensor([2.2266, 2.2715, 3.0801], dtype=torch.float16)
'''

def main(args, matrix_type='64x64', seconds=2):
    config = load_config(args.config)
    spectrogram_cfg = config['spectrogram_params'][f'sec_{seconds}'][matrix_type]
    # data = EpicDatasetSSL(
    #     src_dir=root_dir,
    #     annotations=annotations_dir,
    #     filename=filename_training,
    #     transforms_accl=transforms.Normalize(mean=[-24.0869, -28.0400, -27.4174], std=[17.0260, 14.2892, 15.4472]),
    #     transforms_gyro=transforms.Normalize(mean=[-42.8106, -42.6817, -43.3577], std=[13.2689, 12.8669, 11.9387]),
    #     )

    # data = EpicDataset(
    #     src_dir=root_dir,
    #     annotations=annotations_dir,
    #     filename=filename_training,
    #     transforms_accl=transforms.Normalize(mean=[-16.9617, -23.7787, -21.7584], std=[16.9230, 15.2566, 16.0062]),
    #     transforms_gyro=transforms.Normalize(mean=[-39.3884, -39.3591, -40.8935], std=[15.7324, 15.0534, 14.3891]),
    #     )

    # data = WearDatasetSSL(
    #     src_dir=root_dir,
    #     annotations=annotations_dir,
    #     filename=filename_training,
    #     transforms=transforms.Normalize(
    #         mean=[-25.1615, -23.0049, -24.6689, -26.6896, -24.6670, -26.2407, -26.5644,
    #                 -23.9310, -26.9497, -26.7723, -24.0176, -27.1356],
    #         std=[15.8498, 14.6364, 14.0719, 19.2485, 18.4719, 17.9037, 19.2159, 18.5185,
    #                 16.9717, 19.1566, 18.6452, 17.0249]),
    #     )

    # data = WearDataset(
    #     src_dir=config['dataset']['root_dir'],
    #     annotations=config['dataset']['annotations_dir'],
    #     filename=config['dataset']['filename_training'],
    #     window_size=spectrogram_cfg['window_size'],
    #     overlap_in_s=spectrogram_cfg['overlap_in_s'],
    #     n_fft=spectrogram_cfg['n_fft'],
    #     hop_length=spectrogram_cfg['hop_length'],
    #     sampling_rate=config['dataset']['sampling_rate'],
    #     downsampling_rate=spectrogram_cfg['downsampling_rate'],
    #     resizes=spectrogram_cfg['resizes']
    # )

    if args.dataset == 'egoexo4d':
        accl_stats = {
            'psum': torch.zeros(3),
            'psum_sq': torch.zeros(3),
            'count': 0
        }
    elif args.dataset == 'wear':
        accl_stats = {
            'psum': torch.zeros(12),
            'psum_sq': torch.zeros(12),
            'count': 0
        }

    cfg = load_config(args.config)
    # config['device'] = rank
    if args.dataset == 'wear':
        cfg['dataset']['filename'] = args.split_file

    dataset = make_dataset(
        name=args.dataset,
        is_pretrain=False,
        task_name = cfg['task_name'],
        **cfg['dataset'],
        **cfg['spectrogram_params'][f'sec_{seconds}'][matrix_type]
    )
    # Create a data loader
    batch_size = 512
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=20)
    dataset_len = len(dataset)
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    accl_tot = None
    # for i, data in enumerate(dataloader):
    for i, (data, labels)  in enumerate(dataloader):
        print(f'{i}/{dataset_len//batch_size}')
        accl = data.to(DEVICE)
        # if accl_tot is None:
        #     accl_tot = accl.type(torch.float16)
        # else:
        #     accl_tot = torch.cat((accl_tot, accl.type(torch.float16)), axis=0)

    # for i, (data, target) in enumerate(dataloader):
        # accl, gyro = data[:, :3], data[:, 3:]
        # accl = accl.to(DEVICE)
        # gyro = gyro.to(DEVICE)
        accl_stats['psum'] += accl.sum(axis = [0, 2, 3])
        accl_stats['psum_sq'] += (accl ** 2).sum(axis = [0, 2, 3])
        accl_stats['count'] += accl.shape[0]


    

    # accl_stats['psum'].to('cpu')
    # accl_stats['psum_sq'].to('cpu')
    # accl_stats['count'].to('cpu')
    # gyro_stats['psum'].to('cpu')
    # gyro_stats['psum_sq'].to('cpu')
    # gyro_stats['count'].to('cpu')

    div = accl_stats['count'] * accl.shape[2] * accl.shape[3]
    
    accl_mean = accl_stats['psum'] / div

    accl_std  = torch.sqrt(accl_stats['psum_sq'] / div - accl_mean ** 2)

    # div = gyro_stats['count'] * 64 * 64
    # gyro_mean = gyro_stats['psum'] / div
    # gyro_std  = torch.sqrt(gyro_stats['psum_sq'] / div - gyro_mean ** 2)
    
    # accl_mean = accl_tot.mean(dim=(0, 2, 3))
    # accl_std = accl_tot.std(dim=(0, 2, 3))
    # accl_mean = accl_tot.mean(dim=(0, 2))
    # accl_std = accl_tot.std(dim=(0, 2))
    print(f'{accl_mean=}, {accl_std=}')

    # save_path = config['mean_std']['save_path']
    split = args.split_file.split('_')[-1].split('.')[0]
    save_path = os.path.join(args.save_path, f'split_{split}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.dataset == 'egoexo4d':
        torch.save(accl_mean, os.path.join(args.save_path, f'accl_mean_left.pt'))
        torch.save(accl_std, os.path.join(args.save_path, f'accl_std_left.pt'))
    elif args.dataset == 'wear':
        torch.save(accl_mean, os.path.join(save_path, f'accl_mean.pt'))
        torch.save(accl_std, os.path.join(args.save_path, f'split_{split}', f'accl_std.pt'))
    # print(f'{gyro_mean=}, {gyro_std=}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute mean and std for the dataset')
    parser.add_argument('--config', default='./configs/IMU-MAE/wear_inertial_ft.yaml')
    parser.add_argument('--dataset', default='wear')
    parser.add_argument('--split_file', default='wear_split_1.pkl')
    parser.add_argument('--save_path', default='./data/WEAR/mean_std')
    args = parser.parse_args()
    matrix_type = '128x320'
    seconds = 2
    main(args, matrix_type=matrix_type, seconds=seconds)