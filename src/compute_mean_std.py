import torch
import torchvision.transforms as transforms
from data.epic_dataset_ssl import EpicDatasetSSL
from data.epic_dataset import EpicDataset
from torch.utils.data import DataLoader
import os


root_dir = os.path.join('/data', 'EPIC-KITCHENS')
annotations_dir = os.path.join('data', 'annotations')
train = True
filename_training = 'EPIC_100_train_clean_split.pkl'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


if __name__ == '__main__':
    # data = EpicDatasetSSL(
    #     src_dir=root_dir,
    #     annotations=annotations_dir,
    #     filename=filename_training,
    #     transforms_accl=transforms.Normalize(mean=[-24.0869, -28.0400, -27.4174], std=[17.0260, 14.2892, 15.4472]),
    #     transforms_gyro=transforms.Normalize(mean=[-42.8106, -42.6817, -43.3577], std=[13.2689, 12.8669, 11.9387]),
    #     )
    data = EpicDataset(
        src_dir=root_dir,
        annotations=annotations_dir,
        filename=filename_training,
        transforms_accl=transforms.Normalize(mean=[-16.9617, -23.7787, -21.7584], std=[16.9230, 15.2566, 16.0062]),
        transforms_gyro=transforms.Normalize(mean=[-39.3884, -39.3591, -40.8935], std=[15.7324, 15.0534, 14.3891]),
        )

    # Create a data loader
    dataloader = DataLoader(data, batch_size=32, shuffle=train)

    accl_stats = {
        'psum': torch.zeros(3).to(DEVICE),
        'psum_sq': torch.zeros(3).to(DEVICE),
        'count': torch.tensor(0).to(DEVICE),
    }

    gyro_stats = {
        'psum': torch.zeros(3).to(DEVICE),
        'psum_sq': torch.zeros(3).to(DEVICE),
        'count': torch.tensor(0).to(DEVICE),
    }

    # for i, data in enumerate(dataloader):
    for i, (data, target) in enumerate(dataloader):
        accl, gyro = data[:, :3], data[:, 3:]
        accl = accl.to(DEVICE)
        gyro = gyro.to(DEVICE)
        accl_stats['psum'] += accl.sum(dim = (0, 2, 3))
        accl_stats['psum_sq'] += (accl ** 2).sum(dim = (0, 2, 3))
        accl_stats['count'] += accl.shape[0]

        gyro_stats['psum'] += gyro.sum(dim = (0, 2, 3))
        gyro_stats['psum_sq'] += (gyro ** 2).sum(dim = (0, 2, 3))
        gyro_stats['count'] += gyro.shape[0]

    

    accl_stats['psum'].to('cpu')
    accl_stats['psum_sq'].to('cpu')
    accl_stats['count'].to('cpu')
    gyro_stats['psum'].to('cpu')
    gyro_stats['psum_sq'].to('cpu')
    gyro_stats['count'].to('cpu')

    div = accl_stats['count'] * 64 * 64
    
    accl_mean = accl_stats['psum'] / div
    accl_std  = torch.sqrt(accl_stats['psum_sq'] / div - accl_mean ** 2)

    div = gyro_stats['count'] * 64 * 64
    gyro_mean = gyro_stats['psum'] / div
    gyro_std  = torch.sqrt(gyro_stats['psum_sq'] / div - gyro_mean ** 2)
    
    print(f'{accl_mean=}, {accl_std=}')
    print(f'{gyro_mean=}, {gyro_std=}')
