import torch
import torchvision.transforms as transforms
from data.epic_dataset_ssl import EpicDatasetSSL
from torch.utils.data import DataLoader
import os

#TODO: Implement resnet feature extraction
#TODO: Implement the SSL pipeline

root_dir = os.path.join('/data', 'EPIC-KITCHENS')
annotations_dir = os.path.join('data', 'annotations')
train = True
filename_training = 'EPIC_100_train_clean.pkl'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)


'''
Mean and std for default values of class EpicDatasetSSL:
accl_mean=tensor([-24.0039, -28.0563, -27.5905], device='cuda:0')
accl_std=tensor([17.0473, 14.1716, 15.3116], device='cuda:0')


gyro_mean=tensor([-42.7268, -42.6332, -43.2766], device='cuda:0'),
gyro_std=tensor([13.3456, 12.9086, 11.9457], device='cuda:0')
'''


if __name__ == '__main__':
    data = EpicDatasetSSL(
        src_dir=root_dir,
        annotations=annotations_dir,
        filename=filename_training
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

    for i, (accl, gyro) in enumerate(dataloader):
        accl = accl.to(DEVICE)
        gyro = gyro.to(DEVICE)
        accl_stats['psum'] += accl.sum(dim = (0, 2, 3))
        accl_stats['psum_sq'] += (accl ** 2).sum(dim = (0, 2, 3))
        accl_stats['count'] += accl.shape[0]

        gyro_stats['psum'] += gyro.sum(dim = (0, 2, 3))
        gyro_stats['psum_sq'] += (gyro ** 2).sum(dim = (0, 2, 3))
        gyro_stats['count'] += gyro.shape[0]
        print(accl.shape, gyro.shape)

    

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
