import torch
import torchvision.transforms as transforms
from data.epic_dataset_ssl import EpicDatasetSSL
from data.epic_dataset import EpicDataset
from torch.utils.data import DataLoader
import os
from data.wear_dataset_ssl import WearDatasetSSL
from data.wear_dataset import WearDataset


# root_dir = os.path.join('/data', 'EPIC-KITCHENS')
# annotations_dir = os.path.join('data', 'annotations')
# train = True
# filename_training = 'EPIC_100_train_clean.pkl'
root_dir = os.path.join('/data2', 'WEAR')
annotations_dir = os.path.join('data', 'WEAR', 'annotations')
train = True
filename_training = 'wear_annotations_refactored_train.pkl'
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
'''


if __name__ == '__main__':
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

    data = WearDataset(
        src_dir=root_dir,
        annotations=annotations_dir,
        filename=filename_training
        )


    # Create a data loader
    batch_size = 32
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=train)

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
    dataset_len = len(data.annotations)
    accl_tot = None

    for i, data in enumerate(dataloader):
        accl = data.to(DEVICE)
        if accl_tot is None:
            accl_tot = accl
        else:
            accl_tot = torch.cat((accl_tot, accl), dim=0)
    # for i, (data, target) in enumerate(dataloader):
        # accl, gyro = data[:, :3], data[:, 3:]
        # accl = accl.to(DEVICE)
        # gyro = gyro.to(DEVICE)
        # accl_stats['psum'] += accl.sum(dim = (0, 2, 3))
        # accl_stats['psum_sq'] += (accl ** 2).sum(dim = (0, 2, 3))
        # accl_stats['count'] += accl.shape[0]

        # gyro_stats['psum'] += gyro.sum(dim = (0, 2, 3))
        # gyro_stats['psum_sq'] += (gyro ** 2).sum(dim = (0, 2, 3))
        # gyro_stats['count'] += gyro.shape[0]

    

    # accl_stats['psum'].to('cpu')
    # accl_stats['psum_sq'].to('cpu')
    # accl_stats['count'].to('cpu')
    # gyro_stats['psum'].to('cpu')
    # gyro_stats['psum_sq'].to('cpu')
    # gyro_stats['count'].to('cpu')

    # div = accl_stats['count'] * 64 * 64
    
    # accl_mean = accl_stats['psum'] / div
    # accl_std  = torch.sqrt(accl_stats['psum_sq'] / div - accl_mean ** 2)

    # div = gyro_stats['count'] * 64 * 64
    # gyro_mean = gyro_stats['psum'] / div
    # gyro_std  = torch.sqrt(gyro_stats['psum_sq'] / div - gyro_mean ** 2)
    accl_mean = accl_tot.mean(dim=(0, 2, 3))
    accl_std = accl_tot.std(dim=(0, 2, 3))
    print(f'{accl_mean=}, {accl_std=}')
    # print(f'{gyro_mean=}, {gyro_std=}')