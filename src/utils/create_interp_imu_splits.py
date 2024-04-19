import pandas as pd
import numpy as np
import datetime
import os
import logging

from src.utils.utils import interpolate_data, load_data, save_data


def create_interp_imu_splits(
        source_root_dir: os.PathLike,
        dst_root_dir: os.PathLike,
        annotations_dir: os.PathLike,
        filename_split: str
        ):
    '''
    Create the interpolated IMU splits.
    Args:
        • source_root_dir: str. The path of the source directory.
        • dst_root_dir: str. The path of the destination directory.
        • annotations_dir: str. The path of the annotations directory.
        • filename_split: str. The name of the split file.
        • train: bool. True if the split is for training, False otherwise.
    '''

    assert os.path.exists(source_root_dir), 'The directory does not exist'
    assert os.path.isfile(os.path.join(annotations_dir, filename_split)), f'The file {filename_split} does not exist'

    if not os.path.exists(dst_root_dir):
        os.makedirs(dst_root_dir)
        logging.debug(f'Directory created: {dst_root_dir=}')
    
    data = pd.read_pickle(os.path.join(annotations_dir, filename_split))
    data = data.dropna()
    data = data.reset_index(drop=True)

    video_list = data['video_id'].unique().tolist()

    for video_id in video_list:
        pid = video_id.split('_')[0]

        subdir = os.path.join(pid, 'meta_data')

        source_root_dir_complete = os.path.join(source_root_dir, subdir)
        dst_dir_complete = os.path.join(dst_root_dir, subdir)

        if os.path.isfile(os.path.join(dst_dir_complete, video_id + '-accl.csv')) :
            logging.debug(f'The file {video_id}-accl.csv already exists')
            continue
        if os.path.isfile(os.path.join(dst_dir_complete, video_id + '-gyro.csv')):
            logging.debug(f'The file {video_id}-gyro.csv already exists')
            continue
        
        accl, gyro = load_data(os.path.join(source_root_dir, subdir))

        accl_interp = interpolate_data(accl, is_accl=True)
        save_data(
            accl_interp,
            dst_dir=dst_dir_complete,
            video_id=video_id,
            is_accl=True)

        gyro_interp = interpolate_data(gyro, is_accl=False)
        save_data(
            data=gyro_interp,
            dst_dir=dst_dir_complete,
            video_id=video_id,
            is_accl=False)
