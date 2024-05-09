import pandas as pd
import numpy as np
import datetime
import os
import logging

from src.utils.create_interp_imu_splits import create_interp_imu_splits

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    source_root_dir = os.path.join('/data', 'EPIC-KITCHENS')
    dst_root_dir = os.path.join('/data2', 'EPIC-KITCHENS')
    annotations_dir = os.path.join('..', 'data', 'annotations')
    filename = 'EPIC_100_train_clean.pkl'

    create_interp_imu_splits(source_root_dir, dst_root_dir, annotations_dir, filename)

    filename = 'EPIC_100_validation_clean.pkl'
    create_interp_imu_splits(source_root_dir, dst_root_dir, annotations_dir, filename)