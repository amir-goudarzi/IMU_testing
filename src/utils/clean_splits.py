'''
File originally created to generate new splits when some files are missing from the original one.
'''

import pandas as pd
import os

def clean_split(root_dir, annotations_dir, train=True):
    split = 'EPIC_100_train.pkl' if train else f'EPIC_100_validation.pkl'
    assert os.path.exists(root_dir), 'The directory does not exist'
    assert os.path.isfile(os.path.join(annotations_dir, split)), 'The file epic_kitchens_100_train.csv does not exist'

    data = pd.read_pickle(os.path.join(annotations_dir, split))
    data = data.dropna()
    data = data.reset_index(drop=True)
    new_data = pd.DataFrame(columns=data.columns.tolist())

    for idx, row in data.iterrows():
        participant_id = row['participant_id']
        video_id = row['video_id']
        clip_path = os.path.join(root_dir, participant_id, 'meta_data', video_id)
        if os.path.isfile(clip_path + '-accl.csv') and os.path.isfile(clip_path + '-gyro.csv'):
            new_data.loc[len(new_data.index)] = row.tolist()
        else:
            continue
    save_split = 'EPIC_100_train_clean.pkl' if train else f'EPIC_100_validation_clean.pkl'
    new_data.to_pickle(os.path.join('data', 'annotations', save_split))
