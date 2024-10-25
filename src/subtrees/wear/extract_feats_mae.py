# ------------------------------------------------------------------------
# Main script to commence baseline experiments on WEAR dataset
# ------------------------------------------------------------------------
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
# ------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import argparse
import datetime
import json
import os
from pprint import pprint
import sys
import time

import pandas as pd
import numpy as np
import neptune
import wandb
from neptune.types import File
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

import torch
from torch.nn import functional as F
from inertial_baseline.train import run_inertial_network
from utils.torch_utils import fix_random_seed
from utils.os_utils import Logger, load_config
from camera_baseline.actionformer.libs.core.config import _update_config
import matplotlib.pyplot as plt
from camera_baseline.actionformer.main import run_actionformer
from camera_baseline.actionformer.libs.datasets import make_dataset, make_data_loader
from camera_baseline.tridet.main import run_tridet
from accelerate import Accelerator, GradScalerKwargs
from safetensors.torch import load_file
from camera_baseline.actionformer.libs.modeling import make_meta_arch

sys.path.append(os.path.join('../../../src'))
sys.path.append(os.path.join('src'))
from models.utils_mae import load_vit3d_model, load_mae_model_2d
from audiomae_pretrain import modeling
from utils.os_utils import load_config
from features.imu_preprocessing import SpectrogramsGenerator

# Disable NCCL P2P and IB
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"


def epoch(args, specgram_transform, model, train_loader, config, accelerator, dummy_model, training):
    seconds_bin = args.seconds * 50 * 4 * 3 # 2 seconds * 50 Hz * 4 sensors * 3 channels
    model.eval()
    map_fn = torch.vmap(specgram_transform, in_dims=0, randomness='same')

        

    # # loop over train set
    # for _, video_list in enumerate(train_loader, 0):
    #     # batched_inputs, batched_masks = preprocessing(
    #     #     video_list=video_list, 
    #     #     max_seq_len=config['dataset']['max_seq_len'], 
    #     #     max_div_factor=dummy_model.max_div_factor,
    #     #     device=accelerator.device,
    #     #     training=training)
    #     '''
    #     video_list = {'video_id'        : video_item['id'],
    #         'feats'           : feats,      # C x T
    #         'segments'        : segments,   # N x 2
    #         'labels'          : labels,     # N
    #         'fps'             : video_item['fps'],
    #         'duration'        : video_item['duration'],
    #         'feat_stride'     : feat_stride,
    #         'feat_num_frames' : self.num_frames}
    #     '''
    #     if len(video_list) == 1:
    #         batched_inputs = [video_list[0]['feats'][:seconds_bin, :].nan_to_num().permute(1,0)]
    #     else:
    #         batched_inputs = [video_list[i]['feats'][:seconds_bin, :].nan_to_num().permute(1,0) for i in range(len(video_list))]
        # b = 1 # Mini-batch
        # B = len(video_list)
    dirname = config['dataset']['feat_folder']
    
    for filename in os.listdir(dirname):
        if os.path.isdir(os.path.join(dirname, filename)):
            continue
        features = np.load(os.path.join(dirname, filename))
        feat_tot = np.zeros((features.shape[0], 2048 + 768))
        feat_tot[:, 768:] = features[:, seconds_bin:]

        b = 1
        batched_inputs = torch.tensor(features[:, :seconds_bin]).nan_to_num().to(torch.float32)
        dummy_i3d = torch.rand((1, 2048))
        # forward the model (wo. grad)
        for i, vid in enumerate(batched_inputs):
            with accelerator.autocast():
                vid = vid.to(accelerator.device)
                feats = None
                vid = vid.unsqueeze(0).reshape(1, 12, 2 * 50)
                vid = map_fn(vid)
                _, c, h, w = vid.shape
                

                # TODO: uncomment for vit3d
                # vid = vid.reshape(b, 3, 4, h, w)

                if args.imu_i3d:
                    vid = (vid, dummy_i3d)
                with torch.no_grad():
                    output, _, _, _ = model(vid, mask_ratio=0.0)
                    output = output.mean(dim = 1)
            
                feat_tot[i, :768] = output.squeeze(0).cpu().numpy()

        if args.fromscratch:
            save_path_mae = os.path.join(config['dataset']['feat_folder'], 'mae', 'fromscratch', 'split_' + str(args.split), args.finetune.split('/')[-2])
        elif args.imu_i3d:
            save_path_mae = os.path.join(config['dataset']['feat_folder'], 'mae', 'imu_i3d', 'split_' + str(args.split), args.finetune.split('/')[-2])
        else:
            save_path_mae = os.path.join(config['dataset']['feat_folder'], 'mae', 'split_' + str(args.split), args.finetune.split('/')[-2])
            
        if not os.path.exists(save_path_mae):
            os.makedirs(save_path_mae)
        
        np.save(os.path.join(save_path_mae, filename), feat_tot)

        # for i, video_item in enumerate(video_list):
        #     save_path_mae = os.path.join(config['dataset']['feat_folder'], 'mae', 'split_' + str(args.split), args.finetune.split('/')[-2])
        #     if not os.path.exists(save_path_mae):
        #         os.makedirs(save_path_mae)
        #     tmp = torch.cat((feat_tot[i], video_item['feats'][seconds_bin:, :].cpu()), dim=0).permute(1,0).numpy()
        #     # tmp = feat_tot[i].numpy()

        #     np.save(os.path.join(save_path_mae, video_item['video_id'] + '.npy'), tmp)

def main(args):
    run = None
    accelerator = Accelerator()
    args.gpu = accelerator.device
    config = load_config(args.config)
    config['init_rand_seed'] = args.seed
    config['devices'] = [args.gpu]

    ts = datetime.datetime.fromtimestamp(int(time.time()))
    log_dir = os.path.join('logs', config['name'], str(ts) + '_' + args.run_id)
    sys.stdout = Logger(os.path.join(log_dir, 'log.txt'))

    # save the current cfg
    with open(os.path.join(log_dir, 'cfg.txt'), 'w') as fid:
        pprint(config, stream=fid)
        fid.flush()

    rng_generator = fix_random_seed(config['init_rand_seed'], include_cuda=True)    

    all_v_pred = np.array([])
    all_v_gt = np.array([])
    all_v_mAP = np.empty((0, len(config['dataset']['tiou_thresholds'])))

    i = args.split - 1
    anno_split = config['anno_json'][i]
    with open(anno_split) as f:
        file = json.load(f)
    anno_file = file['database']
    config['labels'] = ['null'] + list(file['label_dict'])
    config['label_dict'] = dict(zip(config['labels'], list(range(len(config['labels'])))))
    train_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Training']
    val_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Validation']

    print('Split {} / {}'.format(i + 1, len(config['anno_json'])))
    if args.eval_type == 'split':
        name = 'split_' + str(i)
    elif args.eval_type == 'loso':
        name = 'sbj_' + str(i)
    config['dataset']['json_anno'] = anno_split
    if config['name'] == 'tadtr':
        config['dataset']['json_info'] = config['info_json'][i]
    
    config = _update_config(config)

    train_dataset = make_dataset(config['dataset_name'], True, config['train_split'], **config['dataset'])
    val_dataset = make_dataset(config['dataset_name'], False, config['val_split'], **config['dataset'])
    train_loader = make_data_loader(train_dataset, True, rng_generator, **config['loader'])
# set bs = 1, and disable shuffle
    val_loader = make_data_loader(val_dataset, False, None, 1, config['loader']['num_workers'])

    dummy_model = make_meta_arch(args, config['model']['model_name'], **config['model'])

    cfg_mae = load_config(args.config_mae)
    cfg_mae['model']['extract_feats'] = True

    # TODO: uncomment for vit3d
    # model = load_vit3d_model(args.seconds, args.matrix_type, cfg_mae)
    model = modeling(
            seconds=args.seconds,
            matrix_type=args.matrix_type,
            audio_exp=True,
            cfg=cfg_mae
        )
    # model = load_mae_model_2d(
    #     finetune=args.finetune,
    #     eval=False,
    #     model=model
    # )
    checkpoint_model = load_file(os.path.join(args.finetune, 'model.safetensors'))
    print("Load pre-trained checkpoint from: %s" % args.finetune)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    model, train_loader, val_loader = accelerator.prepare(model, train_loader, val_loader)

    mean_std_path = cfg_mae['spectrogram_params'][f'sec_{args.seconds}'][args.matrix_type]['mean_std_path']
    mean, std = torch.load(os.path.join(mean_std_path, 'accl_mean.pt')), torch.load(os.path.join(mean_std_path, 'accl_std.pt'))
    del cfg_mae['spectrogram_params'][f'sec_{args.seconds}'][args.matrix_type]['mean_std_path']

    specgram_transform = SpectrogramsGenerator(**cfg_mae['spectrogram_params'][f'sec_{args.seconds}'][args.matrix_type],
                                                mean=mean, std=std, device=accelerator.device)

    epoch(args, specgram_transform, model, train_loader, config, accelerator, dummy_model, training=True)
    epoch(args, specgram_transform, model, val_loader, config, accelerator, dummy_model, training=False)

def combined_to_inertial(args):
    config = load_config(args.config)

    
    if args.fromscratch:
        combined_path_mae = os.path.join(config['dataset']['feat_folder'], 'mae', 'fromscratch', 'split_' + str(args.split), args.finetune.split('/')[-2])
        save_path = os.path.join('/data2/WEAR/processed/inertial_features', '120_frames_60_stride', 'mae', 'fromscratch', 'split_' + str(args.split), args.finetune.split('/')[-2])
    elif args.imu_i3d:
        combined_path_mae = os.path.join(config['dataset']['feat_folder'], 'mae', 'imu_i3d', 'split_' + str(args.split), args.finetune.split('/')[-2])
        save_path = os.path.join('/data2/WEAR/processed/inertial_features', '120_frames_60_stride', 'mae', 'imu_i3d', 'split_' + str(args.split), args.finetune.split('/')[-2])
    else:
        combined_path_mae = os.path.join(config['dataset']['feat_folder'], 'mae', 'split_' + str(args.split), args.finetune.split('/')[-2])
        save_path = os.path.join('/data2/WEAR/processed/inertial_features', '120_frames_60_stride', 'mae', 'split_' + str(args.split), args.finetune.split('/')[-2])

    for filename in os.listdir(combined_path_mae):
        if os.path.isdir(os.path.join(combined_path_mae, filename)):
            continue
        features = np.load(os.path.join(combined_path_mae, filename))
        features = features[:, :768]

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, filename), features)

def inertial_to_combined(args):
    config = load_config(args.config)
    baseline_path = os.path.join(config['dataset']['feat_folder'], 'mae', 'split_' + str(args.split), args.finetune.split('/')[-2])
    if args.fromscratch:
        inertial_path = os.path.join('/data2/WEAR/processed/inertial_features', '120_frames_60_stride', 'fromscratch', 'split_' + str(args.split), args.finetune.split('/')[-2])
        save_path = os.path.join(config['dataset']['feat_folder'], 'fromscratch', 'split_' + str(args.split), args.finetune.split('/')[-2])
    elif args.imu_i3d:
        inertial_path = os.path.join('/data2/WEAR/processed/inertial_features', '120_frames_60_stride', 'mae', 'imu_i3d', 'split_' + str(args.split), args.finetune.split('/')[-2])
        save_path = os.path.join(config['dataset']['feat_folder'], 'mae', 'imu_i3d', 'split_' + str(args.split), args.finetune.split('/')[-2])
    else:
        inertial_path = os.path.join('/data2/WEAR/processed/inertial_features', '120_frames_60_stride', 'mae', 'split_' + str(args.split), args.finetune.split('/')[-2])
        save_path = os.path.join(config['dataset']['feat_folder'], 'mae', 'split_' + str(args.split), args.finetune.split('/')[-2])
    
    for filename in os.listdir(inertial_path):
        if os.path.isdir(os.path.join(inertial_path, filename)):
            continue
        inertial_features = np.load(os.path.join(inertial_path, filename))
        combined_features = np.load(os.path.join(baseline_path, filename))

        combined_features[:, :768] = inertial_features

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, filename), combined_features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WEAR baseline experiments')
    parser.add_argument('--config', type=str, default='configs/config_wear.json', help='path to config file')
    parser.add_argument('--config_mae', type=str, default='configs/config_mae.json', help='path to config file')
    parser.add_argument('--run_id', type=str, default='0', help='run id')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', help='gpu id')
    parser.add_argument('--eval_type', type=str, default='split', help='split or loso')
    parser.add_argument('--finetune', type=str, default='', help='path to finetune model')
    parser.add_argument('--seconds', type=int, default=2)
    parser.add_argument('--matrix_type', type=str, default='128x320')
    parser.add_argument('--mask_ratio', type=float, default=0.9)
    parser.add_argument('--split', type=int, default=1)
    parser.add_argument('--fromscratch', action='store_true', default=False)
    parser.add_argument('--imu_i3d', action='store_true', default=False)
    parser.add_argument('--combined_to_inertial', action='store_true', default=False)
    parser.add_argument('--inertial_to_combined', action='store_true', default=False)

    args = parser.parse_args()
    if args.combined_to_inertial and not args.inertial_to_combined:
        combined_to_inertial(args)
    elif args.inertial_to_combined and not args.combined_to_inertial:
        inertial_to_combined(args)
    else:
        main(args)