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

from inertial_baseline.train import run_inertial_network
from utils.torch_utils import fix_random_seed
from utils.os_utils import Logger, load_config
import matplotlib.pyplot as plt
from camera_baseline.actionformer.main import run_actionformer
from camera_baseline.tridet.main import run_tridet
from accelerate import Accelerator, GradScalerKwargs

def main(args):
    if args.neptune:
        run = neptune.init_run(
        project="",
        api_token=""
        )
    else:
        run = None

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

    feats_path = None

    if config['dataset']['feat_folder'].endswith((
        'mae',
        'fromscratch',
        'imu_i3d',
        'rgb'
        )):
        splits = [config['anno_json'][i].split('wear_')[1].split('.')[0] for i in range(len(config['anno_json']))]
        feats_path = [
            os.path.join(config['dataset']['feat_folder'], splits[i], 'mask_ratio{' + str(args.mask_ratio) + '}_' + 'mae_vit_base_patch16') 
            for i in range(len(splits))]


    for i, anno_split in enumerate(config['anno_json']):
        with open(anno_split) as f:
            file = json.load(f)
        anno_file = file['database']
        config['labels'] = ['null'] + list(file['label_dict'])
        config['label_dict'] = dict(zip(config['labels'], list(range(len(config['labels'])))))
        train_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Training']
        val_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Validation']

        # Adaptation for MAE pre-extracted features.
        if feats_path is not None:
            config['dataset']['feat_folder'] = feats_path[i]

        print('Split {} / {}'.format(i + 1, len(config['anno_json'])))
        if args.eval_type == 'split':
            name = 'split_' + str(i)
        elif args.eval_type == 'loso':
            name = 'sbj_' + str(i)
        config['dataset']['json_anno'] = anno_split
        if config['name'] == 'tadtr':
            config['dataset']['json_info'] = config['info_json'][i]

        seed = config['init_rand_seed']
        kwargs = GradScalerKwargs()
        run = Accelerator(mixed_precision="bf16", kwargs_handlers=[kwargs], log_with="wandb")
        run.init_trackers(f"{args.run_id}", config=config, init_kwargs={"wandb":{"name":f"{name}", "tags":[f'{seed=}']}})

        if config['name'] == 'deepconvlstm' or config['name'] == 'attendanddiscriminate':
            t_losses, v_losses, v_mAP, v_preds, v_gt = run_inertial_network(train_sbjs, val_sbjs, config, log_dir, args.ckpt_freq, args.resume, rng_generator, run)
        elif config['name'] == 'actionformer' or config['name'] == 'actionformer_vit_pretrained':
            t_losses, v_losses, v_mAP, v_preds, v_gt = run_actionformer(val_sbjs, config, log_dir, args.ckpt_freq, args.resume, rng_generator, run, args, split=i)
        elif config['name'] == 'tridet' or config['name'] == 'actionformer_vit_pretrained':
            t_losses, v_losses, v_mAP, v_preds, v_gt = run_tridet(val_sbjs, config, log_dir, args.ckpt_freq, args.resume, rng_generator, run, args, split=i)
        

        # t_losses, v_losses, v_mAP, v_preds, v_gt = run.gather_for_metrics(t_losses, v_losses, v_mAP, v_preds, v_gt)
        # raw results
        conf_mat = confusion_matrix(v_gt, v_preds, normalize='true', labels=range(len(config['labels'])))
        v_acc = conf_mat.diagonal()/conf_mat.sum(axis=1)
        v_prec = precision_score(v_gt, v_preds, average=None, zero_division=1, labels=range(len(config['labels'])))
        v_rec = recall_score(v_gt, v_preds, average=None, zero_division=1, labels=range(len(config['labels'])))
        v_f1 = f1_score(v_gt, v_preds, average=None, zero_division=1, labels=range(len(config['labels'])))

        # print to terminal
        if args.eval_type == 'split':
            block1 = '\nFINAL RESULTS SPLIT {}'.format(i + 1)
        elif args.eval_type == 'loso':
            block1 = '\nFINAL RESULTS SUBJECT {}'.format(i)
        block2 = 'TRAINING:\tavg. loss {:.2f}'.format(np.nanmean(t_losses))
        block3 = 'VALIDATION:\tavg. loss {:.2f}'.format(np.nanmean(v_losses))
        block4 = ''
        block4  += '\n\t\tAvg. mAP {:>4.2f} (%) '.format(np.nanmean(v_mAP) * 100)
        for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], v_mAP):
            block4 += 'mAP@' + str(tiou) +  ' {:>4.2f} (%) '.format(tiou_mAP*100)
        block5 = ''
        block5  += '\t\tAcc {:>4.2f} (%)'.format(np.nanmean(v_acc) * 100)
        block5  += ' Prec {:>4.2f} (%)'.format(np.nanmean(v_prec) * 100)
        block5  += ' Rec {:>4.2f} (%)'.format(np.nanmean(v_rec) * 100)
        block5  += ' F1 {:>4.2f} (%)\n'.format(np.nanmean(v_f1) * 100)

        print('\n'.join([block1, block2, block3, block4, block5]))
                                
        all_v_mAP = np.append(all_v_mAP, v_mAP[None, :], axis=0)
        all_v_gt = np.append(all_v_gt, v_gt)
        all_v_pred = np.append(all_v_pred, v_preds)

        # save raw confusion matrix
        _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
        conf_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=config['labels'])
        conf_disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
        ax.set_title('Confusion Matrix ' + name + ' (raw)')
        plt.savefig(os.path.join(log_dir, name + '_raw.png'))
        plt.close()
        run.wait_for_everyone()
        if run is not None:
            # run['conf_matrices'].append(_, name=name + '_raw')
            run.log({'conf_matrix': wandb.Image(os.path.join(log_dir, name + '_raw.png') )})
        run.end_training()
        run.wait_for_everyone()
        run = None

    kwargs = GradScalerKwargs()
    run = Accelerator(kwargs_handlers=[kwargs], log_with="wandb")
    run.init_trackers(f"{args.run_id}", config=config, init_kwargs={"wandb":{"name":f"Average Statistics", "tags":[f'{seed=}']}})
    # final raw results across all splits
    conf_mat = confusion_matrix(all_v_gt, all_v_pred, normalize='true', labels=range(len(config['labels'])))
    v_acc = conf_mat.diagonal()/conf_mat.sum(axis=1)
    v_prec = precision_score(all_v_gt, all_v_pred, average=None, zero_division=1, labels=range(len(config['labels'])))
    v_rec = recall_score(all_v_gt, all_v_pred, average=None, zero_division=1, labels=range(len(config['labels'])))
    v_f1 = f1_score(all_v_gt, all_v_pred, average=None, zero_division=1, labels=range(len(config['labels'])))

    # print final results to terminal
    block1 = '\nFINAL AVERAGED RESULTS:'
    block2 = ''
    block2  += '\n\t\tAvg. mAP {:>4.2f} (%) '.format(np.nanmean(all_v_mAP) * 100)
    for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], all_v_mAP.T):
        block2 += 'mAP@' + str(tiou) +  ' {:>4.2f} (%) '.format(np.nanmean(tiou_mAP)*100)
    block2  += '\n\t\tAcc {:>4.2f} (%)'.format(np.nanmean(v_acc) * 100)
    block2  += ' Prec {:>4.2f} (%)'.format(np.nanmean(v_prec) * 100)
    block2  += ' Rec {:>4.2f} (%)'.format(np.nanmean(v_rec) * 100)
    block2  += ' F1 {:>4.2f} (%)'.format(np.nanmean(v_f1) * 100)
    
    print('\n'.join([block1, block2]))

    # save final raw confusion matrix
    _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
    ax.set_title('Confusion Matrix Total (raw)')
    conf_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=config['labels']) 
    conf_disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
    plt.savefig(os.path.join(log_dir, 'all_raw.png'))
    plt.close()
    if run is not None:
        run.log({'conf_matrix': wandb.Image(os.path.join(log_dir, 'all_raw.png'))})
    run.end_training()

    # submit final values to neptune 
    if run is not None:
        df = pd.DataFrame(columns=['metric', 'value'])
        data = {
            'final_avg_mAP': np.nanmean(all_v_mAP),
            'final_accuracy': np.nanmean(v_acc),
            'final_precision': np.nanmean(v_prec),
            'final_recall': np.nanmean(v_rec),
            'final_f1': np.nanmean(v_f1)
            }
        for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], all_v_mAP.T):
            data['final_mAP@' + str(tiou)] = np.nanmean(tiou_mAP)
        run.log({'final_results': wandb.Table(data=[[key, value] for key, value in data.items()], columns=['metric', 'value'])})
    run.wait_for_everyone()
    print("ALL FINISHED")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/60_frames_30_stride/tridet_combined.yaml')
    parser.add_argument('--config_mae', default=None)
    parser.add_argument('--eval_type', default='split')
    parser.add_argument('--neptune', default=False, type=bool)
    parser.add_argument('--run_id', default='test', type=str)
    parser.add_argument('--seed', default=42, type=int)       
    parser.add_argument('--ckpt-freq', default=-1, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--gpu', default='cuda:0', type=str)

    # For AudioMAE compatibility 
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--audio_exp', action='store_true', help='audio exp')
    parser.add_argument('--seconds', type=int, default=2)
    parser.add_argument('--matrix_type', type=str, default='64x64')
    parser.add_argument('--use_custom_patch', type=bool, default=False, help='use custom patch with overlapping and override timm PatchEmbed')
    parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
    parser.add_argument('--mask_t_prob', default=0.0, type=float, help='T masking ratio (percentage of removed patches).') #  
    parser.add_argument('--mask_f_prob', default=0.0, type=float, help='F masking ratio (percentage of removed patches).') #  
    parser.add_argument('--mask_ratio', default=0.9, type=float, help='rand masking ratio (percentage of removed patches).') #  
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size.')
    parser.set_defaults(audio_exp=True)
    args = parser.parse_args()
    main(args)  

