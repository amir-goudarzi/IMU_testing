# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import pickle as pkl
import pandas as pd

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models.layers import to_2tuple

import submodules.AudioMAE.util.misc as misc
from submodules.AudioMAE.util.datasets import build_dataset
from submodules.AudioMAE.util.pos_embed import interpolate_pos_embed, interpolate_pos_embed_audio, interpolate_patch_embed_audio, interpolate_pos_embed_img2audio
from submodules.AudioMAE.util.misc import NativeScalerWithGradNormCount as NativeScaler
import submodules.AudioMAE.util.lr_decay as lrd

import submodules.AudioMAE.models_vit as models_vit

from submodules.AudioMAE.engine_finetune_as import train_one_epoch, evaluate #, train_one_epoch_av, evaluate_av
from submodules.AudioMAE.dataset import AudiosetDataset, DistributedWeightedSampler, DistributedSamplerWrapper
from timm.models.vision_transformer import PatchEmbed

from torch.utils.data import WeightedRandomSampler
from data.epic_dataset import EpicDataset


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=2e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=4, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.5,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=527, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # For audioset
    parser.add_argument('--audio_exp', action='store_true', help='audio exp')
    parser.add_argument("--data_train", type=str, default='/checkpoint/berniehuang/ast/egs/audioset/data/datafiles/train_video.json', help="training data json")
    parser.add_argument("--data_eval", type=str, default='/checkpoint/berniehuang/ast/egs/audioset/data/datafiles/eval_video.json', help="validation data json")
    parser.add_argument("--label_csv", type=str, default='/checkpoint/berniehuang/ast/egs/audioset/data/class_labels_indices.csv', help="csv with class labels")
    parser.add_argument("--weight_csv", type=str, default='/checkpoint/berniehuang/mae/data/audioset/weight_train_all.csv', help="weight file")
    
    parser.add_argument('--freqm', help='frequency mask max length', type=int, default=192)
    parser.add_argument('--timem', help='time mask max length', type=int, default=48)
    #parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
    parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used", choices=["audioset", "esc50", "speechcommands", "k400"])
    parser.add_argument("--use_fbank", type=bool, default=False)
    parser.add_argument("--use_soft", type=bool, default=False)
    parser.add_argument("--fbank_dir", type=str, default="/checkpoint/berniehuang/ast/egs/esc50/data/ESC-50-master/fbank", help="fbank dir") 
    parser.set_defaults(audio_exp=False)
    #parser.add_argument("--distributed", type=bool, default=True)
    parser.add_argument('--first_eval_ep', default=0, type=int, help='do eval after first_eval_ep')
    parser.add_argument('--use_custom_patch', type=bool, default=False, help='use custom patch with overlapping and override timm PatchEmbed')
    parser.add_argument('--source_custom_patch', type=bool, default=False, help='the pre-trained model already use custom patch')
    parser.add_argument('--roll_mag_aug', type=bool, default=False, help='use roll_mag_aug')
    parser.add_argument('--mask_t_prob', default=0.0, type=float, help='T masking ratio (percentage of removed patches).') #  
    parser.add_argument('--mask_f_prob', default=0.0, type=float, help='F masking ratio (percentage of removed patches).') #  
    #parser.add_argument('--split_pos', type=bool, default=False, help='use splitted pos emb')
    parser.add_argument('--weight_sampler', type=bool, default=False, help='use weight_sampler')
    parser.add_argument('--epoch_len', default=200000, type=int, help='num of samples/epoch with weight_sampler')
    parser.add_argument('--distributed_wrapper', type=bool, default=False, help='use distributedwrapper for weighted sampler')
    parser.add_argument('--replacement', type=bool, default=False, help='use weight_sampler')
    parser.add_argument('--mask_2d', type=bool, default=True, help='use 2d masking')
    parser.add_argument('--load_video', type=bool, default=False, help='load video')
    parser.add_argument('--av_fusion', type=bool, default=False, help='load video')
    parser.add_argument('--n_frm', default=6, type=int, help='num of frames for video')
    parser.add_argument('--replace_with_mae', type=bool, default=False, help='replace_with_mae')
    parser.add_argument('--load_imgnet_pt', type=bool, default=False, help='when img_pt_ckpt, if load_imgnet_pt, use img_pt_ckpt to initialize audio branch, if not, keep audio branch random')
    return parser


class PatchEmbed_new(nn.Module):
    """ Flexible Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride) # with overlapped patches
        #self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        #self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        #self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        _, _, h, w = self.get_output_shape(img_size) # n, emb_dim, h, w
        self.patch_hw = (h, w)
        self.num_patches = h*w

    def get_output_shape(self, img_size):
        # todo: don't be lazy..
        return self.proj(torch.randn(1, self.in_chans, img_size[0], img_size[1])).shape 

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        #assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


def extract_embeddings(model, dataloader, device):
    model.eval()
    feature_embeddings = pd.DataFrame(columns=['feature_embeddings', 'target'])

    with torch.no_grad():
        for i, (input, target) in enumerate(dataloader):
            input = input.to(device)

            # output = model(input).last_hidden_state[:, 0].cpu()
            output = model(input, classification=False)
            for idx, out in enumerate(output.cpu().numpy()):
                feature_embeddings.loc[len(feature_embeddings)] = [out, target[idx].cpu().numpy()]
            # if target not in feature_embeddings:
            #     feature_embeddings[target] = [output]
            # else:
            #     feature_embeddings[target].append(output)
    embeddings_type = 'global_pool' if model.global_pool else 'cls_token'
    save_path = os.path.join('data','EPIC-KITCHENS', 'IMU', 'audio_mae', embeddings_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # pkl.dump(feature_embeddings, open(os.path.join(save_path, 'feature_embeddings.pkl'), 'wb'))
    feature_embeddings.to_pickle(os.path.join(save_path, 'feature_embeddings.pkl'))


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    root_dir = os.path.join('/data', 'EPIC-KITCHENS')
    annotations_dir = os.path.join('data', 'annotations')
    train = True
    filename_training = 'EPIC_100_train_clean.pkl'

    dataset_train = EpicDataset(
        src_dir=root_dir,
        annotations=annotations_dir,
        filename=filename_training
    )
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    model_dict = {attr: getattr(models_vit, attr) for attr in dir(models_vit)}

    model = model_dict[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
        mask_2d=args.mask_2d,
        use_custom_patch=args.use_custom_patch,
        ## remove video part for A-MAE
        #load_video=args.load_video,
        # n_frm=args.n_frm,
        #split_pos=args.split_pos,
        #av_fusion=args.av_fusion,
    )
    if args.audio_exp:
        img_size=(1024,128) # 1024, 128
        in_chans=1
        emb_dim = 768
        if args.model == "vit_small_patch16":
            emb_dim = 384
        if args.use_custom_patch:
            model.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=16, in_chans=1, embed_dim=emb_dim, stride=10)
            model.pos_embed = nn.Parameter(torch.zeros(1, 1212 + 1, emb_dim), requires_grad=False)  # fixed sin-cos embedding
        else:
            model.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=(16,16), in_chans=1, embed_dim=emb_dim, stride=16) # no overlap. stride=img_size=16
            num_patches = model.patch_embed.num_patches
            #num_patches = 512 # assume audioset, 1024//16=64, 128//16=8, 512=64x8
            model.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim), requires_grad=False)  # fixed sin-cos embedding
    else:
        # audio_exp = False for EPIC-KITCHENS
        img_size = (64,64)
        in_chans = 6
        emb_dim = 768
        if args.model == "vit_small_patch16":
            emb_dim = 384
        if args.use_custom_patch:
            model.patch_embed = PatchEmbed_new(img_size=img_size,
                                                  patch_size=16,
                                                  in_chans=in_chans,
                                                  embed_dim=emb_dim,
                                                  stride=10)
            model.pos_embed = nn.Parameter(torch.zeros(1, 1212 + 1, emb_dim), requires_grad=False)
        else:
            model.patch_embed = PatchEmbed_new(img_size=img_size,
                                                  patch_size=(16,16),
                                                  in_chans=in_chans,
                                                  embed_dim=emb_dim,
                                                  stride=16)
            num_patches = model.patch_embed.num_patches
            model.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim), requires_grad=False)

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()

        if not args.eval:
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # manually initialize fc layer
        if not args.eval:
            trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    loss_scaler = NativeScaler()

    if args.use_soft:
        criterion = SoftTargetCrossEntropy() 
    else:
        criterion = nn.BCEWithLogitsLoss() # works better
    

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # if args.eval:
    #     test_stats = evaluate(data_loader_val, model, device, args.dist_eval)
    #     with open('aps.txt', 'w') as fp:
    #         aps=test_stats['AP']
    #         aps=[str(ap) for ap in aps]
    #         fp.write('\n'.join(aps))
    #     print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['mAP']:.4f}")
    #     exit(0)

    feature_embeddings = extract_embeddings(model, data_loader_train, device)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
