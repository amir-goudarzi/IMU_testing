import accelerate.optimizer
from audiomae_pretrain import get_args_parser, modeling

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, barrier
import os
import sys
import json

from copy import deepcopy
from accelerate import Accelerator, GradScalerKwargs

import timm.optim.optim_factory as optim_factory

from utils.os_utils import load_config

sys.path.append('.')
sys.path.append('..')
sys.path.append(os.path.join('submodules', 'AudioMAE'))
from subtrees.AudioMAE.engine_pretrain import train_one_epoch
from subtrees.AudioMAE.util.misc import NativeScalerWithGradNormCount as NativeScaler
from subtrees.AudioMAE.util.misc import AcceleratorScalerWithGradNormCount as AcceleratorScaler
import subtrees.AudioMAE.util.misc as misc
from torch.utils.tensorboard import SummaryWriter

from data.dataset import make_dataset


def main(args):
    cfg = load_config(args.config)

    # Check to choose if you want to use the SummaryWriter (Tensorboard) or not.
    # Don't use it if you want to log with wandb.

    dataloader, model, optimizer = load_train_objs(cfg, args)

    kwargs = GradScalerKwargs()
    accelerator = Accelerator(mixed_precision="fp16", kwargs_handlers=[kwargs])
    device = accelerator.device

    if accelerator.is_main_process and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    loss_scaler = AcceleratorScaler(accelerator=accelerator)
    dataloader, model, optimizer = accelerator.prepare(dataloader, model, optimizer)

    for epoch in range(args.epochs):
        train_stats = train_one_epoch(
            model,
            dataloader,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
            task_name=cfg['task_name'],
            accelerator=accelerator
        )

        if args.output_dir and (epoch % args.save_every_epoch == 0 or epoch + 1 == args.epochs):
            accelerator.save_state(output_dir=os.path.join(args.output_dir, "accelerator_state"))
            # misc.save_model(
            #     args=args, model=model, model_without_ddp=model.module, optimizer=optimizer,
            #     loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and accelerator.is_main_process:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
    accelerator.end_training()

def load_train_objs(cfg, args):

    if args.dataset == 'wear_ssl':
        train_set = make_dataset(
        name=args.dataset,
        is_pretrain=True,
        **cfg['dataset'],
        **cfg['spectrogram_params'][f'sec_{args.seconds}'][args.matrix_type]
    )
    else:
        train_set = make_dataset(
            name=args.dataset,
            is_pretrain=True,
            task_name = cfg['task_name'],
            preload=True,
            **cfg['dataset'],
            **cfg['spectrogram_params'][f'sec_{args.seconds}'][args.matrix_type]
        )

    dataloader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    model = modeling(
        seconds=args.seconds,
        matrix_type=args.matrix_type,
        audio_exp=args.audio_exp,
        cfg=cfg
    )
    world_size = args.nodes * args.gpus_per_node
    eff_batch_size = args.batch_size * args.accum_iter * world_size

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    # loss_scaler = NativeScaler()
    # loss_scaler = AcceleratorScaler(accelerator=accelerator)

    return dataloader, model, optimizer

if __name__ == "__main__":
    parent_args = get_args_parser()
    parent_args.add_argument("--nodes", type=int, required=True)
    parent_args.add_argument("--gpus_per_node", type=int, required=True)

    args = parent_args.parse_args()
    # print(args)
    # sys.exit(0)
    main(args)