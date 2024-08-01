from audiomae_ft import get_args_parser, modeling, get_mixup, load_model

import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import sys
import json
import time
import datetime

from accelerate import Accelerator, GradScalerKwargs

from utils.os_utils import load_config

sys.path.append('.')
sys.path.append('..')
sys.path.append(os.path.join('submodules', 'AudioMAE'))
from subtrees.AudioMAE.engine_finetune import train_one_epoch, evaluate
from subtrees.AudioMAE.util.misc import NativeScalerWithGradNormCount as NativeScaler
from subtrees.AudioMAE.util.misc import AcceleratorScalerWithGradNormCount as AcceleratorScaler
import subtrees.AudioMAE.util.lr_decay as lrd
from torch.utils.tensorboard import SummaryWriter

from data.dataset import make_dataset


def main(args):
    cfg = load_config(args.config)

    # Check to choose if you want to use the SummaryWriter (Tensorboard) or not.
    # Don't use it if you want to log with wandb.

    train_loader, valid_loader, model, optimizer, criterion = load_train_objs(cfg, args)
    load_model(args.finetune, args.eval, model)
    mixup_fn = get_mixup(args)
    kwargs = GradScalerKwargs()
    accelerator = Accelerator(mixed_precision="fp16", kwargs_handlers=[kwargs])
    device = accelerator.device

    if accelerator.is_main_process and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # if accelerator.is_main_process:
    #     model = load_model(accelerator, args.finetune, args.eval, model)

    # model, optimizer = accelerator.load_state(output_dir=os.path.join(args.output_dir, "accelerator_state"))
    # accelerator.wait_for_everyone()

    loss_scaler = AcceleratorScaler(accelerator=accelerator)
    train_loader, valid_loader, model, optimizer = accelerator.prepare(train_loader, valid_loader, model, optimizer)

    if accelerator.is_main_process:
        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        max_mAP = 0.0
        max_mAcc = 0.0

    for epoch in range(args.epochs):
        train_stats = train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            epoch,
            # device,
            loss_scaler,
            max_norm=args.clip_grad,
            mixup_fn=mixup_fn,
            log_writer=log_writer,
            accelerator=accelerator,
            args=args,
            task_name=cfg['task_name']
        )
        if accelerator.is_main_process:
            if args.output_dir and (epoch % args.save_every_epoch == 0 or epoch + 1 == args.epochs):
                accelerator.wait_for_everyone()
                accelerator.save_state(output_dir=os.path.join(args.output_dir, "accelerator_state"))
                # misc.save_model(
                #     args=args, model=model, model_without_ddp=model.module, optimizer=optimizer,
                #     loss_scaler=loss_scaler, epoch=epoch)
            if epoch >= args.first_eval_ep:
                test_stats = evaluate(valid_loader, model, device, args.dist_eval, out_dir=args.output_dir)
                print(f"mAP of the network on the {len(valid_loader)} test images: {test_stats['mAP']:.4f}")
                print(f"Accuracy of the network on the {len(valid_loader)} test images: {test_stats['mAcc']:.4f}")
                max_mAP = max(max_mAP, test_stats["mAP"])
                max_mAcc = max(max_mAcc, test_stats["mAcc"])
                print(f'Max mAP: {max_mAP:.4f}')
                print(f'Max mAcc: {max_mAcc:.4f}')
            else:
                test_stats ={'mAP': 0.0, 'mAcc': 0.0}
                print(f'too new to evaluate!')
            

            if log_writer is not None:
                log_writer.add_scalar('perf/mAP', test_stats['mAP'], epoch)
                log_writer.add_scalar('perf/Acc', test_stats['mAcc'], epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,}

            if args.output_dir:
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
    accelerator.end_training()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def load_train_objs(cfg, args):

    if args.dataset == 'wear':
        train_set = make_dataset(
            name=args.dataset,
            is_pretrain=False,
            **cfg['dataset'],
            **cfg['spectrogram_params'][f'sec_{args.seconds}'][args.matrix_type]
        )
        val_set = make_dataset(
            name=args.dataset,
            is_pretrain=False,
            **cfg['dataset'],
            **cfg['spectrogram_params'][f'sec_{args.seconds}'][args.matrix_type]
        )
    else:
        preload = False

        # FIXME: Uncomment for debugging
        # cfg['dataset_train']['preload'] = preload
        # cfg['dataset_valid']['preload'] = preload

        train_set = make_dataset(
            name=args.dataset,
            is_pretrain=False,
            task_name = cfg['task_name'],
            **cfg['dataset_train'],
            **cfg['spectrogram_params'][f'sec_{args.seconds}'][args.matrix_type]
        )
        val_set = make_dataset(
            name=args.dataset,
            is_pretrain=False,
            task_name = cfg['task_name'],
            **cfg['dataset_valid'],
            **cfg['spectrogram_params'][f'sec_{args.seconds}'][args.matrix_type]
        )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    model = modeling(
        seconds=args.seconds,
        matrix_type=args.matrix_type,
        cfg=cfg,
        finetune=args.finetune,
        eval=args.eval
    )

    world_size = args.nodes * args.gpus_per_node
    eff_batch_size = args.batch_size * args.accum_iter * world_size

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    param_groups = lrd.param_groups_lrd(model, args.weight_decay,
        no_weight_decay_list=model.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    # loss_scaler = NativeScaler()

    criterion = BCEWithLogitsLoss()

    return train_loader, valid_loader, model, optimizer, criterion

if __name__ == "__main__":
    parent_args = get_args_parser()
    parent_args.add_argument("--nodes", type=int, required=True)
    parent_args.add_argument("--gpus_per_node", type=int, required=True)

    args = parent_args.parse_args()
    main(args)