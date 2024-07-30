# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable
from time import time

import torch

from .util import misc
from .util import lr_sched
import accelerate
from torch.utils.tensorboard import SummaryWriter


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None | SummaryWriter,
                    args=None,
                    task_name=None,
                    accelerator=None | accelerate.Accelerator):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # set model epoch
    model.epoch = epoch

    #choose forward function
    train_forward = train_fn(task_name)

    # for data_iter_step, (samples, _labels, _vids) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        with accelerator.accumulate(model):
        # we use a per iteration (instead of per epoch) lr scheduler
            if data_iter_step % accum_iter == 0:
                lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

            autocast = accelerator if accelerator is None else torch.cuda.amp
            loss_value, loss_total = train_forward(model, device, samples, autocast, args)

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            #loss /= accum_iter
            loss_total = loss_total / accum_iter
            loss_scaler(loss_total, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

        # torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)


        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0 and accelerator.is_main_process:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            # accelerator.log({
            #     "lr": lr,
            #     "loss": loss_value_reduce,
            # }, step=epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_fn(task_name):
    if task_name == "imu_omnivore":
        return imu_omnivore
    elif task_name == "imu":
        return imu
    else:
        raise NotImplementedError(f"Task {task_name} not implemented")
    
    
# TODO: write imu_omnivore function for finetuning.
def imu_omnivore(model, device, samples, args):
    imu, omnivore = samples
    imu = imu.to(device, non_blocking=True)
    omnivore = omnivore.to(device, non_blocking=True)
    with torch.cuda.amp.autocast():
        emb_enc, mask, ids_restore, _ = model.forward_encoder(imu, args.mask_ratio, mask_2d=model.mask_2d)

        omnivore = omnivore.repeat(1, emb_enc.shape[1], 1)
        pred, _, _ = model.forward_decoder(emb_enc, ids_restore)  # [N, L, p*p*3]
        
        loss_a = model.forward_loss(imu, pred, mask, norm_pix_loss=model.norm_pix_loss)
    loss_value = loss_a.item()
    loss_total = loss_a
    
    return loss_value, loss_total


def imu(model, device, samples, autocast, args):
    # samples = samples.to(device, non_blocking=True)
    with autocast.autocast():
        samples = samples.type(torch.bfloat16)
        loss_a, _, _, _ = model(samples, mask_ratio=args.mask_ratio)
    loss_value = loss_a.item()
    loss_total = loss_a

    return loss_value, loss_total
