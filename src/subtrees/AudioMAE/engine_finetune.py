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
from typing import Iterable, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from timm.data import Mixup
from timm.utils import accuracy

from .util import misc
from .util import lr_sched
import accelerate

def train_one_epoch(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        # device: torch.device,
        epoch: int,
        loss_scaler,
        max_norm: float = 0,
        mixup_fn: Optional[Mixup] = None,
        log_writer=None | SummaryWriter,
        accelerator=None | accelerate.Accelerator,
        args=None,
        task_name='imu'
    ):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    #choose forward function
    train_forward = train_fn(task_name)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        with accelerator.accumulate(model):
            # we use a per iteration (instead of per epoch) lr scheduler
            if data_iter_step % accum_iter == 0:
                lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

            # samples = samples.to(device, non_blocking=True)
            # targets = targets.to(device, non_blocking=True)

            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)

            outputs, loss = train_forward(model, samples, targets, criterion, accelerator, args)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss /= accum_iter
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=False,
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

            # torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0 and accelerator.is_main_process:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
        data_loader: Iterable,
        model: torch.nn.Module,
        device,
        accelerator: None | accelerate.Accelerator,
        args,
        task_name='imu',
    ):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    train_forward = train_fn(task_name)
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        # images = images.to(device, non_blocking=True)
        # target = target.to(device, non_blocking=True)

        # compute output
        output, loss = train_forward(model, images, target, criterion, accelerator, args)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

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


def imu(model, samples, targets, criterion, autocast, args):
    # samples = samples.to(device, non_blocking=True)
    with autocast.autocast():
        outputs = model(
            samples,
            mask_t_prob=args.mask_t_prob,
            mask_f_prob=args.mask_f_prob
        )
        loss = criterion(outputs, targets)

    return outputs, loss
