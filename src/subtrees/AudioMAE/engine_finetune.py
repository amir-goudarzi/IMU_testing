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
import numpy as np
import sys
import matplotlib.pyplot as plt
from typing import Iterable, Optional
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import os
import wandb

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import one_hot

from timm.data import Mixup
from timm.utils import accuracy
import wandb.plot
from .util.stat import calculate_stats

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
        loss_scaler = None,
        max_norm: float = 0,
        mixup_fn: Optional[Mixup] = None,
        log_writer=None | SummaryWriter,
        accelerator=None | accelerate.Accelerator,
        args=None,
        task_name='imu',
        num_classes=68
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

            # samples = samples.to(device, non_blocking=True)
            # targets = targets.to(device, non_blocking=True)

            if mixup_fn is not None:
                if type(samples) is torch.Tensor:
                    samples, targets = mixup_fn(samples, targets)
                else:
                    imu, omni = samples
                    imu, targets = mixup_fn(imu, targets)
                    samples = (imu, omni)

            else:
                targets = one_hot(targets, num_classes).to(torch.float32)

            outputs, loss = train_forward(model, samples, targets, criterion, accelerator, mask_t_prob=args.mask_t_prob, mask_f_prob=args.mask_f_prob)
            if loss_scaler is None:
                accelerator.backward(loss)
            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss /= accum_iter
            if data_iter_step % accum_iter == 0:
                lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

            if loss_scaler is not None:
                loss_scaler(loss, optimizer, clip_grad=max_norm,
                            parameters=model.parameters(), create_graph=False,
                            update_grad=(data_iter_step + 1) % accum_iter == 0)
            else:
                optimizer.step()

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
    accelerator.log({'train_loss': metric_logger.loss.global_avg, 'lr': metric_logger.lr.global_avg}, step=epoch)
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
        data_loader: Iterable,
        model: torch.nn.Module,
        accelerator: None | accelerate.Accelerator,
        args,
        epoch: int,
        task_name='imu',
        class_labels=None | pd.DataFrame
    ):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    lb = LabelBinarizer()
    lb.fit(range(model.module.num_classes))

    # switch to evaluation mode
    model.eval()
    train_forward = train_fn(task_name)
    outputs = []
    targets = []
    outputs_conf_matrix = []
    targets_conf_matrix = []

    for images, target in metric_logger.log_every(data_loader, 10, header):
        # images = batch[0]
        # target = batch[-1]
        # images = images.to(device, non_blocking=True)
        # target = target.to(device, non_blocking=True)
        # compute output
        
        if type(images) is list:
            images, vid = images
            batch_size = images.shape[0]
            images = (images, vid)
        else:
            batch_size = images.shape[0]

        output, loss = train_forward(model, images, target, criterion, accelerator)
        output, target = accelerator.gather_for_metrics((output, target))
        
        
        # acc1, acc5 = accuracy(output, torch.argmax(target, dim=1), topk=(1, 5))
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        target = torch.tensor(lb.transform(target.cpu().numpy())).type(torch.float32).to(accelerator.device)
        outputs.append(output)
        targets.append(target)
        outputs_conf_matrix.append(output.argmax(dim=1))
        targets_conf_matrix.append(target.argmax(dim=1))
    
    outputs=torch.cat(outputs).cpu().numpy()
    targets=torch.cat(targets).cpu().numpy()
    outputs_conf_matrix=torch.cat(outputs_conf_matrix).cpu().numpy()
    targets_conf_matrix=torch.cat(targets_conf_matrix).cpu().numpy()

    stats = calculate_stats(outputs, targets)

    mAP = np.mean([stat['AP'] for stat in stats])
    mAUC = np.mean([stat['auc'] for stat in stats])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    
    if accelerator.is_main_process:
        # wandb = accelerator.get_tracker('wandb')

        # disp = ConfusionMatrixDisplay(
        #         confusion_matrix(targets_conf_matrix, outputs_conf_matrix, labels=range(model.module.num_classes), normalize='all'),
        #         display_labels=class_labels
        #     )
        # disp.plot()
        # plot_path = os.path.join(args.output_dir, f'confusion_matrix_{epoch}.png')
        # plt.savefig(plot_path)
        # plt.close()

        if epoch + 1 == args.epochs:
            per_class_accuracy = confusion_matrix(targets_conf_matrix, outputs_conf_matrix, labels=range(model.module.num_classes))
            per_class_accuracy = per_class_accuracy.diagonal() / per_class_accuracy.sum(axis=1)
            per_class_accuracy = {class_labels[i]: acc for i, acc in enumerate(per_class_accuracy)}
            data = [[label, acc] for label, acc in per_class_accuracy.items()]
            table = wandb.Table(data=data, columns=["class", "accuracy"])
            accelerator.log({
                'valid_loss': metric_logger.loss.global_avg,
                'acc1': metric_logger.acc1.global_avg,
                'acc5': metric_logger.acc5.global_avg,
                'mAP': mAP,
                'mAUC': mAUC,
                'per_class_accuracy': wandb.plot.bar(table, "class", "accuracy", title="Per-class accuracy"),
            }, step=epoch)
        else:
            accelerator.log({
                'valid_loss': metric_logger.loss.global_avg,
                'acc1': metric_logger.acc1.global_avg,
                'acc5': metric_logger.acc5.global_avg,
                'mAP': mAP,
                'mAUC': mAUC,
            }, step=epoch)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_fn(task_name):
    if task_name == "imu_omnivore":
        return imu_omnivore
    elif task_name == "imu":
        return imu
    else:
        raise NotImplementedError(f"Task {task_name} not implemented")

# TODO: write imu_omnivore function for finetuning.
def imu_omnivore(model, samples, targets, criterion, autocast, mask_t_prob=0.0, mask_f_prob=0.0):
    # samples = samples.to(device, non_blocking=True)
    with autocast.autocast():
        outputs = model(
            samples,
            mask_t_prob=mask_t_prob,
            mask_f_prob=mask_f_prob
        )
        loss = criterion(outputs, targets)

    return outputs, loss


def imu(model, samples, targets, criterion, accelerator: accelerate.Accelerator, mask_t_prob=0.0, mask_f_prob=0.0):
    # samples = samples.to(device, non_blocking=True)
    with accelerator.autocast():
        outputs = model(
            samples,
            mask_t_prob=mask_t_prob,
            mask_f_prob=mask_f_prob
        )
        loss = criterion(outputs, targets)
    return outputs, loss
