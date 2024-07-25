# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import functools
import torch

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def _cosine_decay_warmup(epoch, warmup_epochs, total_epochs):
    """
    Linear warmup from 0 --> 1.0, then decay using cosine decay to 0.0
    """
    if epoch <= warmup_epochs:
        multiplier = epoch / warmup_epochs
    else:
        multiplier = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        multiplier = 0.5 * (1 + math.cos(math.pi * multiplier))
    return multiplier

def CosineAnnealingLRWarmup(optimizer, T_max, T_warmup) -> torch.optim.lr_scheduler.LRScheduler:
    _decay_func = functools.partial(
    _cosine_decay_warmup, 
    warmup_iterations=T_warmup, total_iterations=T_max
        )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_func)
    return scheduler