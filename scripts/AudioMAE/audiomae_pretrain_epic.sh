#!/bin/bash


dataset=EPIC_KITCHENS
experiments_dir=./reports/experiments/audio_mae
patch=8
model=mae_vit_small_patch$patch
mask_ratio=0.5

CUDA_VISIBLE_DEVICES=0,1 python src/audiomae_pretrain.py \
--log_dir $experiments_dir/epic100_exp/mask_ratio{$mask_ratio}_$model \
--output_dir $experiments_dir/epic100_exp/mask_ratio{$mask_ratio}_$model \
--model $model \
--epochs 60 \
--blr 2e-4 \
--batch_size 16 \
--warmup_epochs 3 \
--mask_ratio $mask_ratio
