#!/bin/bash


dataset=wear
experiments_dir=./reports/experiments/audio_mae/$dataset/pretrain
patch=8
model=mae_vit_small_patch$patch

for mask_ratio in 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    CUDA_VISIBLE_DEVICES=0,1 python src/audiomae_pretrain.py \
    --log_dir $experiments_dir/mask_ratio{$mask_ratio}_$model \
    --output_dir $experiments_dir/mask_ratio{$mask_ratio}_$model \
    --model $model \
    --epochs 60 \
    --blr 2e-4 \
    --batch_size 16 \
    --warmup_epochs 3 \
    --mask_ratio $mask_ratio \
    --dataset $dataset
done
