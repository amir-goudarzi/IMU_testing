#!/bin/bash


dataset=wear
experiments_dir=./reports/experiments/audio_mae/$dataset
patch=8
model=vit_small_patch$patch

for mask_ratio in 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    CUDA_VISIBLE_DEVICES=0,1 python src/audiomae_ft.py \
        --model $model \
        --dataset $dataset \
        --epochs 60 \
        --output_dir $experiments_dir/finetune/mask_ratio{$mask_ratio}_vit_small_patch8 \
        --log_dir $experiments_dir/finetune/mask_ratio{$mask_ratio}_vit_small_patch8 \
        --finetune $experiments_dir/pretrain/mask_ratio{$mask_ratio}_mae_vit_small_patch8/checkpoint-59.pth \
        --blr 0.001 \
        --batch_size 16 \
        --warmup_epochs 4 \
        --mask_2d True \
        --mask_t_prob 0.2 \
        --mask_f_prob 0.2 \
        --mixup 0.5 \
        --nb_classes 18
done
