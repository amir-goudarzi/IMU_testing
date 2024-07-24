#!/bin/bash


dataset=egoexo4d
matrix_type="128x320"
experiments_dir=./reports/experiments/audio_mae/$matrix_type/$dataset/pretrain
patch=16
model=mae_vit_small_patch$patch

# for mask_ratio in 0.3 0.4 0.5 0.6 0.7 0.8 0.9
for mask_ratio in 0.8
do
    python src/audiomae_pretrain.py \
    --log_dir $experiments_dir/mask_ratio{$mask_ratio}_$model \
    --output_dir $experiments_dir/mask_ratio{$mask_ratio}_$model \
    --config ./configs/IMU-MAE/egoexo4d_accl_omni.yaml \
    --model $model \
    --epochs 32 \
    --blr 2e-4 \
    --weight_decay 0.0001 \
    --batch_size 128 \
    --warmup_epochs 3 \
    --mask_ratio $mask_ratio \
    --dataset $dataset \
    --matrix_type 128x320 \
    --seconds 2
done
