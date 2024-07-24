#!/bin/bash


dataset=wear
matrix_type="128x320"
experiments_dir=./reports/experiments/audio_mae/correct/$matrix_type/$dataset/pretrain
patch=16
model=mae_vit_small_patch$patch

for split in 1 2 3
do
    for mask_ratio in 0.6 0.7 0.8 0.9
    do
        python src/audiomae_pretrain.py \
        --log_dir $experiments_dir/sec2_split_$split/mask_ratio{$mask_ratio}_$model \
        --output_dir $experiments_dir/sec2_split_$split/mask_ratio{$mask_ratio}_$model \
        --model $model \
        --epochs 60 \
        --blr 2e-4 \
        --batch_size 64 \
        --warmup_epochs 3 \
        --mask_ratio $mask_ratio \
        --dataset $dataset \
        --config ./configs/IMU-MAE/spectrograms_wear.yaml \
        --seconds 2 \
        --matrix_type $matrix_type \
        --filename_split "wear_split_$split.pkl"
    done
done