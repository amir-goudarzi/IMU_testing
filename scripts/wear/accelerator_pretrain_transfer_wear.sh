#!/bin/bash

dataset=wear_ssl
matrix_type="128x320"
patch=16
experiments_dir=./reports/experiments/audio_mae/$matrix_type/$dataset/2d
model=mae_vit_base_patch$patch
nodes=1
gpus_per_node=2
MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
mask_ratio=0.8
for split in 1 2 3
do
    accelerate launch --main_process_port $MASTER_PORT src/dist_pretrain_accelerate.py \
        --log_dir $experiments_dir/split_$split/mask_ratio{$mask_ratio}_$model/ \
        --output_dir $experiments_dir/split_$split/mask_ratio{$mask_ratio}_$model/ \
        --config ./configs/IMU-MAE/WEAR/split_$split/wear_inertial_pt.yaml \
        --resume ./reports/experiments/audio_mae/128x320/egoexo4d/imu_omni/pretrain/mask_ratio{$mask_ratio}_mae_vit_base_patch16/accelerator_state \
        --split $split \
        --model $model \
        --epochs 32 \
        --blr 1e-3 \
        --weight_decay 0.0001 \
        --batch_size 32 \
        --warmup_epochs 3 \
        --mask_ratio $mask_ratio \
        --dataset $dataset \
        --matrix_type 128x320 \
        --seconds 2 \
        --nodes $nodes \
        --gpus_per_node $gpus_per_node \
        --norm_pix_loss
done
