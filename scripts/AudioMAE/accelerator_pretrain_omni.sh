#!/bin/bash

dataset=egoexo4d
matrix_type="128x320"
experiments_dir=./reports/experiments/audio_mae/$matrix_type/$dataset/imu_omni/pretrain
patch=16
model=mae_vit_base_patch$patch
nodes=1
gpus_per_node=2
MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))

for mask_ratio in 0.7
do
    accelerate launch --main_process_port $MASTER_PORT src/dist_pretrain_accelerate.py \
        --log_dir $experiments_dir/mask_ratio{$mask_ratio}_$model/ \
        --output_dir $experiments_dir/mask_ratio{$mask_ratio}_$model/ \
        --config ./configs/IMU-MAE/egoexo4d_accl_omni_pt.yaml \
        --model $model \
        --epochs 32 \
        --blr 2e-4 \
        --weight_decay 0.0001 \
        --batch_size 128 \
        --warmup_epochs 3 \
        --mask_ratio $mask_ratio \
        --dataset $dataset \
        --matrix_type 128x320 \
        --seconds 2 \
        --nodes $nodes \
        --gpus_per_node $gpus_per_node \
        --norm_pix_loss
done