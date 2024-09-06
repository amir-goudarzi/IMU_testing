#!/bin/bash

dataset=egoexo4d
matrix_type="128x320"
experiments_dir=./reports/experiments/audio_mae/$matrix_type/$dataset/imu_omni/finetuning
patch=16
model=vit_base_patch$patch
nodes=1
gpus_per_node=2
MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
pretrain_mask_ratio=0.9

# for mask_ratio in 0.1 0.2 0.3 0.4
for mask_ratio in 0.4
do
    accelerate launch --main_process_port $MASTER_PORT src/dist_ft_accelerate.py \
        --log_dir $experiments_dir/mask_ratio{$mask_ratio}_$model \
        --output_dir $experiments_dir/mask_ratio{$mask_ratio}_$model \
        --config ./configs/IMU-MAE/egoexo4d_accl_omni_ft.yaml \
        --finetune ./reports/experiments/audio_mae/128x320/egoexo4d/imu_omni/pretrain/mask_ratio{$pretrain_mask_ratio}_mae_$model/accelerator_state \
        --model $model \
        --epochs 30 \
        --blr 2e-4 \
        --weight_decay 0.0001 \
        --batch_size 128 \
        --warmup_epochs 4 \
        --mask_t_prob $mask_ratio \
        --mask_f_prob $mask_ratio \
        --mixup 0.0 \
        --dataset $dataset \
        --matrix_type 128x320 \
        --seconds 2 \
        --nodes $nodes \
        --gpus_per_node $gpus_per_node
done
