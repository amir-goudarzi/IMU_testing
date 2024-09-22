#!/bin/bash

dataset=egoexo4d
matrix_type="128x320"
experiments_dir=./reports/experiments/audio_mae/$matrix_type/$dataset/linprob/label_balance
patch=16
model=vit_base_patch$patch
nodes=1
gpus_per_node=2
MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))

for mask_ratio in 0.6 0.7 0.8 0.9
# for mask_ratio in 0.6 0.9
# --weight_decay 0.0001 \
do
    accelerate launch --main_process_port $MASTER_PORT src/dist_ft_accelerate.py \
        --log_dir $experiments_dir/mask_ratio{$mask_ratio}_$model \
        --output_dir $experiments_dir/mask_ratio{$mask_ratio}_$model \
        --config ./configs/IMU-MAE/egoexo4d_accl_linprob.yaml \
        --finetune ./reports/experiments/audio_mae/128x320/egoexo4d/pretrain/mask_ratio{$mask_ratio}_mae_$model/accelerator_state \
        --model $model \
        --epochs 30 \
        --blr 1e-4 \
        --batch_size 128 \
        --weight_decay 1e-6 \
        --warmup_epochs 3 \
        --mask_t_prob 0.0 \
        --mixup 0.0 \
        --dataset $dataset \
        --matrix_type 128x320 \
        --seconds 2 \
        --nodes $nodes \
        --gpus_per_node $gpus_per_node \
        --label_balance
done
