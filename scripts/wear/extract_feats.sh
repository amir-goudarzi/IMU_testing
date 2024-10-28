#!/bin/bash
backwards=../../..
dataset=wear_ssl
matrix_type="128x320"
patch=16
experiments_dir=./reports/experiments/audio_mae/$matrix_type/$dataset
model=mae_vit_base_patch$patch
nodes=1
gpus_per_node=2
MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
mask_ratio=0.9
config=./configs/120_frames_60_stride/actionformer_inertial.yaml
config_mae=$backwards/configs/IMU-MAE/WEAR

cd src/subtrees/wear
for split in 1 2 3
do
    accelerate launch --main_process_port $MASTER_PORT extract_feats_mae.py \
        --config $config \
        --config_mae $config_mae/split_$split/wear_combined_pt.yaml \
        --finetune $backwards/reports/experiments/audio_mae/128x320/wear_ssl/2d/imu/split_$split/mask_ratio{$mask_ratio}_mae_vit_base_patch16/accelerator_state \
        --seconds 2 \
        --matrix_type $matrix_type \
        --eval_type split \
        --mask_ratio $mask_ratio \
        --split $split \
        --imu_i3d_fromscratch
    
    accelerate launch --main_process_port $MASTER_PORT extract_feats_mae.py \
        --config $config \
        --config_mae $config_mae/split_$split/wear_combined_pt.yaml \
        --finetune $backwards/reports/experiments/audio_mae/128x320/wear_ssl/2d/imu/split_$split/mask_ratio{$mask_ratio}_mae_vit_base_patch16/accelerator_state \
        --seconds 2 \
        --matrix_type $matrix_type \
        --eval_type split \
        --mask_ratio $mask_ratio \
        --split $split \
        --combined_to_inertial \
        --imu_i3d_fromscratch
done
