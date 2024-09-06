#!/bin/bash

backwards=../../..
# config=configs/120_frames_60_stride/actionformer_mae.yaml
config=./configs/wear/actionformer_mae.yaml
eval_type=split
run_id=actionformer_mae
seconds=2
patch=16
model=vit_base_patch$patch
nodes=1
matrix_type=128x320
pretrain_mask_ratio=0.9

config_mae=configs/IMU-MAE/spectrograms_wear.yaml
finetune="./reports/experiments/audio_mae/128x320/egoexo4d/pretrain/mask_ratio{$pretrain_mask_ratio}_mae_$model/accelerator_state"


cd src/subtrees/wear
python main.py \
    --config $config \
    --eval_type $eval_type \
    --run_id $run_id \
    --config_mae $backwards/$config_mae \
    --finetune $backwards/$finetune \
    --seconds $seconds \
    --matrix_type $matrix_type