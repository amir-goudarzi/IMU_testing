#!/bin/bash

backwards=../../..
config=configs/120_frames_60_stride/actionformer_mae.yaml
eval_type=split
run_id=actionformer_mae
config_mae=configs/IMU-MAE/spectrograms_wear.yaml
finetune="reports/experiments/audio_mae/wear/pretrain/sec2_split_1/mask_ratio{0.9}_mae_vit_small_patch8/checkpoint-29.pth"
seconds=2
matrix_type=64x64


cd src/subtrees/wear
python main.py \
    --config $config \
    --eval_type $eval_type \
    --run_id $run_id \
    --config_mae $backwards/$config_mae \
    --finetune $backwards/$finetune \
    --seconds $seconds \
    --matrix_type $matrix_type