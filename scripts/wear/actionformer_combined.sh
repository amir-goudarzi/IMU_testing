#!/bin/bash

backwards=../../..
# config=configs/120_frames_60_stride/actionformer_mae.yaml
config=./configs/120_frames_60_stride/actionformer_combined.yaml
eval_type=split
run_id=actionformer_combined
seconds=2
patch=16
model=vit_base_patch$patch
nodes=1
matrix_type=128x320
pretrain_mask_ratio=0.9

MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))

cd src/subtrees/wear
for seed in 2 3
do
    accelerate launch --main_process_port $MASTER_PORT main.py \
        --config $config \
        --eval_type $eval_type \
        --run_id $run_id \
        --seconds $seconds \
        --seed $seed
done