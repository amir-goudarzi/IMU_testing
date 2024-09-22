#!/bin/bash

backwards=../../..
# config=configs/120_frames_60_stride/actionformer_mae.yaml
config=./configs/120_frames_60_stride/tridet_camera.yaml
eval_type=split
run_id=tridet_camera
seconds=2

MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))

cd src/subtrees/wear
for seed in 1 2 3
do
    accelerate launch --main_process_port $MASTER_PORT main.py \
        --config $config \
        --eval_type $eval_type \
        --run_id $run_id \
        --seconds $seconds \
        --seed $seed
done
