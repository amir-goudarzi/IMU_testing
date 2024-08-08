#!/bin/bash

#SBATCH --job-name=prova_accelerate_gatanasio
#SBATCH -D /home/gatanasio/SSPT_IMU
#SBATCH --output=O-%x.%j
#SBATCH --error=E-%x.%j
#SBATCH --nodes=2                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:2                # number of GPUs per node
#SBATCH --time=00:30:00             # maximum execution time (HH:MM:SS)
#SBATCH --mem=51200
#SBATCH --partition=cuda
#SBATCH --mail-user=giuseppe.atanasio@studenti.polito.it

module load nvidia/cudasdk/11.8_test

######################
### Set enviroment ###
######################
# eval "$(conda shell.bash hook)"
source ../.bashrc
conda deactivate
conda activate audio-mae

export GPUS_PER_NODE=2
######################

######################
#### Set network #####
######################
# head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))

echo MASTER_ADDR: $MASTER_ADDR
echo MASTER_PORT: $MASTER_PORT
echo MACHINE_RANK: $SLURM_NODEID
echo NUM_NODES: $SLURM_NNODES
######################

# Add this after accelerate launch if necessary
# --config_file /home/gatanasio/.cache/huggingface/accelerate/default_config.yaml
export LAUNCHER="accelerate launch \
    --multi_gpu \
    --machine_rank $SLURM_NODEID \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --mixed_precision fp16 \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --rdzv_backend c10d \
    "

####################
##### Set args #####
####################
dataset=egoexo4d
matrix_type="128x320"
experiments_dir=./reports/experiments/audio_mae/$matrix_type/$dataset/linprob
patch=16
model=vit_small_patch$patch

export SCRIPT="$HOME/src/dist_ft_accelerate.py"
export SCRIPT_ARGS=" \
    --mixed_precision fp16 \
    --log_dir $experiments_dir/mask_ratio{$mask_ratio}_$model \
    --output_dir $experiments_dir/mask_ratio{$mask_ratio}_$model \
    --config ./configs/IMU-MAE/egoexo4d_accl_omni_linprob.yaml \
    --finetune ./reports/experiments/audio_mae/128x320/egoexo4d/pretrain/mask_ratio{$mask_ratio}_mae_$model/accelerator_state \
    --model $model \
    --epochs 1 \
    --blr 0.001 \
    --weight_decay 0.0001 \
    --batch_size 8 \
    --warmup_epochs 4 \
    --mask_t_prob $mask_ratio \
    --mixup 0.5 \
    --dataset $dataset \
    --matrix_type 128x320 \
    --seconds 2 \
    --nodes $SLURM_JOB_NUM_NODES \
    --gpus_per_node $GPUS_PER_NODE
    "
    
# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS" 
# srun $CMD
$CMD
