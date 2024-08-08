#!/bin/bash

#SBATCH --job-name=prova_accelerate_gatanasio
#SBATCH -D /home/gatanasio/SSPT_IMU
#SBATCH --output=O-%x.%j
#SBATCH --error=E-%x.%j
#SBATCH --nodes=1                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2                # number of GPUs per node
#SBATCH --time=00:30:00             # maximum execution time (HH:MM:SS)
#SBATCH --mem=100G
#SBATCH --partition=cuda
#SBATCH --mail-user=giuseppe.atanasio@studenti.polito.it

module purge
module --ignore-cache load "nvidia/cudasdk/11.8_test"

######################
### Set enviroment ###
######################
# eval "$(conda shell.bash hook)"
source ../.bashrc
conda deactivate
conda deactivate
conda activate dist_torch

export GPUS_PER_NODE=2
NNODES=$SLURM_NNODES
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)
######################

######################
#### Set network #####
######################
# head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

export MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))

echo MASTER_ADDR: $master_addr
echo MASTER_PORT: $MASTER_PORT
echo MACHINE_RANK: $SLURM_NODEID
echo NUM_NODES: $SLURM_NNODES
echo NODESLIST: $SLURM_JOB_NODELIST
######################

# Set up accelerate config.
export ACCELERATE_CONFIG_YAML="/home/gatanasio/SSPT_IMU/accelerate_config_"$SLURM_JOB_ID".yaml"
# export CREATE_CFG="cat <<EOT > \"\$ACCELERATE_CONFIG_YAML\"
# compute_environment: LOCAL_MACHINE
# distributed_type: MULTI_GPU
# downcast_bf16: 'no'
# gpu_ids: all
# mixed_precision: fp16
# rdzv_backend: c10d
# same_network: true
# tpu_env: []
# tpu_use_cluster: false
# tpu_use_sudo: false
# use_cpu: false
# EOT"

export CREATE_CFG="cat <<EOT > \"\$ACCELERATE_CONFIG_YAML\"
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: \$SLURM_NODEID
main_process_ip: '\$MASTER_ADDR'
main_process_port: \$MASTER_PORT
main_training_function: main
mixed_precision: fp16
num_machines: \$SLURM_JOB_NUM_NODES
num_processes: \$((SLURM_JOB_NUM_NODES * GPUS_PER_NODE))
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOT"

bash -c "$CREATE_CFG"

# Add this after accelerate launch if necessary
# --config_file /home/gatanasio/.cache/huggingface/accelerate/default_config.yaml
export LAUNCHER="accelerate launch --config_file $ACCELERATE_CONFIG_YAML \
--main_process_ip $MASTER_ADDR \
--main_process_port $MASTER_PORT \
--machine_rank \$SLURM_PROCID \
--num_processes $NUM_PROCESSES \
--num_machines $NNODES \
"

####################
##### Set args #####
####################
dataset=egoexo4d
matrix_type="128x320"
experiments_dir=./reports/experiments/audio_mae/$matrix_type/$dataset/linprob
patch=16
model=vit_small_patch$patch
mask_ratio=0.8

export SCRIPT="src/dist_ft_accelerate.py"
export SCRIPT_ARGS=" \
--log_dir $experiments_dir/mask_ratio{$mask_ratio}_$model \
--output_dir $experiments_dir/mask_ratio{$mask_ratio}_$model \
--config ./configs/IMU-MAE/egoexo4d_accl_omni_linprob_dist.yaml \
--finetune ./reports/experiments/audio_mae/128x320/egoexo4d/pretrain/mask_ratio{$mask_ratio}_mae_$model/accelerator_state \
--model $model \
--epochs 1 \
--blr 0.001 \
--weight_decay 0.0001 \
--batch_size 128 \
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

srun bash -c "$CMD"
# $CMD
