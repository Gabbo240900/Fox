#!/bin/bash

#SBATCH --job-name=100H_training
#SBATCH --account=vcu@h100
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --output=error_files/100H_training%j.out
#SBATCH --error=error_files/100H_training%j.err
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=100:00:00



# ENVIRONMENT SETUP

module purge

module load arch/h100

module load pytorch-gpu/py3/2.6.0

source $WORK/envs/cophylo_pt26/bin/activate


# Use all the CPUs Slurm gave this task
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

# NCCL / distributed sanity
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0


# Set cache directories to avoid cluster warnings
export XDG_CACHE_HOME="$ALL_CCFRWORK/.cache"
export MPLCONFIGDIR="$ALL_CCFRWORK/.cache/mpl"
export HF_HOME="$ALL_CCFRWORK/.cache/huggingface"
export WANDB_CACHE_DIR="$ALL_CCFRWORK/.cache/wandb"
export WANDB_DATA_DIR="$JOBSCRATCH/wandb_data"

mkdir -p "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$HF_HOME"



# ECHO
set -x

# W&B settings
export WANDB_ENTITY=cophylo_team
export WANDB_PROJECT=CoPhyloformer
export WANDB_NAME=100H_training_asym
export WANDB_MODE=offline
export WANDB_HTTP_TIMEOUT=120
export WANDB__SERVICE_WAIT=300

# Architecture
export HIDDEN_DIM=256
export AXIAL_LAYERS=4
export CROSS_LAYERS=2
export USE_OPM=1
export USE_DIST_MATRIX=1
export GRADIENT_CHECKPOINTING=1
export USE_FLEX_ATTENTION=0

# Training settings
export MID_EPOCH_VALS=2
export NUM_WORKERS=16
export CKPT_DIR=checkpoints
export BATCH_SIZE=32
export HOST_MAX_LEAVES=50
export PARA_MAX_LEAVES=128
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
srun python train.py
