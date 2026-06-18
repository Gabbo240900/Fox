#!/bin/bash

#SBATCH --job-name=20H_training
#SBATCH --account=vcu@h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --output=error_files/20H_training%j.out
#SBATCH --error=error_files/20H_training%j.err
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=20:00:00



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
export WANDB_NAME=40ep_hgt2_tail2_dropout
export WANDB_MODE=offline
export WANDB_HTTP_TIMEOUT=120
export WANDB__SERVICE_WAIT=300

# Architecture
export HIDDEN_DIM=256
export AXIAL_LAYERS=2
export CROSS_LAYERS=2
export USE_OPM=0
export USE_DIST_MATRIX=1
export GRADIENT_CHECKPOINTING=0
export USE_FLEX_ATTENTION=1
export USE_COMPILE=0
export FIND_UNUSED_PARAMETERS=0
export USE_BUCKETED_BATCHES=1
export BUCKET_SIZE=4
export DROPOUT=0.10
export TIME_DROPOUT=0.15   # prob. of zeroing sim_time per sample (undated-tree / time-agnostic mode); 0 = off

# Training settings
export EPOCHS=40
export EARLY_STOP_PATIENCE=8
export EARLY_STOP_MIN_DELTA=0.0
export MID_EPOCH_VALS=1
export NUM_WORKERS=16
export CKPT_DIR=checkpoints_fast
export BATCH_SIZE=32
export GRAD_ACCUM=4
export VAL_BATCH_MULT=1
export LR=1e-4
export WEIGHT_DECAY=0.05
export HGT_LOSS_WEIGHT=1.0
export TAIL_WEIGHT=1.0
export UNDER_PENALTY=1.0
export HOST_MAX_LEAVES=51
export PARA_MAX_LEAVES=142
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"


if [ -f "$CKPT_DIR/last_epoch.ckpt" ]; then
    echo "[Resume] Resuming from last_epoch.ckpt"
    srun python train.py resume "$CKPT_DIR/last_epoch.ckpt"
elif [ -f "$CKPT_DIR/latest.ckpt" ]; then
    echo "[Resume] Resuming from latest.ckpt (mid-epoch)"
    srun python train.py resume "$CKPT_DIR/latest.ckpt"
else
    echo "[Train] Starting from scratch"
    srun python train.py train
fi
