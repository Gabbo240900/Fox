#!/bin/bash

#SBATCH --job-name=20H_resume
#SBATCH --account=vcu@h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --output=error_files/resume_training%j.out
#SBATCH --error=error_files/resume_training%j.err
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=20:00:00



# ENVIRONMENT SETUP

module purge

module load arch/h100

module load pytorch-gpu/py3/2.4.0

module load miniforge/24.9.0

conda activate cophylo


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

mkdir -p "$WANDB_DATA_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$HF_HOME"



# ECHO
set -x

# W&B settings — keep identical to the original run
export WANDB_ENTITY=cophylo_team
export WANDB_PROJECT=CoPhyloformer
export WANDB_NAME=20H_asym_test
export WANDB_MODE=offline
export WANDB_HTTP_TIMEOUT=120
export WANDB__SERVICE_WAIT=300

# ── RESUME: only line you need to change ────────────────────────────────────
# Replace with the actual checkpoint you want to resume from.
# Run ID is read automatically from the checkpoint — no WANDB_RUN_ID needed.
export RESUME_CKPT="checkpoints/val_checkpoint_epoch1_step21687.pth"
# ────────────────────────────────────────────────────────────────────────────

echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
srun python train.py
