#!/bin/bash

#SBATCH --job-name=20H_training
#SBATCH --account=vcu@h100 
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --output=error_files/20H_training%j.out 
#SBATCH --error=error_files/20H_training%j.err 
#SBATCH --constraint=h100 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1    
#SBATCH --gres=gpu:2           
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
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0


# Set cache directories to avoid cluster warnings
export XDG_CACHE_HOME="$ALL_CCFRWORK/.cache"
export MPLCONFIGDIR="$ALL_CCFRWORK/.cache/mpl"
export HF_HOME="$ALL_CCFRWORK/.cache/huggingface"  
export WANDB_DATA_DIR="$JOBSCRATCH/wandb_data"
export WANDB_CACHE_DIR="$JOBSCRATCH/wandb_cache"

mkdir -p "$WANDB_DATA_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$HF_HOME" 


# ECHO 
set -x 

# STEP 3: Training phase 
export WANDB_ENTITY=cophylo_team           
export WANDB_PROJECT=CoPhyloformer
export WANDB_NAME=20H_training_MAE_norm
export WANDB_MODE=offline    # "online", "offline", or "disabled" 
export WANDB_HTTP_TIMEOUT=120
export WANDB__SERVICE_WAIT=300     

echo "Copying dataset from ALLCCFRSCRATCH to JOBSCRATCH..."
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"

export TIMING_PROFILE=1
export TIMING_EVERY=50

srun python train.py
