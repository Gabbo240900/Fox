#!/bin/bash

#SBATCH --job-name=dev_training
#SBATCH --account=vcu@h100 
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --output=error_files/dev_training%j.out 
#SBATCH --error=error_files/dev_training%j.err 
#SBATCH --constraint=h100 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24 
#SBATCH --time=2:00:00 



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

# STEP 3: Training phase 
export WANDB_ENTITY=cophylo_team           
export WANDB_PROJECT=CoPhyloformer
export WANDB_NAME=Dev_training_overfitting_asym
export WANDB_MODE=offline    # "online", "offline", or "disabled" 
export WANDB_HTTP_TIMEOUT=120
export WANDB__SERVICE_WAIT=300     


export USE_OPM=1
export USE_DIST_MATRIX=1
export MID_EPOCH_VALS=0
export SAVE_MID_EPOCH_VAL_CKPTS=0
export COLLECT_TRAIN_PREDICTIONS=1 #IF 0 train-vs-val scatter plots are skipped.
export WANDB_WATCH=0  #IF 0 no parameter/gradient histograms in W&B.
export GRADIENT_CHECKPOINTING=1 #IF 0 impact: faster, but higher GPU memory usage (can OOM on larger configs).
export TRAIN_NUM_WORKERS=8
export VAL_NUM_WORKERS=8
export DISABLE_CHECKPOINTS=1
export OVERFIT_MODE=0


echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
srun python train.py 
