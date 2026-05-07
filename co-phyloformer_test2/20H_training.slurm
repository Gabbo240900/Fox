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
export WANDB_NAME=20H_asymmetree_old_msa
export WANDB_MODE=offline    # "online", "offline", or "disabled" 
export WANDB_HTTP_TIMEOUT=120
export WANDB__SERVICE_WAIT=300     
export ASYMMETREE_TRAIN_PREENCODED_DIR=/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_asymmetree/generated_trees/new_train
export ASYMMETREE_VAL_PREENCODED_DIR=/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_asymmetree/generated_trees/new_val
export CKPT_DIR=checkpoints

echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
if [ -f "$CKPT_DIR/last_epoch.ckpt" ]; then
    echo "[Resume] Resuming from last_epoch.ckpt"
    srun python train.py resume "$CKPT_DIR/last_epoch.ckpt"
elif [ -f "$CKPT_DIR/latest.ckpt" ]; then
    echo "[Resume] Resuming from latest.ckpt"
    srun python train.py resume "$CKPT_DIR/latest.ckpt"
else
    echo "[Train] Starting from scratch"
    srun python train.py train
fi
