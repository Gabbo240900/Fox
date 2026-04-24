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
#SBATCH --signal=B:USR1@120   # send USR1 to batch script 2 min before time limit



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
export WANDB_NAME=Dev_training_overfitting_asym
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
export DROPOUT=0.05

# Training settings
export EPOCHS=5
export MID_EPOCH_VALS=1
export NUM_WORKERS=16
export CKPT_DIR=checkpoints
export BATCH_SIZE=80
export GRAD_ACCUM=4
export LR=1e-4
export WEIGHT_DECAY=0.01
export HGT_LOSS_WEIGHT=1.5
export TAIL_WEIGHT=1.0
export UNDER_PENALTY=1.5
export HOST_MAX_LEAVES=51
export PARA_MAX_LEAVES=142
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Resubmit helper ────────────────────────────────────────────────────────────
# Called either by the USR1 trap (time limit approaching) or after normal exit
# if the epoch target hasn't been reached yet.
_RESUBMITTED=0

_resubmit() {
    if [ $_RESUBMITTED -eq 0 ]; then
        _RESUBMITTED=1
        echo "[Chain] Resubmitting dev job for continuation..."
        sbatch "$SLURM_SUBMIT_DIR/dev_training.slurm"
    fi
}

# Trap USR1: fires 2 min before the 2h wall-time limit.
# Submits the next job while the current one still has time to checkpoint.
trap '_resubmit' USR1

# ── Launch training (backgrounded so the trap can fire mid-run) ────────────────
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"

if [ -f "$CKPT_DIR/last_epoch.ckpt" ]; then
    echo "[Resume] Resuming from last_epoch.ckpt"
    srun python train.py resume "$CKPT_DIR/last_epoch.ckpt" &
elif [ -f "$CKPT_DIR/latest.ckpt" ]; then
    echo "[Resume] Resuming from latest.ckpt (mid-epoch)"
    srun python train.py resume "$CKPT_DIR/latest.ckpt" &
else
    echo "[Train] Starting from scratch"
    srun python train.py train &
fi

wait $!  # wait for training; USR1 trap can fire here

# ── Post-completion check (only reached if training finished before time limit) ─
# If the trap already fired we already resubmitted — don't do it twice.
# Read the last completed epoch from the checkpoint.
DONE_EPOCH=$(python3 -c "
import torch, os, sys
p = os.path.join('$CKPT_DIR', 'last_epoch.ckpt')
if os.path.exists(p):
    ckpt = torch.load(p, map_location='cpu', weights_only=False)
    print(ckpt.get('epoch', -1))
else:
    print(-1)
" 2>/dev/null || echo "-1")

TARGET_EPOCH=$(( EPOCHS - 1 ))   # epochs are 0-indexed in checkpoint

if [ $_RESUBMITTED -eq 0 ] && [ "$DONE_EPOCH" -lt "$TARGET_EPOCH" ]; then
    echo "[Chain] Completed epoch $DONE_EPOCH / target $TARGET_EPOCH — resubmitting"
    _resubmit
else
    echo "[Chain] All $EPOCHS epochs complete (last epoch saved: $DONE_EPOCH). Done."
fi
