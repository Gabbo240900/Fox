#!/bin/bash

#SBATCH --job-name=post_process
#SBATCH --partition=cpu_p1
#SBATCH --account=vcu@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --output=error_files/post_process%j.out
#SBATCH --error=error_files/post_process%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00

# CPU post-processing job for latest checkpoint.

module purge

module load miniforge/24.9.0
conda activate cophylo

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

export XDG_CACHE_HOME="${ALL_CCFRWORK:-$HOME}/.cache"
export MPLCONFIGDIR="${ALL_CCFRWORK:-$HOME}/.cache/mpl"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" error_files

# Defaults can be overridden at submit time:
# sbatch --export=ALL,CKPT_ROOT=checkpoints,TRAIN_PREENCODED_DIR=...,VAL_PREENCODED_DIR=... post_process.slurm
CKPT_ROOT="${CKPT_ROOT:-checkpoints}"
TRAIN_PREENCODED_DIR="${TRAIN_PREENCODED_DIR:-/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_asymmetree/generated_trees/train_preencoded}"
VAL_PREENCODED_DIR="${VAL_PREENCODED_DIR:-/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_asymmetree/generated_trees/val_preencoded}"
OUTPUT_ROOT="${OUTPUT_ROOT:-post_process_outputs}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
HOST_MAX_LEAVES="${HOST_MAX_LEAVES:-50}"
PARA_MAX_LEAVES="${PARA_MAX_LEAVES:-128}"
TRAIN_SCATTER_MAX="${TRAIN_SCATTER_MAX:-2000}"

set -x

srun python post_process.py \
  --ckpt-root "$CKPT_ROOT" \
  --train-dir "$TRAIN_PREENCODED_DIR" \
  --val-dir "$VAL_PREENCODED_DIR" \
  --output-root "$OUTPUT_ROOT" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --host-max-leaves "$HOST_MAX_LEAVES" \
  --para-max-leaves "$PARA_MAX_LEAVES" \
  --train-scatter-max "$TRAIN_SCATTER_MAX"
