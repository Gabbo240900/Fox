#!/bin/bash

#SBATCH --job-name=post_process
#SBATCH --partition=cpu_p1
#SBATCH --account=vcu@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --output=error_files/post_process%j.out
#SBATCH --error=error_files/post_process%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00

# Plot-only post-processing: reads CSVs written by train.py, no inference.

module purge

module load arch/h100
module load pytorch-gpu/py3/2.6.0
module load miniforge/24.9.0
source $WORK/envs/cophylo_pt26/bin/activate

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

export XDG_CACHE_HOME="${ALL_CCFRWORK:-$HOME}/.cache"
export MPLCONFIGDIR="${ALL_CCFRWORK:-$HOME}/.cache/mpl"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" error_files

# Defaults can be overridden at submit time:
# sbatch --export=ALL,CKPT_DIR=checkpoints,OUTPUT_DIR=plots post_process.slurm
CKPT_DIR="${CKPT_DIR:-checkpoints_fox}"
OUTPUT_DIR="${OUTPUT_DIR:-post_process_outputs}"
TRAIN_CSV="${TRAIN_CSV:-${CKPT_DIR}/train_predictions.csv}"
VAL_CSV="${VAL_CSV:-${CKPT_DIR}/val_predictions.csv}"

set -x

srun python post_process.py \
  --output-dir "$OUTPUT_DIR" \
  --train-csv  "$TRAIN_CSV" \
  --val-csv    "$VAL_CSV"
