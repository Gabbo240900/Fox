#!/bin/bash
#SBATCH --account=vcu@cpu
#SBATCH --job-name=wandb_sync
#SBATCH --time=20:00:00    # adjust as needed
#SBATCH --partition=prepost   # or other lightweight partition
#SBATCH --output=wandb_sync.log

module purge
module load miniforge/24.9.0
conda activate cophylo

wandb sync --sync-all wandb/offline-run-*
