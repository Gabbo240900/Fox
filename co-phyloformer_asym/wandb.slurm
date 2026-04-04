#!/bin/bash
#SBATCH --account=vcu@cpu
#SBATCH --job-name=wandb_sync
#SBATCH --time=20:00:00    # adjust as needed
#SBATCH --partition=prepost
#SBATCH --output=wandb_sync.log

module purge
module load miniforge/24.9.0
conda activate cophylo


while squeue -u $USER | grep -q 20H_trai; do
    wandb sync --sync-all wandb/offline-run-* >/dev/null 2>&1
    sleep 60
done

wandb sync --sync-all wandb/offline-run-*
