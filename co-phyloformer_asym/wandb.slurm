#!/bin/bash
#SBATCH --account=vcu@cpu
#SBATCH --job-name=wandb_sync
#SBATCH --time=20:00:00
#SBATCH --partition=prepost
#SBATCH --output=error_files/wandb_sync%j.log

module purge
module load miniforge/24.9.0
conda activate cophylo

# Sync while any training job (dev/20H/100H) is still running
while squeue -u $USER | grep -qE "dev_trai|20H_trai|100H_tra"; do
    wandb sync --sync-all wandb/offline-run-* >/dev/null 2>&1
    sleep 60
done

# Final sync after job finishes
wandb sync --sync-all wandb/offline-run-*
