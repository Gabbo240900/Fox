#!/bin/bash
#SBATCH --account=vcu@cpu
#SBATCH --job-name=wandb_sync
#SBATCH --time=20:00:00
#SBATCH --partition=prepost
#SBATCH --output=error_files/wandb_sync%j.log

module purge
module load arch/h100
module load pytorch-gpu/py3/2.6.0
source $WORK/envs/cophylo_pt26/bin/activate

# Sync while any training job (dev/20H/100H) is still running
while squeue -u $USER | grep -qE "dev_train|20H_train|100H_train"; do
    wandb sync --sync-all wandb/offline-run-* >/dev/null 2>&1
    sleep 60
done

# Final sync after job finishes
wandb sync --sync-all wandb/offline-run-*
