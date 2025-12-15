#!/bin/bash
#SBATCH --job-name=label_analysis
#SBATCH --partition=cpu_p1
#SBATCH --account=vcu@cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00 
#SBATCH --output=label_analysis_%j.out
#SBATCH --error=label_analysis_%j.err

set -x
module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
module load miniforge/24.9.0
conda activate cophylo

python analyze_labels.py \
  --workers 20 



