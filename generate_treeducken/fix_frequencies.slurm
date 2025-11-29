#!/bin/bash
#SBATCH --job-name=fixfreq
#SBATCH --account=vcu@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --output=fixfreq%j.out 
#SBATCH --error=fixfreq%j.err  
#SBATCH --time=20:00:00


module purge
module load miniforge/24.9.0
conda activate cophylo

python fix_frequencies.py