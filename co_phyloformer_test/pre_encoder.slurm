#!/bin/bash
#SBATCH --job-name=preencode
#SBATCH --partition=cpu_p1
#SBATCH --account=vcu@cpu
#SBATCH --time=100:00:00
#SBATCH --qos=qos_cpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --output=preencode_%j.out
#SBATCH --error=preencode_%j.err  


module purge
module load r/4.4.1
module load miniforge/24.9.0
conda activate cophylo

python pre_encoder.py
