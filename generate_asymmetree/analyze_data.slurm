#!/bin/bash
#SBATCH --job-name=data_analysis
#SBATCH --partition=cpu_p1
#SBATCH --account=vcu@cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00 
#SBATCH --output=data_analysis%j.out
#SBATCH --error=data_analysis%j.err

set -x
module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
module load miniforge/24.9.0
conda activate cophylo

python analyze_data.py /lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_asymmetree/generated_trees/Datasets\
  --exts .tgl --out_dir plots/
