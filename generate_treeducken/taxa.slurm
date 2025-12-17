#!/bin/bash
#SBATCH --job-name=taxa_stats
#SBATCH --partition=cpu_p1
#SBATCH --account=vcu@cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00 
#SBATCH --output=taxa_stats%j.out 
#SBATCH --error=taxa_stats%j.err  


set -x
module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
module load miniforge/24.9.0
conda activate cophylo

python taxa.py /lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/Datasets \
  --workers 20 \
  --csv_out taxa_summary.csv \
  --outliers_csv taxa_outliers_p99.csv \
  --remove




