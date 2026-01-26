#!/bin/bash
#SBATCH --job-name=tree_analysis
#SBATCH --partition=cpu_p1
#SBATCH --account=vcu@cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00 
#SBATCH --output=tree_analysis%j.out 
#SBATCH --error=tree_analysis%j.err  

set -x
module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
module load miniforge/24.9.0
conda activate cophylo

python analyze_trees.py /lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/Dataset_final \
  --workers 20 \
  --csv_out /lustre/fswork/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/trees_analysis_new/analysis_trees_new.csv\
  --out_dir  /lustre/fswork/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/trees_analysis_new/ \
  --extract_extreme_cosp



