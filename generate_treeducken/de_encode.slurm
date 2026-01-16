#!/bin/bash
#SBATCH --job-name=de_encode
#SBATCH --partition=cpu_p1
#SBATCH --account=vcu@cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00 
#SBATCH --output=de_encode%j.out
#SBATCH --error=de_encode%j.err

set -x
module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
module load miniforge/24.9.0
conda activate cophylo

python pt_to_tgl.py \
  --in_dir /lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/extreme_cosp_data/cosp_0 \
  --out_dir /lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/extreme_cosp_data_tgl/cosp_0

python pt_to_tgl.py \
  --in_dir /lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/extreme_cosp_data/cosp_1 \
  --out_dir /lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/extreme_cosp_data_tgl/cosp_1