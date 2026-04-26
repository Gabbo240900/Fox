#!/bin/bash
#SBATCH --job-name=filter_data
#SBATCH --partition=cpu_p1
#SBATCH --account=vcu@cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00 
#SBATCH --output=filter_data%j.out
#SBATCH --error=filter_data%j.err

set -x
module purge
module load arch/h100
module load pytorch-gpu/py3/2.6.0
source $WORK/envs/cophylo_pt26/bin/activate

python filter_data.py /lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_asymmetree/generated_trees/asym_preencoded --min_taxa 5 \
  --max_taxa 142

