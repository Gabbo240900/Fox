#!/bin/bash
#SBATCH --job-name=simulate_trees
#SBATCH --partition=cpu_p1
#SBATCH --account=vcu@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=100:00:00 
#SBATCH --output=alisim%j.out 
#SBATCH --error=alisim%j.err  


set -euo pipefail
module purge
module load r/4.4.1
export R_LIBS_USER=$ALL_CCFRWORK/r_envs/treeducken_env/lib
export MPLCONFIGDIR=$ALL_CCFRWORK/tmp/mpl_cache
export TRANSFORMERS_CACHE=$ALL_CCFRWORK/tmp/transformers_cache

module load miniforge/24.9.0
conda activate cophylo

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
export VECLIB_MAXIMUM_THREADS=$OMP_NUM_THREADS
export BLIS_NUM_THREADS=$OMP_NUM_THREADS


# ECHO
set -x 

python /lustre/fswork/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/alisim.py /lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/Dataset_final \
  --substitution LG \
  --gamma GC \
  --iqtree /lustre/fswork/projects/rech/vcu/commun/Co-Phyloformer/bin/bin_linux/iqtree_2.2.0 \
  --length 500 \
  --max-attempts 1 \
  --allow-duplicate-sequences \
  --n_cores ${SLURM_CPUS_PER_TASK} \
  --temp-dir /lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/alisim_tmp

