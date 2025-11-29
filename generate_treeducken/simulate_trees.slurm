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
#SBATCH --output=simulate_trees%j.out 
#SBATCH --error=simulate_trees%j.err  


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

# STEP 0: remove former datasets
rm -rf /lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/associations/*
rm -rf /lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/Datasets/*
rm -rf /lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/host_trees/*
rm -rf /lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/summaries/*
rm -rf /lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/symb_tree/*
# STEP 1: generate host trees
python /lustre/fswork/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/simulate_input_files.py \
  --h_lambda 0.5 1.2 \
  --c_lambda 0.7 1.6 \
  --s_lambda 0.5 1.2 \
  --s_her 0.05 0.3 \
  --num_trees 2000 \
  --n_cores ${SLURM_CPUS_PER_TASK}

# STEP 2: Simulate alignments 
python alisim.py /lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/Datasets \
  --substitution LG \
  --gamma GC \
  --iqtree /lustre/fswork/projects/rech/vcu/commun/Co-Phyloformer/bin/bin_linux/iqtree_2.2.0 \
  --length 500 \
  --max-attempts 1 \
  --allow-duplicate-sequences \
  --n_cores ${SLURM_CPUS_PER_TASK} \
  --temp-dir ../alisim_tmp


# STEP 3: move and rename generated datasets
python /lustre/fswork/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/dataset_renaming.py \
  --src "/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/Datasets" \
  --dst "/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/validation_datasates"
