#!/bin/bash

#SBATCH --job-name=simulate_trees

#SBATCH --partition=prepost 

#SBATCH --output=simulate_trees%j.out 

#SBATCH --error=simulate_trees%j.err  

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1 

#SBATCH --hint=nomultithread

#SBATCH --time=10:00:00 

# WORKING DIRECTORY 
cd $WORK/Co-phyloformer/generate_host_freq/


# ENVIRONMENT SETUP

module purge

module load miniforge/24.9.0

conda activate myenv 

# ECHO

set -x 

# STEP 1: Run simulations

python simulate_input_trees.py --num_trees 500 \
	--min_leaves 15 --max_leaves 50 \
    	--output_dir ./generated_trees/ \
    	--output_dir_tgl ./generated_trees/Datasets/ \
    	--jar_path ./cophylogeny-ML/code/coala/TGLGenerator.jar


# STEP 2: ALIGN SEQUENCES

python alisim.py generated_trees/Datasets \
    --substitution LG \
    --gamma GC \
    --iqtree ../bin/bin_linux/iqtree_2.2.0 \
    --length 250 \
    --max-attempts 1 \
    --allow-duplicate-sequences


