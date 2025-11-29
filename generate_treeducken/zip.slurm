#!/bin/bash
#SBATCH --job-name=zipper
#SBATCH --account=vcu@cpu
#SBATCH --output=zipper_%j.out
#SBATCH --error=zipper_%j.err
#SBATCH --ntasks=1
#SBATCH --time=05:00:00  
#SBATCH --partition=prepost 

cd $ALL_CCFRSCRATCH/Co-Phyloformer/generate_treeducken/generated_trees/



zip -r Dataset_final.zip Dataset_final && mv Dataset_final.zip $ALL_CCFRWORK/Co-Phyloformer/generate_treeducken/generated_trees/
