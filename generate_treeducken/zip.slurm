#!/bin/bash
#SBATCH --job-name=zipper
#SBATCH --account=vcu@cpu
#SBATCH --output=zipper_%j.out
#SBATCH --error=zipper_%j.err
#SBATCH --ntasks=1
#SBATCH --time=05:00:00  
#SBATCH --partition=prepost 

cd $ALL_CCFRSCRATCH/Co-Phyloformer/generate_treeducken/generated_trees/



zip -r preencoded_pt.zip preencoded_pt && mv preencoded_pt.zip $ALL_CCFRWORK/Co-Phyloformer/generate_treeducken/generated_trees/
