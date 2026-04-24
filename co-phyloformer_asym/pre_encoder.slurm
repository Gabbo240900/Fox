#!/bin/bash
#SBATCH --job-name=preencode
#SBATCH --partition=cpu_p1
#SBATCH --account=vcu@cpu
#SBATCH --time=100:00:00
#SBATCH --qos=qos_cpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --output=preencode_%j.out
#SBATCH --error=preencode_%j.err  


module purge

module load arch/h100

module load pytorch-gpu/py3/2.6.0

source $WORK/envs/cophylo_pt26/bin/activate


python pre_encoder.py
