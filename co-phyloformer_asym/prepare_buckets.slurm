#!/bin/bash
#SBATCH --job-name=prepare_buckets
#SBATCH --partition=cpu_p1
#SBATCH --account=vcu@cpu
#SBATCH --time=08:00:00
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --output=prepare_buckets_%j.out
#SBATCH --error=prepare_buckets_%j.err

module purge

module load arch/h100
module load pytorch-gpu/py3/2.6.0

source $WORK/envs/cophylo_pt26/bin/activate

python prepare_buckets.py
