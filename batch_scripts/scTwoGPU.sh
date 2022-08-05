#!/bin/bash

#SBATCH -J tf_benchmarking
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:30:00
#SBATCH --mem=32GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH -o tf_two_geforce3090_%j.o
#SBATCH -e tf_two_geforce3090_%j.e
#SBATCH --constraint=geforce3090


# Print key runtime properties for records
echo Master process running on `hostname`
echo Directory is `pwd`
echo Starting execution at `date`
echo Current PATH is $PATH

export SINGULARITY_BINDPATH="$HOME/scratch"

CONTAINER=$HOME/nvidia_tf_2_6_0_py3.simg
SCRIPT=$HOME/custom-benchmark.py

# Run The Job Through Singularity
singularity exec --nv $CONTAINER python3 $SCRIPT --num_epochs=3 --num_data=1000 --has_two_gpu
