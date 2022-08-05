#!/bin/bash

#SBATCH -J tf_benchmarking
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:45:00
#SBATCH --mem=32GB
#SBATCH -o tf_no_gpu_%j.o
#SBATCH -e tf_no_gpu_%j.e

# Print key runtime properties for records
echo Master process running on `hostname`
echo Directory is `pwd`
echo Starting execution at `date`
echo Current PATH is $PATH

export SINGULARITY_BINDPATH="$HOME/scratch"

CONTAINER=$HOME/nvidia_tf_2_6_0_py3.simg
SCRIPT=$HOME/custom-benchmark.py

singularity exec --nv $CONTAINER python3 $SCRIPT --num_epochs=3 --num_data=1000
