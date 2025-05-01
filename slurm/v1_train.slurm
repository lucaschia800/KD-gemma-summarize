#!/bin/bash

#SBATCH --job-name=v1_mistral
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lbc800@uw.edu

#SBATCH --account=stf
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=15
#SBATCH --mem=70G
#SBATCH --gpus=l40:1
#SBATCH --time=10:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --chdir=/gscratch/stf/lbc800
#SBATCH --export=all
#SBATCH --output=%x_%j.out

# Modules to use (optional).
# Your programs to run.
apptainer exec --nv --cleanenv --bind /gscratch /gscratch/stf/lbc800/containers/transformers-updated.sif python3 /gscratch/stf/lbc800/python_scripts/train_CLIP.py
