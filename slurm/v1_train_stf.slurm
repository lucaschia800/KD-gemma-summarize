#!/bin/bash

#SBATCH --job-name=v1_mistral
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lbc800@uw.edu

#SBATCH --account=stf
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=150G
#SBATCH --gpus=l40s:2
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00 

#SBATCH --chdir=/gscratch/stf/lbc800
#SBATCH --export=all
#SBATCH --output=/mistral-KD/output-logs/%x_%j.out

# Modules to use (optional).
# Your programs to run.


apptainer exec --nv --cleanenv --bind /gscratch /gscratch/stf/lbc800/containers/KD-Container-v1.sif deepspeed /gscratch/stf/lbc800/mistral-KD/start.py
