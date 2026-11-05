#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 1:30:00
#SBATCH --gpus=v100-16:1
#SBATCH -J ssis_setup
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
cd $PROJECT/rl-lighting-smoke
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
bash smoke/env/setup_ssis.sh
