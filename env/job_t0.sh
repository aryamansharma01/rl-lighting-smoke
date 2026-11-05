#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 0:30:00
#SBATCH --gpus=v100-16:1
#SBATCH -J t0_ssis_demo
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
# T0: run SSISv2 on the repo's demo images (and mirrors), draw overlays, check class semantics.
bash $PROJECT/rl-lighting-smoke/smoke/env/run_t0.sh
