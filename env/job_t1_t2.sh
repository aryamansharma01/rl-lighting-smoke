#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 1:30:00
#SBATCH --gpus=v100-16:1
#SBATCH -J t1_t2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
# T1 (generation) then T2 (SSIS verifier) in one allocation to avoid a second queue wait.
set -e
bash $PROJECT/rl-lighting-smoke/smoke/env/job_t1.sh
bash $PROJECT/rl-lighting-smoke/smoke/env/job_t2.sh
