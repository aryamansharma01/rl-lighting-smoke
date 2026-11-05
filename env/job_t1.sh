#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 1:30:00
#SBATCH --gpus=v100-16:1
#SBATCH -J t1_generate
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
# T1: SD3.5-M base images, 24 prompts x 8 seeds at 40 (full) and 10 (reduced) steps, 512px, cfg 4.5, fp16.
set -ex
cd $PROJECT/rl-lighting-smoke
source smoke/env/psc_env.sh
module load anaconda3/2024.10-1
eval "$(conda shell.bash hook)"; conda activate $PROJECT/envs/gen
nvidia-smi --query-gpu=name,memory.total --format=csv
P=smoke/prompts/t1_prompts.jsonl; O=smoke/results/T1
python smoke/gen/generate.py --prompts $P --out $O --steps 40 --seeds 8 ${GEN_DTYPE:+--dtype $GEN_DTYPE}
python smoke/gen/generate.py --prompts $P --out $O --steps 10 --seeds 8 ${GEN_DTYPE:+--dtype $GEN_DTYPE}
python smoke/gen/contact_sheets.py --root $O --steps 40
python smoke/gen/contact_sheets.py --root $O --steps 10
# T3 pages right after T1 (they don't depend on the verifier): all 192 images, 30 shared
python smoke/label/make_label_pages.py --manifest $O/manifest_40steps.jsonl --root $O
