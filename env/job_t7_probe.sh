#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 0:30:00
#SBATCH --gpus=v100-32:1
#SBATCH -J t7_probe
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
# T7 memory probe: 3 Flow-GRPO epochs (2 groups x 4, 10 SDE steps, batch 1, ckpt on, random reward) at 512px;
# only if 512 OOMs, retry at 384px.
set -ex
cd $PROJECT/rl-lighting-smoke
source smoke/env/psc_env.sh
module load anaconda3/2024.10-1
eval "$(conda shell.bash hook)"; conda activate $PROJECT/envs/gen
nvidia-smi --query-gpu=name,memory.total --format=csv
O=smoke/results/T7
python smoke/gen/flowgrpo_probe.py --out $O/probe_512 --resolution 512 --group 4 --epochs 3
if grep -q '"status": "oom"' $O/probe_512/probe.json; then
  python smoke/gen/flowgrpo_probe.py --out $O/probe_384 --resolution 384 --group 4 --epochs 3
fi
