#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 1:00:00
#SBATCH --gpus=v100-16:1
#SBATCH -J t2_verify
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
# T2: SSIS on T1 images (+ mirrors for the 40-step set), then direction scoring + flip test.
set -ex
cd $PROJECT/rl-lighting-smoke
source smoke/env/psc_env.sh
module load anaconda3/2024.10-1 cuda/11.7.1
eval "$(conda shell.bash hook)"; conda activate $PROJECT/envs/ssis
S=third_party/SSIS; R=smoke/results
python smoke/verifier/ssis_detect.py --ssis-root $S --images $R/T1/images_40steps --out $R/T2/raw_40steps --flip
python smoke/verifier/ssis_detect.py --ssis-root $S --images $R/T1/images_10steps --out $R/T2/raw_10steps
for n in 40 10; do
  python smoke/verifier/direction.py --raw $R/T2/raw_${n}steps --manifest $R/T1/manifest_${n}steps.jsonl --out $R/T2/scores_${n}steps.jsonl
done
python smoke/analysis/t2_flip.py --scores $R/T2/scores_40steps.jsonl --out $R/T2/t2_summary.json
python smoke/analysis/overlay.py --images $R/T1/images_40steps --raw $R/T2/raw_40steps --out $R/T2/overlays_40steps --manifest $R/T1/manifest_40steps.jsonl
# offline sweep of gate/length/area cutoffs from the saved per-pair data (labels added later in T4)
python smoke/analysis/gate_sweep.py --scores $R/T2/scores_40steps.jsonl --out $R/T2/gate_sweep
