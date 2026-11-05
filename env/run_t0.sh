#!/bin/bash
# T0 body: SSISv2 on the repo's demo images (+ mirrors), class-semantics check, overlays.
# Called from job_t0.sh and at the end of setup_ssis.sh (saves one queue wait).
set -ex
cd $PROJECT/rl-lighting-smoke
source smoke/env/psc_env.sh
module load anaconda3/2024.10-1 cuda/11.7.1
eval "$(conda shell.bash hook)"; conda activate $PROJECT/envs/ssis
S=third_party/SSIS
python smoke/verifier/ssis_detect.py --ssis-root $S --images $S/demo/samples --out smoke/results/T0/raw --flip
python smoke/analysis/t0_check.py --images $S/demo/samples --raw smoke/results/T0/raw --out smoke/results/T0/t0_summary.json
python smoke/analysis/overlay.py --images $S/demo/samples --raw smoke/results/T0/raw --out smoke/results/T0/overlays --conf-th 0.3
