#!/bin/bash
# Generation env for SD3.5-Medium (diffusers) on PSC. Torch version per the course PSC guide.
#   bash smoke/env/setup_gen.sh      (login node is fine: pure pip installs)
set -euxo pipefail
source $PROJECT/rl-lighting-smoke/smoke/env/psc_env.sh
ROOT=$PROJECT/rl-lighting-smoke
ENV=$PROJECT/envs/gen
module load anaconda3/2024.10-1
eval "$(conda shell.bash hook)"
[ -d "$ENV" ] || conda create -y -p "$ENV" python=3.11
conda activate "$ENV"
# big wheels pre-staged in $PROJECT/wheels/gen (login-node bandwidth ~0.5 MB/s; see relay notes in README)
pip install --no-index --find-links $PROJECT/wheels/gen torch==2.5.1+cu124 torchvision==0.20.1+cu124
pip install "diffusers==0.31.0" "transformers==4.46.3" "accelerate==1.1.1" sentencepiece protobuf \
    "huggingface_hub<1.0" pillow matplotlib lpips pandas scipy
pip freeze > "$ROOT/smoke/env/gen_pip_freeze.txt"
python -c "import torch, diffusers, transformers; print(torch.__version__, diffusers.__version__, transformers.__version__)"
