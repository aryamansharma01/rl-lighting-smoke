#!/bin/bash
# Install SSIS/SSISv2 (Wang et al., TPAMI 2022) on PSC Bridges-2.
# Run as a GPU batch job (CUDA ops compile against the node's nvcc):
#   sbatch smoke/env/job_setup_ssis.sh
# Pins: python 3.9, torch 1.13.1+cu117 (matches PSC module cuda/11.7.1 exactly),
#       detectron2 @ DETECTRON2_COMMIT, SSIS @ 9a1cac2.
set -euxo pipefail
source $PROJECT/rl-lighting-smoke/smoke/env/psc_env.sh

ROOT=$PROJECT/rl-lighting-smoke
ENV=$PROJECT/envs/ssis
SSIS=$ROOT/third_party/SSIS
DETECTRON2_COMMIT=${DETECTRON2_COMMIT:-247691a1c292654867deb35cabb27f18d75a1363}

module load anaconda3/2024.10-1 cuda/11.7.1
eval "$(conda shell.bash hook)"
[ -d "$ENV" ] || conda create -y -p "$ENV" python=3.9
conda activate "$ENV"
export TORCH_CUDA_ARCH_LIST="7.0;8.0;8.6+PTX"  # V100, A100-class, PTX for L40S/H100
export FORCE_CUDA=1
export MAX_JOBS=8

pip install "numpy<2" "Pillow<10"   # torch 1.13 is built against numpy 1.x
# big wheels pre-staged in $PROJECT/wheels/ssis
pip install --no-index --find-links $PROJECT/wheels/ssis torch==1.13.1+cu117 torchvision==0.14.1+cu117
pip install "numpy<2" cython opencv-python-headless scikit-image "kornia==0.5.6" pycocotools gdown tqdm matplotlib "Pillow<10"
pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git@${DETECTRON2_COMMIT}"
# pysobatools must be built inside its repo checkout
ISD=$ROOT/third_party/InstanceShadowDetection
[ -d "$ISD" ] || (git clone https://github.com/stevewongv/InstanceShadowDetection "$ISD" && git -C "$ISD" checkout 3e63c18)
# its _mask.pyx has "# distutils: sources = ../common/maskApi.c" (wrong relative path; setup.py already lists it)
(cd "$ISD/PythonAPI" && sed -i '/distutils: sources/d' pysobatools/_mask.pyx \
   && rm -f pysobatools/*.so pysobatools/_mask.c && pip install --no-build-isolation .)

# SSIS's setup.py install_requires via pip (numpy kept < 2), then build without easy_install deps
pip install "numpy<2" "termcolor>=1.1" "yacs>=0.1.6" tabulate cloudpickle tensorboard python-Levenshtein Polygon3 shapely
cd "$SSIS"
python setup.py build develop --no-deps

python - <<'EOF'
import torch, detectron2, adet
from detectron2 import __version__ as d2v
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
print("detectron2", d2v)
from adet.layers import ml_nms  # compiled op
print("adet import OK")
EOF
pip freeze > "$ROOT/smoke/env/ssis_pip_freeze.txt"
(cd $PROJECT && pip show detectron2 | head -3) > "$ROOT/smoke/env/ssis_detectron2_info.txt"

# --- checkpoints (Google Drive folder from the SSIS README) ---
python "$ROOT/smoke/env/download_ssis_ckpt.py" --ssis "$SSIS" || echo "ckpt download step failed (fine if checkpoints were relayed)"

# --- T0 right away (same GPU job) ---
set +u; conda deactivate; bash "$ROOT/smoke/env/run_t0.sh"
