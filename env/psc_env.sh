# Sourced by every setup/job script: keep ALL caches and envs in the personal $PROJECT dir,
# nothing in $HOME (limited quota/inodes) and nothing in the shared group dir.
export CONDA_PKGS_DIRS=$PROJECT/.conda_pkgs
export PIP_CACHE_DIR=$PROJECT/.pip_cache
export XDG_CACHE_HOME=$PROJECT/.cache
export TORCH_HOME=$PROJECT/.cache/torch
export MPLCONFIGDIR=$PROJECT/.cache/matplotlib
export HF_HOME=$PROJECT/hf                              # model weights
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token     # token from `huggingface-cli login`
mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR $XDG_CACHE_HOME $HF_HOME
export SD35M_PATH=$PROJECT/models/sd35m                 # SD3.5-M @ b940f67, relayed (env/relay_sd35m.py)
export PYTHONNOUSERSITE=1                               # ignore ~/.local user packages (has an unrelated torch)
