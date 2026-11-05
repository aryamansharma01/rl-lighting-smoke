# Environment and commands

The results in `REPORT.md` were produced on **Modal** (no queue). The PSC Bridges-2 scripts in this
folder are kept and fixed, but no PSC job produced results (all were waiting in the queue).

## Pinned code and weights

| Component | Version |
|---|---|
| SD3.5-Medium | `stabilityai/stable-diffusion-3.5-medium` @ `b940f670f0eda2d07fbb75229e779da1ad11eb80` (gated; needs an HF token with access) |
| SSIS / SSISv2 | `github.com/stevewongv/SSIS` @ `9a1cac2` |
| SSISv2 checkpoint | `model_ssisv2_final.pth`, SHA-256 `1fed82fb4f79217ea9c730d55c8759a9fdc0c36429965a67dab09c9b1bf004c6` (Google Drive folder from the SSIS README; `download_ssis_ckpt.py`) |
| detectron2 | `facebookresearch/detectron2` @ `247691a1c292654867deb35cabb27f18d75a1363` |
| pysobatools | `stevewongv/InstanceShadowDetection` @ `3e63c18`, `PythonAPI/` (see patch below) |

## Modal images (`modal_app.py`, Modal client 1.6.3)

**Generation / T7 probe** (`gen_image`): Debian slim, Python 3.11, torch 2.5.1+cu124, torchvision 0.20.1+cu124,
diffusers 0.31.0, transformers 4.46.3, accelerate 1.1.1, peft 0.13.2, huggingface_hub 0.36.2.
Full freeze: `gen_pip_freeze_modal.txt`.

**SSIS** (`ssis_image`): `nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04` + Python 3.11, torch 2.0.1+cu117,
torchvision 0.15.2+cu117, numpy 1.23.5, Pillow 9.5.0, kornia 0.5.6, opencv-python-headless 4.11.0.86,
scikit-image 0.24.0, pycocotools 2.0.11; CUDA ops built for `TORCH_CUDA_ARCH_LIST=8.0;8.6+PTX`.
Full freeze: `ssis_pip_freeze_modal.txt`.

Build fixes needed for the SSIS image (all in `modal_app.py`; the same fixes are in `setup_ssis.sh` for PSC):
1. `DEBIAN_FRONTEND=noninteractive`, otherwise `tzdata` waits for input forever.
2. `CC=gcc CXX=g++`: Modal's Python is built with clang, so torch's extension builder looks for `clang++`.
3. Ubuntu 22.04 (gcc 11), because Modal's Python passes `-march=x86-64-v3`, which gcc 9 rejects.
4. pysobatools: delete the `# distutils: sources = ../common/maskApi.c` line from `pysobatools/_mask.pyx`
   (a wrong relative path; `setup.py` already lists the source) and the stale py3.6 `.so`/`_mask.c`.
5. SSIS: install its `install_requires` with pip first (numpy kept < 2), then `python setup.py build develop --no-deps`
   (otherwise easy_install fetches a numpy 2.x source tarball).

**Differences from the PSC plan:** Modal no longer supports Python 3.9, so the SSIS env uses Python 3.11 with
torch 2.0.1 instead of Python 3.9 with torch 1.13.1. The same checkpoint and SSIS commit are used; T0 confirmed
the classes and directions on the SSIS demo images.

**GPUs:** T1 on an L40S, SSIS (T0/T2) on an A10, T7 probe on an A100-40GB with usable memory capped at 31.7 GB
(`torch.cuda.set_per_process_memory_fraction`) to emulate a V100-32.

## Commands

One-time setup:
```bash
pip install modal && modal token new
# Modal dashboard -> Secrets -> create "huggingface" with key HF_TOKEN
modal volume put rl-lighting-smoke model_ssisv2_final.pth /ssis_ckpt/model_ssisv2_final.pth   # optional; else gdown
```

Reduced plan (what produced the report; `fast_mode.json` present):
```bash
cd smoke/env
modal run --detach modal_app.py --skip-t7      # SD3.5-M download, T0 + T1 in parallel, T3 pages, then T2
modal run --detach modal_app.py --step t7      # T7 memory probe (A100 capped at 31.7 GB)
modal volume get rl-lighting-smoke results ../results_modal       # fetch everything
cp ../results_modal/T3/label_pages/* ../label/                     # labeling pages -> annotators
```

Analysis (local, any Python with numpy/matplotlib/Pillow):
```bash
cd smoke
python analysis/fast_metrics.py --scores results_modal/T2_scores_40steps.jsonl --label-dir label --out results_modal/fast
python analysis/t6_from_probe.py --probe results_modal/T7/probe_512/probe.json \
    --ssis results_modal/T2/raw_40steps/ssis_stats.json --out results_modal/T6
python analysis/make_figures.py --results results_modal --label-dir label --out results_modal/figures
```

The full plan (192 images, 10- and 40-step sets, 30 shared labels, gate sweep): delete `env/fast_mode.json`
and run the same commands; use `t4_metrics.py`, `t5_signal.py` and `gate_sweep.py` for the extra analyses.

`fast_mode.json` (read at runtime by the scripts): seeds 0–3, 40 steps only, no gate sweep, no shared label set.

## Seeds and determinism

Seeds 0..N−1 per prompt, one `torch.Generator("cuda").manual_seed(seed)` per image, so left/right prompts of the
same object and scene share initial noise. Exact pixels depend on the GPU type: images from an L40S and a V100
with the same seed are close but not identical. Labels are tied to the images they were made from
(`label/key.csv` maps opaque item ids to image ids).

## PSC Bridges-2 (not used for results)

`psc_env.sh` keeps all caches in `$PROJECT` (private); `setup_gen.sh` and `setup_ssis.sh` build conda envs
(Python 3.11 / torch 2.5.1+cu124 and Python 3.9 / torch 1.13.1+cu117); `job_*.sh` are the Slurm jobs
(GPU-shared, v100-16, or v100-32 for the T7 probe). The login nodes' outbound bandwidth was about 0.5 MB/s, so
weights were relayed through a laptop (`relay_sd35m.py`). The allocation has no CPU partition.
