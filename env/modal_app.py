"""Run T0, T1, T2, the T3 pages and the T7 memory probe on Modal instead of PSC (no queue).

Same scripts, prompts, seeds and settings as the PSC jobs: the repo is mounted at /root/rl with
smoke/results pointing into a persistent Modal volume, so every command matches the job_*.sh files.

GPUs: T1 on L40S, SSIS (T0/T2) on A10G (sm_86, compiled natively against CUDA 11.7),
T7 probe on A100-40GB with usable memory capped at 31.7 GB to emulate a V100-32.
Throughput numbers from these GPUs are not V100 numbers (noted in the report).

One-time setup: `modal token new`; a Modal secret named `huggingface` holding HF_TOKEN.
Run (from smoke/env):   modal run --detach modal_app.py            # whole pipeline, survives laptop sleep
Fetch results:          modal volume get rl-lighting-smoke results ../results_modal
"""
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
SMOKE = HERE.parent
REPO = "/root/rl"
VOL = "/vol"
SD35M_REV = "b940f670f0eda2d07fbb75229e779da1ad11eb80"
SSIS_COMMIT = "9a1cac2"
D2_COMMIT = "247691a1c292654867deb35cabb27f18d75a1363"
CKPT_SUB = {"model_ssis_final.pth": "tools/output/SSIS_MS_R_101_bifpn_with_offset_class",
            "model_ssisv2_final.pth": "tools/output/SSISv2_MS_R_101_bifpn_with_offset_class_maskiouv2_da_bl"}

app = modal.App("rl-lighting-smoke")
vol = modal.Volume.from_name("rl-lighting-smoke", create_if_missing=True)
IGNORE = ["results", "results_modal", "**/__pycache__", "**/.DS_Store", "label/*.html", "label/*.csv",
          "label/selection.json"]

gen_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("diffusers==0.31.0", "transformers==4.46.3", "accelerate==1.1.1", "sentencepiece", "protobuf",
                 "huggingface_hub<1.0", "pillow", "matplotlib", "pandas", "scipy", "peft==0.13.2")
    .add_local_dir(SMOKE, f"{REPO}/smoke", ignore=IGNORE)
)

ssis_image = (
    # Modal no longer supports Python 3.9 (PSC env), so: Python 3.11 + torch 2.0.1+cu117, the oldest torch with
    # cp311 wheels, same CUDA 11.7 toolkit; numpy 1.23.5 keeps the deprecated aliases old detectron2/AdelaiDet use.
    modal.Image.from_registry("nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04", add_python="3.11")
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})  # tzdata otherwise waits for input
    .apt_install("git", "build-essential", "ninja-build", "libgl1", "libglib2.0-0")
    # CC/CXX: Modal's Python is built with clang (torch would look for clang++); its -march=x86-64-v3 needs gcc >= 11 -> ubuntu22.04
    .env({"FORCE_CUDA": "1", "TORCH_CUDA_ARCH_LIST": "8.0;8.6+PTX", "MAX_JOBS": "8",
          "CUDA_HOME": "/usr/local/cuda", "CC": "gcc", "CXX": "g++"})
    .pip_install("numpy==1.23.5", "Pillow<10", "setuptools<70", "wheel")
    .pip_install("torch==2.0.1+cu117", "torchvision==0.15.2+cu117",
                 extra_index_url="https://download.pytorch.org/whl/cu117")
    .pip_install("numpy==1.23.5", "cython", "opencv-python-headless", "scikit-image", "kornia==0.5.6", "pycocotools",
                 "gdown", "tqdm", "matplotlib", "Pillow<10")
    # separate layers so a failure in a later step doesn't recompile detectron2
    .run_commands(f"pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git@{D2_COMMIT}'")
    # pysobatools: its _mask.pyx has '# distutils: sources = ../common/maskApi.c' (wrong relative to the build dir;
    # setup.py already lists ./common/maskApi.c), so drop it; also drop the stale py3.6 build artifacts
    .run_commands("git clone https://github.com/stevewongv/InstanceShadowDetection /opt/ISD && cd /opt/ISD && git checkout 3e63c18"
                  " && cd /opt/ISD/PythonAPI && sed -i '/distutils: sources/d' pysobatools/_mask.pyx"
                  " && rm -f pysobatools/*.so pysobatools/_mask.c && pip install --no-build-isolation .")
    # SSIS's setup.py install_requires, via pip with numpy held at 1.23.5 (easy_install would fetch numpy 2.x source)
    .pip_install("numpy==1.23.5", "termcolor>=1.1", "yacs>=0.1.6", "tabulate", "cloudpickle", "tensorboard",
                 "python-Levenshtein", "Polygon3", "shapely")
    .run_commands(f"git clone https://github.com/stevewongv/SSIS /opt/SSIS && cd /opt/SSIS && git checkout {SSIS_COMMIT}"
                  " && python setup.py build develop --no-deps")
    .add_local_dir(SMOKE, f"{REPO}/smoke", ignore=IGNORE)
)


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(["bash", "-c", "set -euo pipefail; " + cmd], cwd=REPO, check=True)


def link_results():
    """smoke/results -> volume, third_party/SSIS -> /opt/SSIS (so PSC commands work unchanged)."""
    Path(f"{VOL}/results").mkdir(parents=True, exist_ok=True)
    r = Path(f"{REPO}/smoke/results")
    if not r.is_symlink():
        if r.exists():
            subprocess.run(["rm", "-rf", str(r)], check=True)
        r.symlink_to(f"{VOL}/results")
    if Path("/opt/SSIS").exists():
        Path(f"{REPO}/third_party").mkdir(exist_ok=True)
        t = Path(f"{REPO}/third_party/SSIS")
        if not t.exists():
            t.symlink_to("/opt/SSIS")
        for name, sub in CKPT_SUB.items():  # checkpoints live on the volume
            src = Path(f"{VOL}/ssis_ckpt/{name}")
            dst = Path("/opt/SSIS") / sub / name
            if src.exists() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.symlink_to(src)


@app.function(image=gen_image, volumes={VOL: vol}, secrets=[modal.Secret.from_name("huggingface")], timeout=3600)
def download_sd35m():
    import json
    from huggingface_hub import snapshot_download
    dest = f"{VOL}/models/sd35m"
    if Path(dest, "transformer/diffusion_pytorch_model.safetensors").exists():
        print("SD3.5-M already on volume")
        return
    snapshot_download("stabilityai/stable-diffusion-3.5-medium", revision=SD35M_REV, local_dir=dest,
                      allow_patterns=["model_index.json", "scheduler/*", "text_encoder/*", "text_encoder_2/*",
                                      "text_encoder_3/*", "tokenizer/*", "tokenizer_2/*", "tokenizer_3/*",
                                      "transformer/*", "vae/*"],
                      ignore_patterns=["*fp8*", "*.fp16.*"])
    sizes = {str(p.relative_to(dest)): p.stat().st_size for p in Path(dest).rglob("*") if p.is_file()}
    print(json.dumps(sizes, indent=1))
    vol.commit()


@app.function(image=gen_image, gpu="L40S", volumes={VOL: vol}, timeout=3 * 3600)
def t1():
    link_results()
    env = f"SD35M_PATH={VOL}/models/sd35m"
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
    sh(f"P=smoke/prompts/t1_prompts.jsonl; O=smoke/results/T1; "
       f"{env} python smoke/gen/generate.py --prompts $P --out $O --steps 40 --seeds 8 && "
       f"{env} python smoke/gen/generate.py --prompts $P --out $O --steps 10 --seeds 8 && "
       "python smoke/gen/contact_sheets.py --root $O --steps 40 && "
       "python smoke/gen/contact_sheets.py --root $O --steps 10")
    # T3 pages (independent of the verifier); saved to the volume next to the results
    sh("python smoke/label/make_label_pages.py --manifest smoke/results/T1/manifest_40steps.jsonl --root smoke/results/T1"
       " --out-dir smoke/results/T3/label_pages")
    sh("pip freeze > smoke/results/T1/gen_pip_freeze.txt")
    vol.commit()


@app.function(image=ssis_image, gpu="A10G", volumes={VOL: vol}, timeout=3600)
def t0():
    Path(f"{VOL}/ssis_ckpt").mkdir(parents=True, exist_ok=True)
    if not Path(f"{VOL}/ssis_ckpt/model_ssisv2_final.pth").exists():
        sh(f"python smoke/env/download_ssis_ckpt.py --ssis /tmp/ckpt && "
           f"find /tmp/ckpt -name '*.pth' -exec cp {{}} {VOL}/ssis_ckpt/ \\; && sha256sum {VOL}/ssis_ckpt/*.pth")
        vol.commit()
    link_results()
    sh("python -c \"import torch, detectron2, adet; from adet.layers import ml_nms; "
       "print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); print('adet OK')\"")
    sh("mkdir -p smoke/results/T0 && pip freeze > smoke/results/T0/ssis_pip_freeze.txt")
    sh("S=third_party/SSIS; "
       "python smoke/verifier/ssis_detect.py --ssis-root $S --images $S/demo/samples --out smoke/results/T0/raw --flip && "
       "python smoke/analysis/t0_check.py --images $S/demo/samples --raw smoke/results/T0/raw --out smoke/results/T0/t0_summary.json && "
       "python smoke/analysis/overlay.py --images $S/demo/samples --raw smoke/results/T0/raw --out smoke/results/T0/overlays --conf-th 0.3")
    vol.commit()


@app.function(image=ssis_image, gpu="A10G", volumes={VOL: vol}, timeout=2 * 3600)
def t2():
    vol.reload()
    link_results()
    sh("S=third_party/SSIS; R=smoke/results; "
       "python smoke/verifier/ssis_detect.py --ssis-root $S --images $R/T1/images_40steps --out $R/T2/raw_40steps --flip && "
       "python smoke/verifier/ssis_detect.py --ssis-root $S --images $R/T1/images_10steps --out $R/T2/raw_10steps && "
       "for n in 40 10; do python smoke/verifier/direction.py --raw $R/T2/raw_${n}steps "
       "--manifest $R/T1/manifest_${n}steps.jsonl --out $R/T2/scores_${n}steps.jsonl; done && "
       "python smoke/analysis/t2_flip.py --scores $R/T2/scores_40steps.jsonl --out $R/T2/t2_summary.json && "
       "python smoke/analysis/overlay.py --images $R/T1/images_40steps --raw $R/T2/raw_40steps "
       "--out $R/T2/overlays_40steps --manifest $R/T1/manifest_40steps.jsonl && "
       "python smoke/analysis/gate_sweep.py --scores $R/T2/scores_40steps.jsonl --out $R/T2/gate_sweep")
    vol.commit()


@app.function(image=gen_image, gpu="A100-40GB", volumes={VOL: vol}, timeout=3600)
def t7_probe():
    link_results()
    env = f"SD35M_PATH={VOL}/models/sd35m"
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
    sh(f"{env} python smoke/gen/flowgrpo_probe.py --out smoke/results/T7/probe_512 --resolution 512 "
       "--group 4 --epochs 3 --mem-cap-gb 31.7")
    if '"status": "oom"' in Path(f"{VOL}/results/T7/probe_512/probe.json").read_text():
        sh(f"{env} python smoke/gen/flowgrpo_probe.py --out smoke/results/T7/probe_384 --resolution 384 "
           "--group 4 --epochs 3 --mem-cap-gb 31.7")
    vol.commit()


@app.function(image=gen_image, timeout=6 * 3600)
def pipeline(skip_t7: bool = False):
    """Runs remotely, so it keeps going if the laptop disconnects (use `modal run --detach`)."""
    download_sd35m.remote()
    h0, h1 = t0.spawn(), t1.spawn()
    h7 = None if skip_t7 else t7_probe.spawn()
    errors = {}
    for name, h in (("t0", h0), ("t1", h1)):
        try:
            h.get()
            print(name, "OK", flush=True)
        except Exception as e:  # keep going so the other results are still produced
            errors[name] = repr(e)
            print(name, "FAILED", e, flush=True)
    if "t0" not in errors and "t1" not in errors:
        try:
            t2.remote()
            print("t2 OK", flush=True)
        except Exception as e:
            errors["t2"] = repr(e)
    if h7 is not None:
        try:
            h7.get()
            print("t7_probe OK", flush=True)
        except Exception as e:
            errors["t7_probe"] = repr(e)
    print("PIPELINE DONE", errors or "no errors", flush=True)
    return errors


@app.function(image=gen_image, volumes={VOL: vol})
def check():
    """Cheap CPU check that the mounts/symlinks work before spending GPU time."""
    link_results()
    sh("ls -la smoke && ls smoke/results/ && touch smoke/results/.write_test && ls smoke/label")
    vol.commit()


@app.local_entrypoint()
def main(step: str = "all", skip_t7: bool = False):
    fn = dict(all=lambda: pipeline.remote(skip_t7), download=download_sd35m.remote, t0=t0.remote,
              t1=t1.remote, t2=t2.remote, t7=t7_probe.remote, check=check.remote)[step]
    print(fn())
