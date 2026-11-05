"""Relay SD3.5-Medium weights to PSC through a laptop.

Why: outbound bandwidth from the Bridges-2 login nodes measured ~0.5 MB/s (Nov 2026), i.e. hours
for 16 GB, while laptop download (~25 MB/s) + scp to the data-transfer node (~11 MB/s) takes
~35 min. Files are downloaded one at a time, uploaded, then deleted, so peak local disk use is
one file (<= 5 GB).

Needs a local HF login with SD3.5-M access. Only the files diffusers needs are copied
(non-variant text encoders, no fp8/single-file checkpoints).

Usage: python relay_sd35m.py --tmp /some/scratch/dir \
          --dest asharmai@data.bridges2.psc.edu:/ocean/projects/cis260267p/asharmai/models/sd35m
On PSC: StableDiffusion3Pipeline.from_pretrained("$PROJECT/models/sd35m")
"""
import argparse
import subprocess
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "stabilityai/stable-diffusion-3.5-medium"
REVISION = "b940f670f0eda2d07fbb75229e779da1ad11eb80"
FILES = [
    "model_index.json", "scheduler/scheduler_config.json",
    "text_encoder/config.json", "text_encoder/model.safetensors",
    "text_encoder_2/config.json", "text_encoder_2/model.safetensors",
    "text_encoder_3/config.json", "text_encoder_3/model.safetensors.index.json",
    "text_encoder_3/model-00001-of-00002.safetensors", "text_encoder_3/model-00002-of-00002.safetensors",
    "tokenizer/merges.txt", "tokenizer/special_tokens_map.json", "tokenizer/tokenizer_config.json", "tokenizer/vocab.json",
    "tokenizer_2/merges.txt", "tokenizer_2/special_tokens_map.json", "tokenizer_2/tokenizer_config.json", "tokenizer_2/vocab.json",
    "tokenizer_3/special_tokens_map.json", "tokenizer_3/spiece.model", "tokenizer_3/tokenizer.json", "tokenizer_3/tokenizer_config.json",
    "transformer/config.json", "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp", required=True, type=Path)
    ap.add_argument("--dest", required=True, help="user@data.bridges2.psc.edu:/abs/path")
    args = ap.parse_args()
    host, root = args.dest.split(":", 1)
    login = host.split("@")[0] + "@bridges2.psc.edu"
    dirs = sorted({str(Path(root) / Path(f).parent) for f in FILES})
    subprocess.run(["ssh", login, "mkdir -p " + " ".join(dirs)], check=True)

    for f in FILES:
        t = time.time()
        local = Path(hf_hub_download(REPO, f, revision=REVISION, local_dir=args.tmp))
        t_dl = time.time() - t
        subprocess.run(["scp", "-q", str(local), f"{host}:{root}/{f}"], check=True)
        size = local.stat().st_size
        local.unlink()
        print(f"{f}: {size / 1e9:.3f} GB  download {t_dl:.0f}s  total {time.time() - t:.0f}s", flush=True)
    # verify sizes on the remote side
    out = subprocess.run(["ssh", login, f"cd {root} && find . -type f -printf '%s %p\\n' | sort -k2"],
                         check=True, capture_output=True, text=True).stdout
    print(out)
    print("RELAY DONE")


if __name__ == "__main__":
    main()
