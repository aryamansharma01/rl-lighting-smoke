"""T1: generate base-model images with SD3.5-Medium (diffusers).

Settings follow Flow-GRPO's SD3.5-M configs (config/grpo.py, commit 879042c):
  resolution 512, guidance_scale 4.5, eval_num_steps 40 (full), num_steps 10 (train/reduced),
  mixed_precision fp16 (config/base.py).

Each image is saved as PNG with a JSON sidecar; a manifest.jsonl lists all images.
Seeds: seed k (0..N-1) for every prompt, so left/right versions of the same
object/scene share initial noise (common random numbers).

Usage:
  python generate.py --prompts ../prompts/t1_prompts.jsonl --out ../results/T1 \
      --steps 40 --seeds 8 [--limit-prompts 2]
"""
import argparse
import json
import os
import time
import sys
from pathlib import Path

import torch
from diffusers import StableDiffusion3Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "env"))
import fast_mode  # noqa: E402  (runtime switch, see env/fast_mode.py)

MODEL = "stabilityai/stable-diffusion-3.5-medium"
REVISION = "b940f670f0eda2d07fbb75229e779da1ad11eb80"
# local copy relayed by env/relay_sd35m.py (same revision); falls back to the Hub id
MODEL_PATH = os.environ.get("SD35M_PATH", MODEL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--batch", type=int, default=8, help="images per pipeline call")
    ap.add_argument("--limit-prompts", type=int, default=0)
    args = ap.parse_args()
    fast_mode.check_steps(args.steps)
    if fast_mode.cfg().get("seeds"):
        args.seeds = min(args.seeds, fast_mode.cfg()["seeds"])

    dtype = dict(fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32)[args.dtype]
    prompts = [json.loads(l) for l in open(args.prompts)]
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]

    out = Path(args.out)
    img_dir = out / f"images_{args.steps}steps"
    img_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    pipe = StableDiffusion3Pipeline.from_pretrained(MODEL_PATH, torch_dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    load_s = time.time() - t0

    # Encode every prompt once with the pipeline's own encode_prompt (CFG negative = ""), then
    # drop the three text encoders (~11 GB fp16) so denoising fits on a 16 GB V100. This is
    # numerically the same computation the pipeline does internally per call.
    t = time.time()
    for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        getattr(pipe, name).to("cuda")
    embeds = {}
    with torch.no_grad():
        for p in prompts:
            pe, npe, ppe, nppe = pipe.encode_prompt(prompt=p["prompt"], prompt_2=None, prompt_3=None,
                                                    device="cuda", do_classifier_free_guidance=True)
            embeds[p["prompt_id"]] = (pe, npe, ppe, nppe)
    text_s = time.time() - t
    pipe.text_encoder = pipe.text_encoder_2 = pipe.text_encoder_3 = None
    torch.cuda.empty_cache()
    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")
    torch.cuda.reset_peak_memory_stats()

    def run(pid, n, steps, gens=None):
        pe, npe, ppe, nppe = embeds[pid]
        rep = lambda x: x.repeat(n, *([1] * (x.dim() - 1)))
        return pipe(prompt_embeds=rep(pe), negative_prompt_embeds=rep(npe),
                    pooled_prompt_embeds=rep(ppe), negative_pooled_prompt_embeds=rep(nppe),
                    num_inference_steps=steps, guidance_scale=args.guidance,
                    height=args.resolution, width=args.resolution, generator=gens).images

    manifest = open(out / f"manifest_{args.steps}steps.jsonl", "w")
    timings, n_bad = [], 0
    run(prompts[0]["prompt_id"], 1, 2)  # warm-up (excluded from timing)

    for p in prompts:
        seeds = list(range(args.seeds))
        for b in range(0, len(seeds), args.batch):
            chunk = seeds[b: b + args.batch]
            gens = [torch.Generator("cuda").manual_seed(s) for s in chunk]
            torch.cuda.synchronize()
            t = time.time()
            images = run(p["prompt_id"], len(chunk), args.steps, gens)
            torch.cuda.synchronize()
            timings.append(((time.time() - t), len(chunk)))
            for s, im in zip(chunk, images):
                # fp16 overflow shows up as an all-black image
                extrema = im.convert("L").getextrema()
                bad = extrema[1] < 5
                n_bad += bad
                name = f"{p['prompt_id']}_s{s}_n{args.steps}"
                im.save(img_dir / f"{name}.png")
                meta = dict(image=str((img_dir / f"{name}.png").relative_to(out)), image_id=name,
                            seed=s, steps=args.steps, resolution=args.resolution,
                            guidance=args.guidance, dtype=args.dtype, model=MODEL, revision=REVISION,
                            black_image=bool(bad), **p)
                (img_dir / f"{name}.json").write_text(json.dumps(meta, indent=1))
                manifest.write(json.dumps(meta) + "\n")
        print(f"{p['prompt_id']} done ({time.time() - t0:.0f}s elapsed)", flush=True)
    manifest.close()

    tot_s = sum(t for t, _ in timings)
    tot_n = sum(n for _, n in timings)
    stats = dict(steps=args.steps, resolution=args.resolution, dtype=args.dtype,
                 batch=args.batch, n_images=tot_n, sec_per_image=tot_s / tot_n,
                 model_load_s=load_s, text_encode_s_all_prompts=text_s,
                 peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9,
                 gpu=torch.cuda.get_device_name(0), black_images=n_bad,
                 torch=torch.__version__)
    (out / f"gen_stats_{args.steps}steps.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
