"""T7 memory/stability probe: a few Flow-GRPO steps for SD3.5-M on one V100-32 in fp16.

It learns nothing on purpose (random rewards). It answers: does one Flow-GRPO epoch (SDE sampling with
log-probs, then the clipped-ratio policy loss + KL to the reference) fit in memory at the real
resolution, and does fp16 produce NaN/inf? It also measures the train/sample cost ratio used by T6.

Re-implemented after Flow-GRPO (github.com/yifan123/flow_grpo, SD3 recipe), not imported from it:
  * sampling: SDE step with log-prob (flow_grpo/diffusers_patch/sd3_sde_with_logprob.py), noise_level 0.7
  * LoRA r=32, alpha=64 on the SD3 attention projections; LoRA weights fp32, base fp16, autocast fp16
  * reference policy for the KL = the same transformer with the adapter disabled (no second copy)
  * loss = mean(max(-A*ratio, -A*clip(ratio, 1-eps, 1+eps))) + beta*KL, eps=1e-4, beta=0.04
  * train on int(num_steps * timestep_fraction) timesteps per sample, batch 1, grads accumulated
    over the timesteps of one sample, AdamW lr 3e-4, grad clip 1.0, GradScaler for fp16
  * gradient checkpointing on the transformer, text embeddings precomputed, text encoders freed

Usage: python flowgrpo_probe.py --out ../results/T7/probe_512 --resolution 512 --group 4 --epochs 3
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline
from peft import LoraConfig

MODEL_PATH = os.environ.get("SD35M_PATH", "stabilityai/stable-diffusion-3.5-medium")
LORA_TARGETS = ["attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
                "attn.to_k", "attn.to_out.0", "attn.to_q", "attn.to_v"]
PROMPTS = ["a photo of a red ball on an open grass field, lit by strong sunlight from the left",
           "a photo of a wooden chair in an empty parking lot, its shadow falls to the left"]


def sde_step(model_out, sigma, sigma_next, sample, sigma_max, noise_level, prev_sample=None, generator=None):
    """One reverse SDE step of the flow ODE (Flow-GRPO). Returns prev_sample, log_prob (per item), mean, std."""
    model_out, sample = model_out.float(), sample.float()
    sigma = sigma.view(-1, 1, 1, 1).float()
    sigma_next = sigma_next.view(-1, 1, 1, 1).float()
    dt = sigma_next - sigma
    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * noise_level
    mean = (sample * (1 + std_dev_t ** 2 / (2 * sigma) * dt)
            + model_out * (1 + std_dev_t ** 2 * (1 - sigma) / (2 * sigma)) * dt)
    std = std_dev_t * torch.sqrt(-dt)
    if prev_sample is None:
        noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
        prev_sample = mean + std * noise
    log_prob = (-((prev_sample.detach() - mean) ** 2) / (2 * std ** 2) - torch.log(std)
                - math.log(math.sqrt(2 * math.pi))).mean(dim=(1, 2, 3))
    return prev_sample, log_prob, mean, std


def finite(x):
    return bool(torch.isfinite(x).all())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--n-prompts", type=int, default=2, help="groups per epoch")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--timestep-fraction", type=float, default=0.99)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--noise-level", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--clip", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--no-ckpt", action="store_true", help="disable gradient checkpointing")
    ap.add_argument("--mem-cap-gb", type=float, default=0,
                    help="cap usable GPU memory (e.g. 31.7 to emulate a V100-32 on a bigger card)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev, dt16 = "cuda", torch.float16
    if args.mem_cap_gb:
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_cap_gb / total))
    torch.manual_seed(0)
    log = dict(args={k: str(v) for k, v in vars(args).items()}, gpu=torch.cuda.get_device_name(0),
               torch=torch.__version__, epochs=[], status="started")

    def save():
        (args.out / "probe.json").write_text(json.dumps(log, indent=1))

    pipe = StableDiffusion3Pipeline.from_pretrained(MODEL_PATH, torch_dtype=dt16)
    for n in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        getattr(pipe, n).to(dev)
    prompts = PROMPTS[: args.n_prompts]
    with torch.no_grad():
        emb = [pipe.encode_prompt(prompt=p, prompt_2=None, prompt_3=None, device=dev,
                                  do_classifier_free_guidance=True) for p in prompts]
    pipe.text_encoder = pipe.text_encoder_2 = pipe.text_encoder_3 = None
    torch.cuda.empty_cache()
    tr, vae, sched = pipe.transformer.to(dev), pipe.vae.to(dev), pipe.scheduler
    tr.requires_grad_(False)
    vae.requires_grad_(False)
    tr.add_adapter(LoraConfig(r=args.rank, lora_alpha=2 * args.rank, init_lora_weights="gaussian",
                              target_modules=LORA_TARGETS))
    params = [p for p in tr.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()  # LoRA weights in fp32 (mixed precision), base stays fp16
    if not args.no_ckpt:
        tr.enable_gradient_checkpointing()
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4, eps=1e-8)
    scaler = torch.cuda.amp.GradScaler()
    log["lora_params_M"] = sum(p.numel() for p in params) / 1e6
    log["mem_after_load_gb"] = torch.cuda.memory_allocated() / 1e9

    sched.set_timesteps(args.steps, device=dev)
    timesteps, sigmas = sched.timesteps, sched.sigmas.to(dev)
    sigma_max = sigmas[1].item()
    T_train = int(args.steps * args.timestep_fraction)
    lat_hw = args.resolution // 8
    C = tr.config.in_channels

    def predict(x, t, pe, ppe):
        """CFG velocity for a batch of latents x; pe/ppe are [neg; pos] stacked embeddings."""
        n = x.shape[0]
        xin = torch.cat([x, x]).to(dt16)
        enc = torch.cat([pe[0].expand(n, -1, -1), pe[1].expand(n, -1, -1)])
        pool = torch.cat([ppe[0].expand(n, -1), ppe[1].expand(n, -1)])
        with torch.autocast("cuda", dtype=dt16):
            out = tr(hidden_states=xin, timestep=t.expand(2 * n), encoder_hidden_states=enc,
                     pooled_projections=pool, return_dict=False)[0]
        u, c = out.float().chunk(2)
        return u + args.guidance * (c - u)

    rng = np.random.default_rng(0)
    try:
        for ep in range(args.epochs):
            e = dict(epoch=ep)
            torch.cuda.reset_peak_memory_stats()
            # ---- sampling (policy = LoRA on), one group per prompt, whole group in one batch
            tr.eval()
            samples = []
            torch.cuda.synchronize(); t0 = time.time()
            with torch.no_grad():
                for k, (pe, npe, ppe, nppe) in enumerate(emb):
                    PE, PPE = (npe, pe), (nppe, ppe)
                    g = torch.Generator(dev).manual_seed(1000 * ep + k)
                    x = torch.randn(args.group, C, lat_hw, lat_hw, generator=g, device=dev, dtype=torch.float32)
                    lat, lps = [x], []
                    for i, t in enumerate(timesteps):
                        v = predict(x, t, PE, PPE)
                        x, lp, _, _ = sde_step(v, sigmas[i].expand(args.group), sigmas[i + 1].expand(args.group),
                                               x, sigma_max, args.noise_level, generator=g)
                        lat.append(x); lps.append(lp)
                    img = vae.decode((x.to(dt16) / vae.config.scaling_factor) + vae.config.shift_factor,
                                     return_dict=False)[0]
                    reward = rng.normal(size=args.group)  # random reward: probe only
                    adv = (reward - reward.mean()) / (reward.std() + 1e-4)
                    samples.append(dict(k=k, lat=torch.stack(lat, 1), lp=torch.stack(lps, 1),
                                        adv=torch.tensor(adv, device=dev, dtype=torch.float32),
                                        finite_lat=finite(x), finite_img=finite(img),
                                        img_max=float(img.abs().max())))
            torch.cuda.synchronize()
            n_img = args.group * len(emb)
            e.update(sample_s=time.time() - t0, sample_s_per_image=(time.time() - t0) / n_img,
                     sample_peak_gb=torch.cuda.max_memory_allocated() / 1e9,
                     sample_finite=all(s["finite_lat"] and s["finite_img"] for s in samples))
            # ---- training: batch 1, accumulate over the T_train timesteps of each sample
            tr.train()
            torch.cuda.reset_peak_memory_stats()
            losses, kls, ratios, n_skipped, n_nonfinite = [], [], [], 0, 0
            torch.cuda.synchronize(); t0 = time.time()
            for s in samples:
                pe, npe, ppe, nppe = emb[s["k"]]
                PE, PPE = (npe, pe), (nppe, ppe)
                for j in range(args.group):
                    opt.zero_grad(set_to_none=True)
                    for i in range(T_train):
                        x, x_next = s["lat"][j:j + 1, i], s["lat"][j:j + 1, i + 1]
                        sg, sgn = sigmas[i].view(1), sigmas[i + 1].view(1)
                        v = predict(x, timesteps[i], PE, PPE)
                        _, lp, mean, std = sde_step(v, sg, sgn, x, sigma_max, args.noise_level, prev_sample=x_next)
                        with torch.no_grad():
                            tr.disable_adapters()
                            v_ref = predict(x, timesteps[i], PE, PPE)
                            tr.enable_adapters()
                            _, _, mean_ref, _ = sde_step(v_ref, sg, sgn, x, sigma_max, args.noise_level, prev_sample=x_next)
                        kl = ((mean - mean_ref) ** 2).mean(dim=(1, 2, 3)) / (2 * std.view(-1) ** 2)
                        ratio = torch.exp(lp - s["lp"][j:j + 1, i])
                        a = s["adv"][j:j + 1]
                        pg = torch.maximum(-a * ratio, -a * torch.clamp(ratio, 1 - args.clip, 1 + args.clip)).mean()
                        loss = (pg + args.beta * kl.mean()) / T_train
                        if not finite(loss):
                            n_nonfinite += 1
                        scaler.scale(loss).backward()
                        losses.append(float(loss) * T_train); kls.append(float(kl.mean())); ratios.append(float(ratio.mean()))
                    scaler.unscale_(opt)
                    gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
                    scale_before = scaler.get_scale()
                    scaler.step(opt)
                    scaler.update()
                    n_skipped += int(scaler.get_scale() < scale_before)  # inf/NaN grads -> step skipped
            torch.cuda.synchronize()
            tt = time.time() - t0
            e.update(train_s=tt, train_s_per_image=tt / n_img, train_s_per_timestep=tt / (n_img * T_train),
                     train_peak_gb=torch.cuda.max_memory_allocated() / 1e9,
                     loss_mean=float(np.mean(losses)), kl_mean=float(np.mean(kls)),
                     ratio_min=float(np.min(ratios)), ratio_max=float(np.max(ratios)),
                     nonfinite_losses=n_nonfinite, optimizer_steps_skipped=n_skipped,
                     last_grad_norm=float(gn), lora_finite=all(finite(p) for p in params),
                     grad_scale=scaler.get_scale())
            # c_train: cost of one training timestep relative to one sampling step, per image (T6 input)
            e["c_train_measured"] = e["train_s_per_timestep"] / (e["sample_s_per_image"] / args.steps)
            log["epochs"].append(e)
            print(json.dumps(e), flush=True)
            save()
        log["status"] = "ok"
    except torch.cuda.OutOfMemoryError as err:
        log["status"] = "oom"
        log["error"] = str(err).splitlines()[0]
    log["peak_gb_overall"] = torch.cuda.max_memory_allocated() / 1e9
    log["total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    log["mem_cap_gb"] = args.mem_cap_gb or None
    save()
    print(json.dumps({k: v for k, v in log.items() if k != "epochs"}, indent=1))


if __name__ == "__main__":
    main()
