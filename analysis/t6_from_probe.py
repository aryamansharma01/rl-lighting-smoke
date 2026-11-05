"""T6 (rough): Flow-GRPO GPU-hour estimate from the measured T7 probe + SSIS timing.

Per training epoch, per image (all measured, not assumed):
  sample = probe sample_s_per_image   (10-step SDE sampling with CFG, group batched, includes VAE decode)
  reward = SSIS sec/image              (T2)
  train  = T_train * probe train_s_per_timestep   (batch 1, LoRA, grad checkpointing, + KL reference pass)
GPU-hours = images/epoch * (sample + reward + train) * epochs / 3600.
Probe epochs after the first are averaged (the first includes warm-up).

The probe ran on an A100-40GB capped at 31.7 GB; SSIS on an A10. V100 hours are a rough scaling:
V100 fp16 tensor throughput is ~125 TFLOPS vs ~312 on A100 (about 2.5x), and the V100 has no
FlashAttention kernels; we report x2 and x3 as a range. Replace with real V100 numbers when PSC's probe runs.

Usage: python t6_from_probe.py --probe ../results_modal/T7/probe_512/probe.json \
           --ssis ../results_modal/T2/raw_40steps/ssis_stats.json --out ../results_modal/T6
"""
import argparse
import json
from pathlib import Path

import numpy as np

CONFIGS = {
    # Flow-GRPO SD3 recipe (config/grpo.py @879042c): 48 groups x 24 images, 10 train steps, tf 0.99
    "flowgrpo_default_100ep": dict(groups=48, group_size=24, train_steps=10, timestep_fraction=0.99, epochs=100),
    "flowgrpo_default_1000ep": dict(groups=48, group_size=24, train_steps=10, timestep_fraction=0.99, epochs=1000),
    "cutdown_8x8_100ep": dict(groups=8, group_size=8, train_steps=10, timestep_fraction=0.99, epochs=100),
    "cutdown_8x8_300ep_tf0.5": dict(groups=8, group_size=8, train_steps=10, timestep_fraction=0.5, epochs=300),
    "cutdown_16x8_200ep": dict(groups=16, group_size=8, train_steps=10, timestep_fraction=0.99, epochs=200),
}
V100_FACTORS = (2.0, 3.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, type=Path)
    ap.add_argument("--ssis", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    p = json.load(open(args.probe))
    eps = p["epochs"][1:] or p["epochs"]
    t_sample = float(np.mean([e["sample_s_per_image"] for e in eps]))
    t_step = float(np.mean([e["train_s_per_timestep"] for e in eps]))
    c_train = float(np.mean([e["c_train_measured"] for e in eps]))
    t_ssis = json.load(open(args.ssis))["sec_per_image"]

    est = {}
    for k, c in CONFIGS.items():
        imgs = c["groups"] * c["group_size"]
        T = int(c["train_steps"] * c["timestep_fraction"])
        per = dict(sample=t_sample, reward=t_ssis, train=T * t_step)
        tot = sum(per.values())
        h = imgs * tot * c["epochs"] / 3600
        est[k] = dict(images_per_epoch=imgs, T_train=T, sec_per_image=per, sec_per_image_total=tot,
                      gpu_hours_measured_gpu=h, gpu_hours_v100_range=[h * f for f in V100_FACTORS],
                      wall_hours_per_epoch_1gpu=imgs * tot / 3600)
    out = dict(measured=dict(probe_gpu=p["gpu"], mem_cap_gb=p.get("mem_cap_gb"),
                             peak_gb_overall=p.get("peak_gb_overall"), status=p["status"],
                             sample_s_per_image_10steps=t_sample, train_s_per_timestep=t_step,
                             c_train_measured=c_train, ssis_s_per_image=t_ssis,
                             ssis_gpu=json.load(open(args.ssis))["gpu"]),
               v100_scaling_assumption=V100_FACTORS, configs=CONFIGS, estimates=est)
    json.dump(out, open(args.out / "t6_estimate.json", "w"), indent=1)

    L = [f"Measured on {p['gpu']} (capped at {p.get('mem_cap_gb')} GB) + SSIS on {out['measured']['ssis_gpu']}; "
         f"sample {t_sample:.2f} s/img, train {t_step:.2f} s/timestep, SSIS {t_ssis:.3f} s/img, c_train {c_train:.1f}.",
         "",
         "| config | images/epoch | s/image (sample + reward + train) | GPU-h (measured GPU) | GPU-h on V100 (×2–×3, rough) |",
         "|---|---|---|---|---|"]
    for k, e in est.items():
        s = e["sec_per_image"]
        lo, hi = e["gpu_hours_v100_range"]
        L.append(f"| {k} | {e['images_per_epoch']} | {s['sample']:.2f} + {s['reward']:.2f} + {s['train']:.2f} = "
                 f"{e['sec_per_image_total']:.1f} | {e['gpu_hours_measured_gpu']:.1f} | {lo:.0f}–{hi:.0f} |")
    (args.out / "t6_table.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
