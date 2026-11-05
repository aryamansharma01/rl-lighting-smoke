"""T6: throughput summary + Flow-GRPO GPU-hour estimate from measured timings.

Cost model per training epoch (one "step" in Flow-GRPO's logging = one epoch of sampling + training):
  images/epoch      = groups * group_size
  sample            = images * t_gen(train_steps)                      (measured, includes CFG)
  reward            = images * t_ssis                                    (measured)
  train             = images * T_train * c_train * t_gen(train_steps) / train_steps
     T_train  = int(train_steps * timestep_fraction) denoising steps backpropagated per image
     c_train  = cost of one training timestep relative to one sampling step:
                fwd+bwd ~ 3 fwd, + 1 ref-model fwd for the KL term (beta > 0), LoRA bwd is a bit
                cheaper than full bwd -> we use 4 (an assumption, flagged in the report).
Throughput numbers come from smoke/results/T1/gen_stats_*.json and T2/raw_*/ssis_stats.json.

Usage: python t6_compute.py --results ../results --out ../results/T6
"""
import argparse
import json
from pathlib import Path


def run(cfg, t_gen_train, t_ssis, c_train=4.0, n_gpus_for_eval=1):
    imgs = cfg["groups"] * cfg["group_size"]
    T = int(cfg["train_steps"] * cfg["timestep_fraction"])
    per_img = dict(sample=t_gen_train, reward=t_ssis,
                   train=T * c_train * t_gen_train / cfg["train_steps"])
    per_img_total = sum(per_img.values())
    epoch_s = imgs * per_img_total
    total_h = epoch_s * cfg["epochs"] / 3600
    return dict(images_per_epoch=imgs, T_train=T, sec_per_image=per_img, sec_per_image_total=per_img_total,
                gpu_sec_per_epoch=epoch_s, epochs=cfg["epochs"], gpu_hours=total_h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--c-train", type=float, default=None,
                    help="override; default = measured by the T7 probe if present, else 4.0 (assumption)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # c_train measured by the T7 probe (V100-32, batch 1, checkpointing): median over epochs after the first
    probe = args.results / "T7/probe_512/probe.json"
    c_src = "assumed"
    if args.c_train is None:
        args.c_train = 4.0
        if probe.exists():
            pj = json.load(open(probe))
            cs = [e["c_train_measured"] for e in pj.get("epochs", [])[1:]] or \
                 [e["c_train_measured"] for e in pj.get("epochs", [])]
            if cs:
                args.c_train, c_src = float(sorted(cs)[len(cs) // 2]), f"measured ({probe})"
    else:
        c_src = "command line"

    g40 = json.load(open(args.results / "T1/gen_stats_40steps.json"))
    g10 = json.load(open(args.results / "T1/gen_stats_10steps.json"))
    s = json.load(open(args.results / "T2/raw_40steps/ssis_stats.json"))
    t10, t40, ts = g10["sec_per_image"], g40["sec_per_image"], s["sec_per_image"]

    # resolution scaling for a cut-down config: transformer cost ~ tokens^1.x; use (res/512)^2 as a floor
    configs = {
        # Flow-GRPO pickscore_sd3 / geneval_sd3 (config/grpo.py @879042c): 48 groups x 24 images,
        # 10 train steps, 40 eval steps, 512px, timestep_fraction 0.99, LoRA r=32, 32 GPUs.
        "flowgrpo_default_100ep": dict(groups=48, group_size=24, train_steps=10, timestep_fraction=0.99, epochs=100, res=512),
        "flowgrpo_default_1000ep": dict(groups=48, group_size=24, train_steps=10, timestep_fraction=0.99, epochs=1000, res=512),
        # cut-down: 8 prompts/epoch x group 8, 10 steps, 512px, 100 epochs (= the T7-sized run x ~1)
        "cutdown_8x8_100ep": dict(groups=8, group_size=8, train_steps=10, timestep_fraction=0.99, epochs=100, res=512),
        # cut-down + train on half the timesteps (Flow-GRPO-Fast-like idea), 300 epochs
        "cutdown_8x8_300ep_tf0.5": dict(groups=8, group_size=8, train_steps=10, timestep_fraction=0.5, epochs=300, res=512),
        "cutdown_16x8_200ep": dict(groups=16, group_size=8, train_steps=10, timestep_fraction=0.99, epochs=200, res=512),
    }
    est = {k: run(c, t10, ts, args.c_train) for k, c in configs.items()}
    out = dict(measured=dict(gpu=g40["gpu"], dtype=g40["dtype"], resolution=g40["resolution"],
                             sec_per_image_10steps=t10, sec_per_image_40steps=t40,
                             gen_batch=g40["batch"], peak_vram_gen_gb=max(g10["peak_vram_gb"], g40["peak_vram_gb"]),
                             ssis_sec_per_image=ts, ssis_peak_vram_gb=s["peak_vram_gb"],
                             ssis_input_min_size=s.get("min_size_test")),
               c_train=args.c_train, c_train_source=c_src, configs=configs, estimates=est)
    json.dump(out, open(args.out / "t6_compute.json", "w"), indent=1)

    lines = ["| config | images/epoch | s/img (sample+reward+train) | GPU-s/epoch | epochs | GPU-hours |",
             "|---|---|---|---|---|---|"]
    for k, e in est.items():
        p = e["sec_per_image"]
        lines.append(f"| {k} | {e['images_per_epoch']} | {p['sample']:.2f}+{p['reward']:.2f}+{p['train']:.2f}"
                     f" = {e['sec_per_image_total']:.2f} | {e['gpu_sec_per_epoch']:.0f} | {e['epochs']} | {e['gpu_hours']:.1f} |")
    (args.out / "t6_table.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(out["measured"], indent=1))
    print(f"c_train = {args.c_train:.2f} ({c_src})")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
