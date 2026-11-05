"""T5: is there a learning signal?

1. Group variance of reward = gate * R_dir over the 8 seeds of each prompt (full steps).
2. Best-of-8 vs random, judged by human labels (needs T3 labels; skipped if absent).
   The labeled set always contains each prompt's verifier top-1; the other labeled images of a
   group were drawn uniformly at random from the remaining 7 seeds, so
     random-sample accuracy = 1/8 * acc(top1) + 7/8 * acc(labeled non-top1)
   is an unbiased estimate for a uniformly random seed.
3. Reduced (10) vs full (40) steps for the same prompt/seed: gate and side agreement.

Usage (from smoke/analysis):
  python t5_signal.py --full ../results/T2/scores_40steps.jsonl --reduced ../results/T2/scores_10steps.jsonl \
      --label-dir ../label --out ../results/T5
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def load(path):
    return {json.loads(l)["image_id"]: json.loads(l) for l in open(path) if not json.loads(l)["flipped"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("--reduced")
    ap.add_argument("--label-dir", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--std-th", type=float, default=0.05)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    full = load(args.full)
    res = {}

    # 1. group variance
    groups = defaultdict(list)
    for r in full.values():
        groups[r["prompt_id"]].append(r)
    stds, all_gated_out, rows = [], 0, []
    for pid, g in sorted(groups.items()):
        rew = np.array([r["reward"] for r in g])
        gates = np.array([r["gate"] for r in g])
        s = float(rew.std())
        stds.append(s)
        all_gated_out += int(gates.sum() == 0)
        rows.append(dict(prompt_id=pid, n=len(g), gate_rate=float(gates.mean()), mean_reward=float(rew.mean()),
                         std_reward=s, n_positive=int((rew > 0).sum()), n_negative=int((rew < 0).sum())))
    stds = np.array(stds)
    usable = (stds >= args.std_th)
    res["group_variance"] = dict(
        n_groups=len(stds), frac_all_gated_out=all_gated_out / len(stds),
        frac_std_below_th=float((stds < args.std_th).mean()), std_th=args.std_th,
        frac_usable=float(usable.mean()), median_std=float(np.median(stds)),
        pass_=bool(usable.mean() >= 0.5), per_group=rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.hist(stds, bins=np.linspace(0, 1.05, 22), color="#457b9d", edgecolor="white")
        ax.axvline(args.std_th, color="#e63946", ls="--", label=f"std = {args.std_th}")
        ax.set_xlabel("std of reward (gate x R_dir) within a group of 8 samples")
        ax.set_ylabel("# prompts")
        ax.set_title(f"Reward spread per prompt: {usable.mean():.0%} of groups usable")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out / "reward_spread_hist.png", dpi=150)
    except ImportError:
        pass

    # 2. best-of-N vs random using human labels
    if args.label_dir and list(args.label_dir.glob("labels_*.csv")):
        from t4_metrics import SIDE, ground_truth, load_labels
        key, per_image, _ = load_labels(args.label_dir)
        gt = {i: ground_truth(l) for i, l in per_image.items() if l}
        meta = {r["image_id"]: r for r in key.values()}
        correct = lambda i: float(SIDE.get(gt[i]) == meta[i]["target_dir"])
        top_acc, other_acc, per_prompt = [], [], []
        for pid, g in groups.items():
            top = max(g, key=lambda r: (r["reward"], -r["seed"]))["image_id"]
            others = [r["image_id"] for r in g if r["image_id"] != top and r["image_id"] in gt and gt[r["image_id"]] != "X"]
            if top not in gt or gt[top] == "X" or not others:
                continue
            t, o = correct(top), float(np.mean([correct(i) for i in others]))
            top_acc.append(t)
            other_acc.append(o)
            per_prompt.append(dict(prompt_id=pid, top1=top, top1_correct=t, others_acc=o, n_others=len(others)))
        if top_acc:
            top_m, oth_m = float(np.mean(top_acc)), float(np.mean(other_acc))
            rand_m = top_m / 8 + 7 * oth_m / 8
            # bootstrap over prompts for a CI on the gap
            rng = random.Random(0)
            n = len(top_acc)
            gaps = []
            for _ in range(5000):
                idx = [rng.randrange(n) for _ in range(n)]
                tm = np.mean([top_acc[k] for k in idx])
                om = np.mean([other_acc[k] for k in idx])
                gaps.append(tm - (tm / 8 + 7 * om / 8))
            res["best_of_n"] = dict(n_prompts=n, best_of_8_acc=round(100 * top_m, 1),
                                    random_sample_acc=round(100 * rand_m, 1),
                                    gap_points=round(100 * (top_m - rand_m), 1),
                                    gap_95ci=[round(100 * float(np.percentile(gaps, q)), 1) for q in (2.5, 97.5)],
                                    pass_=bool(100 * (top_m - rand_m) >= 15), per_prompt=per_prompt)

    # 3. reduced vs full steps
    if args.reduced:
        red = load(args.reduced)
        pairs = []
        for r in full.values():
            # match on prompt/seed (image ids differ by the _n<steps> suffix)
            rid = r["image_id"].rsplit("_n", 1)[0] + f"_n{next(iter(red.values()))['steps']}"
            if rid in red:
                pairs.append((r, red[rid]))
        both = [(a, b) for a, b in pairs if a["gate"] and b["gate"]]
        res["reduced_vs_full"] = dict(
            n_pairs=len(pairs),
            gate_rate_full=float(np.mean([a["gate"] for a, _ in pairs])),
            gate_rate_reduced=float(np.mean([b["gate"] for _, b in pairs])),
            gate_agreement=float(np.mean([a["gate"] == b["gate"] for a, b in pairs])),
            n_both_gated=len(both),
            side_agreement=float(np.mean([a["inferred_side"] == b["inferred_side"] for a, b in both])) if both else None,
            reward_corr=float(np.corrcoef([a["reward"] for a, _ in pairs], [b["reward"] for _, b in pairs])[0, 1]),
            verifier_side_eq_target_full=float(np.mean([a["inferred_side"] == a["target_dir"] for a, _ in both])) if both else None,
            verifier_side_eq_target_reduced=float(np.mean([b["inferred_side"] == b["target_dir"] for _, b in both])) if both else None,
        )
        res["reduced_vs_full"]["pass_"] = bool((res["reduced_vs_full"]["side_agreement"] or 0) >= 0.8)

    json.dump(res, open(args.out / "t5_signal.json", "w"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("per_group", "per_prompt")}
                      for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
