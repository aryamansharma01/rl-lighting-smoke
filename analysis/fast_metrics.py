"""First-results metrics for the reduced plan (24 prompts x 4 seeds, 40 steps, 1 label per image).

Computes only:
  1. base-model side accuracy (human label side == prompted side), overall and by phrasing
  2. verifier coverage (gate rate) and direction accuracy vs human labels at default cutoffs
  3. flip consistency (mirror image gets the opposite side, both gated)
  4. within-group reward std (groups = the 4 seeds of a prompt), reward = gate * cos
  5. best-of-4 (verifier top-1) vs random sample, judged by human labels
95% intervals: Wilson for proportions, bootstrap over prompts for the best-of-4 gap.

Works with only T2 scores (items 2-4 partially: coverage, flip, std) or with labels too.
Usage:
  python fast_metrics.py --scores ../results/T2/scores_40steps.jsonl --label-dir ../label --out ../results/fast
"""
import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SIDE = {"L": "left", "R": "right"}


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def prop(flags):
    k, n = int(sum(flags)), len(flags)
    return dict(pct=round(100 * k / n, 1) if n else None, k=k, n=n, ci95=wilson(k, n))


def load_labels(label_dir):
    """image_id -> label; with several annotators per image, majority (ties -> 'X')."""
    key = {r["item_id"]: r for r in csv.DictReader(open(label_dir / "key.csv"))}
    per = defaultdict(list)
    files = sorted(label_dir.glob("labels_*.csv"))
    for f in files:
        for r in csv.DictReader(open(f)):
            if r["label"] and r["item_id"] in key:
                per[key[r["item_id"]]["image_id"]].append(r["label"])
    gt = {}
    for i, ls in per.items():
        vals, counts = np.unique(ls, return_counts=True)
        gt[i] = vals[counts.argmax()] if (counts == counts.max()).sum() == 1 else "X"
    meta = {r["image_id"]: r for r in key.values()}
    return gt, meta, [f.name for f in files]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, type=Path)
    ap.add_argument("--label-dir", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--std-th", type=float, default=0.05)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(args.scores)]
    orig = {r["image_id"]: r for r in rows if not r["flipped"]}
    flip = {r["image_id"]: r for r in rows if r["flipped"]}
    res = dict(n_images=len(orig), n_mirrored=len(flip))

    # coverage, flip consistency (no labels needed)
    res["verifier_coverage_all"] = prop([r["gate"] for r in orig.values()])
    both = [i for i in orig if i in flip and orig[i]["gate"] and flip[i]["gate"]]
    res["flip_consistency"] = prop([orig[i]["inferred_side"] != flip[i]["inferred_side"] for i in both])
    res["verifier_agrees_with_prompt_on_gated"] = prop(
        [r["inferred_side"] == r["target_dir"] for r in orig.values() if r["gate"]])

    # group reward std, groups = seeds of a prompt
    groups = defaultdict(list)
    for r in orig.values():
        groups[r["prompt_id"]].append(r)
    stds = {p: float(np.std([r["reward"] for r in g])) for p, g in groups.items()}
    res["group_reward_std"] = dict(group_size=sorted({len(g) for g in groups.values()}),
                                   n_groups=len(stds), mean=round(float(np.mean(list(stds.values()))), 3),
                                   median=round(float(np.median(list(stds.values()))), 3),
                                   frac_groups_std_gt_th=prop([s > args.std_th for s in stds.values()]),
                                   std_threshold=args.std_th,
                                   frac_groups_all_gated=prop([all(r["gate"] for r in g) for g in groups.values()]),
                                   frac_groups_none_gated=prop([not any(r["gate"] for r in g) for g in groups.values()]))

    if args.label_dir and list(args.label_dir.glob("labels_*.csv")):
        gt, meta, files = load_labels(args.label_dir)
        res["label_files"] = files
        lab = [i for i in orig if i in gt and gt[i] != "X"]
        res["n_labeled"] = len(lab)
        res["label_counts"] = {k: sum(gt[i] == k for i in lab) for k in ("L", "R", "A", "N")}
        correct = {i: SIDE.get(gt[i]) == orig[i]["target_dir"] for i in lab}

        # 1. base-model side accuracy (A/N count as not correct), and among clear L/R images
        res["base_side_accuracy"] = dict(
            all=prop([correct[i] for i in lab]),
            on_LR_only=prop([correct[i] for i in lab if gt[i] in SIDE]),
            by_phrasing={ph: prop([correct[i] for i in lab if orig[i]["phrasing"] == ph])
                         for ph in sorted({orig[i]["phrasing"] for i in lab})},
            by_target={t: prop([correct[i] for i in lab if orig[i]["target_dir"] == t]) for t in ("left", "right")})

        # 2. verifier accuracy + coverage vs human labels (default cutoffs)
        lr = [i for i in lab if gt[i] in SIDE]
        gated_lr = [i for i in lr if orig[i]["gate"]]
        res["verifier_vs_humans"] = dict(
            coverage_on_human_LR=prop([orig[i]["gate"] for i in lr]),
            accuracy_on_gated_LR=prop([orig[i]["inferred_side"] == SIDE[gt[i]] for i in gated_lr]),
            false_gate_rate_on_human_N=prop([orig[i]["gate"] for i in lab if gt[i] == "N"]),
            gate_rate_on_human_A=prop([orig[i]["gate"] for i in lab if gt[i] == "A"]),
            bar_pct=85)

        # 5. best-of-4 vs random (all seeds labeled, so random = mean over the group)
        tops, rands, per_prompt = [], [], []
        for p, g in sorted(groups.items()):
            g = [r for r in g if r["image_id"] in correct]
            if len(g) < 2:
                continue
            top = max(g, key=lambda r: (r["reward"], -r["seed"]))
            tops.append(float(correct[top["image_id"]]))
            rands.append(float(np.mean([correct[r["image_id"]] for r in g])))
            per_prompt.append(dict(prompt_id=p, top1=top["image_id"], top1_correct=tops[-1], random_acc=rands[-1],
                                   n=len(g), top1_reward=round(top["reward"], 3)))
        if tops:
            rng = random.Random(0)
            n = len(tops)
            gaps = sorted(np.mean([tops[k] - rands[k] for k in idx])
                          for idx in ([rng.randrange(n) for _ in range(n)] for _ in range(5000)))
            res["best_of_n"] = dict(n_prompts=n, best_acc_pct=round(100 * np.mean(tops), 1),
                                    random_acc_pct=round(100 * np.mean(rands), 1),
                                    gap_pts=round(100 * (np.mean(tops) - np.mean(rands)), 1),
                                    gap_ci95_pts=[round(100 * gaps[124], 1), round(100 * gaps[4874], 1)],
                                    per_prompt=per_prompt)

    json.dump(res, open(args.out / "fast_metrics.json", "w"), indent=1)

    def f(p):
        if not p or p.get("n", 0) == 0:
            return "n/a"
        ci = f" [{p['ci95'][0]}–{p['ci95'][1]}]" if p.get("ci95") else ""
        return f"{p['pct']}% ({p['k']}/{p['n']}){ci}"

    L = ["| Metric | Value (95% CI) |", "|---|---|"]
    if "base_side_accuracy" in res:
        b = res["base_side_accuracy"]
        L.append(f"| Base-model side accuracy, all labeled (A/N = wrong) | {f(b['all'])} |")
        L.append(f"| Base-model side accuracy, human-L/R images only | {f(b['on_LR_only'])} |")
        for ph, v in b["by_phrasing"].items():
            L.append(f"| … phrasing = \"{ph}\" | {f(v)} |")
        v = res["verifier_vs_humans"]
        L.append(f"| **Verifier direction accuracy (gated, human L/R)** (bar ≥ 85%) | {f(v['accuracy_on_gated_LR'])} |")
        L.append(f"| Verifier coverage on human-L/R images | {f(v['coverage_on_human_LR'])} |")
        L.append(f"| Verifier false gate rate on human-N images | {f(v['false_gate_rate_on_human_N'])} |")
    L.append(f"| Verifier coverage, all images | {f(res['verifier_coverage_all'])} |")
    L.append(f"| Flip consistency (both gated) | {f(res['flip_consistency'])} |")
    g = res["group_reward_std"]
    L.append(f"| Groups (4 seeds) with reward std > {g['std_threshold']} | {f(g['frac_groups_std_gt_th'])}; "
             f"mean std {g['mean']}, median {g['median']} |")
    if "best_of_n" in res:
        b = res["best_of_n"]
        L.append(f"| Best-of-4 vs random sample (human-judged) | {b['best_acc_pct']}% vs {b['random_acc_pct']}%, "
                 f"gap {b['gap_pts']} pts [{b['gap_ci95_pts'][0]}, {b['gap_ci95_pts'][1]}], {b['n_prompts']} prompts |")
    (args.out / "fast_table.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
