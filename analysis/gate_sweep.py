"""Offline sweep of the verifier's validity cutoffs, from the per-pair data saved in T2.

Recomputes gate and inferred side for every image under a grid of
  conf_th          (pair confidence = min(obj score, shadow score); SSIS itself keeps scores >= 0.1)
  min_len_frac     (|shadow centroid - contact| >= frac * image width)
  min_shadow_frac  (shadow area >= frac * image area)
with the same aggregation as verifier/direction.py (confidence-weighted mean of unit vectors).
No second SSIS run is needed.

Metrics per setting (non-mirrored images unless noted):
  coverage        fraction of images with gate = 1
  flip_consistent among images where original AND mirror are gated: mirror side is opposite
  prompt_agree    among gated images: inferred side == prompted side (NOT accuracy: base model may be wrong)
  with --label-dir (T3 labels): accuracy vs human majority label on gated images labeled L/R,
                    and false_gate_rate = gate = 1 on images humans labeled N.

Usage:
  python gate_sweep.py --scores ../results/T2/scores_40steps.jsonl --out ../results/T2/gate_sweep [--label-dir ../label]
"""
import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

CONF = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
LEN = [0.0, 0.01, 0.02, 0.04, 0.08]
AREA = [0.0, 5e-4, 1e-3, 2e-3, 5e-3]
DEFAULT = (0.3, 0.02, 1e-3)


def decide(rec, conf_th, min_len, min_area):
    H, W = rec["hw"]
    vecs, w = [], []
    for p in rec["pairs"]:
        if "light_vec" not in p:  # empty mask
            continue
        if p["conf"] >= conf_th and p["length"] >= min_len * W and p["sh_area"] >= min_area * H * W:
            vecs.append(p["light_vec"])
            w.append(p["conf"])
    if not vecs:
        return 0, None
    r = (np.asarray(vecs) * np.asarray(w)[:, None]).sum(0)
    return 1, ("left" if r[0] < 0 else "right")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--label-dir", type=Path)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "env"))
    import fast_mode
    if fast_mode.cfg().get("skip_gate_sweep"):
        fast_mode.skip("gate sweep")
    args.out.mkdir(parents=True, exist_ok=True)

    recs = [json.loads(l) for l in open(args.scores)]
    orig = {r["image_id"]: r for r in recs if not r["flipped"]}
    flip = {r["image_id"]: r for r in recs if r["flipped"]}

    gt = {}
    if args.label_dir and list(args.label_dir.glob("labels_*.csv")):
        sys.path.insert(0, str(Path(__file__).parent))
        from t4_metrics import ground_truth, load_labels
        _, per_image, _ = load_labels(args.label_dir)
        gt = {i: ground_truth(l) for i, l in per_image.items() if l}
    side_of_label = {"L": "left", "R": "right"}

    rows = []
    for c, l, a in itertools.product(CONF, LEN, AREA):
        d = {i: decide(r, c, l, a) for i, r in orig.items()}
        df = {i: decide(r, c, l, a) for i, r in flip.items()}
        gated = [i for i in d if d[i][0]]
        both = [i for i in gated if i in df and df[i][0]]
        row = dict(conf_th=c, min_len_frac=l, min_shadow_frac=a, n=len(d),
                   coverage=len(gated) / max(len(d), 1),
                   flip_n=len(both),
                   flip_consistent=(np.mean([d[i][1] != df[i][1] for i in both]) if both else None),
                   prompt_agree=(np.mean([d[i][1] == orig[i]["target_dir"] for i in gated]) if gated else None))
        if gt:
            lr = [i for i in gated if gt.get(i) in side_of_label]
            n_imgs = [i for i in d if gt.get(i) == "N"]
            row.update(acc_n=len(lr),
                       accuracy=(np.mean([d[i][1] == side_of_label[gt[i]] for i in lr]) if lr else None),
                       false_gate_rate=(np.mean([d[i][0] for i in n_imgs]) if n_imgs else None),
                       n_no_shadow=len(n_imgs))
        rows.append(row)

    with open(args.out / "gate_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    # coverage vs (accuracy if labels else prompt agreement), default setting highlighted
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ykey = "accuracy" if gt else "prompt_agree"
    pts = [r for r in rows if r[ykey] is not None]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter([r["coverage"] for r in pts], [r[ykey] for r in pts], s=12, c="#888", label="grid settings")
    dflt = [r for r in pts if (r["conf_th"], r["min_len_frac"], r["min_shadow_frac"]) == DEFAULT]
    if dflt:
        ax.scatter([dflt[0]["coverage"]], [dflt[0][ykey]], s=60, c="#e63946", label="default (0.3, 2%, 0.1%)")
    if gt:
        ax.axhline(0.85, ls="--", c="#457b9d", lw=1, label="85% bar")
    ax.set_xlabel("coverage (fraction of images with gate = 1)")
    ax.set_ylabel("direction accuracy vs human labels" if gt else "agreement with prompted side")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "coverage_vs_accuracy.png", dpi=150)

    best = dflt[0] if dflt else None
    print(json.dumps(dict(n_settings=len(rows), metric=ykey, default=best), indent=1, default=float))


if __name__ == "__main__":
    main()
