"""T4: core metrics from human labels (label/labels_*.csv + label/key.csv) and verifier scores.

Ground truth per image: the single annotator's label, or for the 30 shared images the majority
vote (3 different labels -> 'X' = unresolved, excluded from GT-based metrics).

Also sweeps the verifier confidence threshold (re-scoring raw SSIS detections) so coverage vs
accuracy can be read off together.

Usage (from smoke/analysis):
  python t4_metrics.py --label-dir ../label --scores ../results/T2/scores_40steps.jsonl \
      --raw ../results/T2/raw_40steps --out ../results/T4
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verifier"))
from direction import load_record, verify  # noqa: E402

CATS = ["L", "R", "A", "N"]
SIDE = {"L": "left", "R": "right"}


def fleiss_kappa(table):
    """table: (n_items, n_cats) counts, each row sums to n raters."""
    t = np.asarray(table, float)
    n = t.sum(1)[0]
    p_j = t.sum(0) / t.sum()
    P_i = ((t * t).sum(1) - n) / (n * (n - 1))
    P_bar, P_e = P_i.mean(), (p_j ** 2).sum()
    return float((P_bar - P_e) / (1 - P_e)) if P_e < 1 else float("nan")


def load_labels(label_dir):
    key = {r["item_id"]: r for r in csv.DictReader(open(label_dir / "key.csv"))}
    per_image = defaultdict(dict)  # image_id -> {annotator: label}
    files = sorted(label_dir.glob("labels_*.csv"))
    for f in files:
        for r in csv.DictReader(open(f)):
            if r["label"]:
                per_image[key[r["item_id"]]["image_id"]][r["annotator"]] = r["label"]
    return key, per_image, [f.name for f in files]


def ground_truth(labels):
    c = Counter(labels.values())
    lab, n = c.most_common(1)[0]
    if len(labels) >= 3 and n == 1:
        return "X"
    if len(labels) == 2 and n == 1:
        return "X"
    return lab


def pct(x):
    return None if x is None else round(100 * x, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-dir", required=True, type=Path)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--raw", type=Path, help="raw SSIS npz dir for the threshold sweep")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    key, per_image, files = load_labels(args.label_dir)
    scores = {json.loads(l)["image_id"]: json.loads(l) for l in open(args.scores) if not json.loads(l)["flipped"]}
    meta = {r["image_id"]: r for r in key.values()}

    # --- inter-annotator agreement on shared images
    shared = [i for i, r in meta.items() if r["shared"] == "1"]
    full = [i for i in shared if len(per_image[i]) >= 3]
    table = [[Counter(per_image[i].values())[c] for c in CATS] for i in full]
    kappa = fleiss_kappa(table) if full else None
    pairwise = []
    for i in full:
        labs = list(per_image[i].values())
        pairwise += [labs[a] == labs[b] for a in range(len(labs)) for b in range(a + 1, len(labs))]
    # agreement on the binary side question only (images every rater called L or R)
    lr_only = [i for i in full if all(v in "LR" for v in per_image[i].values())]

    gt = {i: ground_truth(l) for i, l in per_image.items() if l}

    def verifier_metrics(sc):
        lr = [i for i, g in gt.items() if g in "LR"]
        cov = [sc[i]["gate"] for i in lr]
        acc = [sc[i]["inferred_side"] == SIDE[gt[i]] for i in lr if sc[i]["gate"]]
        n_imgs = [i for i, g in gt.items() if g == "N"]
        return dict(n_human_LR=len(lr), coverage=np.mean(cov) if cov else None,
                    n_gated_LR=len(acc), direction_accuracy=np.mean(acc) if acc else None,
                    n_human_N=len(n_imgs),
                    false_gate_rate=np.mean([sc[i]["gate"] for i in n_imgs]) if n_imgs else None,
                    gate_rate_on_human_A=np.mean([sc[i]["gate"] for i, g in gt.items() if g == "A"])
                    if any(g == "A" for g in gt.values()) else None)

    vm = verifier_metrics(scores)
    # offset-based alternative (same gate)
    lr = [i for i, g in gt.items() if g in "LR" and scores[i]["gate"]]
    offset_acc = np.mean([scores[i]["offset_side"] == SIDE[gt[i]] for i in lr]) if lr else None

    # --- base model accuracy vs prompt
    def base_acc(ids):
        labs = [gt[i] for i in ids if gt[i] != "X"]
        tgt = [meta[i]["target_dir"] for i in ids if gt[i] != "X"]
        if not labs:
            return None
        shares = Counter(labs)
        return dict(n=len(labs), accuracy=pct(np.mean([SIDE.get(l) == t for l, t in zip(labs, tgt)])),
                    shares={c: pct(shares[c] / len(labs)) for c in CATS},
                    opposite_side=pct(np.mean([l in "LR" and SIDE[l] != t for l, t in zip(labs, tgt)])))

    ids = list(gt)
    base = dict(all=base_acc(ids),
                by_direction={d: base_acc([i for i in ids if meta[i]["target_dir"] == d]) for d in ("left", "right")},
                by_phrasing={p: base_acc([i for i in ids if meta[i]["phrasing"] == p]) for p in ("light", "shadow")},
                by_object={o: base_acc([i for i in ids if meta[i]["object"] == o]) for o in sorted({m["object"] for m in meta.values()})})

    # --- threshold sweep (coverage vs accuracy)
    sweep = []
    if args.raw:
        recs = {i: load_record(args.raw / f"{i}.npz") for i in gt}
        for th in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            sc = {i: verify(recs[i], meta[i]["target_dir"], conf_th=th) for i in gt}
            m = verifier_metrics(sc)
            sweep.append(dict(conf_th=th, coverage=pct(m["coverage"]), accuracy=pct(m["direction_accuracy"]),
                              false_gate=pct(m["false_gate_rate"]), n_gated_LR=m["n_gated_LR"]))

    res = dict(
        label_files=files, n_labeled_images=len(gt), gt_counts=Counter(gt.values()),
        agreement=dict(n_shared_full=len(full), fleiss_kappa_4cat=kappa,
                       pairwise_pct_agreement=pct(np.mean(pairwise)) if pairwise else None,
                       n_shared_all_raters_LR=len(lr_only),
                       side_agreement_when_all_LR=pct(np.mean([len(set(per_image[i].values())) == 1 for i in lr_only])) if lr_only else None),
        verifier={k: (pct(v) if isinstance(v, float) else v) for k, v in vm.items()},
        verifier_offset_direction_accuracy=pct(offset_acc),
        base_model=base, conf_threshold_sweep=sweep,
        conf_th_used=next(iter(scores.values()))["params"]["conf_th"],
    )
    json.dump(res, open(args.out / "t4_metrics.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))

    # per-image table for failure analysis
    with open(args.out / "t4_per_image.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "prompt_id", "target", "phrasing", "human_gt", "raters", "gate",
                    "verifier_side", "offset_side", "R_dir", "n_valid_pairs"])
        for i in sorted(gt):
            s = scores[i]
            w.writerow([i, meta[i]["prompt_id"], meta[i]["target_dir"], meta[i]["phrasing"], gt[i],
                        "|".join(f"{a}:{l}" for a, l in sorted(per_image[i].items())), s["gate"],
                        s["inferred_side"], s["offset_side"], round(s["R_dir"], 3), s["n_valid"]])

    # figure: base-model label shares by direction and phrasing
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        groups = [("left", None), ("right", None), (None, "light"), (None, "shadow")]
        names, data = [], []
        for d, p in groups:
            sel = [i for i in ids if gt[i] != "X" and (d is None or meta[i]["target_dir"] == d)
                   and (p is None or meta[i]["phrasing"] == p)]
            if not sel:
                continue
            tgt_side = lambda i: "correct" if SIDE.get(gt[i]) == meta[i]["target_dir"] else (
                "opposite" if gt[i] in "LR" else {"A": "ambiguous", "N": "no shadow"}[gt[i]])
            c = Counter(tgt_side(i) for i in sel)
            names.append(f"target={d}" if d else f"phrasing={p}")
            data.append([100 * c[k] / len(sel) for k in ("correct", "opposite", "ambiguous", "no shadow")])
        data = np.array(data)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        left = np.zeros(len(names))
        colors = ["#2a9d8f", "#e76f51", "#e9c46a", "#8d99ae"]
        for k, lab in enumerate(("correct", "opposite", "ambiguous", "no shadow")):
            ax.barh(names, data[:, k], left=left, color=colors[k], label=lab)
            left += data[:, k]
        ax.set_xlabel("% of human-labeled base-model images")
        ax.set_title("SD3.5-M base model: human-judged light direction vs prompt")
        ax.legend(ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.45))
        fig.tight_layout()
        fig.savefig(args.out / "base_accuracy.png", dpi=150)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
