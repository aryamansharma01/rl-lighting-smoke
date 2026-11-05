"""Figures for REPORT.md (reduced plan, Modal results).

fig1_accuracy.png   side accuracy judged by humans: base model (overall, by phrasing) vs verifier best-of-4,
                    plus verifier accuracy / coverage / flip consistency, with 95% Wilson intervals
fig2_reward_vs_human.png  per-prompt rewards of the 4 seeds, colored by whether humans judged the image correct
fig3_overlays.png   verifier overlays: two correct cases and two failure modes
(T7/T6 numbers are tabulated in the report; a chart would add nothing.)

Palette: reference categorical slots 1-2 (blue #2a78d6, orange #eb6834), neutral gray for context marks.
Usage: python make_figures.py --results ../results_modal --label-dir ../label --out ../results_modal/figures
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a3a29c"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SIDE = {"L": "left", "R": "right"}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": SURF})


def bars(ax, rows, ref_lines, title):
    """rows: (label, metric dict with pct/ci95/k/n, color)."""
    y = list(range(len(rows)))[::-1]
    for yi, (lab, m, col) in zip(y, rows):
        ax.barh(yi, m["pct"], height=0.55, color=col, zorder=2)
        lo, hi = m["ci95"]
        ax.plot([lo, hi], [yi, yi], color=INK, lw=1.2, zorder=3)
        ax.text(min(hi + 1.5, 101), yi, f"{m['pct']:.0f}%  ({m['k']}/{m['n']})", va="center", fontsize=9, color=INK2)
    for x, lab in ref_lines:
        ax.axvline(x, color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
        ax.text(x, -0.62, lab, ha="center", va="top", fontsize=8.5, color=INK2,
                bbox=dict(facecolor=SURF, edgecolor="none", pad=0.5))
    ax.set_ylim(-0.95, len(rows) - 0.5)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)


def fig1(m, out):
    b = m["base_side_accuracy"]
    bo = m["best_of_n"]
    n = bo["n_prompts"]
    k_best = round(bo["best_acc_pct"] * n / 100)
    from fast_metrics import wilson
    best = dict(pct=bo["best_acc_pct"], k=k_best, n=n, ci95=wilson(k_best, n))
    v = m["verifier_vs_humans"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 5.2), gridspec_kw=dict(height_ratios=[4, 3], hspace=0.55))
    bars(a1, [("Base model: random sample", b["all"], GRAY),
              ("  prompt says “light from …”", b["by_phrasing"]["light"], GRAY),
              ("  prompt says “shadow falls …”", b["by_phrasing"]["shadow"], GRAY),
              ("Verifier’s best of 4 seeds", best, BLUE)],
         [(50, "chance")], "Does the image follow the prompted light side? (human-judged, 95% CI)")
    bars(a2, [("Verifier accuracy (gated, human L/R)", v["accuracy_on_gated_LR"], BLUE),
              ("Verifier coverage (human L/R)", v["coverage_on_human_LR"], BLUE),
              ("Flip consistency", m["flip_consistency"], BLUE)],
         [(85, "85%"), (95, "95%")], "Verifier vs human labels")
    a2.set_xlabel("percent")
    fig.savefig(out / "fig1_accuracy.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig2(scores, gt, out):
    orig = [s for s in scores if not s["flipped"]]
    by_p = defaultdict(list)
    for s in orig:
        by_p[s["prompt_id"]].append(s)
    order = sorted(by_p, key=lambda p: (by_p[p][0]["object"], by_p[p][0]["scene"], by_p[p][0]["target_dir"]))
    fig, ax = plt.subplots(figsize=(7.2, 7.4))
    seen = set()
    for yi, p in enumerate(order[::-1]):
        g = by_p[p]
        ax.plot([min(s["reward"] for s in g), max(s["reward"] for s in g)], [yi, yi], color=GRID, lw=2, zorder=1)
        top = max(g, key=lambda s: (s["reward"], -s["seed"]))
        for s in g:
            lab = gt.get(s["image_id"])
            if lab in SIDE:
                ok = SIDE[lab] == s["target_dir"]
                col, name = (BLUE, "humans: follows prompt") if ok else (ORANGE, "humans: wrong side")
            else:
                col, name = GRAY, "humans: ambiguous"
            ax.scatter(s["reward"], yi, s=46, color=col, edgecolor=SURF, linewidth=1.5, zorder=3,
                       label=name if name not in seen else None)
            seen.add(name)
            if s is top:
                ax.scatter(s["reward"], yi, s=150, facecolor="none", edgecolor=INK, linewidth=1, zorder=4,
                           label="verifier’s top-1" if "top" not in seen else None)
                seen.add("top")
        s0 = g[0]
        ax.text(1.06, yi, f"{s0['object']} · {s0['scene'].split()[-1]} · {s0['target_dir']} · {s0['phrasing']}",
                va="center", fontsize=8, color=INK2)
    ax.axvline(0, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.set_yticks([])
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("reward = gate · cos(inferred light, prompted side)   (4 seeds per prompt)")
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.set_title("Reward spreads within every group, and its top-1 matches humans in 22/24 prompts",
                 loc="left", fontsize=10.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=4, frameon=False, fontsize=8.5,
              handletextpad=0.3, columnspacing=1.2)
    fig.savefig(out / "fig2_reward_vs_human.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig3(results, out):
    import textwrap
    cases = [("p12_s0_n40", "Correct: sun on the left, shadow falls right (reward +0.93)"),
             ("p00_s1_n40", "Correct but near-vertical: backlit ball, slight lean left (reward +0.30)"),
             ("p19_s1_n40", "Failure: contact point lands on one leg of an oblique bench (reward −0.89)"),
             ("p16_s2_n40", "Failure: SSIS swaps the object and shadow classes (reward −0.81)")]
    fig, axs = plt.subplots(2, 2, figsize=(8.4, 9.2), gridspec_kw=dict(hspace=0.22, wspace=0.06))
    for ax, (iid, cap) in zip(axs.flat, cases):
        ax.imshow(Image.open(results / "T2/overlays_40steps" / f"{iid}.jpg"))
        ax.set_axis_off()
        ax.set_title(iid + "\n" + textwrap.fill(cap, 52), fontsize=8.5, loc="left", color=INK)
    fig.text(0.125, 0.07, "Blue = SSIS object mask, red = shadow mask, yellow = inferred light-from direction, "
             "green = prompted side.", fontsize=8.5, color=INK2)
    fig.savefig(out / "fig3_overlays.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--label-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from fast_metrics import load_labels
    gt, _, _ = load_labels(args.label_dir)
    m = json.load(open(args.results / "fast/fast_metrics.json"))
    scores = [json.loads(l) for l in open(args.results / "T2_scores_40steps.jsonl")]
    fig1(m, args.out)
    fig2(scores, gt, args.out)
    fig3(args.results, args.out)
    print("wrote", sorted(p.name for p in args.out.glob("*.png")))


if __name__ == "__main__":
    main()
