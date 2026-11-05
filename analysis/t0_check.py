"""T0: summarize SSIS raw detections on demo images and check class semantics.

Shadows are dark: if class 1 really is 'Shadow', its masks should have much lower mean
luminance than class-0 ('Object') masks. Also reports instance/pair counts and scores.
Usage: python t0_check.py --images DIR --raw DIR --out results/T0/t0_summary.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verifier"))
from direction import load_record  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True, type=Path)
ap.add_argument("--raw", required=True, type=Path)
ap.add_argument("--out", required=True, type=Path)
args = ap.parse_args()

per_image, lum = {}, {0: [], 1: []}
for npz in sorted(args.raw.glob("*.npz")):
    if npz.stem.endswith("__flip"):
        continue
    img = next(p for p in args.images.iterdir() if p.stem == npz.stem)
    gray = np.asarray(Image.open(img).convert("L")).astype(float)
    rec = load_record(npz)
    rows = []
    for c, s, p, m, off in zip(rec["cls"], rec["score"], rec["pair"], rec["masks"], rec["offset"]):
        L = float(gray[m].mean()) if m.any() else None
        if L is not None:
            lum[int(c)].append(L)
        rows.append(dict(cls=int(c), score=round(float(s), 3), pair=int(p), area=int(m.sum()),
                         mean_lum=None if L is None else round(L, 1), offset=np.round(off, 3).tolist()))
    per_image[npz.stem] = dict(n_instances=len(rows), n_pairs=len(set(rec["pair"].tolist())), instances=rows)

summary = dict(
    mean_luminance_class0=float(np.mean(lum[0])) if lum[0] else None,
    mean_luminance_class1=float(np.mean(lum[1])) if lum[1] else None,
    class1_is_darker_in_frac_of_pairs=None,
    per_image=per_image)
# per pair: is the class-1 member darker than its class-0 partner?
darker = []
for im in per_image.values():
    by_pair = {}
    for r in im["instances"]:
        by_pair.setdefault(r["pair"], {})[r["cls"]] = r["mean_lum"]
    darker += [d[1] < d[0] for d in by_pair.values() if 0 in d and 1 in d and None not in d.values()]
summary["class1_is_darker_in_frac_of_pairs"] = float(np.mean(darker)) if darker else None
summary["n_pairs_total"] = len(darker)
args.out.parent.mkdir(parents=True, exist_ok=True)
json.dump(summary, open(args.out, "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "per_image"}, indent=1))
for k, v in per_image.items():
    print(k, v["n_instances"], "instances,", v["n_pairs"], "pairs")
