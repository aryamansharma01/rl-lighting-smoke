"""Draw verifier overlays: object masks (blue), shadow masks (red), contact point, shadow centroid,
inferred light-from arrow (yellow, drawn from the object toward the light) and target arrow (green).

Usage:
  python overlay.py --images DIR --raw DIR --out DIR [--ids id1 id2 ...] [--scores jsonl] [--conf-th 0.3]
If --ids is omitted, every npz in --raw is drawn.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verifier"))
from direction import TARGETS, load_record, verify  # noqa: E402


def arrow(d, start, vec, length, color, width=4):
    end = (start[0] + vec[0] * length, start[1] + vec[1] * length)
    d.line([tuple(start), end], fill=color, width=width)
    # arrow head
    ang = np.arctan2(vec[1], vec[0])
    for s in (+1, -1):
        a = ang + np.pi + s * 0.45
        d.line([end, (end[0] + 14 * np.cos(a), end[1] + 14 * np.sin(a))], fill=color, width=width)


def draw(img_path, rec, res, target=None, caption=""):
    im = Image.open(img_path).convert("RGB")
    if rec["masks"].shape[1:] != (im.size[1], im.size[0]):
        raise ValueError("mask/image size mismatch")
    base = np.asarray(im).astype(float)
    over = base.copy()
    for c, m in zip(rec["cls"], rec["masks"]):
        col = np.array([40, 90, 255]) if c == 0 else np.array([255, 40, 40])
        over[m] = 0.45 * base[m] + 0.55 * col
    out = Image.fromarray(over.astype(np.uint8))
    d = ImageDraw.Draw(out)
    L = 0.22 * im.size[0]
    for p in res["pairs"]:
        if "contact" not in p:
            continue
        c, s = p["contact"], p["shadow_centroid"]
        col = "white" if p["valid"] else (150, 150, 150)
        d.ellipse([c[0] - 5, c[1] - 5, c[0] + 5, c[1] + 5], outline=col, width=3)
        d.line([tuple(c), tuple(s)], fill=col, width=2)
    if res["gate"]:
        anchor = res["pairs"][[p["valid"] for p in res["pairs"]].index(True)]["contact"]
        arrow(d, anchor, res["inferred_light_vec"], L, (255, 220, 0))
    if target:
        arrow(d, (im.size[0] / 2, 18), TARGETS[target], L * 0.6, (0, 220, 0), width=3)
    txt = (f"gate={res['gate']} pairs={res['n_valid']}/{res['n_pairs']} "
           f"inferred={res['inferred_side']} R_dir={res['R_dir']:+.2f}")
    d.rectangle([0, im.size[1] - 34, im.size[0], im.size[1]], fill=(0, 0, 0))
    d.text((4, im.size[1] - 32), txt, fill="white")
    d.text((4, im.size[1] - 17), caption[:90], fill=(200, 200, 200))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--manifest", type=Path, help="to look up target_dir per image")
    ap.add_argument("--conf-th", type=float, default=0.3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {}
    if args.manifest:
        meta = {json.loads(l)["image_id"]: json.loads(l) for l in open(args.manifest)}
    ids = args.ids or [p.stem for p in sorted(args.raw.glob("*.npz"))]
    for iid in ids:
        base_id = iid.replace("__flip", "")
        img = next((args.images / f"{base_id}{ext}" for ext in (".png", ".jpg") if (args.images / f"{base_id}{ext}").exists()), None)
        rec = load_record(args.raw / f"{iid}.npz")
        target = meta.get(base_id, {}).get("target_dir")
        if target and iid.endswith("__flip"):
            target = {"left": "right", "right": "left"}[target]
        res = verify(rec, target, conf_th=args.conf_th)
        im = Image.open(img)
        if iid.endswith("__flip"):
            tmp = args.out / "_flip_tmp.png"
            im.transpose(Image.FLIP_LEFT_RIGHT).save(tmp)
            img = tmp
        draw(img, rec, res, target, caption=f"{iid} target={target}").save(args.out / f"{iid}.jpg", quality=90)
    print(f"wrote {len(ids)} overlays to {args.out}")


if __name__ == "__main__":
    main()
