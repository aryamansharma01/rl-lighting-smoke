"""One contact sheet per prompt (all seeds in a grid, header with prompt id / target / phrasing).

Usage: python contact_sheets.py --root ../results/T1 --steps 40
"""
import argparse
import json
import textwrap
from collections import defaultdict
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "env"))
import fast_mode  # noqa: E402  (runtime switch, see env/fast_mode.py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--thumb", type=int, default=256)
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()
    fast_mode.check_steps(args.steps)

    root = Path(args.root)
    rows = [json.loads(l) for l in open(root / f"manifest_{args.steps}steps.jsonl")]
    by_prompt = defaultdict(list)
    for r in rows:
        by_prompt[r["prompt_id"]].append(r)

    out_dir = root / f"contact_sheets_{args.steps}steps"
    out_dir.mkdir(exist_ok=True)
    T, C = args.thumb, args.cols
    for pid, items in sorted(by_prompt.items()):
        items.sort(key=lambda r: r["seed"])
        nrow = (len(items) + C - 1) // C
        header = 70
        sheet = Image.new("RGB", (C * T, header + nrow * (T + 16)), "white")
        d = ImageDraw.Draw(sheet)
        p = items[0]
        title = f"{pid} | light from {p['target_dir'].upper()} | phrasing={p['phrasing']} | {args.steps} steps"
        d.text((6, 4), title, fill="black")
        for i, line in enumerate(textwrap.wrap(p["prompt"], width=C * T // 6)[:3]):
            d.text((6, 20 + 14 * i), line, fill=(60, 60, 60))
        for i, r in enumerate(items):
            im = Image.open(root / r["image"]).convert("RGB").resize((T, T))
            x, y = (i % C) * T, header + (i // C) * (T + 16)
            sheet.paste(im, (x, y))
            d.text((x + 4, y + T + 2), f"seed {r['seed']}", fill="black")
        sheet.save(out_dir / f"{pid}.jpg", quality=90)
    print(f"wrote {len(by_prompt)} sheets to {out_dir}")


if __name__ == "__main__":
    main()
