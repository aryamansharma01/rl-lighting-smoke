"""T8 (optional): VLM judge baseline for light direction, run locally on Apple silicon with MLX.

Model: Qwen2.5-VL-7B-Instruct, 4-bit MLX conversion (mlx-community/Qwen2.5-VL-7B-Instruct-4bit),
falls back to the 3B model with --model if 7B does not fit in memory. Greedy decoding, max 4 tokens.
Each image is also judged mirrored (--flip) to measure the left/right bias and flip consistency.

Compared against: human majority labels (T3), the SSIS verifier (T2), and the prompted side.

Usage:
  python t8_vlm.py --images ../results/T1/images_40steps --manifest ../results/T1/manifest_40steps.jsonl \
      --out ../results/T8 [--scores ../results/T2/scores_40steps.jsonl] [--label-dir ../label] [--flip] [--limit N]
Re-running resumes: images already in t8_vlm.jsonl are skipped.
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

QUESTION = ("Look at the shadow cast by the main object in this image. From which side of the image is the "
            "light coming? Answer with exactly one word: left, right, ambiguous, or none "
            "(none = no visible cast shadow).")
WORDS = ("left", "right", "ambiguous", "none")


def parse(text):
    t = text.strip().lower()
    for w in WORDS:
        if t.startswith(w):
            return w
    hits = [w for w in WORDS if w in t]
    return hits[0] if len(hits) == 1 else "unparsed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default="mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
    ap.add_argument("--scores", type=Path)
    ap.add_argument("--label-dir", type=Path)
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    meta = {json.loads(l)["image_id"]: json.loads(l) for l in open(args.manifest)}
    ids = sorted(meta)[: args.limit or None]
    out_path = args.out / "t8_vlm.jsonl"
    done = set()
    if out_path.exists():
        done = {(r["image_id"], r["flipped"]) for r in map(json.loads, open(out_path))}

    t0 = time.time()
    model, processor = load(args.model)
    load_s = time.time() - t0
    prompt = apply_chat_template(processor, model.config, QUESTION, num_images=1)
    tmp = Path(tempfile.mkdtemp())
    times = []
    with open(out_path, "a") as f:
        for iid in ids:
            for flipped in ([False, True] if args.flip else [False]):
                if (iid, flipped) in done:
                    continue
                path = args.images / f"{iid}.png"
                if flipped:
                    p2 = tmp / f"{iid}__flip.png"
                    ImageOps.mirror(Image.open(path)).save(p2)
                    path = p2
                t = time.time()
                res = generate(model, processor, prompt, [str(path)], max_tokens=4, temperature=0.0, verbose=False)
                text = res.text if hasattr(res, "text") else str(res)
                times.append(time.time() - t)
                target = meta[iid]["target_dir"]
                if flipped:
                    target = {"left": "right", "right": "left"}[target]
                f.write(json.dumps(dict(image_id=iid, flipped=flipped, raw=text, answer=parse(text),
                                        target_dir=target, prompt_id=meta[iid]["prompt_id"], seed=meta[iid]["seed"],
                                        model=args.model, sec=times[-1])) + "\n")
                f.flush()
            print(f"{iid} ({np.mean(times):.1f}s/img)", flush=True)

    rows = [json.loads(l) for l in open(out_path)]
    orig = {r["image_id"]: r for r in rows if not r["flipped"]}
    flip = {r["image_id"]: r for r in rows if r["flipped"]}
    side = lambda a: a if a in ("left", "right") else None
    summ = dict(model=args.model, n=len(orig), load_s=load_s,
                sec_per_image=float(np.mean([r["sec"] for r in rows])),
                answer_counts={w: sum(r["answer"] == w for r in orig.values()) for w in WORDS + ("unparsed",)},
                prompt_agree_on_lr=None, flip_consistent=None)
    lr = [r for r in orig.values() if side(r["answer"])]
    if lr:
        summ["prompt_agree_on_lr"] = float(np.mean([r["answer"] == r["target_dir"] for r in lr]))
    both = [i for i in orig if i in flip and side(orig[i]["answer"]) and side(flip[i]["answer"])]
    if both:
        summ["flip_n"] = len(both)
        summ["flip_consistent"] = float(np.mean([orig[i]["answer"] != flip[i]["answer"] for i in both]))
    if args.scores and args.scores.exists():
        sc = {r["image_id"]: r for r in map(json.loads, open(args.scores)) if not r["flipped"]}
        b = [i for i in orig if side(orig[i]["answer"]) and i in sc and sc[i]["gate"]]
        summ["verifier_agree_n"] = len(b)
        summ["verifier_agree"] = float(np.mean([orig[i]["answer"] == sc[i]["inferred_side"] for i in b])) if b else None
    if args.label_dir and list(args.label_dir.glob("labels_*.csv")):
        sys.path.insert(0, str(Path(__file__).parent))
        from t4_metrics import ground_truth, load_labels
        _, per_image, _ = load_labels(args.label_dir)
        gt = {i: ground_truth(l) for i, l in per_image.items() if l}
        S = {"L": "left", "R": "right"}
        lab = [i for i in orig if gt.get(i) in S]
        cov = [i for i in lab if side(orig[i]["answer"])]
        summ.update(human_lr_n=len(lab), coverage_on_human_lr=len(cov) / max(len(lab), 1),
                    accuracy_on_covered=(float(np.mean([orig[i]["answer"] == S[gt[i]] for i in cov])) if cov else None),
                    accuracy_all_human_lr=(float(np.mean([orig[i]["answer"] == S[gt[i]] for i in lab])) if lab else None),
                    says_lr_on_human_N=(float(np.mean([bool(side(orig[i]["answer"])) for i in orig if gt.get(i) == "N"]))
                                        if any(gt.get(i) == "N" for i in orig) else None))
    (args.out / "t8_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
