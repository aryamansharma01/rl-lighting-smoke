"""Light direction + reward from SSIS object-shadow pairs (pure numpy, no GPU).

verify(record, target) -> dict(gate, inferred_light_vec, R_dir, R_cons, pairs, confidences, ...)

Conventions (image coordinates: x to the right, y DOWN):
  light-from vector points from the object TOWARD the light, i.e. -(shadow - contact).
  target "left"  -> (-1, 0)     target "right" -> (+1, 0)

Per pair:
  contact c = mean (x,y) of object-mask pixels in the bottom BAND_FRAC of the object's bbox
  v         = centroid(shadow mask) - c
  light     = -v / |v|
Pair confidence = min(object score, shadow score); a pair is valid if
  confidence >= CONF_TH, shadow area >= MIN_SHADOW_FRAC of the image, and
  |v| >= MIN_LEN_FRAC of the image width (a shadow sitting right under the contact
  point has no usable direction).
Aggregation: confidence-weighted mean of unit vectors, then normalized. This avoids
the 180-degree wraparound of "from the left" that breaks a median of raw angles.
gate = 1 iff at least one valid pair.
R_dir = cos(angle(inferred, target)) = inferred . target (both unit); 0 when gate = 0.
R_side = 1 if the inferred side matches the target side, else 0 (binary, logged only; 0 when gate = 0).
reward = gate * R_dir (the cosine, not R_side).
R_cons = weighted mean resultant length of the valid pairs' unit vectors, None if < 2 pairs.

Also reports an independent estimate from SSIS's own association offsets
(shadow offset points shadow->object = light-from; object offset points object->shadow),
under "offset_*" keys. It's logged only, not used in the reward.

CLI: score every raw npz in a folder and write a jsonl joined with generation metadata:
  python direction.py --raw ../results/T2/raw_40steps --manifest ../results/T1/manifest_40steps.jsonl \
      --out ../results/T2/scores_40steps.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

OBJECT, SHADOW = 0, 1
CONF_TH = 0.3          # reward gate threshold on pair confidence (logged with every result)
BAND_FRAC = 0.10       # bottom 10% of the object bbox defines the contact region
MIN_SHADOW_FRAC = 1e-3 # shadow must cover >= 0.1% of the image
MIN_LEN_FRAC = 0.02    # |v| >= 2% of image width
TARGETS = {"left": np.array([-1.0, 0.0]), "right": np.array([1.0, 0.0])}


def load_record(path):
    z = np.load(path)
    rec = {k: z[k] for k in z.files}
    H, W = (int(x) for x in rec["hw"])
    n = len(rec["cls"])
    rec["masks"] = np.unpackbits(rec["masks"], count=n * H * W).reshape(n, H, W).astype(bool)
    return rec


def contact_point(mask, band_frac=BAND_FRAC):
    ys, xs = np.nonzero(mask)
    y0, y1 = ys.min(), ys.max()
    cut = y1 - band_frac * (y1 - y0 + 1)
    sel = ys >= cut
    return np.array([xs[sel].mean(), ys[sel].mean()])


def centroid(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()])


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def side_of(vec):
    if vec is None:
        return None
    return "left" if vec[0] < 0 else "right"


def weighted_mean_dir(vecs, w):
    if len(vecs) == 0:
        return None, 0.0
    r = (np.asarray(vecs) * np.asarray(w)[:, None]).sum(0) / np.sum(w)
    return unit(r), float(np.linalg.norm(r))


def verify(rec, target=None, conf_th=CONF_TH, band_frac=BAND_FRAC,
           min_shadow_frac=MIN_SHADOW_FRAC, min_len_frac=MIN_LEN_FRAC):
    H, W = rec["masks"].shape[1:] if len(rec["cls"]) else tuple(int(x) for x in rec["hw"])
    pairs = []
    for pid in np.unique(rec["pair"]):
        idx = np.nonzero(rec["pair"] == pid)[0]
        cls = rec["cls"][idx]
        if len(idx) != 2 or sorted(cls.tolist()) != [OBJECT, SHADOW]:
            continue  # malformed pair (SSIS should always give one object + one shadow)
        o, s = (idx[cls == OBJECT][0], idx[cls == SHADOW][0])
        om, sm = rec["masks"][o], rec["masks"][s]
        info = dict(pair=int(pid), obj_score=float(rec["score"][o]), sh_score=float(rec["score"][s]),
                    conf=float(min(rec["score"][o], rec["score"][s])),
                    obj_area=int(om.sum()), sh_area=int(sm.sum()))
        reasons = []
        if om.sum() == 0 or sm.sum() == 0:
            reasons.append("empty_mask")
        else:
            c = contact_point(om, band_frac)
            v = centroid(sm) - c
            info.update(contact=c.tolist(), shadow_centroid=centroid(sm).tolist(),
                        shadow_vec=v.tolist(), length=float(np.linalg.norm(v)))
            info["light_vec"] = unit(-v).tolist()
            # SSIS offset-based estimate (direction only; scale is irrelevant)
            info["offset_light_vec"] = unit(unit(rec["offset"][s]) - unit(rec["offset"][o])).tolist()
            if np.linalg.norm(v) < min_len_frac * W:
                reasons.append("shadow_too_short")
        if info["conf"] < conf_th:
            reasons.append("low_conf")
        if info["sh_area"] < min_shadow_frac * H * W:
            reasons.append("shadow_too_small")
        info["valid"] = not reasons
        info["invalid_reasons"] = reasons
        pairs.append(info)

    valid = [p for p in pairs if p["valid"]]
    gate = int(len(valid) > 0)
    w = [p["conf"] for p in valid]
    inferred, resultant = weighted_mean_dir([p["light_vec"] for p in valid], w)
    off_inferred, _ = weighted_mean_dir([p["offset_light_vec"] for p in valid], w)

    out = dict(gate=gate, n_pairs=len(pairs), n_valid=len(valid),
               inferred_light_vec=None if inferred is None else inferred.tolist(),
               inferred_side=side_of(inferred),
               inferred_angle_deg=None if inferred is None else float(np.degrees(np.arctan2(-inferred[1], inferred[0]))),
               offset_light_vec=None if off_inferred is None else off_inferred.tolist(),
               offset_side=side_of(off_inferred),
               R_dir=0.0, R_side=0, R_cons=resultant if len(valid) >= 2 else None,
               confidences=w, pairs=pairs, hw=[int(H), int(W)],
               params=dict(conf_th=conf_th, band_frac=band_frac,
                           min_shadow_frac=min_shadow_frac, min_len_frac=min_len_frac))
    if target is not None and gate:
        out["R_dir"] = float(np.dot(inferred, TARGETS[target]))
        out["R_side"] = int(side_of(inferred) == target)
    out["reward"] = gate * out["R_dir"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--conf-th", type=float, default=CONF_TH)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "env"))
    import fast_mode
    fast_mode.check_steps(args.raw)

    meta = {json.loads(l)["image_id"]: json.loads(l) for l in open(args.manifest)}
    n = 0
    with open(args.out, "w") as f:
        for p in sorted(args.raw.glob("*.npz")):
            stem = p.stem
            flipped = stem.endswith("__flip")
            image_id = stem[: -len("__flip")] if flipped else stem
            m = meta[image_id]
            # a mirrored image's true target is the opposite side
            target = m["target_dir"]
            if flipped:
                target = {"left": "right", "right": "left"}[target]
            r = verify(load_record(p), target, conf_th=args.conf_th)
            r.update(image_id=image_id, flipped=flipped, target_dir=target,
                     prompt_id=m["prompt_id"], seed=m["seed"], steps=m["steps"],
                     phrasing=m["phrasing"], object=m["object"], scene=m["scene"])
            f.write(json.dumps(r) + "\n")
            n += 1
    print(f"scored {n} files -> {args.out}")


if __name__ == "__main__":
    main()
