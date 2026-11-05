"""T2 summary: gate rate, pair stats, flip test, verifier-side vs prompted-side (no human labels).

Usage: python t2_flip.py --scores ../results/T2/scores_40steps.jsonl --out ../results/T2/t2_summary.json
"""
import argparse
import json
from collections import Counter

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.scores)]
    orig = {r["image_id"]: r for r in rows if not r["flipped"]}
    flip = {r["image_id"]: r for r in rows if r["flipped"]}

    gated = [r for r in orig.values() if r["gate"]]
    reasons = Counter(x for r in orig.values() for p in r["pairs"] for x in p["invalid_reasons"])
    confs = [p["conf"] for r in orig.values() for p in r["pairs"]]

    def flip_stats(key):
        both = n_flip = n_mirror_gate_out = 0
        for iid, r in orig.items():
            if not r["gate"] or iid not in flip:
                continue
            m = flip[iid]
            if not m["gate"]:
                n_mirror_gate_out += 1
                continue
            both += 1
            a, b = r[key], m[key]
            if a is not None and b is not None and np.sign(a[0]) == -np.sign(b[0]) != 0:
                n_flip += 1
        n = sum(1 for iid, r in orig.items() if r["gate"] and iid in flip)
        return dict(n_gated_original=n, mirror_gated_out=n_mirror_gate_out, both_gated=both,
                    sign_flip_rate_both_gated=n_flip / both if both else None,
                    sign_flip_rate_counting_gate_out_as_fail=n_flip / n if n else None)

    # Left/right bias of the verifier (on originals; prompts are balanced L/R)
    side_counts = Counter(r["inferred_side"] for r in gated)
    agree_target = [r["inferred_side"] == r["target_dir"] for r in gated]
    # gate flips between original and mirror (should be rare: mirror is the same scene)
    gate_agree = np.mean([orig[i]["gate"] == flip[i]["gate"] for i in orig if i in flip]) if flip else None

    summary = dict(
        n_images=len(orig), conf_th=rows[0]["params"]["conf_th"] if rows else None,
        gate_rate=len(gated) / len(orig) if orig else None,
        pairs_per_image=Counter(r["n_pairs"] for r in orig.values()),
        valid_pairs_per_image=Counter(r["n_valid"] for r in orig.values()),
        invalid_reasons=reasons,
        pair_conf_quantiles=dict(zip(["p10", "p25", "p50", "p75", "p90"],
                                     np.quantile(confs, [.1, .25, .5, .75, .9]).round(3).tolist())) if confs else None,
        flip_test_mask_geometry=flip_stats("inferred_light_vec"),
        flip_test_ssis_offset=flip_stats("offset_light_vec"),
        gate_agreement_original_vs_mirror=gate_agree,
        inferred_side_counts_on_gated=side_counts,
        verifier_side_equals_prompted_side=float(np.mean(agree_target)) if agree_target else None,
        mask_vs_offset_side_agreement=float(np.mean([r["inferred_side"] == r["offset_side"] for r in gated])) if gated else None,
        R_cons_defined=sum(r["R_cons"] is not None for r in orig.values()),
        mean_reward_by_target={t: float(np.mean([r["reward"] for r in orig.values() if r["target_dir"] == t]))
                               for t in ("left", "right")},
    )
    json.dump(summary, open(args.out, "w"), indent=1, default=lambda o: dict(o) if isinstance(o, Counter) else str(o))
    print(json.dumps(summary, indent=1, default=lambda o: dict(o) if isinstance(o, Counter) else str(o)))


if __name__ == "__main__":
    main()
