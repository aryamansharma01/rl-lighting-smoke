"""Synthetic sanity tests for direction.py (no GPU). Run: python test_direction.py"""
import numpy as np

from direction import verify

H = W = 200


def rect(y0, y1, x0, x1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def record(pairs):
    """pairs: list of (object_mask, shadow_mask, obj_score, sh_score)."""
    cls, score, pair, masks, offset = [], [], [], [], []
    for i, (om, sm, so, ss) in enumerate(pairs):
        cls += [0, 1]; score += [so, ss]; pair += [i + 1, i + 1]; masks += [om, sm]
        co, cs = np.argwhere(om).mean(0)[::-1], np.argwhere(sm).mean(0)[::-1]
        offset += [(cs - co) / 128, (co - cs) / 128]
    return dict(cls=np.array(cls), score=np.array(score), pair=np.array(pair),
                masks=np.array(masks).reshape(-1, H, W), offset=np.array(offset),
                hw=np.array([H, W]))


def mirror(rec):
    r = dict(rec)
    r["masks"] = rec["masks"][:, :, ::-1]
    r["offset"] = rec["offset"] * np.array([-1, 1])
    return r


# object: post standing at x 90..110, y 60..120; shadow on the ground to the right
obj = rect(60, 120, 90, 110)
sh_right = rect(112, 124, 110, 170)

r = verify(record([(obj, sh_right, 0.9, 0.8)]), "left")
assert r["gate"] == 1 and r["inferred_side"] == "left", r
assert r["R_dir"] > 0.95, r["R_dir"]
assert r["offset_side"] == "left"
assert r["R_cons"] is None  # single pair -> skipped
assert abs(abs(r["inferred_angle_deg"]) - 180) < 20, r["inferred_angle_deg"]

# mirror: x-component must flip sign exactly (pure geometry)
rm = verify(mirror(record([(obj, sh_right, 0.9, 0.8)])), "right")
assert rm["inferred_side"] == "right" and rm["R_dir"] > 0.95
assert np.isclose(rm["inferred_light_vec"][0], -r["inferred_light_vec"][0], atol=0.02)

# wrong-target reward is negative
assert verify(record([(obj, sh_right, 0.9, 0.8)]), "right")["R_dir"] < -0.95

# wraparound: shadows at ~+-10 degrees around "to the right" -> light from ~180 degrees, not ~0
o1, o2 = rect(30, 60, 20, 30), rect(130, 160, 20, 30)
s_up = rect(40, 50, 60, 120)    # slightly above contact row of o1 (y~58), to the right
s_dn = rect(165, 175, 60, 120)  # slightly below contact row of o2 (y~158), to the right
r2 = verify(record([(o1, s_up, 0.9, 0.9), (o2, s_dn, 0.9, 0.9)]), "left")
assert r2["n_valid"] == 2 and r2["inferred_side"] == "left", r2
assert abs(abs(r2["inferred_angle_deg"]) - 180) < 15, r2["inferred_angle_deg"]
assert r2["R_cons"] is not None and r2["R_cons"] > 0.9
assert r2["R_dir"] > 0.95

# disagreeing pairs -> low consistency
o3 = rect(60, 120, 90, 110)
s_left = rect(112, 124, 20, 90)
r3 = verify(record([(o1, s_up, 0.9, 0.9), (o3, s_left, 0.9, 0.9)]), "left")
assert r3["R_cons"] < 0.5, r3["R_cons"]

# gating: low confidence, no pairs, shadow directly under the object
assert verify(record([(obj, sh_right, 0.9, 0.1)]), "left")["gate"] == 0
assert verify({**record([]), "masks": np.zeros((0, H, W), bool)}, "left")["gate"] == 0
under = rect(115, 119, 92, 108)  # centroid 1.5 px below contact (< 2% of width)
g = verify(record([(obj, under, 0.9, 0.9)]), "left")
assert g["gate"] == 0 and "shadow_too_short" in g["pairs"][0]["invalid_reasons"], g["pairs"]
assert g["reward"] == 0.0

print("all direction tests passed")
