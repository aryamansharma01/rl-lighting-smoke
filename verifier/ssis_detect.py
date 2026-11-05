"""Run SSISv2 on images (and their horizontal mirrors) and dump raw detections.

Runs inside the SSIS conda env (see env/README.md). Does NOT compute rewards;
verifier/direction.py does that from the dumped detections, so gating thresholds
can be changed without re-running the detector.

Per image writes <out>/<image_id>.npz (and <image_id>__flip.npz with --flip):
  cls       (N,)  int   0 = Object, 1 = Shadow (SOBA category order; checked in T0)
  score     (N,)  float SSIS instance score (after mask-IoU rescoring)
  pair      (N,)  int   association id; the two members of a pair share it
  offset    (N,2) float SSIS predicted offset (x,y) to the paired instance centre,
                        in units of 128 px of the resized network input
  location  (N,2) float FCOS location of the instance (resized input coords)
  masks     packed bool masks (N,H,W), unpack with np.unpackbits(..., count=H*W)
  hw        (2,)  original image H, W

Usage (from repo root, SSIS env active):
  python smoke/verifier/ssis_detect.py --ssis-root $SSIS --images results/T1/images_40steps \
      --out results/T2/raw_40steps --flip
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "env"))
import fast_mode  # noqa: E402  (runtime switch, see env/fast_mode.py)


def build_predictor(ssis_root, score_th):
    sys.path.insert(0, str(ssis_root))
    from adet.config import get_cfg
    from detectron2.engine.defaults import DefaultPredictor

    cfg = get_cfg()
    cfg.merge_from_file(str(ssis_root / "configs/SSIS/MS_R_101_BiFPN_SSISv2_demo.yaml"))
    cfg.MODEL.WEIGHTS = str(ssis_root / "tools/output/SSISv2_MS_R_101_bifpn_with_offset_class_maskiouv2_da_bl/model_ssisv2_final.pth")
    # same thresholds the official demo sets (demo.py --confidence-threshold default 0.1);
    # the reward-level confidence gate is applied later in direction.py
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = score_th
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_th
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_th
    cfg.freeze()
    return DefaultPredictor(cfg), cfg


def to_record(inst, hw):
    n = len(inst)
    if n == 0:
        return dict(cls=np.zeros(0, int), score=np.zeros(0), pair=np.zeros(0, int),
                    offset=np.zeros((0, 2)), location=np.zeros((0, 2)),
                    masks=np.packbits(np.zeros((0,), bool)), hw=np.array(hw))
    masks = inst.pred_masks
    masks = masks.numpy() if torch.is_tensor(masks) else np.asarray(masks)
    return dict(
        cls=np.asarray(inst.pred_classes.cpu() if torch.is_tensor(inst.pred_classes) else inst.pred_classes, dtype=int),
        score=inst.scores.cpu().numpy().astype(float),
        pair=np.asarray(inst.pred_associations, dtype=int),
        offset=inst.offset.cpu().numpy().astype(float),
        location=inst.locations.cpu().numpy().astype(float),
        masks=np.packbits(masks.astype(bool).ravel()),
        hw=np.array(hw),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssis-root", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path, help="dir of .png/.jpg")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--flip", action="store_true", help="also run on horizontal mirror")
    ap.add_argument("--score-th", type=float, default=0.1)
    args = ap.parse_args()
    fast_mode.check_steps(args.images)

    args.out.mkdir(parents=True, exist_ok=True)
    pred, cfg = build_predictor(args.ssis_root.resolve(), args.score_th)
    paths = sorted([p for p in args.images.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])

    # warm-up (excluded from timing)
    with torch.no_grad():
        pred(cv2.imread(str(paths[0])))
    torch.cuda.reset_peak_memory_stats()

    times = []
    for p in paths:
        img = cv2.imread(str(p))  # BGR, as the official demo
        variants = [("", img)] + ([("__flip", img[:, ::-1].copy())] if args.flip else [])
        for suffix, im in variants:
            torch.cuda.synchronize()
            t = time.time()
            with torch.no_grad():
                out = pred(im)
            torch.cuda.synchronize()
            times.append(time.time() - t)
            # DefaultPredictor returns model([x])[0] -> processed_results list; [0] -> this image
            inst = out[0]["instances"].to("cpu")
            np.savez_compressed(args.out / f"{p.stem}{suffix}.npz", **to_record(inst, im.shape[:2]))

    stats = dict(n_calls=len(times), sec_per_image=float(np.mean(times)),
                 sec_per_image_median=float(np.median(times)),
                 peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9,
                 gpu=torch.cuda.get_device_name(0), score_th=args.score_th,
                 min_size_test=cfg.INPUT.MIN_SIZE_TEST, torch=torch.__version__)
    (args.out / "ssis_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
