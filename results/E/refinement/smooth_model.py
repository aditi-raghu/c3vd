"""The group's base model: Depth Anything V2 Small + Smooth Depth-Dependent Bias Correction.

  1. official Depth Anything V2 code, ViT-S weights, model.infer_image(bgr, input_size=518)
  2. linear calibration:   depth = clip(-scale * pred + offset, 1e-3, 100)
  3. smooth correction:    depth = clip(depth + interp(depth, bin centres, corrections), 1e-3, 100)

fit_base() fits steps 2 and 3 under any mask and is what refine.py uses. load_final_params() reads the group notebook's
saved parameters from results/DAv2/final_model_parameters.json (fitted on raw > 0) and is kept for the reproduce check;
fit_params() is the notebook's own fit on raw > 0 and is used by notebooks/LumenAware_on_SmallSmooth.ipynb.

The official code + weights live OUTSIDE this repo (not committed):
    git clone https://github.com/DepthAnything/Depth-Anything-V2 ~/Downloads/Depth-Anything-V2
    weights: https://huggingface.co/depth-anything/Depth-Anything-V2-Small -> checkpoints/depth_anything_v2_vits.pth
Set DA2_ROOT, or da2_root in config.yaml, if you cloned it elsewhere.
"""
import json
import os
import sys
from pathlib import Path

import cv2
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # C3VD TIFFs: harmless metadata warning
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from common import CFG, DEVICE  # noqa: E402

DA2_ROOT = Path(os.environ.get("DA2_ROOT", CFG.get("da2_root", Path.home() / "Downloads" / "Depth-Anything-V2"))).expanduser()
FINAL_PARAMS = REPO / "results" / "DAv2" / "final_model_parameters.json"
BINS = np.array([0, 20, 40, 60, 80, 100], dtype=np.float64)
CENTERS = np.array([10, 30, 50, 70], dtype=np.float64)
ALL_CENTERS = (BINS[:-1] + BINS[1:]) / 2
MIN_BIN_PIXELS = 2000


def load_final_params():
    p = json.load(open(FINAL_PARAMS))
    return {"scale": p["scale"], "offset": p["offset"], "centers": np.array(p["bin_centers_mm"], float),
            "corrections": np.array(p["correction_values_mm"], float)}


def apply_smooth(pred, params):
    """Relative DA-V2 output -> depth in mm with the group's linear + smooth correction."""
    linear = np.clip(-params["scale"] * pred + params["offset"], 1e-3, 100.0)
    return np.clip(linear + np.interp(linear, params["centers"], params["corrections"]), 1e-3, 100.0)


def fit_base(preds, gts, valids):
    """Linear + smooth fit on the valid pixels of the given frames: least squares of GT on -pred, then the mean signed
    error per 20 mm bin of linear depth. A bin needs at least MIN_BIN_PIXELS pixels."""
    x = np.concatenate([-p[v] for p, v in zip(preds, valids)]).astype(np.float64)
    y = np.concatenate([g[v] for g, v in zip(gts, valids)]).astype(np.float64)
    xm, ym = x.mean(), y.mean()
    scale = ((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum()
    offset = ym - scale * xm
    lin = np.clip(scale * x + offset, 1e-3, 100.0)
    k = np.clip(np.digitize(lin, BINS) - 1, 0, len(ALL_CENTERS) - 1)
    s, n = np.bincount(k, y - lin, len(ALL_CENTERS)), np.bincount(k, minlength=len(ALL_CENTERS))
    ok = n >= MIN_BIN_PIXELS
    return {"scale": float(scale), "offset": float(offset), "centers": ALL_CENTERS[ok], "corrections": s[ok] / n[ok]}


def fit_params(preds, raws):
    """The group notebook's fit on raw > 0: least squares of GT on -pred, then the mean signed error per predicted-depth
    bin [0,20),[20,40),[40,60),[60,80),[80,100]; the first four are used at centres 10..70 mm."""
    X = np.concatenate([-p[r > 0] for p, r in zip(preds, raws)]).astype(np.float64)
    y = np.concatenate([(r[r > 0].astype(np.float64) / 65535.0 * 100.0) for r in raws])
    A = np.stack([X, np.ones_like(X)], 1)
    (scale, offset), *_ = np.linalg.lstsq(A, y, rcond=None)
    lin = np.clip(scale * X + offset, 1e-3, 100.0); err = y - lin
    corr = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (lin >= lo) & ((lin <= hi) if hi == 100 else (lin < hi))
        corr.append(float(err[m].mean()) if m.any() else 0.0)
    return {"scale": float(scale), "offset": float(offset), "centers": CENTERS, "corrections": np.array(corr[:4])}


class DA2Small:
    """Official Depth Anything V2 Small; infer_image exactly as in the group notebook."""

    def __init__(self):
        import torch
        sys.path.insert(0, str(DA2_ROOT))
        from depth_anything_v2.dpt import DepthAnythingV2
        self.torch = torch
        self.model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
        self.model.load_state_dict(torch.load(DA2_ROOT / "checkpoints" / "depth_anything_v2_vits.pth",
                                              map_location="cpu", weights_only=True))
        self.model = self.model.to(DEVICE).eval()

    def __call__(self, bgr, input_size=518):
        with self.torch.inference_mode():
            return self.model.infer_image(bgr, input_size=input_size).astype(np.float32)
