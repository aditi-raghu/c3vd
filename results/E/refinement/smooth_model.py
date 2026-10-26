"""The group's final model: Depth Anything V2 Small + Smooth Depth-Dependent Bias Correction.

Reimplemented exactly as in results/Depth Anything V2/Depth_Anything_V2_C3VD.ipynb (cells 19, 40-41, 47-48):
  1. official Depth Anything V2 code, ViT-S weights, model.infer_image(bgr, input_size=518)
  2. linear calibration:   depth = clip(-scale * pred + offset, 1e-3, 100)
  3. smooth correction:    depth = clip(depth + interp(depth, [10,30,50,70], corrections), 1e-3, 100)
Final parameters are read (never written) from results/Depth Anything V2/final_model_parameters.json.

The official code + weights live OUTSIDE this repo (not committed):
    git clone https://github.com/DepthAnything/Depth-Anything-V2 ~/Downloads/Depth-Anything-V2
    weights: https://huggingface.co/depth-anything/Depth-Anything-V2-Small -> checkpoints/depth_anything_v2_vits.pth
Set DA2_ROOT if you cloned it elsewhere.
"""
import json
import os
import sys
from pathlib import Path

import cv2
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # C3VD TIFFs: harmless metadata warning
import numpy as np

REPO = Path(__file__).resolve().parents[3]
DA2_ROOT = Path(os.environ.get("DA2_ROOT", Path.home() / "Downloads" / "Depth-Anything-V2"))
FINAL_PARAMS = REPO / "results" / "Depth Anything V2" / "final_model_parameters.json"
BINS = np.array([0, 20, 40, 60, 80, 100])
CENTERS = np.array([10, 30, 50, 70], dtype=np.float64)


def load_final_params():
    p = json.load(open(FINAL_PARAMS))
    return {"scale": p["scale"], "offset": p["offset"], "centers": np.array(p["bin_centers_mm"], float),
            "corrections": np.array(p["correction_values_mm"], float)}


def load_frame(color_path):
    """BGR uint8 image (as the notebook reads it), GT in mm, eval mask gt_raw > 0 (as the notebook)."""
    bgr = cv2.imread(str(color_path))
    raw = cv2.imread(str(color_path).replace("_color.png", "_depth.tiff"), cv2.IMREAD_UNCHANGED)
    return bgr, raw.astype(np.float32) / 65535.0 * 100.0, raw


def apply_smooth(pred, params):
    """Relative DA-V2 output -> depth in mm with the group's linear + smooth correction."""
    linear = np.clip(-params["scale"] * pred + params["offset"], 1e-3, 100.0)
    return np.clip(linear + np.interp(linear, params["centers"], params["corrections"]), 1e-3, 100.0)


def fit_params(preds, raws):
    """Re-derive the calibration on a set of dev frames exactly as the notebook does (for train/val tuning).
    Linear: OLS of GT on -pred over gt_raw > 0 (cell 40). Corrections: mean signed error per predicted-depth
    bin [0,20),[20,40),[40,60),[60,80),[80,100] (cells 41/47); the first four are used at centres 10..70 mm."""
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
    """Official Depth Anything V2 Small; infer_image exactly as in the notebook."""

    def __init__(self):
        import torch
        sys.path.insert(0, str(DA2_ROOT))
        from depth_anything_v2.dpt import DepthAnythingV2
        self.torch = torch
        self.model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
        self.model.load_state_dict(torch.load(DA2_ROOT / "checkpoints" / "depth_anything_v2_vits.pth",
                                              map_location="cpu", weights_only=True))
        dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = self.model.to(dev).eval()

    def __call__(self, bgr, input_size=518):
        with self.torch.inference_mode():
            return self.model.infer_image(bgr, input_size=input_size).astype(np.float32)
