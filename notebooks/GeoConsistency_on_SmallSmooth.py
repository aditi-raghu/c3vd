"""Novelty: geometry-consistency depth correction on top of the group base model (DAv2 Small + smooth correction).

Problem   A predicted depth map can have locally implausible jumps on an otherwise smooth colon wall.
Method    From the predicted depth only: local spike evidence (log-depth vs its 5x5 median) x surface-normal
          inconsistency (pseudo-normals from log-depth gradients vs their 7x7 mean). Where both are high, the depth
          is pulled to the local median, unless a strong RGB edge supports the jump (real folds are protected).
Ablation  Original (base) -> conventional smoothing (Gaussian, bilateral) -> geometry-consistency (full),
          plus the full method without RGB protection and without the normal term.

Base model = Sowmya's DAv2 Small + smooth (notebooks/(New)_DAv2_Model_Comparison.ipynb, results/DAv2_MaskA_Verified):
  DAv2 Small, official infer_image(input_size=518) preprocessing (518 x 644 for C3VD frames)
  linear:  D = clip(scale * (-pred) + offset, 1e-3, 100), least squares on all mask-A dev pixels
  smooth:  D0 = clip(D + interp(D, [10, 30, 50, 70], corrections), 1e-3, 100); corrections = mean (GT - D) in
           [0, 20, 40, 60, 80, 100] mm bins of D, all mask-A dev pixels
Protocol = the other novelties (LumenAware_on_SmallSmooth):
  dev (trans_t1_a, 61 frames)  4-block CV with a 2-frame gap (common.dev_folds). In each fold the base is refitted on
                               the fit frames, and the filter settings are chosen on the fit frames (AbsRel, mask A)
  test (trans_t2_b, 103)       base fitted on all dev (= Sowmya's parameters), settings chosen on all dev; run ONCE
                               with --test (guarded by test_done.json)
  masks                        A (main) and B (second table)
  metrics                      common.compute_metrics / common.aggregate: AbsRel, RMSE, MAE (mm), delta < 1.25;
                               AbsRel wall / lumen (group lumen map, w > 0.5 = lumen); folds better than the base
  geometry extras              AbsRel at boundaries (top 10% GT log-depth gradient, eroded mask A) and on flat wall,
                               spike rate; paired block bootstrap on test (blocks of 10 frames)
Filters run at 1/4 resolution (--work-scale); their change of log-depth is upsampled and added to the full-resolution
base, and all metrics are computed at 1080 x 1350.

Run from the repo root (or anywhere below the folder with common.py and config.yaml):
    python notebooks/GeoConsistency_on_SmallSmooth.py              dev CV only (choose and check settings)
    python notebooks/GeoConsistency_on_SmallSmooth.py --test       dev CV, then the single test run
    python notebooks/GeoConsistency_on_SmallSmooth.py --limit 8    quick check on 8 dev frames (no test)
DAv2 Small: the official code is used if a Depth-Anything-V2 clone with checkpoints/depth_anything_v2_vits.pth is found
(DA2_ROOT or next to the repo); otherwise the Hugging Face port of the same weights with the official preprocessing.

Output: results/GeoConsistency_on_SmallSmooth/ (same files as results/LumenAware_on_SmallSmooth)
"""
import argparse
import itertools
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

HERE = Path(__file__).resolve().parent
ROOT = next((d for d in (HERE, *HERE.parents) if (d / "common.py").exists()), HERE)
sys.path.insert(0, str(ROOT))
from common import (CFG, DEVICE, SEED, aggregate, compute_metrics, dev_folds, list_frames, make_mask,  # noqa: E402
                    save_run_info, save_summary, stream_results_dir, sync)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tifffile  # noqa: E402
import torch  # noqa: E402
from scipy import ndimage  # noqa: E402
from tqdm import tqdm  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------- settings
RUN_NAME = "GeoConsistency_on_SmallSmooth"
INPUT_SIZE = 518                       # same as the group base model
DMAX, EPS = 100.0, 1e-3
BINS = np.array([0, 20, 40, 60, 80, 100], dtype=np.float64)
CENTERS = np.array([10, 30, 50, 70], dtype=np.float64)
N_BLOCKS, GAP = 4, 2
MASKS = ["A", "B"]
SIGMA, TAU = 20, 0.5                   # group lumen detector (only for the wall / lumen columns)
WORK_SCALE = 4                         # filters at 1/4 resolution (set with --work-scale)
K_NORMAL = 10.0                        # pseudo-normal factor (log-depth space)
SPIKE_THR = 0.05                       # |log D - median5(log D)| > this = spike (spike-rate measure)
EDGE_FRAC = 0.10                       # boundary pixels = top 10% GT log-depth gradient
BLOCK, N_BOOT = 10, 2000               # block bootstrap on test frames
GRIDS = {"gauss": {"sigma": [0.5, 1.0, 1.5, 2.0]},
         "bilateral": {"sc": [0.05, 0.1, 0.2]},
         "geometry": {"tau": [0.02, 0.04, 0.08], "ang0": [5.0, 10.0, 20.0], "lam": [0.0, 1.0], "iters": [1, 2, 3]}}
BASE, FULL = "Base: Small + smooth (group)", "+ geometry-consistency (full)"
METHODS = [BASE, "+ Gaussian smoothing", "+ bilateral smoothing", FULL,
           "  - without RGB edge protection", "  - without normal-consistency term"]
SOWMYA_PARAMS = {"scale": 13.321026124555381, "offset": 64.22639769237846,           # results/DAv2_MaskA_Verified
                 "corrections": [1.1854982670035257, -4.087490818371066, 1.6420928046873542, 23.402725890497383]}
SOWMYA_TEST_A = {"absrel": 0.28052515114812326, "rmse": 5.595781375673298, "mae": 4.409883934945352,
                 "delta1": 0.5349643815520103}                                         # small_smooth, mask A, test
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


# ---------------------------------------------------------------- DAv2 Small (official preprocessing)
class DA2Small:
    """Official Depth-Anything-V2 code if available (exact group pipeline); otherwise the Hugging Face port of the same
    weights fed with the official infer_image preprocessing (518 lower bound, multiple of 14, cubic resize, ImageNet
    normalisation, bilinear align_corners=True upsampling)."""

    def __init__(self):
        roots = [Path(os.environ["DA2_ROOT"])] if os.environ.get("DA2_ROOT") else []
        roots += [ROOT.parent / "Depth-Anything-V2", ROOT / "Depth-Anything-V2", Path.home() / "Downloads" / "Depth-Anything-V2"]
        root = next((r for r in roots if (r / "checkpoints" / "depth_anything_v2_vits.pth").exists()), None)
        self.times, self.calls = [], 0
        if root is not None:
            sys.path.insert(0, str(root))
            from depth_anything_v2.dpt import DepthAnythingV2
            self.model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
            self.model.load_state_dict(torch.load(root / "checkpoints" / "depth_anything_v2_vits.pth",
                                                  map_location="cpu", weights_only=True))
            self.model = self.model.to(DEVICE).eval()
            self.source = f"official code ({root})"
        else:
            from transformers import AutoModelForDepthEstimation
            self.model = AutoModelForDepthEstimation.from_pretrained(
                "depth-anything/Depth-Anything-V2-Small-hf").to(DEVICE).eval()
            self.source = "Hugging Face port (depth-anything/Depth-Anything-V2-Small-hf) + official preprocessing"
        print(f"DAv2 Small: {self.source} on {DEVICE}")

    @staticmethod
    def preprocess(bgr, size=INPUT_SIZE):
        h, w = bgr.shape[:2]
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.0
        scale = max(size / h, size / w)                          # 'lower_bound', keep aspect ratio

        def multiple14(x, min_val):
            y = int(np.round(x / 14) * 14)
            return int(np.ceil(x / 14) * 14) if y < min_val else y
        nh, nw = multiple14(scale * h, size), multiple14(scale * w, size)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
        img = (img - MEAN) / STD
        return np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.float32)

    def __call__(self, bgr):
        h, w = bgr.shape[:2]
        sync(DEVICE)
        t0 = time.perf_counter()
        with torch.no_grad():
            if hasattr(self.model, "infer_image"):
                d = torch.from_numpy(self.model.infer_image(bgr, input_size=INPUT_SIZE))
            else:
                x = torch.from_numpy(self.preprocess(bgr)).unsqueeze(0).to(DEVICE)
                d = self.model(pixel_values=x).predicted_depth
                d = torch.nn.functional.interpolate(d[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        sync(DEVICE)
        if self.calls:                                           # first frame = warm-up
            self.times.append(time.perf_counter() - t0)
        self.calls += 1
        return d.float().cpu().numpy()


_MODEL, CACHE = {}, None


def get_pred(split, cp, bgr):
    for folder in (CACHE, ROOT / "results" / "E" / "refinement" / "cache"):      # own cache, Stream E's cache
        if folder is not None:
            f = folder / split / f"{cp.stem.split('_')[0]}_id.npy"
            if f.exists():
                return np.load(f).astype(np.float32)
    if "m" not in _MODEL:
        _MODEL["m"] = DA2Small()
    pred = _MODEL["m"](bgr).astype(np.float32)
    if CACHE is not None:
        f = CACHE / split / f"{cp.stem.split('_')[0]}_id.npy"
        f.parent.mkdir(parents=True, exist_ok=True)
        np.save(f, pred)
    return pred


# ---------------------------------------------------------------- base: Sowmya's DAv2 Small + smooth (mask A)
def fit_base(frames):
    """Closed-form least squares of GT on -pred over all mask-A pixels, then mean residual per bin (Sowmya)."""
    n, sx, sy, sxx, sxy = 0, 0.0, 0.0, 0.0, 0.0
    for F in frames:
        v = F["mask"]["A"]
        x, y = (-F["pred"])[v].astype(np.float64), F["gt"][v].astype(np.float64)
        n += len(x); sx += x.sum(); sy += y.sum(); sxx += np.dot(x, x); sxy += np.dot(x, y)
    scale = (sxy - sx * sy / n) / (sxx - sx * sx / n)
    scale = float(scale)
    offset = float((sy - scale * sx) / n)                   # plain floats, as Sowmya's fit_linear returns
    sums, counts = np.zeros(5), np.zeros(5, dtype=np.int64)
    for F in frames:
        v = F["mask"]["A"]
        lin = np.clip(scale * (-F["pred"]) + offset, EPS, DMAX)
        idx = np.clip(np.searchsorted(BINS, lin, side="right") - 1, 0, 4)
        for k in range(5):
            m = v & (idx == k)
            sums[k] += np.sum(F["gt"][m] - lin[m], dtype=np.float64)
            counts[k] += np.count_nonzero(m)
    corr = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return {"scale": float(scale), "offset": float(offset), "corrections": corr[:4].tolist(), "counts": counts.tolist()}


def apply_base(pred, P):
    lin = np.clip(P["scale"] * (-pred) + P["offset"], EPS, DMAX)
    return np.clip(lin + np.interp(lin, CENTERS, P["corrections"]), EPS, DMAX)      # same dtypes as Sowmya's code


# ---------------------------------------------------------------- group lumen map (for the wall / lumen columns)
def blur(x, sigma=SIGMA):
    return cv2.GaussianBlur(x.astype(np.float32), (0, 0), sigma)


def lumen_map(rgb, pred, fov):
    """w in [0, 1] from darkness x farness (cue 'both'), largest core above TAU; no ground truth."""
    f = fov.astype(np.float32)
    v = rgb.astype(np.float32).max(axis=2) / 255.0
    v_s = blur(v * f) / (blur(f) + 1e-6)
    dark = 1.0 - np.clip(v_s / (np.percentile(v_s[fov], 99) + 1e-6), 0, 1)
    far = np.zeros_like(v_s)
    ranks = (-pred)[fov].argsort().argsort()
    far[fov] = ranks / max(len(ranks) - 1, 1)
    far = blur(far * f) / (blur(f) + 1e-6)
    dark[~fov], far[~fov] = 0, 0
    score = np.sqrt(dark * far)
    mask = (score > TAU) & fov
    lab, n = ndimage.label(mask)
    if n == 0:
        return np.zeros(score.shape, np.float16)
    core = lab == (np.argmax(ndimage.sum(mask, lab, range(1, n + 1))) + 1)
    w = blur(core * score)
    w = w / (w.max() + 1e-6)
    w[~fov] = 0
    return w.astype(np.float16)


# ---------------------------------------------------------------- frames
def load_frame(split, cp):
    bgr = cv2.imread(str(cp))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    raw = tifffile.imread(cp.with_name(cp.name.replace("_color.png", "_depth.tiff")))
    pred = get_pred(split, cp, bgr)
    fov = rgb.max(axis=2) > 10
    F = {"name": cp.stem.split("_")[0], "path": cp, "pred": pred, "gt": raw.astype(np.float32) / 65535.0 * DMAX,
         "fov": fov,
         "mask": {m: make_mask(raw, rgb, m) for m in MASKS}, "w": lumen_map(rgb, pred, fov)}
    H, W = raw.shape
    size = (W // WORK_SCALE, H // WORK_SCALE)
    small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA) if WORK_SCALE > 1 else rgb
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0, (0, 0), 1.0)
    mag = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    F["work"] = {"size": size, "edge": np.clip(mag / (np.percentile(mag, 95) + 1e-6), 0, 1),
                 "fov": cv2.resize(fov.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0}
    inner = cv2.erode(F["mask"]["A"].astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    lg = np.log(np.maximum(F["gt"], EPS))
    grad = np.hypot(cv2.Sobel(lg, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(lg, cv2.CV_32F, 0, 1, ksize=3))
    edge = inner & (grad >= np.quantile(grad[inner], 1 - EDGE_FRAC)) if inner.any() else inner
    F["regions"] = (edge, inner & ~edge)
    return F


# ---------------------------------------------------------------- filters (log-depth, work resolution)
def to_work(D, F):
    L = np.log(D).astype(np.float32)
    return L if WORK_SCALE == 1 else cv2.resize(L, F["work"]["size"], interpolation=cv2.INTER_AREA)


def to_full(D, Ls, Lf):
    delta = Lf - Ls
    if delta.shape != D.shape:
        delta = cv2.resize(delta, (D.shape[1], D.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(D * np.exp(delta), EPS, DMAX).astype(np.float32)


def f_gauss(L, F, sigma):
    return cv2.GaussianBlur(L, (0, 0), sigma)


def f_bilateral(L, F, sc):
    return cv2.bilateralFilter(L, 7, sc, 3)


def geometry_steps(L, F, tau, ang0, lam, iters):
    """Yields log-depth after each iteration: alpha = spike evidence x normal inconsistency x (1 - lam * RGB edge)."""
    protect = lam * F["work"]["edge"]
    for _ in range(iters):
        med = cv2.medianBlur(L, 5)
        s_depth = np.clip((np.abs(L - med) - tau) / tau, 0, 1)
        Lx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        Ly = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        n = np.dstack([-K_NORMAL * Lx, -K_NORMAL * Ly, np.ones_like(L)])
        n /= np.linalg.norm(n, axis=2, keepdims=True)
        nbar = cv2.boxFilter(n, -1, (7, 7))
        nbar /= np.linalg.norm(nbar, axis=2, keepdims=True) + 1e-8
        ang = cv2.dilate(np.degrees(np.arccos(np.clip((n * nbar).sum(2), -1, 1))), np.ones((3, 3), np.uint8))
        alpha = s_depth * np.clip(ang / ang0, 0, 1) * (1.0 - protect)
        L = ((1 - alpha) * L + alpha * med).astype(np.float32)
        yield L


def f_geometry(L, F, tau, ang0, lam, iters):
    for L in geometry_steps(L, F, tau, ang0, lam, iters):
        pass
    return L


FILTERS = {"gauss": f_gauss, "bilateral": f_bilateral, "geometry": f_geometry}


def refine(D0, F, kind, params):
    Ls = to_work(D0, F)
    return to_full(D0, Ls, FILTERS[kind](Ls, F, **params))


def method_depths(F, P):
    D0 = apply_base(F["pred"], P["base"])
    s = P["settings"]
    return {BASE: D0,
            "+ Gaussian smoothing": refine(D0, F, "gauss", s["gauss"]),
            "+ bilateral smoothing": refine(D0, F, "bilateral", s["bilateral"]),
            FULL: refine(D0, F, "geometry", s["geometry"]),
            "  - without RGB edge protection": refine(D0, F, "geometry", {**s["geometry"], "lam": 0.0}),
            "  - without normal-consistency term": refine(D0, F, "geometry", {**s["geometry"], "ang0": 1e-6})}


# ---------------------------------------------------------------- choosing the settings (fit frames only)
def absrel_A(D, F):
    v = F["mask"]["A"]
    g = F["gt"][v].astype(np.float64)
    return float(np.mean(np.abs(D[v].astype(np.float64) - g) / g))


def choose_settings(frames, base):
    """Every grid setting on the given (fit) frames; the lowest mean mask-A AbsRel wins. Returns settings + log."""
    held = [(apply_base(F["pred"], base), F) for F in frames]
    held = [(D0, to_work(D0, F), F) for D0, F in held]
    log, best = [], {}
    for kind in ("gauss", "bilateral"):
        res = []
        for combo in itertools.product(*GRIDS[kind].values()):
            p = dict(zip(GRIDS[kind], combo))
            res.append((np.mean([absrel_A(to_full(D0, Ls, FILTERS[kind](Ls, F, **p)), F) for D0, Ls, F in held]), p))
        log += [{"filter": kind, **p, "absrel_A": a} for a, p in res]
        best[kind] = min(res, key=lambda t: t[0])[1]
    g, res = GRIDS["geometry"], []
    for tau, ang0, lam in itertools.product(g["tau"], g["ang0"], g["lam"]):
        scores = {it: [] for it in g["iters"]}
        for D0, Ls, F in held:
            for it, Lf in enumerate(geometry_steps(Ls, F, tau, ang0, lam, max(g["iters"])), 1):
                if it in scores:
                    scores[it].append(absrel_A(to_full(D0, Ls, Lf), F))
        res += [(np.mean(s), {"tau": tau, "ang0": ang0, "lam": lam, "iters": it}) for it, s in scores.items()]
    log += [{"filter": "geometry", **p, "absrel_A": a} for a, p in res]
    best["geometry"] = min(res, key=lambda t: t[0])[1]
    base_absrel = float(np.mean([absrel_A(D0, F) for D0, _, F in held]))
    return best, pd.DataFrame(log).assign(base_absrel_A=base_absrel)


# ---------------------------------------------------------------- evaluation (same columns as the lumen-aware novelty)
def region_absrel(pred, F, v):
    """AbsRel on a pixel subset (same formula as common.compute_metrics); NaN below 100 pixels, as the group."""
    if v.sum() < 100:
        return np.nan
    p, g = np.clip(pred[v], 1e-3, None).astype(np.float64), F["gt"][v].astype(np.float64)
    return float(np.mean(np.abs(p - g) / g))


def spike_rate(D, F):
    L = to_work(D, F)
    fov = F["work"]["fov"] if WORK_SCALE > 1 else F["fov"]
    return float(np.mean((np.abs(L - cv2.medianBlur(L, 5)) > SPIKE_THR)[fov]))


def eval_frame(F, P, extra):
    lumen = F["w"].astype(np.float32) > 0.5
    edge, flat = F["regions"]
    rows = []
    for label, pred in method_depths(F, P).items():
        spikes = spike_rate(pred, F)
        for m in MASKS:
            v = F["mask"][m]
            rows.append({**extra, "frame": F["name"], "method": label, "mask": m, **compute_metrics(pred, F["gt"], v),
                         "absrel_wall": region_absrel(pred, F, v & ~lumen),
                         "absrel_lumen": region_absrel(pred, F, v & lumen),
                         "absrel_boundary": region_absrel(pred, F, v & edge),
                         "absrel_flat": region_absrel(pred, F, v & flat), "spike_rate": spikes})
    return rows


def summarise(pf):
    out = []
    for m in MASKS:
        sub = pf[pf["mask"] == m]
        if "fold" in sub:
            fold_ab = sub.groupby(["method", "fold"]).absrel.mean().unstack()
            better = fold_ab.lt(fold_ab.loc[BASE], axis=1).sum(axis=1)
        for label in METHODS:
            g = sub[sub.method == label]
            row = {"mask": m, "method": label, **aggregate(g),
                   "absrel_wall": g.absrel_wall.mean(), "absrel_lumen": g.absrel_lumen.mean()}
            if "fold" in sub:
                row["folds_better"] = np.nan if label == BASE else int(better[label])
            row.update({"absrel_boundary": g.absrel_boundary.mean(), "absrel_flat": g.absrel_flat.mean(),
                        "spike_rate": g.spike_rate.mean()})
            out.append(row)
    return pd.DataFrame(out)


def group_rows(tab, split):
    rows = []
    for r in tab.itertuples():
        name = {BASE: "DAv2 Small + smooth (group base)",
                "  - without RGB edge protection": "+ geometry-consistency without RGB edge protection",
                "  - without normal-consistency term": "+ geometry-consistency without normal-consistency term"
                }.get(r.method, r.method)
        name = name if r.method == BASE else f"DAv2 Small + smooth {name}"
        rows.append({"model": name, "output_type": "disparity", "alignment": "global_lsq + smooth",
                     "fit_on": "dev_blocks_cv" if split == "dev" else "dev_all", "split": split, "mask": r.mask,
                     "n_frames": r.n_frames, "absrel": r.absrel, "rmse": r.rmse, "mae": r.mae, "delta1": r.delta1})
    return rows


def block_bootstrap(pf, rng):
    """Paired moving-block bootstrap over test frames (mask A): method - base, same averaging as the table."""
    base = pf[(pf["mask"] == "A") & (pf.method == BASE)].set_index("frame").sort_index()
    n = len(base)
    block = max(1, min(BLOCK, n // 5))
    starts = rng.integers(0, n - block + 1, size=(N_BOOT, int(np.ceil(n / block))))
    idx = (starts[:, :, None] + np.arange(block)).reshape(N_BOOT, -1)[:, :n]
    rows = []
    for label in METHODS[1:]:
        x = pf[(pf["mask"] == "A") & (pf.method == label)].set_index("frame").loc[base.index]
        for col in ("absrel", "rmse", "delta1"):
            if col == "rmse":
                d = np.sqrt(x.mse.mean()) - np.sqrt(base.mse.mean())
                db = np.sqrt(x.mse.values[idx].mean(1)) - np.sqrt(base.mse.values[idx].mean(1))
            else:
                d = x[col].mean() - base[col].mean()
                db = (x[col].values - base[col].values)[idx].mean(1)
            lo, hi = np.percentile(db, [2.5, 97.5])
            good = (hi < 0) if col != "delta1" else (lo > 0)
            bad = (lo > 0) if col != "delta1" else (hi < 0)
            rows.append({"method": label.strip(), "metric": col, "diff_vs_base": d, "ci_low": lo, "ci_high": hi,
                         "verdict": "better" if good else "worse" if bad else "no clear difference",
                         "block_frames": block})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
def fig_ablation(tab, title, path):
    t = tab[tab["mask"] == "A"]
    colors = ["#8c8b86", "#eb6834", "#1baf7a", "#2a78d6", "#9fc3ee", "#9fc3ee"]
    panels = [("absrel", "AbsRel ↓", "%.4f"), ("rmse", "RMSE mm ↓", "%.3f"), ("delta1", "δ<1.25 ↑", "%.4f"),
              ("absrel_boundary", "AbsRel boundary ↓", "%.4f"), ("absrel_flat", "AbsRel flat ↓", "%.4f"),
              ("spike_rate", "Spike rate ↓", "%.5f")]
    fig, axes = plt.subplots(1, len(panels), figsize=(25, 4.2), sharey=True, constrained_layout=True)
    labels = [m.strip() for m in t.method][::-1]
    for ax, (col, lbl, fmt) in zip(axes, panels):
        vals = t[col].values[::-1]
        bars = ax.barh(labels, vals, color=colors[:len(vals)][::-1], height=0.6)
        ax.bar_label(bars, fmt=fmt, padding=3, fontsize=8, color="#52514e")
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        pad = (hi - lo) * 0.6 + 1e-9
        ax.set_xlim(max(lo - pad, 0), hi + pad)
        ax.set_title(lbl, loc="left", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title + "  (mask A; x-axes zoomed, values on the bars)", x=0.01, ha="left")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_examples(split, names, P, path):
    paths = {cp.stem.split("_")[0]: cp for cp in list_frames(CFG, split)}
    fig, axes = plt.subplots(len(names), 6, figsize=(24, 3.6 * len(names)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for r, nm in enumerate(names):
        F = load_frame(split, paths[nm])
        rgb = cv2.cvtColor(cv2.imread(str(paths[nm])), cv2.COLOR_BGR2RGB)
        v = F["mask"]["A"]
        nan = lambda a: np.where(v, a, np.nan)  # noqa: E731
        D = method_depths(F, P)
        D0, Dg = D[BASE], D[FULL]
        ax = axes[r]
        ax[0].imshow(rgb); ax[0].set_title(f"{split} frame {nm}")
        im = ax[1].imshow(nan(F["gt"]), cmap="magma", vmin=0, vmax=DMAX); ax[1].set_title("GT (mm, mask A)")
        ax[2].imshow(nan(D0), cmap="magma", vmin=0, vmax=DMAX); ax[2].set_title(f"Base\nAbsRel {absrel_A(D0, F):.4f}")
        ax[3].imshow(nan(Dg), cmap="magma", vmin=0, vmax=DMAX)
        ax[3].set_title(f"+ geometry-consistency\nAbsRel {absrel_A(Dg, F):.4f}")
        ic = ax[4].imshow(np.where(rgb.max(axis=2) > 10, np.abs(np.log(Dg / D0)), np.nan), cmap="viridis", vmin=0, vmax=0.1)
        ax[4].set_title("|log change| by geometry")
        ie = ax[5].imshow(nan(np.abs(Dg - F["gt"])), cmap="inferno", vmin=0, vmax=30); ax[5].set_title("|Error| (mm)")
        for a_ in ax:
            a_.axis("off")
    fig.colorbar(im, ax=axes[:, 1:4], fraction=0.012, label="Depth (mm)")
    fig.colorbar(ic, ax=axes[:, 4], fraction=0.04, label="|log change|")
    fig.colorbar(ie, ax=axes[:, 5], fraction=0.04, label="|Error| (mm)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- run
def show(tab, title):
    cols = ["method", "absrel", "rmse", "mae", "delta1", "absrel_wall", "absrel_lumen", "absrel_boundary",
            "absrel_flat", "spike_rate"] + (["folds_better"] if "folds_better" in tab else [])
    f4 = "{:.4f}".format
    print(f"\n{title}")
    print(tab[cols].to_string(index=False, formatters={
        "absrel": f4, "rmse": "{:.3f}".format, "mae": "{:.3f}".format, "delta1": f4, "absrel_wall": f4,
        "absrel_lumen": f4, "absrel_boundary": f4, "absrel_flat": f4, "spike_rate": "{:.5f}".format,
        "folds_better": lambda x: "" if pd.isna(x) else f"{int(x)}/{N_BLOCKS}"}))


def main():
    global WORK_SCALE, CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="run the single test after the dev CV")
    ap.add_argument("--limit", type=int, default=0, help="first N dev frames only (quick check; no test)")
    ap.add_argument("--work-scale", type=int, default=4, help="filter resolution = full / this")
    a = ap.parse_args()
    WORK_SCALE = a.work_scale
    out_dir = stream_results_dir(CFG, RUN_NAME + (f"_limit{a.limit}" if a.limit else ""))
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    test_done = out_dir / "test_done.json"
    cache = ROOT / "results" / "cache" / "dav2_small_518"
    probe = cache
    while not probe.exists():
        probe = probe.parent
    CACHE = cache if shutil.disk_usage(probe).free / 1e9 > 3.0 else None      # no cache on a nearly full disk
    if a.test and a.limit:
        sys.exit("--test needs all frames (no --limit).")
    if a.test and test_done.exists():
        sys.exit(f"The test was already run ({test_done} exists). It is run only once.")
    dev_paths = list_frames(CFG, "dev")[:a.limit or None]
    print(f"{RUN_NAME} | {len(dev_paths)} dev frames | device {DEVICE} | masks {MASKS} | work scale 1/{WORK_SCALE} | "
          f"prediction cache {'on' if CACHE else 'off (low disk)'}")

    # ---- dev: 4-block CV; base refitted and settings chosen on the fit frames of each fold ----
    FR = [load_frame("dev", cp) for cp in tqdm(dev_paths, desc="dev frames")]
    folds = dev_folds(len(FR), N_BLOCKS, GAP)
    rows, fold_params, logs = [], [], []
    for k, (fit_ids, held_ids) in enumerate(tqdm(folds, desc="dev folds")):
        base = fit_base([FR[i] for i in fit_ids])
        settings, log = choose_settings([FR[i] for i in fit_ids], base)
        P = {"base": base, "settings": settings}
        fold_params.append(P)
        logs.append(log.assign(fold=k + 1))
        for i in held_ids:
            rows += eval_frame(FR[i], P, {"idx": i, "fold": k + 1})
    dev_pf = pd.DataFrame(rows)
    dev_tab = summarise(dev_pf)
    dev_pf.to_csv(out_dir / "per_frame_dev_cv.csv", index=False)
    dev_tab.to_csv(out_dir / "dev_cv_summary.csv", index=False)

    # ---- parameters for the test: base and settings on all dev frames ----
    base_all = fit_base(FR)
    settings_all, log_all = choose_settings(FR, base_all)
    P_all = {"base": base_all, "settings": settings_all}
    pd.concat(logs + [log_all.assign(fold="all dev")]).to_csv(out_dir / "tuning_dev.csv", index=False)
    with open(out_dir / "params_dev.json", "w", encoding="utf-8") as f:
        json.dump({"base_all_dev": base_all, "base_per_fold": [p["base"] for p in fold_params],
                   "settings_all_dev": settings_all, "settings_per_fold": [p["settings"] for p in fold_params],
                   "base_reference_sowmya": SOWMYA_PARAMS, "grids": GRIDS, "work_scale": WORK_SCALE,
                   "k_normal": K_NORMAL, "spike_thr": SPIKE_THR, "edge_frac": EDGE_FRAC, "sigma": SIGMA, "tau": TAU,
                   "input_size": INPUT_SIZE, "bins": BINS.tolist(), "centers": CENTERS.tolist()}, f, indent=2)
    save_summary(group_rows(dev_tab, "dev"), out_dir, RUN_NAME)
    model = _MODEL.get("m")
    save_run_info(out_dir, DEVICE, SEED, extra={
        "base_model": "DAv2 Small + smooth, Sowmya's mask-A calibration (results/DAv2_MaskA_Verified)",
        "dav2_source": model.source if model else "cached predictions", "input_size": INPUT_SIZE,
        "n_blocks": N_BLOCKS, "gap": GAP, "masks": MASKS, "work_scale": WORK_SCALE,
        "dav2_sec_per_frame": float(np.mean(model.times)) if model and model.times else None})

    for m in MASKS:
        show(dev_tab[dev_tab["mask"] == m], f"Dev held-out, {N_BLOCKS}-block CV on {CFG['dev_sequence']} "
                                             f"({len(FR)} frames), mask {m}")
    print("\nSettings chosen on all dev frames (used for the test):", settings_all)
    if not a.limit:
        d = {k: base_all[k] - SOWMYA_PARAMS[k] for k in ("scale", "offset")}
        dc = float(np.max(np.abs(np.subtract(base_all["corrections"], SOWMYA_PARAMS["corrections"]))))
        ok = abs(d["scale"]) < 0.05 and abs(d["offset"]) < 0.2 and dc < 0.2
        print(f"Base on all dev vs Sowmya's parameters: scale {base_all['scale']:.4f} (hers {SOWMYA_PARAMS['scale']:.4f}),"
              f" offset {base_all['offset']:.4f} (hers {SOWMYA_PARAMS['offset']:.4f}), max correction diff {dc:.3f} mm"
              f" -> {'OK, same base model' if ok else 'CHECK: different predictions or data'}")
    fig_ablation(dev_tab, f"Dev held-out ({len(FR)} frames, {N_BLOCKS}-block CV)", fig_dir / "fig1_ablation_dev.png")
    if not a.test:
        print(f"\nDev results saved to {out_dir}. When the settings look right, run once with --test.")
        return

    # ---- test: run ONCE ----
    del FR
    test_rows = []
    for cp in tqdm(list_frames(CFG, "test"), desc="test frames"):
        test_rows += eval_frame(load_frame("test", cp), P_all, {})
    test_pf = pd.DataFrame(test_rows)
    test_tab = summarise(test_pf)
    boot = block_bootstrap(test_pf, np.random.default_rng(SEED))
    test_pf.to_csv(out_dir / "per_frame_test.csv", index=False)
    test_tab.to_csv(out_dir / "test_summary.csv", index=False)
    boot.to_csv(out_dir / "block_bootstrap_test.csv", index=False)
    save_summary(group_rows(dev_tab, "dev") + group_rows(test_tab, "test"), out_dir, RUN_NAME)
    for m in MASKS:
        show(test_tab[test_tab["mask"] == m], f"Test {CFG['test_sequence']} ({test_pf.frame.nunique()} frames), "
                                              f"mask {m}; base and settings fitted on all dev")
    T = test_tab[test_tab["mask"] == "A"].set_index("method")
    print(f"\nBase vs Sowmya's test numbers (mask A): AbsRel {T.absrel[BASE]:.4f} ({SOWMYA_TEST_A['absrel']:.4f}), "
          f"RMSE {T.rmse[BASE]:.3f} ({SOWMYA_TEST_A['rmse']:.3f}), delta {T.delta1[BASE]:.4f} ({SOWMYA_TEST_A['delta1']:.4f})")
    print("\nPaired block bootstrap on test (mask A, method - base, 95% CI):")
    for r in boot.itertuples():
        print(f"  {r.method:<36} {r.metric:<7} {r.diff_vs_base:+.5f}  [{r.ci_low:+.5f}, {r.ci_high:+.5f}]  {r.verdict}")
    ab = boot[boot.metric == "absrel"]
    full = ab[ab.method == FULL.strip()].iloc[0]
    verdict = (f"Geometry-consistency (full) vs base on test: AbsRel {full.diff_vs_base:+.4f} "
               f"(95% CI [{full.ci_low:+.4f}, {full.ci_high:+.4f}], {full.verdict}); largest change of any filter "
               f"{ab.diff_vs_base.abs().max():.4f} AbsRel.")
    print(f"\nVERDICT: {verdict}")
    with open(test_done, "w", encoding="utf-8") as f:
        json.dump({"date": datetime.now().isoformat(timespec="seconds"), "n_frames": int(test_pf.frame.nunique()),
                   "absrel_mask_A": T.absrel.to_dict(), "verdict": verdict}, f, indent=2)
    fig_ablation(test_tab, f"Test {CFG['test_sequence']}", fig_dir / "fig2_ablation_test.png")
    names = sorted(test_pf.frame.unique())
    fig_examples("test", [names[i] for i in np.linspace(0, len(names) - 1, 3).astype(int)], P_all,
                 fig_dir / "fig3_test_examples.png")
    print(f"Test done. Saved to {out_dir}")


if __name__ == "__main__":
    main()