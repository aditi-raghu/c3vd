import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
import logging, warnings, pickle, itertools
logging.getLogger("tifffile").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================= SETTINGS =======================
MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"
TAG = "da2_small"
MODE = "quick"                       # "quick" | "report"
PRESETS = {"quick": (20, 20), "report": (10, 5)}      # (DEV_STRIDE, TEST_STRIDE)
DEV_STRIDE, TEST_STRIDE = PRESETS[MODE]

ALIGN = "median"                     # "median": per-image GT median scaling (uses GT -> upper bound, like teammates)
                                     # "global": ONE scale/shift fitted on dev only (the fair protocol)
REF_SIGMA = 1.0                      # the "smoothing filter" of the reference pipeline (Gaussian, log-depth). Edit to match teammates

SCALE = 4                            # work at 1/4 resolution
K_NORMAL = 10.0                      # pseudo-normal factor (log-depth space)
MAX_DEPTH_MM = 100.0
DEPTH_CLIP = (0.5, 100.0)
BASE = Path(__file__).resolve().parent
DEV, TEST = "trans_t1_a", "trans_t2_b"
CACHE_DIR = BASE / "cache"
CACHE_DIR.mkdir(exist_ok=True)
# ========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
_model = {}


def get_model():
    if not _model:
        print(f"Loading {MODEL_NAME} on {device}")
        _model["p"] = AutoImageProcessor.from_pretrained(MODEL_NAME)
        _model["m"] = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME).to(device).eval()
    return _model["p"], _model["m"]


def frame_pairs(seq):
    for folder in (BASE / seq / seq, BASE / seq):
        depths = sorted(folder.glob("*_depth.tif*"))
        if depths:
            pairs = [(d.with_name(d.name.split("_depth")[0] + "_color.png"), d) for d in depths]
            return [(c, d) for c, d in pairs if c.exists()]
    raise SystemExit(f"No depth files found for {seq}")


def load_depth_mm(path):
    raw = tifffile.imread(str(path))
    assert raw.dtype == np.uint16
    return raw.astype(np.float32) / 65535.0 * MAX_DEPTH_MM


def load_rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


@torch.no_grad()
def predict_rel(rgb, out_shape):
    proc, model = get_model()
    inputs = proc(images=rgb, return_tensors="pt").to(device)
    out = model(**inputs).predicted_depth.unsqueeze(1)
    out = F.interpolate(out, size=out_shape, mode="bicubic", align_corners=False)
    return out.squeeze().float().cpu().numpy()


# ---------------- 1. cache model predictions ----------------
def build_cache(seq, stride, name):
    f = CACHE_DIR / f"{TAG}_{name}_s{stride}.pkl"
    if f.exists():
        print(f"Loaded cache {f.name}")
        return pickle.load(open(f, "rb"))
    items = []
    for c, d in tqdm(frame_pairs(seq)[::stride], desc=f"predict {name}"):
        gt = load_depth_mm(d)
        h, w = gt.shape[0] // SCALE, gt.shape[1] // SCALE
        gt_s = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
        rgb_s = cv2.resize(load_rgb(c), (w, h), interpolation=cv2.INTER_AREA)
        rgb_full = load_rgb(c)
        items.append(dict(name=d.name, rgb=rgb_s, gt=gt_s,
                          pred=predict_rel(rgb_full, (h, w)).astype(np.float32)))
    pickle.dump(items, open(f, "wb"))
    return items


# ---------------- metrics ----------------
def metrics(pred, gt, mask):
    p, g = pred[mask], gt[mask]
    return (np.mean(np.abs(p - g) / g),                 # AbsRel
            np.sqrt(np.mean((p - g) ** 2)),             # RMSE
            np.mean(np.abs(p - g)),                     # MAE
            np.mean(np.maximum(p / g, g / p) < 1.25))   # delta1


def region_masks(gt, mask):
    """Edge region = top 10% GT log-depth gradient (folds). Smooth region = the rest."""
    Lg = np.log(np.where(mask, gt, np.nan))
    gy, gx = np.gradient(Lg)
    mag = np.hypot(gx, gy)
    ok = np.isfinite(mag)
    thr = np.percentile(mag[ok], 90)
    return (mag >= thr) & ok & mask, (mag < thr) & ok & mask


def spike_rate(D, thr=0.05):
    L = np.log(D).astype(np.float32)
    return float(np.mean(np.abs(L - cv2.medianBlur(L, 5)) > thr))


# ---------------- refinement building blocks (depth in mm) ----------------
def f_identity(D, rgb):
    return D


def f_gauss(D, rgb, sigma=1.0):
    if sigma <= 0:
        return D
    return np.exp(cv2.GaussianBlur(np.log(D).astype(np.float32), (0, 0), sigma))


def f_bilateral(D, rgb, sc=0.1):
    return np.exp(cv2.bilateralFilter(np.log(D).astype(np.float32), 7, sc, 3))


def f_geometry(D, rgb, tau=0.04, ang0=10.0, lam=1.0, iters=2):
    """Penalise isolated depth spikes that also break local normal consistency,
    unless strong RGB edges support them (real folds)."""
    L = np.log(D).astype(np.float32)
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0, (0, 0), 1.0)
    mag = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    edge = np.clip(mag / (np.percentile(mag, 95) + 1e-6), 0, 1)        # 1 = strong RGB edge
    protect = lam * edge
    for _ in range(iters):
        med = cv2.medianBlur(L, 5)
        s_depth = np.clip((np.abs(L - med) - tau) / tau, 0, 1)          # spike evidence
        Lx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        Ly = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        n = np.dstack([-K_NORMAL * Lx, -K_NORMAL * Ly, np.ones_like(L)])
        n /= np.linalg.norm(n, axis=2, keepdims=True)
        nbar = cv2.boxFilter(n, -1, (7, 7))
        nbar /= np.linalg.norm(nbar, axis=2, keepdims=True) + 1e-8
        ang = np.degrees(np.arccos(np.clip((n * nbar).sum(2), -1, 1)))
        ang = cv2.dilate(ang, np.ones((3, 3), np.uint8))
        s_norm = np.clip(ang / ang0, 0, 1)                              # normal-inconsistency evidence
        alpha = s_depth * s_norm * (1.0 - protect)
        L = (1 - alpha) * L + alpha * med
    return np.exp(L)


# ---------------- evaluation ----------------
def evaluate(fn, items, depths):
    rows = []
    for it, D in zip(items, depths):
        m = it["mask"]
        if m.sum() < 100:
            continue
        Dr = np.clip(fn(D, it["rgb"]), *DEPTH_CLIP)
        ar, rm, ma, de = metrics(Dr, it["gt"], m)
        e, s = it["em"], it["sm"]
        rows.append(dict(
            frame=it["name"], absrel=ar, rmse=rm, mae=ma, delta1=de,
            absrel_edge=np.mean(np.abs(Dr[e] - it["gt"][e]) / it["gt"][e]) if e.any() else np.nan,
            absrel_smooth=np.mean(np.abs(Dr[s] - it["gt"][s]) / it["gt"][s]) if s.any() else np.nan,
            spike=spike_rate(Dr)))
    return pd.DataFrame(rows)


def tune(method, grid, items, depths, label):
    """Tune on DEV only, by AbsRel."""
    keys, results = list(grid), []
    for combo in tqdm(list(itertools.product(*grid.values())), desc=f"tune {label}"):
        p = dict(zip(keys, combo))
        df = evaluate(lambda D, rgb, p=p: method(D, rgb, **p), items, depths)
        results.append((df.absrel.mean(), df.rmse.mean(), p))
    results.sort(key=lambda t: t[0])
    pd.DataFrame([{**p, "dev_absrel": a, "dev_rmse": r} for a, r, p in results]) \
        .to_csv(BASE / f"geo_tuning_{label}_{TAG}.csv", index=False)
    print(f"  best {label}: {results[0][2]}  dev AbsRel={results[0][0]:.4f}")
    return results[0][2]


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    return np.percentile([rng.choice(x, len(x)).mean() for _ in range(n)], [2.5, 97.5])


def prep(items):
    for it in items:
        it["mask"] = it["gt"] > 0
        it["em"], it["sm"] = region_masks(it["gt"], it["mask"])
    return items


# ======================= RUN =======================
dev_items = prep(build_cache(DEV, DEV_STRIDE, "dev"))
test_items = prep(build_cache(TEST, TEST_STRIDE, "test"))

dev_median = float(np.median(np.concatenate([it["gt"][it["mask"]][::7] for it in dev_items])))

# global scale/shift (disparity space) fitted on DEV only
P = np.concatenate([it["pred"][it["mask"]][::7] for it in dev_items])
G = np.concatenate([1.0 / it["gt"][it["mask"]][::7] for it in dev_items])
a_g, b_g = np.linalg.lstsq(np.stack([P, np.ones_like(P)], 1), G, rcond=None)[0]
print(f"Align mode: {ALIGN} | dev median {dev_median:.2f} mm | global fit: disp = {a_g:.6g}*pred + {b_g:.6g}")


def align(it):
    if ALIGN == "global":
        D = 1.0 / np.maximum(a_g * it["pred"] + b_g, 1e-6)
    else:   # per-image median scaling (uses this frame's GT -> upper bound)
        dep = 1.0 / np.maximum(it["pred"], 1e-3 * it["pred"].max())
        D = dep * np.median(it["gt"][it["mask"]]) / np.median(dep[it["mask"]])
    return np.clip(D, *DEPTH_CLIP).astype(np.float32)


dev_D0 = [align(it) for it in dev_items]          # aligned, unsmoothed
test_D0 = [align(it) for it in test_items]


def ref(D, rgb):                                   # reference pipeline: DA V2 Small + smoothing filter
    return f_gauss(D, rgb, REF_SIGMA)


# ---- tune on DEV only (all start from the smoothed reference) ----
best_g = tune(lambda D, rgb, sigma: f_gauss(ref(D, rgb), rgb, sigma), {"sigma": [0.5, 1.0, 1.5, 2.0]},
              dev_items, dev_D0, "extra_gauss")
best_b = tune(lambda D, rgb, sc: f_bilateral(ref(D, rgb), rgb, sc), {"sc": [0.05, 0.1, 0.2]},
              dev_items, dev_D0, "extra_bilateral")
best_geo = tune(lambda D, rgb, **p: f_geometry(ref(D, rgb), rgb, **p),
                {"tau": [0.02, 0.04, 0.08], "ang0": [5.0, 10.0, 20.0], "lam": [0.0, 1.0], "iters": [1, 2, 3]},
                dev_items, dev_D0, "geometry")

# ---- evaluate ONCE on TEST ----
REF = "DA2-S + smoothing (reference)"
methods = {
    "Constant (dev median)":                lambda D, rgb: np.full_like(D, dev_median),
    "DA2-S aligned, no smoothing":          f_identity,
    REF:                                    ref,
    "Reference + extra Gaussian":           lambda D, rgb: f_gauss(ref(D, rgb), rgb, **best_g),
    "Reference + bilateral":                lambda D, rgb: f_bilateral(ref(D, rgb), rgb, **best_b),
    "Geometry only (no smoothing)":         lambda D, rgb: f_geometry(D, rgb, **best_geo),
    "Reference + GEOMETRY (full)":          lambda D, rgb: f_geometry(ref(D, rgb), rgb, **best_geo),
    "   - without RGB protection":          lambda D, rgb: f_geometry(ref(D, rgb), rgb, **{**best_geo, "lam": 0.0}),
    "   - without normal term":             lambda D, rgb: f_geometry(ref(D, rgb), rgb, **{**best_geo, "ang0": 1e-6}),
}
per_frame = {k: evaluate(fn, test_items, test_D0) for k, fn in methods.items()}

summary = pd.DataFrame([{
    "Method": k, "AbsRel": d.absrel.mean(), "RMSE_mm": d.rmse.mean(), "MAE_mm": d.mae.mean(),
    "delta<1.25": d.delta1.mean(), "AbsRel_edge": d.absrel_edge.mean(),
    "AbsRel_smooth": d.absrel_smooth.mean(), "SpikeRate": d.spike.mean()} for k, d in per_frame.items()])
summary.to_csv(BASE / f"geo_summary_{TAG}_{ALIGN}.csv", index=False)

line = "=" * 128
print(f"\n{line}\n GEOMETRY REFINEMENT ABLATION | {MODEL_NAME} | align={ALIGN} | ref sigma={REF_SIGMA} | "
      f"test: {TEST} ({len(per_frame[REF])} frames)\n{line}")
print(f" {'Method':<34}{'AbsRel v':>9}{'RMSE v':>8}{'MAE v':>8}{'d<1.25 ^':>10}{'Edge v':>9}{'Smooth v':>10}{'Spikes v':>10}")
print("-" * 128)
for _, r in summary.iterrows():
    print(f" {r.Method:<34}{r.AbsRel:>9.4f}{r.RMSE_mm:>8.3f}{r.MAE_mm:>8.3f}{r['delta<1.25']:>10.4f}"
          f"{r.AbsRel_edge:>9.4f}{r.AbsRel_smooth:>10.4f}{r.SpikeRate:>10.4f}")
print(line)
if ALIGN == "median":
    print(" NOTE: per-image median alignment uses test GT -> upper bound, not a fair result.")
print(f" Tuned on {DEV} only: extra_gauss={best_g}, bilateral={best_b}, geometry={best_geo}")

ref_df = per_frame[REF]
for k in ["Reference + extra Gaussian", "Reference + bilateral", "Reference + GEOMETRY (full)"]:
    for col, nm in [("absrel", "AbsRel"), ("rmse", "RMSE"), ("mae", "MAE")]:
        diff = per_frame[k][col].values - ref_df[col].values
        lo, hi = boot_ci(diff)
        print(f" {k:<30} d{nm:<6} = {diff.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  improved: {np.mean(diff < 0):.0%}")

# ---------------- figures ----------------
names = ["DA2-S aligned, no smoothing", REF, "Reference + extra Gaussian", "Reference + bilateral", "Reference + GEOMETRY (full)"]
short = ["No smooth", "Reference", "+Gauss", "+Bilateral", "+Geometry"]
sub = summary.set_index("Method").loc[names]
fig, ax = plt.subplots(1, 5, figsize=(22, 4))
for a_, (t, col) in zip(ax, [("AbsRel (lower)", "AbsRel"), ("RMSE mm (lower)", "RMSE_mm"), ("MAE mm (lower)", "MAE_mm"),
                              ("delta<1.25 (higher)", "delta<1.25"), ("Spike rate (lower)", "SpikeRate")]):
    bars = a_.bar(short, sub[col], color=["#BBBBBB", "#999999", "#DD8452", "#C44E52", "#4C72B0"])
    a_.bar_label(bars, fmt="%.3f"); a_.set_title(t); a_.tick_params(axis="x", rotation=25)
    a_.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(BASE / f"geo_ablation_{TAG}_{ALIGN}.png", dpi=200); plt.close()

idx = np.linspace(0, len(test_items) - 1, 3).astype(int)
fig, ax = plt.subplots(len(idx), 6, figsize=(20, 3.2 * len(idx)))
for r, i in enumerate(idx):
    it, D = test_items[i], test_D0[i]
    Dref = np.clip(ref(D, it["rgb"]), *DEPTH_CLIP)
    Dgeo = np.clip(methods["Reference + GEOMETRY (full)"](D, it["rgb"]), *DEPTH_CLIP)
    vmax = np.percentile(it["gt"][it["mask"]], 99)
    panels = [(it["rgb"], "Color", None), (np.where(it["mask"], it["gt"], np.nan), "GT (mm)", "magma"),
              (D, "DA2-S aligned", "magma"), (Dref, "Reference (smoothed)", "magma"),
              (Dgeo, "Reference + geometry", "magma"), (np.abs(np.log(Dgeo / Dref)), "|log change| by geometry", "viridis")]
    for c, (img, t, cm) in enumerate(panels):
        kw = dict(cmap=cm, vmin=0, vmax=vmax) if cm == "magma" else (dict(cmap=cm, vmin=0, vmax=0.1) if cm else {})
        ax[r, c].imshow(img, **kw); ax[r, c].set_title(t); ax[r, c].axis("off")
plt.tight_layout(); plt.savefig(BASE / f"geo_qualitative_{TAG}_{ALIGN}.png", dpi=150)
print(f"\n Saved: geo_summary_{TAG}_{ALIGN}.csv, geo_tuning_*_{TAG}.csv, geo_ablation_{TAG}_{ALIGN}.png, geo_qualitative_{TAG}_{ALIGN}.png")