"""DPT (zero-shot) on C3VD with the group protocol. One file.

Put DPT.py in the project folder (next to common.py and config.yaml) or in a subfolder of it, then run:
    python DPT.py                      DPT-Large (Intel/dpt-large)
    python DPT.py --limit 8            quick check on 8 dev and 8 test frames
    python DPT.py --model dpt_hybrid   DPT-Hybrid (same weights family as the MiDaS stream)
    python DPT.py --model da2_small    protocol check: must reproduce the group's lumen-aware table

Group functions: taken from common.py when it is the current version (protocol v2: MASK_RULE, aggregate, dev_folds).
If common.py is older or missing, identical built-in copies of the protocol-v2 functions are used, so the numbers
are the same. The script prints which one it used.

Protocol (code review R2, S3, E5): mask A by default (0 < raw < 65535 and RGB max > 10), metrics per frame,
mean over frames with RMSE = sqrt(mean per-frame MSE), all 61 dev frames, 4-block CV with a 2-frame gap on dev,
test run once with parameters fitted on all dev frames. Calibration chain copied from the lumen-aware isotonic
notebook (Final_Lumen_model.ipynb). Rows starting with "Oracle" use test ground truth (upper bounds only).

Output: results/DPT/<model>_mask<A|B>/  (summary CSV, full tables, per-frame CSVs, params, run_info, figures)
"""
import argparse
import json
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ================================================================ find common.py and config.yaml
HERE = Path(__file__).resolve().parent
ROOT = next((d for d in (HERE, *HERE.parents) if (d / "common.py").exists()), None) or \
       next((d for d in (HERE, *HERE.parents) if (d / "config.yaml").exists()), HERE)
sys.path.insert(0, str(ROOT))
try:
    import common
except ModuleNotFoundError as e:
    if e.name != "common":
        raise
    common = None


# ---- built-in copies of common.py protocol v2 (used only when common.py is missing or older) ----
def _load_config():
    import yaml
    path = ROOT / "config.yaml"
    if not path.exists():
        sys.exit(f"config.yaml not found in {ROOT}. Copy config.example.yaml to config.yaml and set data_root.")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["data_root"] = Path(cfg["data_root"]).expanduser()
    cfg["results_root"] = (ROOT / cfg.get("results_root", "results")).resolve()
    return cfg


def _get_device(name="auto"):
    name = os.environ.get("ENDODEPTH_DEVICE", name).lower()
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _sequence_dir(cfg, split):
    path = Path(cfg["data_root"]) / (cfg["dev_sequence"] if split == "dev" else cfg["test_sequence"])
    if not path.is_dir():
        sys.exit(f"Sequence folder not found: {path}. Check data_root in config.yaml.")
    return path


def _stream_results_dir(cfg, stream):
    path = Path(cfg["results_root"]) / stream
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_mask(raw, rgb, rule):
    if rule == "A":
        return (raw > 0) & (raw < 65535) & (rgb.max(axis=2) > 10)
    if rule == "B":
        return raw > 0
    raise ValueError(f"Unknown mask rule {rule!r}. Use 'A' or 'B'.")


def _list_frames(cfg, split):
    return sorted(_sequence_dir(cfg, split).glob("*_color.png"))


def _load_frame(color_path, rule=None):
    import tifffile
    color_path = Path(color_path)
    rgb = cv2.cvtColor(cv2.imread(str(color_path)), cv2.COLOR_BGR2RGB)
    raw = tifffile.imread(color_path.with_name(color_path.name.replace("_color.png", "_depth.tiff")))
    gt_mm = raw.astype(np.float32) / 65535.0 * 100.0
    return rgb, gt_mm, _make_mask(raw, rgb, (rule or MASK).upper())


def _compute_metrics(pred_mm, gt_mm, valid):
    p = np.clip(pred_mm[valid], 1e-3, None).astype(np.float64)
    g = gt_mm[valid].astype(np.float64)
    e = p - g
    mse = float(np.mean(e ** 2))
    return {"absrel": float(np.mean(np.abs(e) / g)), "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(e))),
            "delta1": float(np.mean(np.maximum(p / g, g / p) < 1.25)), "mse": mse}


def _aggregate(pf):
    return {"absrel": float(pf.absrel.mean()), "mae": float(pf.mae.mean()), "rmse": float(np.sqrt(pf.mse.mean())),
            "delta1": float(pf.delta1.mean()), "n_frames": int(len(pf))}


def _to_relative_depth(pred, output_type):
    return 1.0 / np.clip(pred, 1e-6, None) if output_type == "disparity" else pred


def _align_median(rel, gt, valid):
    return rel * (np.median(gt[valid]) / np.median(rel[valid]))


def _align_lsq(rel, gt, valid):
    A = np.stack([rel[valid], np.ones(valid.sum())], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, gt[valid], rcond=None)
    return rel * a + b


def _dev_folds(n, n_blocks=4, gap=2):
    folds = []
    for held in np.array_split(np.arange(n), n_blocks):
        lo, hi = held.min() - gap, held.max() + gap
        folds.append(([i for i in range(n) if i < lo or i > hi], held.tolist()))
    return folds


_RESULT_COLUMNS = ["model", "output_type", "alignment", "fit_on", "split", "mask",
                   "absrel", "rmse", "mae", "delta1", "n_frames"]


def _save_summary(rows, out_dir, name):
    df = pd.DataFrame(rows)[_RESULT_COLUMNS]
    df.to_csv(Path(out_dir) / f"{name}_summary.csv", index=False)
    return df


def _save_run_info(out_dir, device, seed, extra=None):
    import transformers
    info = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
            "transformers": transformers.__version__, "numpy": np.__version__, "device": str(device), "seed": seed,
            "mask_rule": MASK}
    if extra:
        info.update(extra)
    with open(Path(out_dir) / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


# ---- choose: current common.py (protocol v2) if available, otherwise the built-in copies ----
USE_COMMON = common is not None and hasattr(common, "MASK_RULE") and hasattr(common, "aggregate")
if common is not None and hasattr(common, "CFG"):
    CFG, SEED, DEVICE = common.CFG, common.SEED, common.DEVICE
    sync, stream_results_dir = common.sync, common.stream_results_dir
else:
    CFG = _load_config()
    SEED = CFG.get("seed", 42)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    DEVICE = _get_device(CFG.get("device", "auto"))
    sync, stream_results_dir = _sync, _stream_results_dir
if USE_COMMON:
    MASK = str(common.MASK_RULE).upper()
    list_frames, load_frame, compute_metrics, aggregate = (common.list_frames, common.load_frame,
                                                          common.compute_metrics, common.aggregate)
    dev_folds, save_summary, save_run_info = common.dev_folds, common.save_summary, common.save_run_info
    align_median, align_lsq, to_relative_depth = common.align_median, common.align_lsq, common.to_relative_depth
    SOURCE = f"group functions from {ROOT / 'common.py'} (protocol v2)"
else:
    MASK = str(CFG.get("mask_rule", "A")).upper()
    list_frames, load_frame, compute_metrics, aggregate = _list_frames, _load_frame, _compute_metrics, _aggregate
    dev_folds, save_summary, save_run_info = _dev_folds, _save_summary, _save_run_info
    align_median, align_lsq, to_relative_depth = _align_median, _align_lsq, _to_relative_depth
    SOURCE = ("built-in copies of the protocol-v2 group functions ("
              + ("common.py not found" if common is None else "your common.py is an older version") + ")")
RESULTS = Path(CFG["results_root"])

# ---- settings of the lumen-aware isotonic notebook (they define the group numbers) ----
DMAX = 100.0
BIN_EDGES = np.array([0, 20, 40, 60, 80, 100], dtype=np.float32)
CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
MIN_PIXELS = 2000
FIT_STRIDE = 4
ISO_STRIDE = 16
N_BLOCKS, GAP = 4, 2
CUES = ["both", "dark", "far"]
SIGMA, TAU = 20, 0.5
DAV2_INPUT_SIZE = (700, 868)
MEAN_I = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD_I = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CACHE_ROOT = Path(CFG.get("cache_root", RESULTS / "cache")).expanduser()

MODELS = {
    "dpt_large":  {"hf": "Intel/dpt-large",        "kind": "dpt", "short": "DPT", "name": "DPT-Large"},
    "dpt_hybrid": {"hf": "Intel/dpt-hybrid-midas", "kind": "dpt", "short": "DPT", "name": "DPT-Hybrid"},
    "da2_small":  {"hf": "depth-anything/Depth-Anything-V2-Small-hf", "kind": "dav2", "short": "DAv2", "name": "DAv2 Small"},
    "da2_base":   {"hf": "depth-anything/Depth-Anything-V2-Base-hf",  "kind": "dav2", "short": "DAv2", "name": "DAv2 Base"},
    "da2_large":  {"hf": "depth-anything/Depth-Anything-V2-Large-hf", "kind": "dav2", "short": "DAv2", "name": "DAv2 Large"},
}
OUTPUT_TYPE = "disparity"
ORACLE = "Oracle: linear fitted on test (not fair)"
MED = "Oracle: per-frame median (not fair)"
LSQ = "Oracle: per-frame LSQ (not fair)"
INK, MUTED, FAIR, BASE, UNFAIR = "#0b0b0b", "#52514e", "#2a78d6", "#8c8b86", "#d3d2cc"


# ================================================================ disk space
MODEL_GB = {"dpt_large": 1.4, "dpt_hybrid": 0.5, "da2_small": 0.1, "da2_base": 0.4, "da2_large": 1.4}
DISK_CACHE = True          # False = keep predictions in memory (set in main when the disk is nearly full)


def free_gb(path):
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free / 1e9


def check_space_for_model(key):
    """Stop with a clear message, before the download starts, if the model weights will not fit on the disk."""
    try:
        from huggingface_hub import try_to_load_from_cache
        for name in ("model.safetensors", "pytorch_model.bin"):
            if isinstance(try_to_load_from_cache(MODELS[key]["hf"], name), str):
                return                                                    # already downloaded
    except Exception:
        return
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    free, need = free_gb(hf_home), MODEL_GB[key] + 0.2
    if free < need:
        sys.exit(f"\nNot enough disk space to download {MODELS[key]['hf']} (about {MODEL_GB[key]} GB): "
                 f"{free:.1f} GB free on the disk of {hf_home}.\n"
                 f"Free some space, use a smaller model (--model dpt_hybrid needs {MODEL_GB['dpt_hybrid']} GB), or put "
                 f"the download on another drive: $env:HF_HOME = \"D:\\hf_cache\"")


# ================================================================ model + cached predictions
class Predictor:
    def __init__(self, key):
        spec = MODELS[key]
        self.kind, self.calls, self.input_hw = spec["kind"], 0, None
        check_space_for_model(key)
        print(f"Loading {spec['hf']} on {DEVICE}")
        if self.kind == "dav2":
            from transformers import AutoModelForDepthEstimation
            self.model = AutoModelForDepthEstimation.from_pretrained(spec["hf"]).to(DEVICE).eval()
            self.proc = None
        else:
            from transformers import DPTForDepthEstimation, DPTImageProcessor
            self.model = DPTForDepthEstimation.from_pretrained(spec["hf"]).to(DEVICE).eval()
            self.proc = DPTImageProcessor.from_pretrained(spec["hf"])

    def __call__(self, rgb):
        with torch.no_grad():
            if self.kind == "dav2":
                H, W = DAV2_INPUT_SIZE
                img = np.asarray(Image.fromarray(rgb).resize((W, H), Image.BICUBIC), dtype=np.float32) / 255.0
                x = torch.from_numpy(((img - MEAN_I) / STD_I).transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
            else:
                x = self.proc(images=rgb, return_tensors="pt")["pixel_values"].to(DEVICE)
            self.input_hw = tuple(int(s) for s in x.shape[-2:])
            d = self.model(pixel_values=x).predicted_depth
            d = torch.nn.functional.interpolate(d.unsqueeze(1), size=rgb.shape[:2], mode="bilinear",
                                                align_corners=False)
            return d[0, 0].float().cpu().numpy()


_PREDICTORS = {}
_MEM, _MEM_TIMES = {}, {}      # predictions and timings kept in memory when DISK_CACHE is False


def _cache_file(key, split, cp, suffix=""):
    return CACHE_ROOT / key / split / f"{cp.stem}{suffix}.npy"


def get_pred(key, split, cp, rgb):
    mk = (key, split, cp.stem)
    if mk in _MEM:
        return _MEM[mk]
    f = _cache_file(key, split, cp)
    if f.exists():
        return np.load(f)
    if key not in _PREDICTORS:
        _PREDICTORS[key] = Predictor(key)
    model = _PREDICTORS[key]
    sync(DEVICE)
    t0 = time.perf_counter()
    pred = model(rgb).astype(np.float32)
    sync(DEVICE)
    dt = time.perf_counter() - t0
    timing = np.array([dt, model.calls == 0, *model.input_hw], float)
    model.calls += 1
    if DISK_CACHE:
        f.parent.mkdir(parents=True, exist_ok=True)
        np.save(f, pred)
        np.save(_cache_file(key, split, cp, ".time"), timing)
    else:
        _MEM[mk], _MEM_TIMES[mk] = pred, timing
    return pred


def timing_info(key, split, paths):
    rows = []
    for cp in paths:
        f = _cache_file(key, split, cp, ".time")
        if (key, split, cp.stem) in _MEM_TIMES:
            rows.append(_MEM_TIMES[(key, split, cp.stem)])
        elif f.exists():
            rows.append(np.load(f))
    secs = [r[0] for r in rows if not r[1]]
    if not secs:
        return {"sec_per_frame": None, "note": "no timings in the cache (delete the cache folder to time)"}
    return {"sec_per_frame": float(np.mean(secs)), "n_timed": len(secs),
            "model_input_hw": [int(rows[0][2]), int(rows[0][3])], "note": "warm-up frame not counted"}


# ================================================================ frames
def frame_paths(split, limit=0):
    paths = list_frames(CFG, split)
    return paths[:limit] if limit else paths


def frame_record(key, split, cp):
    rgb, gt, valid = load_frame(cp)
    F = {"name": cp.stem.replace("_color", ""), "rgb": rgb, "gt": gt, "valid": valid,
         "fov": rgb.max(axis=2) > 10, "pred": get_pred(key, split, cp, rgb)}
    make_lumen_maps(F)
    return F


# ================================================================ lumen maps (lumen-aware notebook, cell 3)
def blur(x, sigma=SIGMA):
    return cv2.GaussianBlur(x.astype(np.float32), (0, 0), sigma)


def lumen_cues(rgb, depth_like, fov, sigma=SIGMA):
    f = fov.astype(np.float32)
    v = rgb.astype(np.float32).max(axis=2) / 255.0
    v_s = blur(v * f, sigma) / (blur(f, sigma) + 1e-6)
    dark = 1.0 - np.clip(v_s / (np.percentile(v_s[fov], 99) + 1e-6), 0, 1)
    far = np.zeros_like(v_s)
    ranks = depth_like[fov].argsort().argsort()
    far[fov] = ranks / max(len(ranks) - 1, 1)
    far = blur(far * f, sigma) / (blur(f, sigma) + 1e-6)
    dark[~fov], far[~fov] = 0, 0
    return dark, far


def combine(dark, far, cue):
    return {"dark": dark, "far": far}.get(cue, np.sqrt(dark * far))


def lumen_from_score(score, fov, sigma=SIGMA, tau=TAU):
    mask = (score > tau) & fov
    lab, n = ndimage.label(mask)
    if n == 0:
        return np.zeros(score.shape, np.float16), np.zeros(score.shape, bool)
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    core = lab == (np.argmax(sizes) + 1)
    w = blur(core * score, sigma)
    w = w / (w.max() + 1e-6)
    w[~fov] = 0
    return w.astype(np.float16), core


def make_lumen_maps(F):
    dark, far = lumen_cues(F["rgb"], -F["pred"], F["fov"])
    F["w"] = {}
    for cue in CUES:
        F["w"][cue], core = lumen_from_score(combine(dark, far, cue), F["fov"])
        if cue == "both":
            F["core"] = core


# ================================================================ calibration (lumen-aware notebook, cell 5)
def linear_depth(F, P):
    return np.clip(P["a"] * -F["pred"] + P["b"], 1e-3, DMAX)


def fit_params(frames):
    xs, ys, ws = [], [], {c: [] for c in CUES}
    for F in frames:
        v = F["valid"]
        xs.append(-F["pred"][v][::FIT_STRIDE])
        ys.append(F["gt"][v][::FIT_STRIDE])
        for c in CUES:
            ws[c].append(F["w"][c][v][::FIT_STRIDE])
    x = np.concatenate(xs).astype(np.float64)
    y = np.concatenate(ys).astype(np.float64)
    xm, ym = x.mean(), y.mean()
    a = ((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum()
    b = ym - a * xm
    lin = np.clip(a * x + b, 1e-3, DMAX)
    err = y - lin
    k = np.clip(np.digitize(lin, BIN_EDGES) - 1, 0, len(CENTERS) - 1)

    def binned(weights):
        s = np.bincount(k, weights=err * weights, minlength=len(CENTERS))
        n = np.bincount(k, weights=weights, minlength=len(CENTERS))
        return s, n

    s, n = binned(np.ones_like(err))
    ok = n >= MIN_PIXELS
    curves = {"all": (CENTERS[ok], s[ok] / n[ok])}
    fallback = np.interp(CENTERS, *curves["all"])
    for c in CUES:
        w = np.concatenate(ws[c]).astype(np.float64)
        for part, wt in [("wall", 1.0 - w), ("lumen", w)]:
            s, n = binned(wt)
            curves[f"{part}_{c}"] = (CENTERS, np.where(n >= MIN_PIXELS, s / np.maximum(n, 1e-9), fallback))
    return {"a": float(a), "b": float(b), "median": float(np.median(y)), "curves": curves}


def fit_iso(frames, P, lumen=False):
    xs, ys, ws = [], [], []
    for F in frames:
        v = F["valid"]
        xs.append(linear_depth(F, P)[v][::ISO_STRIDE])
        ys.append(F["gt"][v][::ISO_STRIDE])
        ws.append(F["w"]["both"][v][::ISO_STRIDE].astype(np.float32))
    x, y, w = map(np.concatenate, (xs, ys, ws))
    t = np.log(y)
    iso = lambda wt: IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, t, sample_weight=wt)  # noqa: E731
    if not lumen:
        return {"all": iso(np.ones_like(x))}
    return {"wall": iso(1.0 - w + 1e-6), "lumen": iso(w + 1e-6)}


def fit_inverse(frames):
    """First version's disparity-space fit: 1/GT = a * disparity + b over all dev pixels."""
    x = np.concatenate([F["pred"][F["valid"]][::FIT_STRIDE] for F in frames]).astype(np.float64)
    y = np.concatenate([1.0 / F["gt"][F["valid"]][::FIT_STRIDE] for F in frames]).astype(np.float64)
    (a, b), *_ = np.linalg.lstsq(np.stack([x, np.ones_like(x)], 1), y, rcond=None)
    return {"inv_a": float(a), "inv_b": float(b)}


def fit_all(frames):
    P = fit_params(frames)
    P["iso"] = fit_iso(frames, P)
    P["iso_lumen"] = fit_iso(frames, P, lumen=True)
    P.update(fit_inverse(frames))
    return P


def predict(kind, F, P, cue="both"):
    if kind == "constant":
        return np.full(F["gt"].shape, P["median"], np.float32)
    lin = linear_depth(F, P)
    if kind == "linear":
        return lin
    if kind == "smooth":
        return np.clip(lin + np.interp(lin, *P["curves"]["all"]), 1e-3, DMAX)
    w = F["w"][cue].astype(np.float32)
    if kind == "lumen":
        corr = (1 - w) * np.interp(lin, *P["curves"][f"wall_{cue}"]) + w * np.interp(lin, *P["curves"][f"lumen_{cue}"])
        return np.clip(lin + corr, 1e-3, DMAX)
    flat, M = lin.ravel(), P[kind]
    if kind == "iso":
        t = M["all"].predict(flat)
    else:
        wf = w.ravel()
        t = (1 - wf) * M["wall"].predict(flat) + wf * M["lumen"].predict(flat)
    return np.clip(np.exp(t).reshape(lin.shape), 1e-3, DMAX)


def inverse_depth(F, P):
    return np.clip(1.0 / np.maximum(P["inv_a"] * F["pred"] + P["inv_b"], 1e-6), 1e-3, DMAX)


def oracle_median(F, P):
    return np.clip(align_median(to_relative_depth(F["pred"], OUTPUT_TYPE), F["gt"], F["valid"]), 1e-3, DMAX)


def oracle_lsq(F, P):
    return np.clip(align_lsq(to_relative_depth(F["pred"], OUTPUT_TYPE), F["gt"], F["valid"]), 1e-3, DMAX)


def predictors(short, P_or=None):
    group = [("Constant median", "constant", None), (f"{short} + global linear", "linear", None),
             ("+ smooth correction", "smooth", None), ("+ lumen-aware (darkness only)", "lumen", "dark"),
             ("+ lumen-aware (depth only)", "lumen", "far"), ("+ lumen-aware (both cues)", "lumen", "both"),
             ("Isotonic (log)", "iso", None), ("Lumen-aware isotonic (log)", "iso_lumen", "both")]
    m = {label: (lambda F, P, kind=kind, cue=cue: predict(kind, F, P, cue or "both")) for label, kind, cue in group}
    m[f"{short} + global inverse-depth fit"] = inverse_depth
    m[MED], m[LSQ] = oracle_median, oracle_lsq
    if P_or is not None:
        m[ORACLE] = lambda F, P: predict("linear", F, P_or)
    return m


# ================================================================ metrics
def frame_metrics(pred, F, region=None):
    v = F["valid"] if region is None else (F["valid"] & region)
    if v.sum() < 100:
        return None
    m = compute_metrics(pred.astype(np.float64), F["gt"], v)       # common.compute_metrics
    m.setdefault("mse", m["rmse"] ** 2)
    return m


def frame_rows(F, preds, P, **info):
    lumen_region = F["w"]["both"].astype(np.float32) > 0.5
    rows = []
    for label, fn in preds.items():
        pred = fn(F, P)
        m = frame_metrics(pred, F)
        if m is None:
            continue
        ml, mw = frame_metrics(pred, F, lumen_region), frame_metrics(pred, F, ~lumen_region)
        p = pred[F["valid"]]
        rows.append({"frame": F["name"], **info, "method": label, "mask": MASK, **m,
                     "absrel_wall": mw["absrel"] if mw else np.nan, "absrel_lumen": ml["absrel"] if ml else np.nan,
                     "clip_low": float(np.mean(p <= 1.0001e-3)), "clip_high": float(np.mean(p >= DMAX))})
    return rows


def summarise(pf):
    extra = ["absrel_wall", "absrel_lumen", "clip_low", "clip_high"]
    return pd.DataFrame([{"method": m, "mask": MASK, **aggregate(d), **d[extra].mean().to_dict()}
                         for m, d in pf.groupby("method", sort=False)])


# ================================================================ protocol check
def check_against_group(summary, key, limit):
    print("\nProtocol check against the group's lumen-aware isotonic table")
    if limit:
        print("  skipped (--limit uses a subset of frames)")
        return None
    g, path = None, None
    for f in sorted(RESULTS.glob("*/*_summary.csv"), key=lambda p: p.stat().st_mtime, reverse=True):
        if "lumen" in f.parent.name.lower():
            t = pd.read_csv(f)
            if "model" in t and t.model.str.contains("DAv2 Small Lumen-aware isotonic", regex=False).any():
                g, path = t, f
                break
    if g is None:
        print("  skipped (group table not found in results/)")
        return None
    if (str(g["mask"].iloc[0]) if "mask" in g else "A") != MASK:
        print(f"  skipped: the group table uses another mask than this run (mask {MASK})")
        return None
    for split in ("dev", "test"):
        if summary[summary.split == split].n_frames.max() != g[g.split == split].n_frames.max():
            print(f"  skipped: different number of {split} frames than the group table")
            return None
    g["method"] = [m[len("DAv2 Small "):] for m in g.model]
    name = MODELS[key]["name"]
    ours = summary.assign(method=[m[len(name) + 1:] for m in summary.model])
    shared = ["Constant median"] if key != "da2_small" else list(g.method.unique())
    out = []
    for split in ("dev", "test"):
        for m in shared:
            a, b = ours[(ours.split == split) & (ours.method == m)], g[(g.split == split) & (g.method == m)]
            if len(a) and len(b):
                d = max(abs(float(a[c].iloc[0] - b[c].iloc[0])) for c in ("absrel", "rmse", "mae", "delta1"))
                ok = d <= (1e-6 if m == "Constant median" else 5e-4)
                out.append({"split": split, "method": m, "max_abs_diff": d, "status": "OK" if ok else "DIFFERENT"})
    out = pd.DataFrame(out)
    print(f"  compared with {path}")
    print(out.to_string(index=False))
    print("  -> same protocol as the group." if (out.status == "OK").all() else "  -> CHECK: differs from the group table.")
    return out


# ================================================================ figures
def fig_metrics(tab, title, path):
    t = tab.iloc[::-1]
    colors = [BASE if m == "Constant median" else UNFAIR if m.startswith("Oracle") else FAIR for m in t.method]
    fig, axes = plt.subplots(1, 4, figsize=(20, 0.42 * len(t) + 1.6), sharey=True, constrained_layout=True)
    for ax, (col, lbl) in zip(axes, [("absrel", "AbsRel (lower is better)"), ("rmse", "RMSE, mm (lower is better)"),
                                     ("mae", "MAE, mm (lower is better)"), ("delta1", "δ<1.25 (higher is better)")]):
        bars = ax.barh(t.method, t[col], color=colors, height=0.6)
        ax.bar_label(bars, fmt="%.3f" if col in ("absrel", "delta1") else "%.2f", padding=3, color=MUTED, fontsize=8)
        ax.set_title(lbl, color=INK, fontsize=10, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.margins(x=0.18)
    fig.suptitle(title + "  (grey = oracle rows, use test ground truth)", color=INK, fontsize=12, x=0.01, ha="left")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_examples(key, P, preds, best, test_pf, name, path):
    order = test_pf[test_pf.method == best].set_index("frame").absrel.sort_values()
    pick = [order.index[0], order.index[len(order) // 2], order.index[-1]]
    lin = f"{MODELS[key]['short']} + global linear"
    by_name = {cp.stem.replace("_color", ""): cp for cp in frame_paths("test")}
    fig, axes = plt.subplots(3, 5, figsize=(21, 10), constrained_layout=True)
    for r, nm in enumerate(pick):
        F = frame_record(key, "test", by_name[nm])
        nan = lambda a: np.where(F["valid"], a, np.nan)  # noqa: E731
        d_lin, d_best = preds[lin](F, P), preds[best](F, P)
        ax = axes[r]
        ax[0].imshow(F["rgb"]); ax[0].set_title(f"Test frame {nm}")
        im = ax[1].imshow(nan(F["gt"]), cmap="plasma", vmin=0, vmax=DMAX); ax[1].set_title(f"Ground truth (mask {MASK})")
        ax[2].imshow(nan(d_lin), cmap="plasma", vmin=0, vmax=DMAX)
        ax[2].set_title(f"{lin}\nAbsRel {frame_metrics(d_lin, F)['absrel']:.3f}")
        ax[3].imshow(nan(d_best), cmap="plasma", vmin=0, vmax=DMAX)
        ax[3].set_title(f"{best}\nAbsRel {frame_metrics(d_best, F)['absrel']:.3f}")
        ie = ax[4].imshow(nan(np.abs(d_best - F["gt"])), cmap="inferno", vmin=0, vmax=30); ax[4].set_title("|Error| (mm)")
        for a_ in ax:
            a_.axis("off")
    fig.colorbar(im, ax=axes[:, 1:4], fraction=0.015, label="Depth (mm)")
    fig.colorbar(ie, ax=axes[:, 4], fraction=0.04, label="|Error| (mm)")
    fig.suptitle(f"{name}: best, median and worst test frames ({best}, mask {MASK})", fontsize=12, x=0.01, ha="left")
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ================================================================ run
def to_jsonable(x):
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def fmt_table(tab):
    cols = ["method", "absrel", "rmse", "mae", "delta1", "absrel_wall", "absrel_lumen", "clip_low", "clip_high"]
    f4 = "{:.4f}".format
    return tab[cols].to_string(index=False, formatters={
        "absrel": f4, "rmse": "{:.3f}".format, "mae": "{:.3f}".format, "delta1": f4, "absrel_wall": f4,
        "absrel_lumen": f4, "clip_low": "{:.2%}".format, "clip_high": "{:.2%}".format})


def summary_rows(tab, split, name, short):
    meta = {f"{short} + global inverse-depth fit": ("global_inv_lsq", "dev_blocks_cv", "dev_all"),
            MED: ("median_oracle", "per_frame_gt", "per_frame_gt"),
            LSQ: ("lsq_oracle", "per_frame_gt", "per_frame_gt"),
            ORACLE: ("test_lsq_oracle", "test", "test"),
            "Constant median": ("none", "dev_blocks_cv", "dev_all")}
    rows = []
    for r in tab.itertuples():
        align, fit_dev, fit_test = meta.get(r.method, ("global_lsq", "dev_blocks_cv", "dev_all"))
        rows.append({"model": f"{name} {r.method}", "output_type": OUTPUT_TYPE, "alignment": align,
                     "fit_on": fit_dev if split == "dev" else fit_test, "split": split, "mask": MASK,
                     "absrel": r.absrel, "rmse": r.rmse, "mae": r.mae, "delta1": r.delta1, "n_frames": r.n_frames})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dpt_large", choices=list(MODELS))
    ap.add_argument("--limit", type=int, default=0, help="first N dev and N test frames only (quick check)")
    ap.add_argument("--no-disk-cache", action="store_true",
                    help="keep predictions in memory instead of saving them (automatic when < 3 GB free)")
    a = ap.parse_args()
    global DISK_CACHE
    free = free_gb(CACHE_ROOT)
    if a.no_disk_cache or free < 3.0:
        DISK_CACHE = False
        print(f"Predictions are kept in memory, not saved to disk ({free:.1f} GB free). Results are the same.")

    key, spec = a.model, MODELS[a.model]
    short, name = spec["short"], spec["name"]
    run = f"DPT_{key}_mask{MASK}"
    out_dir = stream_results_dir(CFG, "DPT") / (f"{key}_mask{MASK}" + (f"_limit{a.limit}" if a.limit else ""))
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"{spec['hf']} | device {DEVICE} | seed {SEED} | mask {MASK} | output {out_dir}")
    print(f"Using {SOURCE}")

    # ---- dev: 4-fold block CV, held-out frames only ----
    FR = [frame_record(key, "dev", cp) for cp in tqdm(frame_paths("dev", a.limit), desc="dev frames")]
    preds_dev = predictors(short)
    rows = []
    for k, (fit_ids, held) in enumerate(tqdm(dev_folds(len(FR), N_BLOCKS, GAP), desc="dev block CV")):
        P = fit_all([FR[i] for i in fit_ids])
        for i in held:
            rows += frame_rows(FR[i], preds_dev, P, split="dev", idx=i, fold=k + 1)
    dev_pf = pd.DataFrame(rows)
    dev_tab = summarise(dev_pf)

    # ---- test: fitted on all dev frames, applied once ----
    P = fit_all(FR)
    n_dev = len(FR)
    del FR
    xs, ys = [], []
    for cp in tqdm(frame_paths("test", a.limit), desc="test oracle fit"):
        rgb, gt, valid = load_frame(cp)
        xs.append(-get_pred(key, "test", cp, rgb)[valid][::FIT_STRIDE])
        ys.append(gt[valid][::FIT_STRIDE])
    a_or, b_or = np.polyfit(np.concatenate(xs).astype(np.float64), np.concatenate(ys).astype(np.float64), 1)
    P_or = {"a": float(a_or), "b": float(b_or)}
    preds_test = predictors(short, P_or)
    rows = []
    for cp in tqdm(frame_paths("test", a.limit), desc="test frames"):
        rows += frame_rows(frame_record(key, "test", cp), preds_test, P, split="test")
    test_pf = pd.DataFrame(rows)
    test_tab = summarise(test_pf)
    n_test = test_pf.frame.nunique()

    # ---- save ----
    dev_pf.to_csv(out_dir / "per_frame_heldout.csv", index=False)
    test_pf.to_csv(out_dir / "per_frame_test.csv", index=False)
    dev_tab.to_csv(out_dir / "ablation_heldout.csv", index=False)
    test_tab.to_csv(out_dir / "results_test.csv", index=False)
    summary = save_summary(summary_rows(dev_tab, "dev", name, short) + summary_rows(test_tab, "test", name, short),
                           out_dir, run)
    if "mask" not in summary:
        summary = summary.assign(mask=MASK)
    iso = lambda m: {"x": m.X_thresholds_, "log_depth": m.y_thresholds_}  # noqa: E731
    with open(out_dir / "params_dev.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable({"model": spec["hf"], "mask": MASK, "fitted_on": f"all {n_dev} dev frames",
                               "a": P["a"], "b": P["b"], "median": P["median"], "inv_a": P["inv_a"],
                               "inv_b": P["inv_b"], "test_oracle": P_or,
                               "curves": {k: list(v) for k, v in P["curves"].items()},
                               "isotonic": {"all": iso(P["iso"]["all"]), "wall": iso(P["iso_lumen"]["wall"]),
                                            "lumen": iso(P["iso_lumen"]["lumen"])}}), f, indent=2)
    timing = {"dev": timing_info(key, "dev", frame_paths("dev", a.limit)),
              "test": timing_info(key, "test", frame_paths("test", a.limit))}
    save_run_info(out_dir, DEVICE, SEED, extra={
        "model": spec["hf"], "mask": MASK, "n_blocks": N_BLOCKS, "gap": GAP, "limit": a.limit,
        "dev_frames": n_dev, "test_frames": int(n_test), "timing": timing,
        "metrics": "common.compute_metrics per frame; mean over frames; RMSE = sqrt(mean per-frame MSE)"})

    # ---- print ----
    line = "=" * 132
    print(f"\n{line}\n {name} | dev {CFG['dev_sequence']} held-out ({n_dev} frames, {N_BLOCKS}-fold block CV)"
          f" | mask {MASK}\n{line}\n{fmt_table(dev_tab)}")
    print(f"\n{line}\n {name} | test {CFG['test_sequence']} ({n_test} frames), fitted on all dev frames | mask {MASK}"
          f"\n{line}\n{fmt_table(test_tab)}\n{line}")
    print(" Rows starting with 'Oracle' use test ground truth: upper bounds only.")
    print(" RMSE = sqrt(mean of per-frame MSE); other metrics = mean over frames.")
    lin = test_tab.set_index("method").loc[f"{short} + global linear"]
    print(f" Global linear on test: {lin.clip_low:.2%} of mask pixels get depth <= 0 (clipped), "
          f"{lin.clip_high:.2%} reach 100 mm.")
    t = timing["test"]
    if t.get("sec_per_frame"):
        print(f" Time: {t['sec_per_frame']:.3f} s per frame, model input {t['model_input_hw']}.")

    check = check_against_group(summary, key, a.limit)
    if check is not None:
        check.assign(mask=MASK).to_csv(out_dir / "protocol_check.csv", index=False)

    # ---- figures ----
    fair = test_tab[~test_tab.method.str.startswith("Oracle") & (test_tab.method != "Constant median")]
    best = fair.sort_values("absrel").method.iloc[0]
    fig_metrics(dev_tab, f"{name}: dev held-out ({n_dev} frames), mask {MASK}", fig_dir / "fig1_dev_metrics.png")
    fig_metrics(test_tab, f"{name}: test {CFG['test_sequence']} ({n_test} frames), mask {MASK}",
                fig_dir / "fig2_test_metrics.png")
    fig_examples(key, P, preds_test, best, test_pf, name, fig_dir / "fig3_test_examples.png")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()