"""Shared setup for all streams. Do not change this part without group agreement."""
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent


def load_config(path=None):
    path = Path(path) if path else REPO_ROOT / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Copy config.example.yaml to config.yaml "
            "and set data_root."
        )
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["data_root"] = Path(cfg["data_root"]).expanduser()
    cfg["results_root"] = (REPO_ROOT / cfg.get("results_root", "results")).resolve()
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(name="auto"):
    name = os.environ.get("ENDODEPTH_DEVICE", name).lower()
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    """Wait for the GPU to finish, so that time measurements are correct."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def sequence_dir(cfg, split):
    """split = 'dev' or 'test'."""
    name = cfg["dev_sequence"] if split == "dev" else cfg["test_sequence"]
    path = cfg["data_root"] / name
    if not path.is_dir():
        raise FileNotFoundError(f"Sequence folder not found: {path}")
    return path


def stream_results_dir(cfg, stream):
    """stream = 'A', 'B', 'C', 'D' or 'E'."""
    path = cfg["results_root"] / stream
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_versions(device, seed):
    import transformers
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "seed": seed,
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def save_run_info(out_dir, device, seed, extra=None):
    info = log_versions(device, seed)
    if extra:
        info.update(extra)
    with open(Path(out_dir) / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


CFG = load_config()
SEED = CFG.get("seed", 42)
set_seed(SEED)
DEVICE = get_device(CFG.get("device", "auto"))


# ---------- Shared data, metrics and alignment (group protocol v1) ----------
import cv2
import tifffile
import pandas as pd


def list_frames(cfg, split):
    return sorted(sequence_dir(cfg, split).glob("*_color.png"))


def load_frame(color_path):
    rgb = cv2.cvtColor(cv2.imread(str(color_path)), cv2.COLOR_BGR2RGB)
    depth_path = color_path.with_name(color_path.name.replace("_color.png", "_depth.tiff"))
    raw = tifffile.imread(depth_path)
    gt_mm = raw.astype(np.float32) / 65535.0 * 100.0
    valid = (raw > 0) & (raw < 65535) & (rgb.max(axis=2) > 10)
    return rgb, gt_mm, valid


def compute_metrics(pred_mm, gt_mm, valid):
    p = np.clip(pred_mm[valid], 1e-3, None)
    g = gt_mm[valid]
    return {
        "absrel": float(np.mean(np.abs(p - g) / g)),
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "mae": float(np.mean(np.abs(p - g))),
        "delta1": float(np.mean(np.maximum(p / g, g / p) < 1.25)),
    }


def to_relative_depth(pred, output_type):
    """output_type = 'depth' (ZoeDepth, metric DAv2) or 'disparity' (DAv2 relative, MiDaS, DPT)."""
    if output_type == "disparity":
        return 1.0 / np.clip(pred, 1e-6, None)
    return pred


def align_median(rel, gt, valid):
    """Per-image median scaling. Uses the GT of each frame (standard protocol)."""
    return rel * (np.median(gt[valid]) / np.median(rel[valid]))


def align_lsq(rel, gt, valid):
    """Per-image scale and shift. Uses the GT of each frame."""
    A = np.stack([rel[valid], np.ones(valid.sum())], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, gt[valid], rcond=None)
    return rel * a + b


def fit_global_lsq(rels, gts, valids):
    """One scale and shift for many frames (fit on dev, apply without GT)."""
    P = np.concatenate([r[v] for r, v in zip(rels, valids)])
    G = np.concatenate([g[v] for g, v in zip(gts, valids)])
    A = np.stack([P, np.ones_like(P)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, G, rcond=None)
    return float(a), float(b)


def dev_blocks(n_frames, n_blocks=4):
    """Blocks of neighbouring frames for block cross-validation."""
    return np.array_split(np.arange(n_frames), n_blocks)


RESULT_COLUMNS = ["model", "output_type", "alignment", "fit_on", "split",
                  "absrel", "rmse", "mae", "delta1", "n_frames"]


def save_summary(rows, out_dir, name):
    df = pd.DataFrame(rows)[RESULT_COLUMNS]
    df.to_csv(Path(out_dir) / f"{name}_summary.csv", index=False)
    return df
