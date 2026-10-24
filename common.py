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
