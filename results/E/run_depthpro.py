"""Stream E: Depth Pro (Apple, 2024) zero-shot on raw C3VD frames, compared with the constant-median baseline.

Raw input: frames go into Depth Pro exactly as stored (no undistortion, no highlight removal, no cropping);
only Depth Pro's own preprocessing (resize to 1536x1536, normalise) is applied. No training or fine-tuning.

Usage (from the repo root, env `endodepth`):
    python results/E/run_depthpro.py --limit 2        # quick check on 2 dev frames
    python results/E/run_depthpro.py                  # full dev run
    python results/E/run_depthpro.py --split test     # final test run (settings frozen on dev first)

Scaling (same names as the ZoeDepth notebook, plus one deployable protocol):
    none (metric x1000)   Depth Pro output in mm, nothing fitted                     no GT used
    dev_scale             x one scale = median(GT/pred) over dev pixels              dev GT only (in-sample on dev)
    dev_lsq_depth         a*pred + b, least squares over all dev pixels             dev GT only (= 'Linear' in the DA-V2 stream)
    median                x per-frame median(GT)/median(pred)                        uses this frame's GT (oracle)
    lsq_depth             per-frame least-squares a*pred + b in depth                uses this frame's GT (oracle)
Baselines: constant dev median (dev GT only) and constant per-frame median (oracle).

Masks (both reported, the group uses both):
    gt>0       drop only pixels with no GT (raw depth = 0)                 = Depth Anything V2 stream final table
    valid      also drop 100 mm clamped pixels and near-black pixels       = ZoeDepth notebook / env_check.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import CFG, DEVICE, SEED, sequence_dir, stream_results_dir, save_run_info, sync  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tifffile  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

MODEL_ID = "apple/DepthPro-hf"
STREAM = "E"


def load_frame(color_path):
    rgb = cv2.cvtColor(cv2.imread(str(color_path)), cv2.COLOR_BGR2RGB)
    raw = tifffile.imread(color_path.with_name(color_path.name.replace("_color.png", "_depth.tiff")))
    gt_mm = raw.astype(np.float32) / 65535.0 * 100.0
    masks = {"gt>0": raw > 0,
             "valid": (raw > 0) & (raw < 65535) & (rgb.max(axis=2) > 10)}
    return rgb, gt_mm, masks


def compute_metrics(pred, gt):
    p = np.clip(pred, 1e-3, None)
    return {"mae": float(np.mean(np.abs(p - gt))), "absrel": float(np.mean(np.abs(p - gt) / gt)),
            "rmse": float(np.sqrt(np.mean((p - gt) ** 2))),
            "delta1": float(np.mean(np.maximum(p / gt, gt / p) < 1.25))}


class DepthPro:
    def __init__(self):
        from transformers import AutoImageProcessor, DepthProForDepthEstimation
        self.proc = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.model = DepthProForDepthEstimation.from_pretrained(MODEL_ID).to(DEVICE).eval()

    def __call__(self, rgb):
        """Returns (depth in mm at input resolution, predicted horizontal FoV in degrees, seconds)."""
        with torch.no_grad():
            x = self.proc(images=rgb, return_tensors="pt")["pixel_values"].to(DEVICE)
            sync(DEVICE); t0 = time.perf_counter()
            out = self.model(pixel_values=x)
            sync(DEVICE); dt = time.perf_counter() - t0
            post = self.proc.post_process_depth_estimation(out, target_sizes=[rgb.shape[:2]])[0]
        return post["predicted_depth"].float().cpu().numpy() * 1000.0, float(post["field_of_view"]), dt


def predict_all(split, color_paths, out_dir, state):
    """Predictions are cached per frame in results/E/cache/ (ignored by git), so reruns take seconds."""
    cache = out_dir / "cache" / split
    cache.mkdir(parents=True, exist_ok=True)
    preds, fovs = {}, {}
    for cp in tqdm(color_paths, desc=f"Depth Pro {split}"):
        f, ff = cache / f"{cp.stem}.npy", cache / f"{cp.stem}_fov.npy"
        if not f.exists():
            if "model" not in state:
                state["model"] = DepthPro()
            d, fov, dt = state["model"](cv2.cvtColor(cv2.imread(str(cp)), cv2.COLOR_BGR2RGB))
            np.save(f, d.astype(np.float32)); np.save(ff, np.float32(fov)); state.setdefault("times", []).append(dt)
        preds[cp.stem], fovs[cp.stem] = np.load(f), float(np.load(ff))
    return preds, fovs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--limit", type=int, default=0, help="first N frames only (quick check)")
    a = ap.parse_args()

    out_dir = stream_results_dir(CFG, STREAM)
    state = {}
    dev_paths = sorted(sequence_dir(CFG, "dev").glob("*_color.png"))
    paths = sorted(sequence_dir(CFG, a.split).glob("*_color.png"))
    if a.limit:
        dev_paths, paths = dev_paths[:a.limit], paths[:a.limit]
    print(f"Device {DEVICE} | split {a.split} | {len(paths)} frames | seed {SEED}")

    # constants fitted on dev only: baseline median depth and Depth Pro scale correction (per mask)
    dev_preds, _ = predict_all("dev", dev_paths, out_dir, state)
    dev_cal = {}
    for mk in ("gt>0", "valid"):
        G, R, P = [], [], []
        for cp in dev_paths:
            _, gt, masks = load_frame(cp); m = masks[mk]; p = dev_preds[cp.stem][m]
            G.append(gt[m]); R.append(gt[m] / np.maximum(p, 1e-6)); P.append(p)
        G, P = np.concatenate(G), np.concatenate(P)
        (da, db), *_ = np.linalg.lstsq(np.stack([P, np.ones_like(P)], 1), G, rcond=None)
        dev_cal[mk] = {"median_mm": float(np.median(G)), "dev_scale": float(np.median(np.concatenate(R))),
                       "dev_lsq_a": float(da), "dev_lsq_b_mm": float(db)}
        print(f"dev constants [{mk}]: median depth {dev_cal[mk]['median_mm']:.2f} mm, "
              f"Depth Pro scale x{dev_cal[mk]['dev_scale']:.4f}")

    preds, fovs = (dev_preds, None) if a.split == "dev" else predict_all("test", paths, out_dir, state)
    if fovs is None:
        _, fovs = predict_all("dev", paths, out_dir, state)

    rows = []
    for cp in paths:
        _, gt, masks = load_frame(cp)
        for mk, m in masks.items():
            g, p = gt[m], preds[cp.stem][m]
            A = np.stack([p, np.ones_like(p)], 1); (la, lb), *_ = np.linalg.lstsq(A, g, rcond=None)
            cands = {
                ("Constant median", "dev median"): np.full_like(g, dev_cal[mk]["median_mm"]),
                ("Constant median", "frame median (oracle)"): np.full_like(g, np.median(g)),
                ("Depth Pro", "none (metric x1000)"): p,
                ("Depth Pro", "dev_scale"): p * dev_cal[mk]["dev_scale"],
                ("Depth Pro", "dev_lsq_depth"): dev_cal[mk]["dev_lsq_a"] * p + dev_cal[mk]["dev_lsq_b_mm"],
                ("Depth Pro", "median (oracle)"): p * np.median(g) / np.median(p),
                ("Depth Pro", "lsq_depth (oracle)"): la * p + lb,
            }
            for (method, scaling), x in cands.items():
                rows.append({"split": a.split, "mask": mk, "frame": cp.stem, "method": method, "scaling": scaling,
                             "pred_fov_deg": fovs[cp.stem], **compute_metrics(x, g)})

    per_frame = pd.DataFrame(rows)
    summary = (per_frame.groupby(["mask", "method", "scaling"], sort=False)[["mae", "absrel", "rmse", "delta1"]]
               .mean().reset_index())
    tag = f"depthpro_{a.split}" + (f"_limit{a.limit}" if a.limit else "")
    per_frame.to_csv(out_dir / f"{tag}_per_frame.csv", index=False)
    summary.to_csv(out_dir / f"{tag}_summary.csv", index=False)
    times = state.get("times", [])
    save_run_info(out_dir, DEVICE, SEED, extra={
        "model": MODEL_ID, "split": a.split, "frames": len(paths), "flip_test": False, "fine_tuned": False,
        "input": "raw frames, Depth Pro processor only (1536x1536)", "dev_constants": dev_cal,
        "mean_pred_fov_deg": float(np.mean(list(fovs.values()))),
        "sec_per_frame": float(np.mean(times[1:] if len(times) > 1 else times)) if times else "cached",
    })
    info_path = out_dir / f"run_info_{a.split}.json"           # keep dev and test run info side by side
    new = json.load(open(out_dir / "run_info.json"))
    if new["sec_per_frame"] == "cached" and info_path.exists():  # keep the timing from the run that did inference
        new["sec_per_frame"] = json.load(open(info_path)).get("sec_per_frame", "cached")
    json.dump(new, open(info_path, "w"), indent=2); (out_dir / "run_info.json").unlink()

    pd.set_option("display.width", 140)
    for mk in ("gt>0", "valid"):
        print(f"\n[{a.split}] mask = {mk}")
        print(summary[summary["mask"] == mk].drop(columns="mask").round(4).to_string(index=False))
    print(f"\nMean predicted FoV {np.mean(list(fovs.values())):.1f} deg | saved to {out_dir}/{tag}_*")


if __name__ == "__main__":
    main()
