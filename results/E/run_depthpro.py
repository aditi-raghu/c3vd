"""Depth Pro (Apple, 2024) zero-shot on raw C3VD frames, compared with the constant-median baseline.

Raw input: frames go into Depth Pro exactly as stored (no undistortion, no highlight removal, no cropping);
only Depth Pro's own preprocessing (resize to 1536x1536, normalise) is applied. No training or fine-tuning.

Group protocol (common.py and README.md): mask A for the main table, mask B for the second table; metrics per frame,
mean over frames, RMSE = sqrt(mean per-frame MSE); fits on development data only (4-block CV, 2-frame gap);
test uses fits on all 61 dev frames and runs once.

Usage (from the repo root, env `endodepth`):
    python results/E/run_depthpro.py --limit 2        # quick check on 2 dev frames, in-sample, saves no tables
    python results/E/run_depthpro.py --split dev      # 4-block CV on dev
    python results/E/run_depthpro.py --split test     # one test run; refuses to run a second time

Methods (the same calibrations as every other model in the group's comparison):
    Constant median   median depth of the fit frames                         dev GT only
    Raw metric        Depth Pro output in mm, nothing fitted, no clip        no GT used
    Dev scale         x one scale = median(GT/pred) over the fit pixels      dev GT only
    Linear            a*pred + b, least squares over the fit pixels          dev GT only
    Smooth            Linear + correction from the mean error per 20 mm bin  dev GT only
    Oracle median     x per-frame median(GT)/median(pred)                    uses this frame's GT (oracle)
    Oracle LSQ        per-frame least-squares a*pred + b                     uses this frame's GT (oracle)
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import (CFG, DEPTH_MAX_MM, DEVICE, SEED, aggregate, align_lsq, align_median, compute_metrics, dev_folds,  # noqa: E402
                    list_frames, load_frame, save_run_info, save_summary, stream_results_dir, sync)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

MODEL_ID = "apple/DepthPro-hf"
STREAM = "E"
MASKS = ("A", "B")
CLIP = (1e-3, DEPTH_MAX_MM)
BIN_EDGES = np.array([0, 20, 40, 60, 80, 100], dtype=np.float64)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
MIN_BIN_PIXELS = 2000
ALIGNMENT = {"Constant median": ("none", True), "Raw metric": ("none", False), "Dev scale": ("global_scale", True),
             "Linear": ("global_lsq", True), "Smooth": ("global_lsq + smooth", True),
             "Oracle median": ("median_oracle", False), "Oracle LSQ": ("lsq_oracle", False)}


class DepthPro:
    def __init__(self):
        from transformers import AutoImageProcessor, DepthProForDepthEstimation
        self.proc = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.model = DepthProForDepthEstimation.from_pretrained(MODEL_ID).to(DEVICE).eval()

    def __call__(self, rgb):
        """Returns (depth in mm at input resolution, predicted horizontal FoV in degrees, seconds for the whole prediction)."""
        sync(DEVICE); t0 = time.perf_counter()
        with torch.no_grad():
            x = self.proc(images=rgb, return_tensors="pt")["pixel_values"].to(DEVICE)
            out = self.model(pixel_values=x)
            post = self.proc.post_process_depth_estimation(out, target_sizes=[rgb.shape[:2]])[0]
            depth = post["predicted_depth"].float().cpu().numpy() * 1000.0
        sync(DEVICE); dt = time.perf_counter() - t0
        return depth, float(post["field_of_view"]), dt


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
            rgb = cv2.cvtColor(cv2.imread(str(cp)), cv2.COLOR_BGR2RGB)
            d, fov, dt = state["model"](rgb)
            np.save(f, d.astype(np.float32)); np.save(ff, np.float32(fov)); state.setdefault("times", []).append(dt)
        preds[cp.stem], fovs[cp.stem] = np.load(f), float(np.load(ff))
    return preds, fovs


def fit_on(preds, gts, valids):
    """All fitted constants, from the given development frames and mask only."""
    x = np.concatenate([p[v] for p, v in zip(preds, valids)]).astype(np.float64)
    y32 = np.concatenate([g[v] for g, v in zip(gts, valids)])
    y = y32.astype(np.float64)
    a = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
    b = y.mean() - a * x.mean()
    lin = np.clip(a * x + b, *CLIP)
    k = np.clip(np.digitize(lin, BIN_EDGES) - 1, 0, len(BIN_CENTERS) - 1)
    s, n = np.bincount(k, y - lin, len(BIN_CENTERS)), np.bincount(k, minlength=len(BIN_CENTERS))
    ok = n >= MIN_BIN_PIXELS
    scale = np.median(np.concatenate([g[v] / np.maximum(p[v], 1e-6) for p, g, v in zip(preds, gts, valids)]))
    return {"median_mm": float(np.median(y32)), "dev_scale": float(scale), "linear_a": float(a), "linear_b_mm": float(b),
            "bin_centers_mm": BIN_CENTERS[ok].tolist(), "bin_corrections_mm": (s[ok] / n[ok]).tolist()}


def method_preds(p, g, v, c):
    """Depth map of every method for one frame. Fitted and oracle rows are clipped; the raw row shows the raw failure."""
    lin = np.clip(c["linear_a"] * p + c["linear_b_mm"], *CLIP)
    return {"Constant median": np.full_like(g, c["median_mm"]),
            "Raw metric": p,
            "Dev scale": np.clip(p * c["dev_scale"], *CLIP),
            "Linear": lin,
            "Smooth": np.clip(lin + np.interp(lin, c["bin_centers_mm"], c["bin_corrections_mm"]), *CLIP),
            "Oracle median": np.clip(align_median(p, g, v), *CLIP),
            "Oracle LSQ": np.clip(align_lsq(p, g, v), *CLIP)}


def score_frame(split, mask, fold, stem, p, g, v, c):
    return [{"split": split, "mask": mask, "fold": fold, "frame": stem, "method": name, "oracle": name.startswith("Oracle"),
             **compute_metrics(x, g, v)} for name, x in method_preds(p, g, v, c).items()]


def summarise(per_frame):
    rows = [{"split": sp, "mask": mk, "method": me, "oracle": me.startswith("Oracle"), **aggregate(d)}
            for (sp, mk, me), d in per_frame.groupby(["split", "mask", "method"], sort=False)]
    return pd.DataFrame(rows)


def group_table(out_dir):
    """The group's summary format, once both splits exist."""
    tables = [out_dir / f"depthpro_{s}_summary.csv" for s in ("dev", "test")]
    if not all(t.exists() for t in tables):
        return
    rows = []
    for t in tables:
        for r in pd.read_csv(t).itertuples():
            align, fitted = ALIGNMENT[r.method]
            fit = "per_frame_gt" if r.method.startswith("Oracle") else ("dev_blocks_cv" if r.split == "dev" else "dev_all") if fitted else "none"
            name = "Constant median" if r.method == "Constant median" else f"Depth Pro {r.method}"
            rows.append({"model": name, "output_type": "metric", "alignment": align, "fit_on": fit, "split": r.split,
                         "mask": r.mask, "absrel": r.absrel, "rmse": r.rmse, "mae": r.mae, "delta1": r.delta1,
                         "n_frames": r.n_frames})
    save_summary(rows, out_dir, "DepthPro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--limit", type=int, default=0, help="first N dev frames only: in-sample quick check, saves no tables")
    a = ap.parse_args()
    if a.limit and a.split == "test":
        sys.exit("--limit is for dev only: the test set is scored once.")

    out_dir = stream_results_dir(CFG, STREAM)
    done = out_dir / "depthpro_test_done.json"
    if a.split == "test" and done.exists():
        sys.exit(f"{done.name} exists: the test set is scored once. Delete it only if the protocol changes.")
    state = {}
    dev_paths = list_frames(CFG, "dev")[:a.limit or None]
    test_paths = list_frames(CFG, "test") if a.split == "test" else []
    print(f"Device {DEVICE} | split {a.split} | {len(dev_paths)} dev frames | {len(test_paths)} test frames | seed {SEED}")
    dev_preds, fovs = predict_all("dev", dev_paths, out_dir, state)
    if a.split == "test":
        test_preds, fovs = predict_all("test", test_paths, out_dir, state)

    rows, constants = [], {}
    for mk in MASKS:
        dev = [load_frame(cp, mk)[1:] for cp in dev_paths]
        P = [dev_preds[cp.stem] for cp in dev_paths]
        G, V = [d[0] for d in dev], [d[1] for d in dev]
        stems = [cp.stem.split("_")[0] for cp in dev_paths]
        constants[mk] = fit_on(P, G, V)
        if a.limit:
            rows += [r for i in range(len(P)) for r in score_frame("dev", mk, 0, stems[i], P[i], G[i], V[i], constants[mk])]
        elif a.split == "dev":
            for k, (fit, held) in enumerate(dev_folds(len(P))):
                c = fit_on([P[i] for i in fit], [G[i] for i in fit], [V[i] for i in fit])
                rows += [r for i in held for r in score_frame("dev", mk, k + 1, stems[i], P[i], G[i], V[i], c)]
        else:
            for cp in test_paths:
                _, g, v = load_frame(cp, mk)
                rows += score_frame("test", mk, 0, cp.stem.split("_")[0], test_preds[cp.stem], g, v, constants[mk])
        print(f"  mask {mk} done", flush=True)

    per_frame = pd.DataFrame(rows)
    summary = summarise(per_frame)
    pd.set_option("display.width", 140)
    for mk in MASKS:
        print(f"\n[{a.split}] mask {mk}" + (" (quick check, in-sample)" if a.limit else ""))
        print(summary[summary["mask"] == mk].drop(columns=["split", "mask", "oracle"]).round(4).to_string(index=False))
    print(f"\nMean predicted FoV {np.mean(list(fovs.values())):.1f} deg (the model's own per-frame estimate)")
    if a.limit:
        return

    tag = f"depthpro_{a.split}"
    per_frame.to_csv(out_dir / f"{tag}_per_frame.csv", index=False)
    summary.to_csv(out_dir / f"{tag}_summary.csv", index=False)
    group_table(out_dir)
    times = state.get("times", [])
    info_path = out_dir / f"run_info_{a.split}.json"
    sec = float(np.mean(times[1:] if len(times) > 1 else times)) if times else None
    if sec is None and info_path.exists():                       # keep the timing from the run that did inference
        sec = json.load(open(info_path)).get("sec_per_frame")
    save_run_info(out_dir, DEVICE, SEED, extra={
        "model": MODEL_ID, "split": a.split, "dev_frames": len(dev_paths), "test_frames": len(test_paths), "masks": list(MASKS),
        "flip_test": False, "fine_tuned": False, "input": "raw frames, Depth Pro processor only (1536x1536)",
        "constants_from_all_dev_frames": constants, "n_blocks": 4, "gap": 2,
        "mean_pred_fov_deg": float(np.mean(list(fovs.values()))), "sec_per_frame": sec,
        "timing": "whole prediction per frame, first frame not counted"})
    (out_dir / "run_info.json").replace(info_path)
    if a.split == "test":
        json.dump({"date": datetime.now().isoformat(timespec="seconds"), "n_frames": len(test_paths),
                   "absrel_mask_A": {r.method: r.absrel for r in summary[summary["mask"] == "A"].itertuples()}},
                  open(done, "w"), indent=2)
    print(f"Saved to {out_dir}/{tag}_*")


if __name__ == "__main__":
    main()
