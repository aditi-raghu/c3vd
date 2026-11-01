"""DPT on C3VD, evaluated the same way as the other models (Depth Anything V2, ZoeDepth, MiDaS, Depth Pro).

Needs common.py and config.yaml in the same folder (the group's shared loader, mask and metrics).
Run:  python DPT.py

1. Dev (trans_t1_a): run DPT and fit one scale + shift that turns its output into mm.
2. Test (trans_t2_b): run DPT once with that scale + shift.
3. Table: AbsRel, RMSE (mm), delta < 1.25 per frame, then the mean over frames
   (RMSE = sqrt(mean of per-frame MSE), the group rule). Mask A from common.py.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")          # plain model download (avoids a stall on Windows)
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402
from common import CFG, DEVICE, compute_metrics, list_frames, load_frame  # noqa: E402

MODEL_ID = "Intel/dpt-large"     # or "Intel/dpt-hybrid-midas" (smaller, but the same weights family as MiDaS)
LIMIT = 0                        # e.g. 8 = quick test on the first 8 dev and 8 test frames; 0 = all frames
OUT_DIR = Path(CFG["results_root"]) / "DPT"


def load_model():
    from transformers import DPTForDepthEstimation, DPTImageProcessor
    processor = DPTImageProcessor.from_pretrained(MODEL_ID)
    model = DPTForDepthEstimation.from_pretrained(MODEL_ID).to(DEVICE).eval()
    return processor, model


def predict(processor, model, rgb):
    """DPT output (relative inverse depth: larger = closer), resized to the frame size."""
    with torch.no_grad():
        x = processor(images=rgb, return_tensors="pt")["pixel_values"].to(DEVICE)
        d = model(pixel_values=x).predicted_depth.unsqueeze(1)
        d = torch.nn.functional.interpolate(d, size=rgb.shape[:2], mode="bilinear", align_corners=False)
    return d[0, 0].float().cpu().numpy()


def main():
    dev, test = list_frames(CFG, "dev"), list_frames(CFG, "test")
    if LIMIT:
        dev, test = dev[:LIMIT], test[:LIMIT]
    print(f"{MODEL_ID} on {DEVICE} | {len(dev)} dev frames | {len(test)} test frames")
    processor, model = load_model()

    # 1. Dev: fit depth = a * (-DPT) + b on every 4th valid pixel (same fit as the other models)
    xs, ys = [], []
    for cp in tqdm(dev, desc="dev"):
        rgb, gt, valid = load_frame(cp)
        disp = predict(processor, model, rgb)
        xs.append(-disp[valid][::4])
        ys.append(gt[valid][::4])
    x, y = np.concatenate(xs).astype(np.float64), np.concatenate(ys).astype(np.float64)
    a = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
    b = float(y.mean() - a * x.mean())
    dev_median = float(np.median(y))
    print(f"Fitted on dev: depth = {a:.4f} * (-DPT) + {b:.4f} mm | dev median depth {dev_median:.2f} mm")

    # 2. Test: run once with the dev values
    rows, times = [], []
    for cp in tqdm(test, desc="test"):
        rgb, gt, valid = load_frame(cp)
        t0 = time.perf_counter()
        disp = predict(processor, model, rgb)
        times.append(time.perf_counter() - t0)
        rel = 1.0 / np.clip(disp, 1e-6, None)
        preds = {
            "Constant median (baseline)": np.full(gt.shape, dev_median, np.float32),
            "DPT + scale & shift fitted on dev": np.clip(a * -disp + b, 1e-3, 100.0),
            "DPT + per-frame median scaling (oracle)": np.clip(rel * np.median(gt[valid]) / np.median(rel[valid]),
                                                               1e-3, 100.0),
        }
        for method, pred in preds.items():
            m = compute_metrics(pred, gt, valid)
            rows.append({"frame": cp.stem.replace("_color", ""), "method": method, "absrel": m["absrel"],
                         "mse": m.get("mse", m["rmse"] ** 2), "delta1": m["delta1"]})

    # 3. Table: mean over frames, RMSE = sqrt(mean per-frame MSE)
    per_frame = pd.DataFrame(rows)
    table = per_frame.groupby("method", sort=False).agg(absrel=("absrel", "mean"), mse=("mse", "mean"),
                                                        delta1=("delta1", "mean"))
    table.insert(1, "rmse_mm", np.sqrt(table.pop("mse")))
    table = table.rename(columns={"absrel": "AbsRel", "rmse_mm": "RMSE (mm)", "delta1": "delta<1.25"})

    print(f"\nTest {CFG['test_sequence']} ({len(test)} frames), mask A")
    print(table.round(4).to_string())
    print(f"\nTime: {np.mean(times[1:] if len(times) > 1 else times):.2f} s per frame on {DEVICE}")
    print("Other models on the same test set (from the group's results):")
    print("  Constant median 0.550 / 17.71 mm / 0.275 | DAv2 Small 0.320 / 6.84 mm / 0.512 | "
          "MiDaS DPT-Hybrid 0.424 / 16.83 mm / 0.219 | Depth Pro 0.586 / 13.53 mm / 0.308")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_frame.to_csv(OUT_DIR / "dpt_per_frame.csv", index=False)
    table.to_csv(OUT_DIR / "dpt_results.csv")
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
