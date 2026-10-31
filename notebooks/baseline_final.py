from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

BASE = Path(__file__).resolve().parent
DEV, TEST = "trans_t1_a", "trans_t2_b"
MAX_DEPTH_MM = 100.0                      # uint16 0-65535 -> 0-100 mm


def depth_files(seq):
    # nested layout first: BASE/seq/seq/, then BASE/seq/
    for folder in (BASE / seq / seq, BASE / seq):
        files = sorted(folder.glob("*_depth.tiff")) + sorted(folder.glob("*_depth.tif"))
        if files:
            print(f"{seq}: {len(files)} depth files in {folder}")
            return files
    raise SystemExit(f"No *_depth.tiff files found for {seq} under {BASE / seq}")


def load_depth_mm(path):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.dtype != np.uint16:
        raw = tifffile.imread(str(path))
    assert raw.dtype == np.uint16, f"Expected uint16, got {raw.dtype} ({path.name})"
    return raw.astype(np.float32) / 65535.0 * MAX_DEPTH_MM


def metrics(pred, gt, mask):
    p, g = pred[mask], gt[mask]
    absrel = np.mean(np.abs(p - g) / g)
    rmse = np.sqrt(np.mean((p - g) ** 2))
    delta1 = np.mean(np.maximum(p / g, g / p) < 1.25)
    return absrel, rmse, delta1


# 1. Constant = median GT depth over valid pixels of the DEV sequence
vals = []
for f in tqdm(depth_files(DEV), desc="dev"):
    d = load_depth_mm(f)
    vals.append(d[d > 0][::50])           # subsample to save memory
dev_median = float(np.median(np.concatenate(vals)))
print(f"\nDev median depth ({DEV}): {dev_median:.3f} mm")

# 2. Evaluate on the TEST sequence
rows = []
for f in tqdm(depth_files(TEST), desc="test"):
    gt = load_depth_mm(f)
    mask = gt > 0
    if mask.sum() == 0:
        continue
    a, r, d = metrics(np.full_like(gt, dev_median), gt, mask)
    ao, ro, do = metrics(np.full_like(gt, np.median(gt[mask])), gt, mask)
    rows.append(dict(frame=f.name, absrel=a, rmse=r, delta1=d,
                     absrel_oracle=ao, rmse_oracle=ro, delta1_oracle=do))

df = pd.DataFrame(rows)
df.to_csv(BASE / "baseline_per_frame.csv", index=False)

print(f"\n=== Test ({TEST}), {len(df)} frames, mean over frames ===")
print(f"Constant (dev median {dev_median:.2f} mm): "
      f"AbsRel={df.absrel.mean():.4f}  RMSE={df.rmse.mean():.3f} mm  "
      f"d<1.25={df.delta1.mean():.4f}")
print(f"Oracle per-image median:                 "
      f"AbsRel={df.absrel_oracle.mean():.4f}  RMSE={df.rmse_oracle.mean():.3f} mm  "
      f"d<1.25={df.delta1_oracle.mean():.4f}")