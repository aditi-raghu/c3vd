from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import DPTForDepthEstimation, DPTImageProcessor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Settings ----------------
BASE = Path(__file__).resolve().parent
DEV, TEST = "trans_t1_a", "trans_t2_b"
MODEL_NAME = "Intel/dpt-hybrid-midas"   # lighter. Alternative: "Intel/dpt-large"
DEV_STRIDE = 5          # use every 5th dev frame to fit the global scale/shift
TEST_STRIDE = 1         # 1 = all test frames. Use 10 for a quick trial run
MAX_DEPTH_MM = 100.0
DEPTH_CLIP = (0.5, 100.0)   # predictions are clipped to this range (mm)
N_QUAL = 4              # number of qualitative examples
# ------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device} | Model: {MODEL_NAME}")
processor = DPTImageProcessor.from_pretrained(MODEL_NAME)
model = DPTForDepthEstimation.from_pretrained(MODEL_NAME).to(device).eval()


def frame_pairs(seq):
    for folder in (BASE / seq / seq, BASE / seq):
        depths = sorted(folder.glob("*_depth.tif*"))
        if depths:
            pairs = [(d.with_name(d.name.split("_depth")[0] + "_color.png"), d) for d in depths]
            pairs = [(c, d) for c, d in pairs if c.exists()]
            print(f"{seq}: {len(pairs)} color+depth pairs in {folder}")
            return pairs
    raise SystemExit(f"No depth files found for {seq}")


def load_depth_mm(path):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.dtype != np.uint16:
        raw = tifffile.imread(str(path))
    assert raw.dtype == np.uint16, f"Expected uint16, got {raw.dtype}"
    return raw.astype(np.float32) / 65535.0 * MAX_DEPTH_MM


def load_rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


@torch.no_grad()
def predict_rel(rgb, out_shape):
    """DPT output = relative INVERSE depth (larger = closer), resized to GT size."""
    inputs = processor(images=rgb, return_tensors="pt").to(device)
    out = model(**inputs).predicted_depth.unsqueeze(1)
    out = F.interpolate(out, size=out_shape, mode="bicubic", align_corners=False)
    return out.squeeze().float().cpu().numpy()


def disp_to_depth(disp):
    return np.clip(1.0 / np.maximum(disp, 1e-6), *DEPTH_CLIP)


def metrics(pred, gt, mask):
    p, g = pred[mask], gt[mask]
    absrel = np.mean(np.abs(p - g) / g)
    rmse = np.sqrt(np.mean((p - g) ** 2))
    delta1 = np.mean(np.maximum(p / g, g / p) < 1.25)
    return absrel, rmse, delta1


def fit_scale_shift(pred, gt_disp):
    A = np.stack([pred, np.ones_like(pred)], axis=1)
    coef, *_ = np.linalg.lstsq(A, gt_disp, rcond=None)
    return float(coef[0]), float(coef[1])


# ---------- 1. Constant baseline (dev median, all dev frames) ----------
dev_pairs = frame_pairs(DEV)
vals = []
for _, d in tqdm(dev_pairs, desc="dev median"):
    g = load_depth_mm(d)
    vals.append(g[g > 0][::50])
dev_median = float(np.median(np.concatenate(vals)))
print(f"Constant baseline: {dev_median:.3f} mm")

# ---------- 2. Fit GLOBAL scale/shift on DEV only ----------
P, G = [], []
for c, d in tqdm(dev_pairs[::DEV_STRIDE], desc="dev fit"):
    gt = load_depth_mm(d)
    m = gt > 0
    pred = predict_rel(load_rgb(c), gt.shape)
    P.append(pred[m][::37])
    G.append(1.0 / gt[m][::37])
a_glob, b_glob = fit_scale_shift(np.concatenate(P), np.concatenate(G))
print(f"Global fit (inverse-depth space): gt_disp = {a_glob:.6g} * pred + {b_glob:.6g}")

# ---------- 3. Evaluate on TEST ----------
test_pairs = frame_pairs(TEST)[::TEST_STRIDE]
qual_ids = set(np.linspace(0, len(test_pairs) - 1, N_QUAL).astype(int).tolist())
rows, qual = [], []

for i, (c, d) in enumerate(tqdm(test_pairs, desc="test")):
    gt = load_depth_mm(d)
    mask = gt > 0
    if mask.sum() == 0:
        continue
    rgb = load_rgb(c)
    pred = predict_rel(rgb, gt.shape)

    methods = {}
    methods["const"] = np.full_like(gt, dev_median)
    methods["glob"] = disp_to_depth(a_glob * pred + b_glob)           # fair

    dep = 1.0 / np.maximum(pred, 1e-3 * pred.max())                   # per-image median (uses test GT)
    s = np.median(gt[mask]) / np.median(dep[mask])
    methods["median"] = np.clip(dep * s, *DEPTH_CLIP)

    a_i, b_i = fit_scale_shift(pred[mask][::37], 1.0 / gt[mask][::37])  # per-image LSQ (uses test GT)
    methods["lsq"] = disp_to_depth(a_i * pred + b_i)

    row = {"frame": d.name}
    for k, p in methods.items():
        ar, rm, de = metrics(p, gt, mask)
        row[f"{k}_absrel"], row[f"{k}_rmse"], row[f"{k}_delta1"] = ar, rm, de
    rows.append(row)

    if i in qual_ids:
        qual.append((rgb, gt, methods["glob"], mask))

df = pd.DataFrame(rows)
df.to_csv(BASE / "dpt_per_frame.csv", index=False)

# ---------- 4. Summary table ----------
names = {
    "const": "Constant (dev median)",
    "glob": "DPT + global scale/shift (dev-fit)",
    "median": "DPT + per-image median*",
    "lsq": "DPT + per-image LSQ*",
}
summary = pd.DataFrame({
    "Method": list(names.values()),
    "AbsRel": [df[f"{k}_absrel"].mean() for k in names],
    "RMSE_mm": [df[f"{k}_rmse"].mean() for k in names],
    "delta<1.25": [df[f"{k}_delta1"].mean() for k in names],
})
summary.to_csv(BASE / "dpt_summary.csv", index=False)

line = "=" * 82
print(f"\n{line}\n DPT ({MODEL_NAME}) | dev: {DEV} | test: {TEST} ({len(df)} frames)\n{line}")
print(f" {'Method':<38}{'AbsRel (v)':>12}{'RMSE mm (v)':>14}{'d<1.25 (^)':>13}")
print("-" * 82)
for _, r in summary.iterrows():
    print(f" {r.Method:<38}{r.AbsRel:>12.4f}{r.RMSE_mm:>14.3f}{r['delta<1.25']:>13.4f}")
print(line)
print(" * uses test ground truth -> upper bound only, not a fair result.")
print(" Metrics: mean over frames, valid GT pixels only.\n")

# ---------- 5. Figures ----------
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
colors = ["#999999", "#4C72B0", "#DD8452", "#55A868"]
short = ["Constant", "DPT global", "DPT median*", "DPT LSQ*"]
for a, (t, key) in zip(ax, [("AbsRel (lower better)", "AbsRel"),
                            ("RMSE mm (lower better)", "RMSE_mm"),
                            ("delta<1.25 (higher better)", "delta<1.25")]):
    bars = a.bar(short, summary[key], color=colors)
    a.bar_label(bars, fmt="%.3f")
    a.set_title(t)
    a.tick_params(axis="x", rotation=20)
    a.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(BASE / "dpt_results.png", dpi=200)
plt.close()

fig, ax = plt.subplots(len(qual), 4, figsize=(14, 3.2 * len(qual)))
ax = np.atleast_2d(ax)
for r, (rgb, gt, pr, m) in enumerate(qual):
    vmax = np.percentile(gt[m], 99)
    err = np.where(m, np.abs(pr - gt) / np.maximum(gt, 1e-6), np.nan)
    ax[r, 0].imshow(rgb); ax[r, 0].set_title("Color")
    ax[r, 1].imshow(np.where(m, gt, np.nan), cmap="magma", vmin=0, vmax=vmax); ax[r, 1].set_title("GT depth (mm)")
    ax[r, 2].imshow(pr, cmap="magma", vmin=0, vmax=vmax); ax[r, 2].set_title("DPT (global align)")
    ax[r, 3].imshow(err, cmap="viridis", vmin=0, vmax=1); ax[r, 3].set_title("AbsRel error")
    for a in ax[r]:
        a.axis("off")
plt.tight_layout()
plt.savefig(BASE / "dpt_qualitative.png", dpi=150)
print(" Saved: dpt_summary.csv, dpt_per_frame.csv, dpt_results.png, dpt_qualitative.png")