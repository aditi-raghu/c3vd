"""Figures for the refinement ablation, mask A: sparsification curves (dev CV + test) and the test frames with the largest
and smallest gain."""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from refine import BASE, CFG, MASKS, depth_stack, load_preds, load_split, uncertainty, uncertainty_guided  # noqa: E402

MASK = MASKS[0]
ENSEMBLE_UG = "Ensemble + uncertainty-guided"

# 1. sparsification: AbsRel of the remaining pixels when the most uncertain pixels are removed
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
for ax, tag, title in zip(axes, ["dev_cv", "test"], ["Dev, 4-block CV", f"Test {CFG['test_sequence']}"]):
    c = pd.read_csv(HERE / f"sparsification_{tag}.csv")
    c = c[c["mask"] == MASK]
    q = json.load(open(HERE / f"uncertainty_quality_{tag}.json"))[MASK]
    ax.plot(c.fraction_removed * 100, c.by_uncertainty, "o-", label="remove most uncertain")
    ax.plot(c.fraction_removed * 100, c.oracle, "k--", label="remove largest true error (oracle)")
    ax.plot(c.fraction_removed * 100, c.random, ":", color="grey", label="remove random pixels")
    ax.set(title=f"{title}, mask {MASK}  (Spearman {q['spearman_uncertainty_vs_error']:.2f})", xlabel="pixels removed (%)",
           ylabel="AbsRel of remaining pixels"); ax.grid(alpha=0.3)
axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(HERE / "sparsification.png", dpi=150)

# 2. largest / smallest gain on test for the ensemble + uncertainty-guided variant against the base
pf = pd.read_csv(HERE / "ablation_test_per_frame.csv", dtype={"frame": str})
pf = pf[pf["mask"] == MASK]
piv = pf.pivot(index="frame", columns="variant", values="absrel")
gain = piv[BASE] - piv[ENSEMBLE_UG]
settings = json.load(open(HERE / "chosen_settings.json"))[MASK]; th = settings["test_thresholds"]
saved = settings["base_params_all_dev"]
params = {"scale": saved["scale"], "offset": saved["offset"], "centers": np.array(saved["centers"]), "corrections": np.array(saved["corrections"])}
paths, stems, gts, valids = load_split("test", MASK)
picks = [("largest gain", gain.idxmax()), ("smallest gain", gain.idxmin())]
fig, axes = plt.subplots(2, 6, figsize=(22, 7))
for row, (label, stem) in enumerate(picks):
    i = stems.index(stem)
    bgr, gt, m = cv2.imread(str(paths[i])), gts[i], valids[i]
    stack = depth_stack(load_preds("test", stem), params); D, U = stack["id"], uncertainty(stack)
    R = uncertainty_guided(np.mean(np.stack(list(stack.values())), 0), bgr, U, settings["ug_r"], settings["ug_eps"], th["lo"], th["hi"])
    nan = lambda x: np.where(m, x, np.nan)  # noqa: E731
    e0, e1 = np.abs(D - gt) / np.maximum(gt, 1e-3), np.abs(R - gt) / np.maximum(gt, 1e-3)
    panels = [(bgr[..., ::-1], "RGB", None), (nan(gt), "GT depth (mm)", "plasma"), (nan(D), "Base: Small + Smooth", "plasma"),
              (nan(U), "Uncertainty (rel. spread)", "magma"), (nan(R), "Ensemble + uncertainty-guided", "plasma"),
              (nan(e0 - e1), "AbsRel reduction (blue = better)", "RdBu")]
    for ax, (img, t, cmap) in zip(axes[row], panels):
        kw = {"vmin": 0, "vmax": 60} if cmap == "plasma" else {"vmin": -0.3, "vmax": 0.3} if cmap == "RdBu" else \
             {"vmin": 0, "vmax": np.nanpercentile(img, 99)} if cmap == "magma" else {}
        ax.imshow(img, cmap=cmap, **kw); ax.set_title(t, fontsize=10); ax.axis("off")
    axes[row][0].set_title(f"Test {stem} ({label}: AbsRel {piv.loc[stem, BASE]:.3f} -> {piv.loc[stem, ENSEMBLE_UG]:.3f})", fontsize=10)
fig.tight_layout(); fig.savefig(HERE / "examples_test.png", dpi=110)
print("frames improved:", int((gain > 0).sum()), "/", len(gain), "| picks:", picks)
