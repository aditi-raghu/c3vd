"""Figures for the refinement ablation: sparsification curves (dev + test) and best/worst test frames."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from refine import CFG, depth_stack, load_final_params, load_frame, load_preds, sequence_dir, uncertainty, uncertainty_guided  # noqa: E402

# 1. sparsification: AbsRel of the remaining pixels when the most uncertain pixels are removed
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
for ax, tag, title in zip(axes, ["dev_val", "test"], ["Dev 0045-0060", "Test trans_t2_b"]):
    c = pd.read_csv(HERE / f"sparsification_{tag}.csv"); q = json.load(open(HERE / f"uncertainty_quality_{tag}.json"))
    ax.plot(c.fraction_removed * 100, c.by_uncertainty, "o-", label="remove most uncertain (ours)")
    ax.plot(c.fraction_removed * 100, c.oracle, "k--", label="remove largest true error (oracle)")
    ax.plot(c.fraction_removed * 100, c.random, ":", color="grey", label="remove random pixels")
    ax.set(title=f"{title}  (Spearman {q['spearman_uncertainty_vs_error']:.2f})", xlabel="pixels removed (%)",
           ylabel="AbsRel of remaining pixels"); ax.grid(alpha=0.3)
axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(HERE / "sparsification.png", dpi=150)

# 2. best / worst test frames for variant 4 vs base
pf = pd.read_csv(HERE / "ablation_test_per_frame.csv", dtype={"frame": str})
piv = pf.pivot(index="frame", columns="variant", values="absrel")
gain = piv["0 base (Small + Smooth)"] - piv["4 ensemble + uncertainty-guided"]
settings = json.load(open(HERE / "chosen_settings.json")); th = settings["test_thresholds"]; params = load_final_params()
picks = [("largest gain", gain.idxmax()), ("smallest gain", gain.idxmin())]
fig, axes = plt.subplots(2, 6, figsize=(22, 7))
for row, (label, stem) in enumerate(picks):
    bgr, gt, raw = load_frame(sequence_dir(CFG, "test") / f"{stem}_color.png")
    stack = depth_stack(load_preds("test", stem), params); D, U = stack["id"], uncertainty(stack)
    R = uncertainty_guided(np.mean(np.stack(list(stack.values())), 0), bgr, U, settings["ug_r"], settings["ug_eps"], th["lo"], th["hi"])
    m = raw > 0; nan = lambda x: np.where(m, x, np.nan)
    e0, e1 = np.abs(D - gt) / np.maximum(gt, 1e-3), np.abs(R - gt) / np.maximum(gt, 1e-3)
    panels = [(bgr[..., ::-1], "RGB", None), (nan(gt), "GT depth (mm)", "plasma"), (nan(D), "Base: Small + Smooth", "plasma"),
              (nan(U), "Uncertainty (rel. spread)", "magma"), (nan(R), "Ensemble + unc.-guided", "plasma"),
              (nan(e0 - e1), "AbsRel reduction (green = better)", "RdYlGn")]
    for ax, (img, t, cmap) in zip(axes[row], panels):
        kw = {"vmin": 0, "vmax": 60} if cmap == "plasma" else {"vmin": -0.3, "vmax": 0.3} if cmap == "RdYlGn" else \
             {"vmin": 0, "vmax": np.nanpercentile(img, 99)} if cmap == "magma" else {}
        ax.imshow(img, cmap=cmap, **kw); ax.set_title(t, fontsize=10); ax.axis("off")
    axes[row][0].set_title(f"Test {stem} ({label}: AbsRel {piv.loc[stem, '0 base (Small + Smooth)']:.3f} -> "
                           f"{piv.loc[stem, '4 ensemble + uncertainty-guided']:.3f})", fontsize=10)
fig.tight_layout(); fig.savefig(HERE / "examples_test.png", dpi=110)
print("frames improved:", int((gain > 0).sum()), "/", len(gain), "| picks:", picks)
