"""Report figures and statistics (version 2: group base, mask A). Reads only the CSV/JSON files in c3vd/results."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results"
OUT = REPO / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
LUM, REF = R / "LumenAware_on_SmallSmooth", R / "E" / "refinement"
DAV2 = R / "DAv2_MaskA_Verified"

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GREY, INK, INK2, GRID = "#a3a29c", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GREY, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "figure.dpi": 110,
    "savefig.dpi": 300, "savefig.bbox": "tight", "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def block_ci(x, n_boot=5000, block=10, seed=42):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    n = len(x)
    starts = np.arange(n - block + 1)
    k = int(np.ceil(n / block))
    means = [np.concatenate([x[s:s + block] for s in rng.choice(starts, k)])[:n].mean() for _ in range(n_boot)]
    return np.percentile(means, [2.5, 97.5])


# ------------------------------------------------------------------ data (test, mask A)
dav2 = pd.read_csv(DAV2 / "dav2_all_variants_maskA_official518_test.csv").set_index("method")
zoe = pd.read_csv(R / "ZoeDepth/zoedepth_test_summary.csv").query("mask == 'A'").set_index("scaling")
dp = pd.read_csv(R / "E/depthpro_test_summary.csv").query("mask == 'A'").set_index("method")
dpt = pd.read_csv(R / "DPT/dpt_results_test_maskA.csv").set_index("method")
dpt_pf = pd.read_csv(R / "DPT/dpt_per_frame_test_MaskA.csv")
midas = pd.read_csv(R / "MiDaS/midas_summary_mask_A.csv").set_index("Method")
spec = pd.read_csv(R / "SpecularIllumination_on_SmallSmooth/specular_summary_mask_A.csv").set_index("method")
geo = pd.read_csv(R / "GeoConsistency_on_SmallSmooth/test_summary.csv").query("mask == 'A'").set_index("method")
lt = pd.read_csv(LUM / "test_summary.csv").query("mask == 'A'").set_index("method")
ld = pd.read_csv(LUM / "dev_cv_summary.csv").query("mask == 'A'").set_index("method")
lpf = pd.read_csv(LUM / "per_frame_test.csv", dtype={"frame": str}).query("mask == 'A'")
ref = pd.read_csv(REF / "ablation_test_summary.csv").query("mask == 'A'").set_index("variant")
rpf = pd.read_csv(REF / "ablation_test_per_frame.csv", dtype={"frame": str}).query("mask == 'A'")
lin_pf = pd.read_csv(DAV2 / "small_linear_maskA_test_per_frame.csv", dtype={"frame": str})
base_pf = pd.read_csv(DAV2 / "small_smooth_maskA_test_per_frame.csv", dtype={"frame": str})

dpt_mae = None  # dpt_results.csv has no MAE
BASE = "Base: Small + smooth (group, mask A)"
LISO = "+ lumen-aware isotonic (log)"

# ------------------------------------------------------------------ statistics
W = lpf.pivot(index="frame", columns="method", values="absrel")
W["Small + global linear"] = lin_pf.set_index("frame").absrel
stats = []
pairs = [("Small + global linear", BASE, "smooth correction (base) vs linear"),
         ("+ lumen-aware residual (darkness)", BASE, ""), ("+ lumen-aware residual (depth)", BASE, ""),
         ("+ lumen-aware residual (both)", BASE, ""), ("+ isotonic (log)", BASE, ""), (LISO, BASE, ""),
         (LISO, "+ isotonic (log)", "lumen map on top of isotonic")]
for a, b, note in pairs:
    if a == "Small + global linear":
        a, b = b, a        # report base vs linear
    d = W[a] - W[b]
    lo, hi = block_ci(d.sort_index().values)
    stats.append({"method": a, "reference": b, "mean dAbsRel": d.mean(), "CI lo": lo, "CI hi": hi,
                  "frames better": f"{(d < 0).sum()}/{len(d)}", "Wilcoxon p": wilcoxon(W[a], W[b]).pvalue})
stats = pd.DataFrame(stats)
stats.to_csv(OUT / "paired_tests_calibration.csv", index=False)
ci = []
for m in W.columns:
    lo, hi = block_ci(W[m].sort_index().values)
    ci.append({"method": m, "mean": W[m].mean(), "ci_lo": lo, "ci_hi": hi})
ci = pd.DataFrame(ci)
ci.to_csv(OUT / "bootstrap_ci_calibration.csv", index=False)

# ------------------------------------------------------------------ Fig 1: framework
fig, ax = plt.subplots(figsize=(13, 5.4))
ax.set_xlim(0, 13); ax.set_ylim(0, 5.4); ax.axis("off"); ax.grid(False)
cols = [
    (2.0, "A. Released model (frozen)", ["MiDaS DPT-Hybrid", "DPT-Large", "ZoeDepth NYU+KITTI", "Depth Pro",
                                        "DAv2 S / B / L", "DAv2 Large Indoor / Outdoor"], "#dce9f8"),
    (4.75, "B. Scale alignment", ["None (raw metric)", "Dev-fit scale", "Dev-fit scale + shift",
                                  "Per-frame median*", "Per-frame scale + shift*"], "#dce9f8"),
    (7.5, "C. Calibration (DAv2 Small)", ["N1 Smooth (group base)",
                                          "N2 abl.: lumen residual", "N2 abl.: isotonic only",
                                          "N2 Lumen-iso (final)", "N4 Specular illumination"], "#fde3d6"),
    (10.25, "D. Refinement (on the base)", ["N3 Uncertainty-guided filter", "N3 abl.: 6-pass ensemble",
                                            "N3 abl.: uniform, edge", "N5 Geometry consistency"], "#fde3d6"),
]
for x, head, items, col in cols:
    ax.text(x + 1.1, 5.05, head, ha="center", fontsize=9.5, fontweight="bold", color=INK)
    for j, t in enumerate(items):
        yy = 4.45 - j * 0.68
        bold = ("group base" in t) or ("final" in t)
        if t.startswith("  ="):
            ax.text(x + 1.1, yy + 0.18, t.strip(), ha="center", va="center", fontsize=7.8, color=INK2, style="italic")
            continue
        ax.add_patch(FancyBboxPatch((x, yy - 0.25), 2.2, 0.5, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=col, ec=INK if bold else GREY, lw=1.6 if bold else 0.8))
        ax.text(x + 1.1, yy, t, ha="center", va="center", fontsize=8.3, color=INK, fontweight="bold" if bold else None)
ax.text(0.9, 5.05, "Input", ha="center", fontsize=9.5, fontweight="bold", color=INK)
ax.add_patch(FancyBboxPatch((0.1, 2.6), 1.6, 0.7, boxstyle="round,pad=0,rounding_size=0.08", fc="#f3f2ee", ec=GREY, lw=0.8))
ax.text(0.9, 2.95, "C3VD RGB frame\n1080 x 1350", ha="center", va="center", fontsize=8.4, color=INK)
for x0, x1 in [(1.72, 1.98), (4.22, 4.73), (6.97, 7.48), (9.72, 10.23)]:
    ax.annotate("", (x1, 2.95), (x0, 2.95), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2))
ax.text(6.5, 0.45, "All paths: mask A, metrics per frame then mean over frames (RMSE = sqrt of mean MSE). "
        "Dev = trans_t1_a (61 frames, all fitting). Test = trans_t2_b (103 frames, run once).",
        ha="center", fontsize=8.3, color=INK2)
ax.text(6.5, 0.12, "* uses the ground truth of each test frame (oracle upper bound). N1-N5 = the five group novelties. Bold = N1 (group base) and N2 (final model). abl. = ablation.",
        ha="center", fontsize=8.3, color=INK2)
save(fig, "fig01_framework")

# ------------------------------------------------------------------ Fig 2: model comparison (test, mask A)
rows = [
    ("Constant median (baseline)", dav2 is not None and 0.5498, 17.71, 0.2752, "grey"),
    ("MiDaS DPT-Hybrid, inverse fit", midas.loc["MiDaS inverse", "AbsRel"],
     midas.loc["MiDaS inverse", "RMSE_mm"], midas.loc["MiDaS inverse", "Delta1"], "m"),
    ("DPT-Large, scale + shift", dpt.loc["DPT + scale & shift fitted on dev", "AbsRel"],
     dpt.loc["DPT + scale & shift fitted on dev", "RMSE (mm)"], dpt.loc["DPT + scale & shift fitted on dev", "delta<1.25"], "m"),
    ("Depth Pro, dev scale", dp.loc["Dev scale", "absrel"], dp.loc["Dev scale", "rmse"], dp.loc["Dev scale", "delta1"], "m"),
    ("ZoeDepth, scale + shift", zoe.loc["dev scale + shift", "absrel"], zoe.loc["dev scale + shift", "rmse"],
     zoe.loc["dev scale + shift", "delta1"], "m"),
    ("DAv2 Large Outdoor, linear", *dav2.loc["DAv2 Outdoor + global linear", ["absrel", "rmse", "delta1"]], "m"),
    ("DAv2 Small, linear", *dav2.loc["DAv2 Small + global linear", ["absrel", "rmse", "delta1"]], "m"),
    ("DAv2 Base, linear", *dav2.loc["DAv2 Base + global linear", ["absrel", "rmse", "delta1"]], "m"),
    ("DAv2 Large Indoor, linear", *dav2.loc["DAv2 Indoor + global linear", ["absrel", "rmse", "delta1"]], "m"),
    ("DAv2 Large, linear", *dav2.loc["DAv2 Large + global linear", ["absrel", "rmse", "delta1"]], "m"),
    ("N1: DAv2 Small + smooth (group base)", *lt.loc[BASE, ["absrel", "rmse", "delta1"]], "ours"),
    ("N2: + lumen-aware isotonic (final model)", *lt.loc[LISO, ["absrel", "rmse", "delta1"]], "ours"),
]
fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
names = [r[0] for r in rows]
y = np.arange(len(rows))
for c, (k, ttl) in enumerate([(1, "AbsRel (lower is better)"), (2, "RMSE, mm (lower is better)"), (3, "δ < 1.25 (higher is better)")]):
    ax = axes[c]
    v = np.array([r[k] for r in rows], float)
    cols_ = [GREY if r[4] == "grey" else (C[1] if r[4] == "ours" else C[0]) for r in rows]
    ax.barh(y, v, color=cols_, height=0.62)
    for yi, vi in zip(y, v):
        ax.text(vi, yi, f" {vi:.3f}" if k != 2 else f" {vi:.2f}", va="center", fontsize=8.3, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(names if c == 0 else [""] * len(y))
    ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, v.max() * 1.22); ax.set_title(ttl)
    ax.axhline(9.5, color=INK2, lw=0.8, ls=":")
fig.suptitle("Released models with a dev-fitted correction, the group base and the final model (test, trans_t2_b, mask A)",
             fontsize=12, fontweight="bold", x=0.02, ha="left")
fig.text(0.02, -0.02, "Blue: released models with a deployable dev-fitted scale (and shift). Orange: N1 (group base) and N2 (final model).", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "fig02_model_comparison")

# ------------------------------------------------------------------ Fig 3: alignment protocols (test, mask A)
dpt_med = dpt.loc["DPT + per-frame median scaling (oracle)", "AbsRel"]
prot = {
    "Depth Pro": [dp.loc["Raw metric", "absrel"], dp.loc["Dev scale", "absrel"], dp.loc["Linear", "absrel"],
                  dp.loc["Oracle median", "absrel"], dp.loc["Oracle LSQ", "absrel"]],
    "ZoeDepth": [zoe.loc["none (metric x1000)", "absrel"], zoe.loc["dev scale", "absrel"], zoe.loc["dev scale + shift", "absrel"],
                 zoe.loc["per-frame median (oracle)", "absrel"], zoe.loc["per-frame scale + shift (oracle)", "absrel"]],
    "DPT-Large": [np.nan, np.nan, dpt.loc["DPT + scale & shift fitted on dev", "AbsRel"], dpt_med, np.nan],
    "DAv2 Small": [np.nan, np.nan, dav2.loc["DAv2 Small + global linear", "absrel"],
                   dav2.loc["DAv2 Small + per-image GT median ORACLE", "absrel"], np.nan],
}
cats = ["Raw metric\n(no fit)", "Dev-fit\nscale", "Dev-fit\nscale + shift", "Per-frame\nmedian (oracle)", "Per-frame\nscale + shift (oracle)"]
fig, ax = plt.subplots(figsize=(11, 4.6))
x = np.arange(len(cats)); w = 0.2
for i, (m, v) in enumerate(prot.items()):
    xs = x + (i - 1.5) * w
    ax.bar(xs, v, w * 0.92, color=C[i], label=m)
    for xi, vi in zip(xs, v):
        if not np.isnan(vi):
            ax.text(xi, vi * 1.08, f"{vi:.2f}" if vi < 10 else f"{vi:.0f}", ha="center", fontsize=7.8, color=INK)
ax.axhline(0.5498, color=INK2, ls="--", lw=1, label="Constant-median baseline (0.550)")
ax.set_yscale("log"); ax.set_ylim(0.15, 400)
ax.set_xticks(x); ax.set_xticklabels(cats)
ax.set_ylabel("Test AbsRel (log scale)")
ax.set_title("Scale ambiguity: the same predictions under five alignment protocols (test, mask A)", loc="left")
ax.legend(loc="upper right", ncol=3, fontsize=9)
save(fig, "fig03_alignment_protocol")

# ------------------------------------------------------------------ Fig 4: calibration chain on the base (dev CV + test)
order = [BASE, "+ lumen-aware residual (darkness)", "+ lumen-aware residual (depth)", "+ lumen-aware residual (both)",
         "+ isotonic (log)", LISO]
labels = ["N1: Small + smooth (group base)", "+ lumen residual (darkness)", "+ lumen residual (depth)",
          "+ lumen residual (both)", "+ isotonic (log)", "N2: + lumen-aware isotonic (final)"]
lin_t = dav2.loc["DAv2 Small + global linear"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.9))
for ax, (k, ttl) in zip(axes, [("absrel", "AbsRel ↓"), ("rmse", "RMSE (mm) ↓"), ("delta1", "δ < 1.25 ↑")]):
    yy = np.arange(len(order) + 1); h = 0.38
    dv = np.r_[np.nan, ld.loc[order, k].values]
    tv = np.r_[lin_t[k], lt.loc[order, k].values]
    ax.barh(yy - h / 2, dv, h * 0.92, color=C[0], label="Dev, 4-block CV (held-out frames)")
    ax.barh(yy + h / 2, tv, h * 0.92, color=C[1], label="Test (fitted on all dev, run once)")
    f = "{:.3f}" if k != "rmse" else "{:.2f}"
    for yi, a_, b_ in zip(yy, dv, tv):
        if not np.isnan(a_):
            ax.text(a_, yi - h / 2, " " + f.format(a_), va="center", fontsize=7.6, color=INK)
        ax.text(b_, yi + h / 2, " " + f.format(b_), va="center", fontsize=7.6, color=INK)
    ax.set_yticks(yy); ax.set_yticklabels(["Small + global linear"] + labels if ax is axes[0] else [""] * len(yy))
    ax.invert_yaxis(); ax.set_title(ttl); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, np.nanmax(np.r_[dv, tv]) * 1.3)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=2, fontsize=9, bbox_to_anchor=(0.99, 0.99))
fig.suptitle("Calibration of DAv2 Small 518 (mask A). Each lumen / isotonic method corrects the base.",
             fontsize=12, fontweight="bold", x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.9))
save(fig, "fig04_calibration_chain")

# ------------------------------------------------------------------ Fig 5: per-frame test
fig, axes = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
series = [("Small + global linear", W["Small + global linear"], C[0]), ("Small + smooth (base)", W[BASE], C[1]),
          ("+ lumen-aware isotonic", W[LISO], C[2])]
for n, s, col in series:
    s = s.sort_index()
    axes[0].plot(s.index.astype(int), s.values, color=col, lw=2, label=n)
axes[0].set_ylabel("AbsRel per frame"); axes[0].set_ylim(0, 0.6)
axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
d = (W[LISO] - W[BASE]).sort_index()
axes[1].bar(d.index.astype(int), d.values, color=[C[2] if v < 0 else C[7] for v in d.values], width=0.8)
axes[1].axhline(0, color=INK2, lw=0.8)
axes[1].set_ylabel("Δ AbsRel\n(lumen-iso − base)"); axes[1].set_xlabel("Test frame index (trans_t2_b)")
axes[1].text(0.01, 0.08, f"Lumen-aware isotonic better on {(d < 0).sum()} / {len(d)} frames",
             transform=axes[1].transAxes, fontsize=9, color=INK2)
axes[0].set_title("Per-frame error along the test sequence (mask A)", loc="left")
fig.tight_layout()
save(fig, "fig05_per_frame_test")

# ------------------------------------------------------------------ Fig 6: wall vs lumen
fig, ax = plt.subplots(figsize=(10, 4))
o2 = [BASE, "+ lumen-aware residual (both)", "+ isotonic (log)", LISO]
l2 = ["Small + smooth\n(base)", "+ lumen residual\n(both)", "+ isotonic\n(log)", "+ lumen-aware\nisotonic"]
t2 = lt.loc[o2]
x = np.arange(len(o2))
ax.bar(x - 0.19, t2.absrel_wall, 0.36, color=C[0], label="Wall region (w ≤ 0.5)")
ax.bar(x + 0.19, t2.absrel_lumen, 0.36, color=C[1], label="Detected lumen (w > 0.5)")
for xi, a_, b_ in zip(x, t2.absrel_wall, t2.absrel_lumen):
    ax.text(xi - 0.19, a_ + 0.006, f"{a_:.3f}", ha="center", fontsize=8.5)
    ax.text(xi + 0.19, b_ + 0.006, f"{b_:.3f}", ha="center", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(l2, fontsize=9); ax.grid(axis="x", visible=False)
ax.set_ylabel("Test AbsRel"); ax.legend(); ax.set_ylim(0, 0.38)
ax.set_title("The gain is on the wall, not in the lumen (test, mask A)", loc="left")
save(fig, "fig06_wall_vs_lumen")

# ------------------------------------------------------------------ Fig 7: refinement + sparsification
sp_t = pd.read_csv(REF / "sparsification_test.csv").query("mask == 'A'")
sp_d = pd.read_csv(REF / "sparsification_dev_cv.csv").query("mask == 'A'")
uq_t = json.load(open(REF / "uncertainty_quality_test.json"))["A"]
uq_d = json.load(open(REF / "uncertainty_quality_dev_cv.json"))["A"]
idx = ["Small + Smooth (base)", "Uniform smoothing", "Edge-aware (guided filter)", "Uncertainty-guided",
       "Ensemble mean (control)", "Ensemble + uncertainty-guided"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.4, 1, 1]})
b = ref.loc[idx]; yy = np.arange(len(idx))
axes[0].barh(yy - 0.2, b.absrel, 0.38, color=C[0], label="All pixels")
axes[0].barh(yy + 0.2, b.absrel_edge, 0.38, color=C[1], label="Boundary pixels (top 10% GT gradient)")
for yi, a_, e_ in zip(yy, b.absrel, b.absrel_edge):
    axes[0].text(a_, yi - 0.2, f" {a_:.3f}", va="center", fontsize=8)
    axes[0].text(e_, yi + 0.2, f" {e_:.3f}", va="center", fontsize=8)
axes[0].set_yticks(yy); axes[0].set_yticklabels(["Small + smooth (base)", "Uniform Gaussian", "Edge-aware (guided)",
                                                  "Uncertainty-guided", "Ensemble mean (control)", "Ensemble + uncert.-guided"])
axes[0].invert_yaxis(); axes[0].set_xlim(0, 0.62); axes[0].grid(axis="y", visible=False)
axes[0].set_title("Test AbsRel (mask A)"); axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=2)
for ax, s, uq, ttl in [(axes[1], sp_d, uq_d, "Sparsification, dev CV"), (axes[2], sp_t, uq_t, "Sparsification, test")]:
    ax.plot(s.fraction_removed * 100, s.by_uncertainty, color=C[0], lw=2, marker="o", ms=5, label="Remove most uncertain")
    ax.plot(s.fraction_removed * 100, s.oracle, color=C[2], lw=2, marker="o", ms=5, label="Remove largest error (oracle)")
    ax.plot(s.fraction_removed * 100, s.random, color=GREY, lw=2, ls="--", label="Random")
    ax.set_xlabel("Pixels removed (%)"); ax.set_title(ttl); ax.set_ylim(0, 0.4)
    ax.text(0.97, 0.93, f"Spearman ρ {uq['spearman_uncertainty_vs_error']:.2f}\nAUSE {uq['AUSE']:.3f}",
            transform=ax.transAxes, ha="right", va="center", fontsize=8.5, color=INK2)
axes[1].set_ylabel("AbsRel of remaining pixels"); axes[1].legend(fontsize=8, loc="lower left")
fig.suptitle("Uncertainty-guided refinement on the group base (6 perturbed passes)", fontsize=12, fontweight="bold", x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "fig07_refinement")

# ------------------------------------------------------------------ Fig 8: accuracy vs speed
pts = [("DAv2 Small + linear", 0.064, dav2.loc["DAv2 Small + global linear", "absrel"], C[0]),
       ("DAv2 Small + smooth (base)", 0.064, lt.loc[BASE, "absrel"], C[1]),
       ("+ lumen-aware isotonic", 0.064, lt.loc[LISO, "absrel"], C[2]),
       ("Base + 6-pass ensemble\n+ uncertainty-guided", 0.5, ref.loc["Ensemble + uncertainty-guided", "absrel"], C[6]),
       ("ZoeDepth + scale + shift", json.load(open(R / "ZoeDepth/run_info_test.json"))["sec_per_frame"],
        zoe.loc["dev scale + shift", "absrel"], C[3]),
       ("Depth Pro + dev scale", json.load(open(R / "E/run_info_test.json"))["sec_per_frame"], dp.loc["Dev scale", "absrel"], C[7])]
fig, ax = plt.subplots(figsize=(9, 4.6))
offs = {"DAv2 Small + linear": (9, 4), "DAv2 Small + smooth (base)": (9, -4), "+ lumen-aware isotonic": (9, 0)}
for n, s, a, col in pts:
    ax.scatter(s, a, s=110, color=col, edgecolor="white", linewidth=2, zorder=3)
    ax.annotate(n, (s, a), xytext=offs.get(n, (9, 4)), textcoords="offset points", fontsize=8.5, color=INK, va="center")
ax.axhline(0.5498, color=INK2, ls="--", lw=1); ax.text(0.03, 0.565, "constant median", fontsize=8.5, color=INK2)
ax.axvline(1 / 25, color=C[7], ls=":", lw=1.2); ax.text(0.041, 0.07, "25 fps\n(video rate)", fontsize=8, color=C[7])
ax.set_xscale("log"); ax.set_xlim(0.025, 15); ax.set_ylim(0, 0.68)
ax.set_xlabel("Model inference, seconds per frame (log scale, Apple M4 Pro, MPS)"); ax.set_ylabel("Test AbsRel (mask A)")
ax.set_title("Accuracy against inference time", loc="left")
save(fig, "fig08_accuracy_vs_speed")

print(stats.round(4).to_string())
print(ci.round(4).to_string())
print("DPT per-frame check:", dpt_pf.groupby("method").absrel.mean().round(4).to_dict())


# ------------------------------------------------------------------ Fig 9: the five novelties, each against its own reference
nov = [
    ("N1 Smooth correction (Sowmya)\nref: DAv2 Small + linear", dav2.loc["DAv2 Small + smooth correction", "absrel"], dav2.loc["DAv2 Small + global linear", "absrel"], C[1]),
    ("N2 Lumen-aware isotonic (Aditi)\nref: group base", lt.loc[LISO, "absrel"], lt.loc[BASE, "absrel"], C[2]),
    ("N3 Uncertainty-guided (Eshitha)\nref: group base", ref.loc["Uncertainty-guided", "absrel"], ref.loc["Small + Smooth (base)", "absrel"], C[0]),
    ("N3 + 6-pass ensemble\nref: group base", ref.loc["Ensemble + uncertainty-guided", "absrel"], ref.loc["Small + Smooth (base)", "absrel"], C[0]),
    ("N4 Specular reliability (Shravan)\nref: its base run", spec.loc["C: + Reliability", "AbsRel"], spec.loc["A: Small + Smooth", "AbsRel"], C[0]),
    ("N5 Geometry consistency (Poorvi)\nref: its base run", geo.loc["+ geometry-consistency (full)", "absrel"], geo.loc["Base: Small + smooth (group)", "absrel"], C[0]),
]
fig, ax = plt.subplots(figsize=(11, 4.6))
yy = np.arange(len(nov))
for yi, (n, v, r, col) in zip(yy, nov):
    d = v - r
    ax.barh(yi, d, color=col if d < 0 else C[7], height=0.6)
    ax.text(d - 0.002 if d < 0 else d + 0.002, yi, (f"{d:+.5f}  ({r:.4f} \u2192 {v:.4f})" if abs(d) < 1e-3 else f"{d:+.4f}  ({r:.3f} \u2192 {v:.3f})"), va="center",
            ha="right" if d < 0 else "left", fontsize=9, color=INK)
ax.axvline(0, color=INK, lw=0.8)
ax.set_yticks(yy); ax.set_yticklabels([n[0] for n in nov], fontsize=9); ax.invert_yaxis()
ax.set_xlim(-0.13, 0.03); ax.grid(axis="y", visible=False)
ax.set_xlabel("Change of test AbsRel against the reference (negative = better), mask A")
ax.set_title("The five group novelties (test, trans_t2_b)", loc="left")
fig.tight_layout()
save(fig, "fig09_novelties")
