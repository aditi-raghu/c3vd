"""With vs without the improvement: the base (Small + Smooth) vs the refined variants, per split and mask.

Reads ablation_{dev_cv,test}_per_frame.csv and writes improvement_vs_base_{dev_cv,test}.csv.
Aggregation is the group rule (frame means; RMSE = sqrt(mean per-frame MSE)). The 95% CI of the AbsRel change is a paired
moving-block bootstrap (10-frame blocks, 4000 resamples, seed 42), because neighbouring frames are correlated.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = "Small + Smooth (base)"
COMPARE = ["Uniform smoothing", "Edge-aware (guided filter)", "Uncertainty-guided", "Ensemble mean (control)", "Ensemble + uncertainty-guided"]
METRICS = [("mae", "MAE (mm)", -1), ("absrel", "AbsRel", -1), ("rmse", "RMSE (mm)", -1), ("delta1", "δ<1.25", +1),
           ("absrel_edge", "AbsRel boundary", -1), ("absrel_flat", "AbsRel flat", -1)]


def block_ci(d, block=10, n=4000, seed=42):
    d = np.asarray(d)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(d) - block + 1, size=(n, len(d) // block + 1))
    means = np.array([np.concatenate([d[s:s + block] for s in row])[:len(d)].mean() for row in starts])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


for tag in ("dev_cv", "test"):
    per_frame = pd.read_csv(HERE / f"ablation_{tag}_per_frame.csv", dtype={"frame": str})
    rows = []
    for mk, pf in per_frame.groupby("mask"):
        agg = {v: {"mae": g.mae.mean(), "absrel": g.absrel.mean(), "rmse": np.sqrt(g.mse.mean()), "delta1": g.delta1.mean(),
                   "absrel_edge": g.absrel_edge.mean(), "absrel_flat": g.absrel_flat.mean()}
               for v, g in pf.groupby("variant")}
        piv = pf.pivot(index="frame", columns="variant", values="absrel").sort_index()
        for v in COMPARE:
            for key, name, better in METRICS:
                b, w = agg[BASE][key], agg[v][key]
                rows.append({"split": tag, "mask": mk, "variant": v, "metric": name, "without (base)": round(b, 4), "with improvement": round(w, 4),
                             "difference": round(w - b, 4), "change_%": round(100 * (w - b) / b, 1), "better?": "yes" if (w - b) * better > 0 else "no"})
            lo, hi = block_ci((piv[v] - piv[BASE]).values)
            rows.append({"split": tag, "mask": mk, "variant": v, "metric": "AbsRel change, 95% CI low", "difference": round(lo, 4)})
            rows.append({"split": tag, "mask": mk, "variant": v, "metric": "AbsRel change, 95% CI high", "difference": round(hi, 4)})
            rows.append({"split": tag, "mask": mk, "variant": v, "metric": "frames with lower AbsRel", "with improvement": f"{int((piv[v] < piv[BASE]).sum())}/{len(piv)}"})
    out = pd.DataFrame(rows)
    out.to_csv(HERE / f"improvement_vs_base_{tag}.csv", index=False)
    print(f"\n== {tag} ==\n" + out[(out["mask"] == "A") & out.metric.isin(["AbsRel", "frames with lower AbsRel", "AbsRel change, 95% CI low", "AbsRel change, 95% CI high"])].astype(object).fillna("").to_string(index=False))
