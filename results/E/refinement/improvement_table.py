"""With vs without the improvement: group model (Small + Smooth) vs the refined variants.

Reads ablation_{dev_val,test}_per_frame.csv and writes improvement_vs_base_{dev_val,test}.csv.
Aggregation matches the group's final table (frame means; RMSE = sqrt(mean per-frame MSE)).
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = "0 base (Small + Smooth)"
COMPARE = {"3 uncertainty-guided (ours)": "Uncertainty-guided refinement",
           "4 ensemble + uncertainty-guided": "Ensemble + uncertainty-guided refinement"}
METRICS = [("mae", "MAE (mm)", -1), ("absrel", "AbsRel", -1), ("rmse", "RMSE (mm)", -1), ("delta1", "δ<1.25", +1),
           ("absrel_edge", "AbsRel boundary", -1), ("absrel_flat", "AbsRel flat", -1)]

for tag in ("dev_val", "test"):
    pf = pd.read_csv(HERE / f"ablation_{tag}_per_frame.csv", dtype={"frame": str})
    agg = {v: {"mae": g.mae.mean(), "absrel": g.absrel.mean(), "rmse": np.sqrt(g.mse.mean()), "delta1": g.delta1.mean(),
               "absrel_edge": g.absrel_edge.mean(), "absrel_flat": g.absrel_flat.mean()}
           for v, g in pf.groupby("variant")}
    piv = pf.pivot(index="frame", columns="variant", values="absrel")
    rows = []
    for v, label in COMPARE.items():
        for key, name, better in METRICS:
            b, w = agg[BASE][key], agg[v][key]
            rows.append({"comparison": f"{label} vs Small + Smooth", "metric": name, "without (base)": round(b, 4),
                         "with improvement": round(w, 4), "difference": round(w - b, 4),
                         "change_%": round(100 * (w - b) / b, 1), "better?": "yes" if (w - b) * better > 0 else "no"})
        rows.append({"comparison": f"{label} vs Small + Smooth", "metric": "frames with lower AbsRel",
                     "with improvement": f"{int((piv[v] < piv[BASE]).sum())}/{len(piv)}"})
    out = pd.DataFrame(rows)
    out.to_csv(HERE / f"improvement_vs_base_{tag}.csv", index=False)
    print(f"\n== {tag} ==\n" + out.astype(object).fillna("").to_string(index=False))
