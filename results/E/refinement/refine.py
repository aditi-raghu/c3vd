"""Stream E improvement: boundary-preserving, uncertainty-guided depth refinement of the group's final model
(Depth Anything V2 Small + Smooth Correction, see smooth_model.py).

Idea: run the model under small input perturbations; where the predictions disagree (high uncertainty),
replace the depth by an edge-aware (guided-filter) refined version; where they agree, keep the original
depth, so confident structure and anatomical boundaries are preserved.

Ablation (all on top of the group's model, same mask and metrics as the group's final table):
  0 base                    group model, unchanged
  C ensemble mean           mean of the perturbed predictions, no filtering  (control: is any gain just averaging?)
  1 uniform                 Gaussian smoothing everywhere
  2 edge-aware              guided filter (He et al. 2013) everywhere, colour image as guide
  3 uncertainty-guided      guided filter blended in only where perturbation uncertainty is high   <- proposed
  4 ensemble + unc.-guided  variant 3 applied to the perturbation mean (the passes are already needed for
                            the uncertainty); same frozen settings as 3, added after the dev check, before test

Stages (run from the repo root, env endodepth):
  python results/E/refinement/refine.py --stage predict      # perturbed predictions for dev + test (cached)
  python results/E/refinement/refine.py --stage reproduce    # check the base reproduces the group's numbers
  python results/E/refinement/refine.py --stage tune         # choose filter settings on dev only
  python results/E/refinement/refine.py --stage test         # run the frozen settings once on test

Protocol (same as the group notebook): tune with calibration fitted on dev frames 0000-0044 and scored on
0045-0060; test uses the group's final parameters (all 61 dev frames). Test GT is never used for any choice.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[2]))
from smooth_model import DA2Small, apply_smooth, fit_params, load_final_params, load_frame  # noqa: E402
from common import CFG, SEED, sequence_dir  # noqa: E402

CACHE = HERE / "cache"
N_TRAIN = 45                      # group split: dev 0000-0044 train, 0045-0060 validation
EDGE_FRAC = 0.10                  # top 10% GT depth-gradient pixels = "boundary" pixels
PERTURBATIONS = {                 # name -> (input transform, input size, undo flip?)
    "id":      (lambda x: x, 518, False),
    "flip":    (lambda x: x[:, ::-1].copy(), 518, True),
    "bright+": (lambda x: np.clip(x.astype(np.float32) * 1.1, 0, 255).astype(np.uint8), 518, False),
    "bright-": (lambda x: np.clip(x.astype(np.float32) * 0.9, 0, 255).astype(np.uint8), 518, False),
    "s476":    (lambda x: x, 476, False),
    "s560":    (lambda x: x, 560, False),
}


# ------------------------------------------------------------------ data + predictions
def frames(split):
    return sorted(sequence_dir(CFG, split).glob("*_color.png"))


def predict(split):
    """Raw DA-V2 outputs per perturbation, cached. 'id' kept in float32 (= exact group model), others float16."""
    (CACHE / split).mkdir(parents=True, exist_ok=True)
    model, times = None, []
    for i, cp in enumerate(frames(split)):
        for name, (tf, size, unflip) in PERTURBATIONS.items():
            f = CACHE / split / f"{cp.stem.split('_')[0]}_{name}.npy"
            if f.exists():
                continue
            model = model or DA2Small()
            t0 = time.perf_counter()
            p = model(tf(cv2.imread(str(cp))), size)
            times.append(time.perf_counter() - t0)
            np.save(f, (p[:, ::-1] if unflip else p).astype(np.float32 if name == "id" else np.float16))
        print(f"  {split} {cp.stem} ({i + 1}/{len(frames(split))})", end="\r", flush=True)
    if times:
        print(f"\n  {split}: {np.mean(times) * 1000:.0f} ms per forward pass")


def load_preds(split, stem):
    return {n: np.load(CACHE / split / f"{stem}_{n}.npy").astype(np.float32) for n in PERTURBATIONS}


def depth_stack(preds, params):
    """Calibrated depth (mm) for every perturbation; base = the 'id' pass = the group model."""
    return {n: apply_smooth(p, params) for n, p in preds.items()}


def uncertainty(stack):
    d = np.stack(list(stack.values()))
    return (d.std(0) / np.maximum(d.mean(0), 1e-3)).astype(np.float32)   # relative spread across perturbations


# ------------------------------------------------------------------ refinement filters
def fov_mask(bgr):
    return (bgr.max(axis=2) > 10).astype(np.float32)   # image-based (no GT): excludes the black corners


def uniform(D, bgr, sigma):
    f = fov_mask(bgr)                                    # normalised convolution: no bleeding from black corners
    num = cv2.GaussianBlur(D * f, (0, 0), sigma); den = cv2.GaussianBlur(f, (0, 0), sigma)
    return np.where(f > 0, num / np.maximum(den, 1e-6), D)


def edge_aware(D, bgr, r, eps):
    guide = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return cv2.ximgproc.guidedFilter(guide, D.astype(np.float32), r, eps)


def uncertainty_guided(D, bgr, U, r, eps, lo, hi):
    w = np.clip((U - lo) / max(hi - lo, 1e-9), 0.0, 1.0)  # 0 = confident (keep), 1 = uncertain (refine)
    return w * edge_aware(D, bgr, r, eps) + (1 - w) * D


# ------------------------------------------------------------------ metrics
def frame_metrics(pred, gt, raw):
    m = raw > 0
    p, g = np.clip(pred[m], 1e-3, None), gt[m]
    out = {"mae": np.mean(np.abs(g - p)), "absrel": np.mean(np.abs(g - p) / g), "mse": np.mean((g - p) ** 2),
           "delta1": np.mean(np.maximum(g / p, p / g) < 1.25)}
    # boundary vs flat pixels: GT log-depth gradient, computed away from the mask border
    inner = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    lg = np.log(np.maximum(gt, 1e-3))
    grad = np.hypot(cv2.Sobel(lg, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(lg, cv2.CV_32F, 0, 1, ksize=3))
    edge = inner & (grad >= np.quantile(grad[inner], 1 - EDGE_FRAC))
    rel = np.abs(gt - np.clip(pred, 1e-3, None)) / np.maximum(gt, 1e-3)
    out["absrel_edge"], out["absrel_flat"] = rel[edge].mean(), rel[inner & ~edge].mean()
    return out


def aggregate(rows):
    """Same aggregation as the group notebook: frame means, RMSE = sqrt(mean of per-frame MSE)."""
    df = pd.DataFrame(rows)
    g = df.groupby("variant", sort=False)
    out = g[["mae", "absrel", "delta1", "absrel_edge", "absrel_flat"]].mean()
    out.insert(2, "rmse", np.sqrt(g["mse"].mean()))
    return out.reset_index()


def uncertainty_quality(U, base, gt, raw, rng, n=20000):
    """Does uncertainty track error? Spearman rho + sparsification (AbsRel after removing most-uncertain px)."""
    from scipy.stats import spearmanr
    m = raw > 0
    k = rng.choice(np.flatnonzero(m), size=min(n, m.sum()), replace=False)
    u, e = U.ravel()[k], (np.abs(base - gt) / np.maximum(gt, 1e-3)).ravel()[k]
    fr = np.linspace(0, 0.9, 10)
    def curve(order):
        return [e[order[int(f * len(e)):]].mean() for f in fr]
    return {"spearman": spearmanr(u, e)[0], "fractions": fr,
            "by_uncertainty": curve(np.argsort(-u)), "oracle": curve(np.argsort(-e)), "random": [e.mean()] * len(fr)}


# ------------------------------------------------------------------ evaluation of one configuration set
def evaluate(split, stems, params, cfg, thresholds):
    rows, uq = [], []
    rng = np.random.default_rng(SEED)
    for stem in stems:
        bgr, gt, raw = load_frame(sequence_dir(CFG, split) / f"{stem}_color.png")
        stack = depth_stack(load_preds(split, stem), params)
        D, U = stack["id"], uncertainty(stack)
        Dm = np.mean(np.stack(list(stack.values())), 0)
        variants = {
            "0 base (Small + Smooth)": D,
            "C ensemble mean (control)": Dm,
            "1 uniform": uniform(D, bgr, cfg["sigma"]),
            "2 edge-aware": edge_aware(D, bgr, cfg["ea_r"], cfg["ea_eps"]),
            "3 uncertainty-guided (ours)": uncertainty_guided(D, bgr, U, cfg["ug_r"], cfg["ug_eps"],
                                                              thresholds["lo"], thresholds["hi"]),
            "4 ensemble + uncertainty-guided": uncertainty_guided(Dm, bgr, U, cfg["ug_r"], cfg["ug_eps"],
                                                                  thresholds["lo"], thresholds["hi"]),
        }
        for name, pred in variants.items():
            rows.append({"split": split, "frame": stem, "variant": name, **frame_metrics(pred, gt, raw)})
        q = uncertainty_quality(U, D, gt, raw, rng); q["frame"] = stem; uq.append(q)
    return rows, uq


def thresholds_from(split, stems, params, p_lo, p_hi):
    """Uncertainty thresholds = percentiles of U over in-FoV pixels of the given (dev) frames."""
    vals = []
    for stem in stems:
        bgr = cv2.imread(str(sequence_dir(CFG, split) / f"{stem}_color.png"))
        U = uncertainty(depth_stack(load_preds(split, stem), params))
        vals.append(U[fov_mask(bgr) > 0][::50])
    v = np.concatenate(vals)
    return {"lo": float(np.percentile(v, p_lo)), "hi": float(np.percentile(v, p_hi)), "p_lo": p_lo, "p_hi": p_hi}


# ------------------------------------------------------------------ stages
def stage_reproduce():
    dev = [p.stem.split("_")[0] for p in frames("dev")]; test = [p.stem.split("_")[0] for p in frames("test")]
    tr, va = dev[:N_TRAIN], dev[N_TRAIN:]
    p_tr = fit_params([load_preds("dev", s)["id"] for s in tr],
                      [load_frame(sequence_dir(CFG, "dev") / f"{s}_color.png")[2] for s in tr])
    print("train-only params (0000-0044):", {k: np.round(v, 4).tolist() if hasattr(v, "tolist") else round(v, 4)
                                             for k, v in p_tr.items()})
    for split, stems, params, ref in [("dev", va, p_tr, "validation_comparison.csv, row Smooth"),
                                      ("test", test, load_final_params(), "final_test_comparison.csv, row Improved")]:
        rows = []
        for s in stems:
            _, gt, raw = load_frame(sequence_dir(CFG, split) / f"{s}_color.png")
            rows.append({"variant": "base", **frame_metrics(apply_smooth(load_preds(split, s)["id"], params), gt, raw)})
        a = aggregate(rows).iloc[0]
        print(f"{split:4s} {len(stems):3d} frames  MAE {a.mae:.4f}  AbsRel {a.absrel:.4f}  RMSE {a.rmse:.4f}  "
              f"d<1.25 {a.delta1:.4f}   (group: {ref})")


GRID = {"sigma": [2, 4, 8, 16], "r": [4, 8, 16, 32], "eps": [1e-4, 1e-3, 1e-2], "pct": [(50, 90), (70, 95), (85, 99)]}


def stage_tune():
    dev = [p.stem.split("_")[0] for p in frames("dev")]
    tr, va = dev[:N_TRAIN], dev[N_TRAIN:]
    params = fit_params([load_preds("dev", s)["id"] for s in tr],
                        [load_frame(sequence_dir(CFG, "dev") / f"{s}_color.png")[2] for s in tr])
    cache = {}
    for s in va:
        bgr, gt, raw = load_frame(sequence_dir(CFG, "dev") / f"{s}_color.png")
        stack = depth_stack(load_preds("dev", s), params)
        cache[s] = (bgr, gt, raw, stack["id"], uncertainty(stack))
    score = lambda fn: float(np.mean([frame_metrics(fn(*cache[s]), cache[s][1], cache[s][2])["absrel"] for s in va]))
    log, best = [], {}
    b = score(lambda bgr, gt, raw, D, U: D); log.append({"variant": "base", "absrel": b})
    res = {sg: score(lambda bgr, gt, raw, D, U: uniform(D, bgr, sg)) for sg in GRID["sigma"]}
    best["sigma"] = min(res, key=res.get); log += [{"variant": "uniform", "sigma": k, "absrel": v} for k, v in res.items()]
    res = {(r, e): score(lambda bgr, gt, raw, D, U: edge_aware(D, bgr, r, e))
           for r, e in itertools.product(GRID["r"], GRID["eps"])}
    best["ea_r"], best["ea_eps"] = min(res, key=res.get)
    log += [{"variant": "edge-aware", "r": k[0], "eps": k[1], "absrel": v} for k, v in res.items()]
    res = {}
    for pl, ph in GRID["pct"]:
        th = thresholds_from("dev", tr, params, pl, ph)
        for r, e in itertools.product(GRID["r"], GRID["eps"]):
            res[(pl, ph, r, e)] = score(lambda bgr, gt, raw, D, U: uncertainty_guided(D, bgr, U, r, e, th["lo"], th["hi"]))
    pl, ph, best["ug_r"], best["ug_eps"] = min(res, key=res.get); best["pct"] = [pl, ph]
    log += [{"variant": "uncertainty-guided", "p_lo": k[0], "p_hi": k[1], "r": k[2], "eps": k[3], "absrel": v}
            for k, v in res.items()]
    pd.DataFrame(log).to_csv(HERE / "tuning_dev_val.csv", index=False)
    json.dump({"chosen_on": "dev 0045-0060, calibration fitted on 0000-0044, criterion AbsRel", "seed": SEED,
               "perturbations": list(PERTURBATIONS), **best}, open(HERE / "chosen_settings.json", "w"), indent=2)
    print("chosen settings:", best)
    th = thresholds_from("dev", tr, params, *best["pct"])
    rows, uq = evaluate("dev", va, params, best, th)
    report(rows, uq, "dev_val")


def stage_test():
    best = json.load(open(HERE / "chosen_settings.json"))
    params = load_final_params()
    dev = [p.stem.split("_")[0] for p in frames("dev")]
    th = thresholds_from("dev", dev, params, *best["pct"])          # thresholds from dev only (all 61, final params)
    json.dump({**best, "test_thresholds": th}, open(HERE / "chosen_settings.json", "w"), indent=2)
    rows, uq = evaluate("test", [p.stem.split("_")[0] for p in frames("test")], params, best, th)
    report(rows, uq, "test")


def report(rows, uq, tag):
    pd.DataFrame(rows).to_csv(HERE / f"ablation_{tag}_per_frame.csv", index=False)
    summ = aggregate(rows); summ.to_csv(HERE / f"ablation_{tag}_summary.csv", index=False)
    pd.set_option("display.width", 160)
    print(f"\n[{tag}] ablation (mask gt_raw>0; RMSE = sqrt(mean per-frame MSE))")
    print(summ.round(4).to_string(index=False))
    fr = uq[0]["fractions"]
    curves = {k: np.mean([q[k] for q in uq], 0) for k in ("by_uncertainty", "oracle", "random")}
    ause = float(np.trapezoid(curves["by_uncertainty"] - curves["oracle"], fr))
    aurg = float(np.trapezoid(curves["random"] - curves["by_uncertainty"], fr))
    rho = float(np.mean([q["spearman"] for q in uq]))
    pd.DataFrame({"fraction_removed": fr, **curves}).to_csv(HERE / f"sparsification_{tag}.csv", index=False)
    json.dump({"spearman_uncertainty_vs_error": rho, "AUSE": ause, "AURG": aurg},
              open(HERE / f"uncertainty_quality_{tag}.json", "w"), indent=2)
    print(f"uncertainty vs error: Spearman {rho:.3f} | AUSE {ause:.4f} (0 = perfect) | AURG {aurg:.4f} (>0 = better than random)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["predict", "reproduce", "tune", "test"])
    a = ap.parse_args()
    if a.stage == "predict":
        predict("dev"); predict("test")
    else:
        {"reproduce": stage_reproduce, "tune": stage_tune, "test": stage_test}[a.stage]()
