"""Uncertainty-guided depth refinement, one of the group's comparison streams, on the base model
(Depth Anything V2 Small + Smooth Correction at input size 518, see smooth_model.py).

Idea: run the model under small input perturbations; where the predictions disagree (high uncertainty),
replace the depth by an edge-aware (guided-filter) refined version; where they agree, keep the original
depth, so confident structure and anatomical boundaries are preserved.

Ablation (all on top of the base model):
  base                           the base model, unchanged
  Uniform smoothing              Gaussian smoothing everywhere
  Edge-aware (guided filter)     guided filter (He et al. 2013) everywhere, colour image as guide
  Uncertainty-guided             guided filter blended in only where perturbation uncertainty is high
  Ensemble mean (control)        mean of the perturbed predictions, no filtering (is any gain just averaging?)
  Ensemble + uncertainty-guided  the uncertainty-guided filter applied to the perturbation mean

Stages (run from the repo root, env `endodepth`):
  python results/E/refinement/refine.py --stage predict      # perturbed predictions for dev + test (cached)
  python results/E/refinement/refine.py --stage reproduce    # check: the notebook's saved parameters give its numbers
  python results/E/refinement/refine.py --stage tune         # choose filter settings by 4-block CV on dev
  python results/E/refinement/refine.py --stage test         # frozen settings on test; refuses to run a second time

Group protocol (common.py and README.md): mask A for the main table, mask B for the second table; metrics per frame, mean
over frames, RMSE = sqrt(mean per-frame MSE). Under each mask the base is fitted and the settings are chosen by 4-block CV
with a 2-frame gap, criterion AbsRel; test uses fits on all 61 dev frames. Test GT is never used for any choice.
"""
import argparse
import itertools
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[2]))
from smooth_model import DA2Small, apply_smooth, fit_base, load_final_params  # noqa: E402
from common import CFG, DEVICE, SEED, aggregate, compute_metrics, dev_folds, list_frames, load_frame, save_run_info  # noqa: E402

CACHE = HERE / "cache"
DONE = HERE / "refinement_test_done.json"
SETTINGS = HERE / "chosen_settings.json"
MASKS = ("A", "B")
BASE = "Small + Smooth (base)"
EDGE_FRAC = 0.10                  # top 10% GT depth-gradient pixels = "boundary" pixels
PERTURBATIONS = {                 # name -> (input transform, input size, undo flip?)
    "id":      (lambda x: x, 518, False),
    "flip":    (lambda x: x[:, ::-1].copy(), 518, True),
    "bright+": (lambda x: np.clip(x.astype(np.float32) * 1.1, 0, 255).astype(np.uint8), 518, False),
    "bright-": (lambda x: np.clip(x.astype(np.float32) * 0.9, 0, 255).astype(np.uint8), 518, False),
    "s476":    (lambda x: x, 476, False),
    "s560":    (lambda x: x, 560, False),
}
GRID = {"sigma": [2, 4, 8, 16], "r": [4, 8, 16, 32], "eps": [1e-4, 1e-3, 1e-2], "pct": [(50, 90), (70, 95), (85, 99)]}


# ------------------------------------------------------------------ data + predictions
def stem_of(cp):
    return cp.stem.split("_")[0]


def predict(split):
    """Raw DA-V2 outputs per perturbation, cached. 'id' kept in float32 (= exact base model), others float16."""
    (CACHE / split).mkdir(parents=True, exist_ok=True)
    model, times = None, []
    paths = list_frames(CFG, split)
    for i, cp in enumerate(paths):
        for name, (tf, size, unflip) in PERTURBATIONS.items():
            f = CACHE / split / f"{stem_of(cp)}_{name}.npy"
            if f.exists():
                continue
            model = model or DA2Small()
            t0 = time.perf_counter()
            p = model(tf(cv2.imread(str(cp))), size)
            times.append(time.perf_counter() - t0)
            np.save(f, (p[:, ::-1] if unflip else p).astype(np.float32 if name == "id" else np.float16))
        print(f"  {split} {cp.stem} ({i + 1}/{len(paths)})", end="\r", flush=True)
    if times:
        print(f"\n  {split}: {np.mean(times) * 1000:.0f} ms per forward pass")


def load_preds(split, stem):
    return {n: np.load(CACHE / split / f"{stem}_{n}.npy").astype(np.float32) for n in PERTURBATIONS}


def load_split(split, mask):
    """Paths, stems, ground truth and valid masks of one split under one mask rule."""
    paths = list_frames(CFG, split)
    frames = [load_frame(cp, mask)[1:] for cp in paths]
    return paths, [stem_of(cp) for cp in paths], [f[0] for f in frames], [f[1] for f in frames]


def depth_stack(preds, params):
    """Calibrated depth (mm) for every perturbation; base = the 'id' pass = the base model."""
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


def variants(bgr, D, Dm, U, cfg, th):
    return {BASE: D,
            "Uniform smoothing": uniform(D, bgr, cfg["sigma"]),
            "Edge-aware (guided filter)": edge_aware(D, bgr, cfg["ea_r"], cfg["ea_eps"]),
            "Uncertainty-guided": uncertainty_guided(D, bgr, U, cfg["ug_r"], cfg["ug_eps"], th["lo"], th["hi"]),
            "Ensemble mean (control)": Dm,
            "Ensemble + uncertainty-guided": uncertainty_guided(Dm, bgr, U, cfg["ug_r"], cfg["ug_eps"], th["lo"], th["hi"])}


# ------------------------------------------------------------------ metrics
def absrel(pred, gt, valid):
    p, g = np.clip(pred[valid], 1e-3, None).astype(np.float64), gt[valid].astype(np.float64)
    return float(np.mean(np.abs(p - g) / g))


def frame_metrics(pred, gt, valid):
    """common.compute_metrics, plus AbsRel on boundary and flat pixels: GT log-depth gradient, away from the mask border."""
    out = compute_metrics(pred, gt, valid)
    inner = cv2.erode(valid.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    lg = np.log(np.maximum(gt, 1e-3))
    grad = np.hypot(cv2.Sobel(lg, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(lg, cv2.CV_32F, 0, 1, ksize=3))
    edge = inner & (grad >= np.quantile(grad[inner], 1 - EDGE_FRAC))
    rel = np.abs(gt - np.clip(pred, 1e-3, None)) / np.maximum(gt, 1e-3)
    out["absrel_edge"], out["absrel_flat"] = float(rel[edge].mean()), float(rel[inner & ~edge].mean())
    return out


def summarise(per_frame):
    rows = []
    for (sp, mk, v), g in per_frame.groupby(["split", "mask", "variant"], sort=False):
        rows.append({"split": sp, "mask": mk, "variant": v, **aggregate(g), "absrel_edge": g.absrel_edge.mean(),
                     "absrel_flat": g.absrel_flat.mean()})
    return pd.DataFrame(rows)


def uncertainty_quality(U, base, gt, valid, rng, n=20000):
    """Does uncertainty track error? Spearman rho + sparsification (AbsRel after removing most-uncertain px)."""
    from scipy.stats import spearmanr
    k = rng.choice(np.flatnonzero(valid), size=min(n, valid.sum()), replace=False)
    u, e = U.ravel()[k], (np.abs(base - gt) / np.maximum(gt, 1e-3)).ravel()[k]
    fr = np.linspace(0, 0.9, 10)

    def curve(order):
        return [e[order[int(f * len(e)):]].mean() for f in fr]
    return {"spearman": spearmanr(u, e)[0], "fractions": fr,
            "by_uncertainty": curve(np.argsort(-u)), "oracle": curve(np.argsort(-e)), "random": [e.mean()] * len(fr)}


def quality_table(uq, split, mask):
    fr = uq[0]["fractions"]
    curves = {k: np.mean([q[k] for q in uq], 0) for k in ("by_uncertainty", "oracle", "random")}
    summary = {"spearman_uncertainty_vs_error": float(np.mean([q["spearman"] for q in uq])),
               "AUSE": float(np.trapezoid(curves["by_uncertainty"] - curves["oracle"], fr)),
               "AURG": float(np.trapezoid(curves["random"] - curves["by_uncertainty"], fr))}
    return summary, pd.DataFrame({"split": split, "mask": mask, "fraction_removed": fr, **curves})


# ------------------------------------------------------------------ thresholds
def thresholds_from(paths, params, pcts):
    """Uncertainty thresholds = percentiles of U over in-FoV pixels of the given development frames."""
    vals = []
    for cp in paths:
        U = uncertainty(depth_stack(load_preds("dev", stem_of(cp)), params))
        vals.append(U[fov_mask(cv2.imread(str(cp))) > 0][::50])
    v = np.concatenate(vals)
    return {pc: {"lo": float(np.percentile(v, pc[0])), "hi": float(np.percentile(v, pc[1])), "p_lo": pc[0], "p_hi": pc[1]} for pc in pcts}


# ------------------------------------------------------------------ stages
def stage_reproduce():
    """The notebook's saved parameters, fitted on raw > 0. Mask B must give the notebook's test row (AbsRel 0.2825)."""
    params = load_final_params()
    for mk in MASKS:
        _, stems, gts, valids = load_split("test", mk)
        pf = pd.DataFrame([compute_metrics(apply_smooth(load_preds("test", s)["id"], params), g, v) for s, g, v in zip(stems, gts, valids)])
        a = aggregate(pf)
        print(f"saved parameters, test, mask {mk}: MAE {a['mae']:.4f}  AbsRel {a['absrel']:.4f}  RMSE {a['rmse']:.4f}  d<1.25 {a['delta1']:.4f}")


def tune_mask(mask):
    paths, stems, gts, valids = load_split("dev", mask)
    ids = [load_preds("dev", s)["id"] for s in stems]
    pcts = [tuple(p) for p in GRID["pct"]]
    cache, fold_th = {}, {}
    for k, (fit, held) in enumerate(dev_folds(len(paths))):
        params = fit_base([ids[i] for i in fit], [gts[i] for i in fit], [valids[i] for i in fit])
        fold_th[k] = thresholds_from([paths[i] for i in fit], params, pcts)
        for i in held:
            st = depth_stack(load_preds("dev", stems[i]), params)
            cache[i] = dict(k=k, bgr=cv2.imread(str(paths[i])), D=st["id"], U=uncertainty(st), Dm=np.mean(np.stack(list(st.values())), 0))
    print(f"  mask {mask}: stacks ready", flush=True)

    def score(fn):
        return float(np.mean([absrel(fn(c), gts[i], valids[i]) for i, c in cache.items()]))
    log = [{"variant": "base", "absrel": score(lambda c: c["D"])}]
    res = {s: score(lambda c: uniform(c["D"], c["bgr"], s)) for s in GRID["sigma"]}
    best = {"sigma": min(res, key=res.get)}
    log += [{"variant": "uniform", "sigma": k, "absrel": v} for k, v in res.items()]
    res = {(r, e): score(lambda c: edge_aware(c["D"], c["bgr"], r, e)) for r, e in itertools.product(GRID["r"], GRID["eps"])}
    best["ea_r"], best["ea_eps"] = min(res, key=res.get)
    log += [{"variant": "edge-aware", "r": k[0], "eps": k[1], "absrel": v} for k, v in res.items()]
    res = {}
    for pc in pcts:
        for r, e in itertools.product(GRID["r"], GRID["eps"]):
            res[(*pc, r, e)] = score(lambda c: uncertainty_guided(c["D"], c["bgr"], c["U"], r, e, fold_th[c["k"]][pc]["lo"], fold_th[c["k"]][pc]["hi"]))
    pl, ph, best["ug_r"], best["ug_eps"] = min(res, key=res.get)
    best["pct"] = [pl, ph]
    log += [{"variant": "uncertainty-guided", "p_lo": k[0], "p_hi": k[1], "r": k[2], "eps": k[3], "absrel": v} for k, v in res.items()]
    print(f"  mask {mask}: chosen", best, flush=True)
    rows, uq = [], []
    rng = np.random.default_rng(SEED)
    for i, c in cache.items():
        th = fold_th[c["k"]][(pl, ph)]
        for name, pred in variants(c["bgr"], c["D"], c["Dm"], c["U"], best, th).items():
            rows.append({"split": "dev", "mask": mask, "fold": c["k"] + 1, "frame": stems[i], "variant": name, **frame_metrics(pred, gts[i], valids[i])})
        uq.append(uncertainty_quality(c["U"], c["D"], gts[i], valids[i], rng))
    return rows, uq, best, pd.DataFrame(log).assign(mask=mask)


def stage_tune():
    rows, logs, settings, quality, curves = [], [], {}, {}, []
    for mk in MASKS:
        r, uq, best, log = tune_mask(mk)
        rows += r; logs.append(log)
        settings[mk] = {"chosen_on": "dev 4-block CV, 2-frame gap, criterion AbsRel", "seed": SEED, "perturbations": list(PERTURBATIONS), **best}
        quality[mk], c = quality_table(uq, "dev", mk)
        curves.append(c)
    pd.concat(logs).to_csv(HERE / "tuning_dev_cv.csv", index=False)
    json.dump(settings, open(SETTINGS, "w"), indent=2)
    report(rows, quality, curves, "dev_cv")


def stage_test():
    if DONE.exists():
        sys.exit(f"{DONE.name} exists: the test set is scored once. Delete it only if the protocol changes.")
    settings = json.load(open(SETTINGS))
    rows, quality, curves = [], {}, []
    for mk in MASKS:
        d_paths, d_stems, d_gts, d_valids = load_split("dev", mk)
        params = fit_base([load_preds("dev", s)["id"] for s in d_stems], d_gts, d_valids)
        cfg = settings[mk]
        pc = tuple(cfg["pct"])
        th = thresholds_from(d_paths, params, [pc])[pc]             # thresholds from all 61 dev frames only
        cfg["test_thresholds"] = th
        cfg["base_params_all_dev"] = {"scale": params["scale"], "offset": params["offset"], "centers": params["centers"].tolist(), "corrections": params["corrections"].tolist()}
        paths, stems, gts, valids = load_split("test", mk)
        uq, rng = [], np.random.default_rng(SEED)
        for cp, s, g, v in zip(paths, stems, gts, valids):
            bgr = cv2.imread(str(cp))
            st = depth_stack(load_preds("test", s), params)
            D, U, Dm = st["id"], uncertainty(st), np.mean(np.stack(list(st.values())), 0)
            for name, pred in variants(bgr, D, Dm, U, cfg, th).items():
                rows.append({"split": "test", "mask": mk, "fold": 0, "frame": s, "variant": name, **frame_metrics(pred, g, v)})
            uq.append(uncertainty_quality(U, D, g, v, rng))
        quality[mk], c = quality_table(uq, "test", mk)
        curves.append(c)
        print(f"  mask {mk}: test done", flush=True)
    json.dump(settings, open(SETTINGS, "w"), indent=2)
    report(rows, quality, curves, "test")
    save_run_info(HERE, DEVICE, SEED, extra={"stream": "uncertainty-guided refinement", "base": "DAv2 Small + Smooth, input 518, fitted under each mask",
                                             "masks": list(MASKS), "n_blocks": 4, "gap": 2})
    t = pd.read_csv(HERE / "ablation_test_summary.csv")
    json.dump({"date": datetime.now().isoformat(timespec="seconds"), "n_frames": 103,
               "absrel_mask_A": {r.variant: r.absrel for r in t[t["mask"] == "A"].itertuples()}}, open(DONE, "w"), indent=2)


def report(rows, quality, curves, tag):
    pf = pd.DataFrame(rows)
    pf.to_csv(HERE / f"ablation_{tag}_per_frame.csv", index=False)
    summ = summarise(pf)
    summ.to_csv(HERE / f"ablation_{tag}_summary.csv", index=False)
    pd.concat(curves).to_csv(HERE / f"sparsification_{tag}.csv", index=False)
    json.dump(quality, open(HERE / f"uncertainty_quality_{tag}.json", "w"), indent=2)
    pd.set_option("display.width", 170)
    print(f"\n[{tag}] ablation (frame means; RMSE = sqrt(mean per-frame MSE))")
    print(summ.drop(columns="split").round(4).to_string(index=False))
    print("uncertainty vs error:", json.dumps(quality))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["predict", "reproduce", "tune", "test"])
    a = ap.parse_args()
    if a.stage == "predict":
        predict("dev"); predict("test")
    else:
        {"reproduce": stage_reproduce, "tune": stage_tune, "test": stage_test}[a.stage]()
