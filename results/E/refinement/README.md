# Uncertainty-guided depth refinement

This is one of the group's comparison streams, reported against the same base as the group's other improvements. It is not the group's proposed method.

**Built on the base model**, Depth Anything V2 Small + Smooth Depth-Dependent Bias Correction, using the official code and ViT-S weights with `infer_image(input_size=518)`. Under each mask the base is fitted on development data with `smooth_model.fit_base`.

## With and without the improvement (test, mask A)

`trans_t2_b`, 103 frames. The CI is a paired moving-block bootstrap (10-frame blocks, 4000 resamples).

| Variant | AbsRel ↓ | RMSE (mm) ↓ | MAE (mm) ↓ | δ<1.25 ↑ | AbsRel boundary ↓ | AbsRel flat ↓ | Change in AbsRel (95% CI) | Frames better |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Small + Smooth (base) | 0.2805 | 5.60 | 4.41 | 0.535 | 0.4408 | 0.2601 | – | – |
| Uniform smoothing | 0.2793 | 5.62 | 4.42 | 0.534 | 0.4390 | 0.2592 | −0.0012 (−0.0024 to −0.0003) | 68/103 |
| Edge-aware (guided filter) | 0.2805 | 5.60 | 4.41 | 0.535 | 0.4409 | 0.2601 | 0.0000 (−0.0001 to 0.0000) | 53/103 |
| Uncertainty-guided | 0.2775 | 5.56 | 4.38 | 0.536 | 0.4341 | 0.2576 | −0.0030 (−0.0035 to −0.0025) | 99/103 |
| Ensemble mean (control) | 0.2586 | 5.25 | 4.06 | 0.580 | 0.4209 | 0.2379 | −0.0220 (−0.0281 to −0.0169) | 94/103 |
| Ensemble + uncertainty-guided | 0.2559 | 5.22 | 4.04 | 0.582 | 0.4152 | 0.2356 | −0.0247 (−0.0308 to −0.0194) | 97/103 |

Most of the gain comes from averaging the six perturbed predictions (see the control row). The refinement adds a little on top.

## Test results, mask B (second table)

| Variant | AbsRel ↓ | RMSE (mm) ↓ | MAE (mm) ↓ | δ<1.25 ↑ | AbsRel boundary ↓ | AbsRel flat ↓ | Change in AbsRel (95% CI) | Frames better |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Small + Smooth (base) | 0.2825 | 5.87 | 4.65 | 0.544 | 0.4388 | 0.2624 | – | – |
| Uniform smoothing | 0.2814 | 5.93 | 4.67 | 0.542 | 0.4369 | 0.2617 | −0.0012 (−0.0023 to −0.0003) | 71/103 |
| Edge-aware (guided filter) | 0.2808 | 5.98 | 4.68 | 0.541 | 0.4372 | 0.2607 | −0.0017 (−0.0027 to −0.0008) | 93/103 |
| Uncertainty-guided | 0.2793 | 5.84 | 4.62 | 0.546 | 0.4338 | 0.2594 | −0.0032 (−0.0038 to −0.0026) | 101/103 |
| Ensemble mean (control) | 0.2608 | 5.56 | 4.31 | 0.588 | 0.4198 | 0.2404 | −0.0218 (−0.0277 to −0.0172) | 96/103 |
| Ensemble + uncertainty-guided | 0.2580 | 5.53 | 4.28 | 0.590 | 0.4161 | 0.2377 | −0.0245 (−0.0305 to −0.0198) | 100/103 |

## Dev results, 4-block CV, mask A

| Variant | AbsRel ↓ | RMSE (mm) ↓ | MAE (mm) ↓ | δ<1.25 ↑ | AbsRel boundary ↓ | AbsRel flat ↓ | Change in AbsRel (95% CI) | Frames better |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Small + Smooth (base) | 0.2783 | 6.07 | 4.69 | 0.510 | 0.4162 | 0.2607 | – | – |
| Uniform smoothing | 0.2776 | 6.08 | 4.69 | 0.511 | 0.4158 | 0.2604 | −0.0006 (−0.0009 to −0.0002) | 49/61 |
| Edge-aware (guided filter) | 0.2783 | 6.07 | 4.69 | 0.510 | 0.4164 | 0.2607 | 0.0000 (0.0000 to +0.0001) | 19/61 |
| Uncertainty-guided | 0.2764 | 6.06 | 4.67 | 0.512 | 0.4127 | 0.2590 | −0.0019 (−0.0028 to −0.0008) | 50/61 |
| Ensemble mean (control) | 0.2552 | 5.60 | 4.28 | 0.552 | 0.3922 | 0.2377 | −0.0231 (−0.0277 to −0.0204) | 60/61 |
| Ensemble + uncertainty-guided | 0.2537 | 5.59 | 4.26 | 0.555 | 0.3901 | 0.2362 | −0.0246 (−0.0294 to −0.0216) | 60/61 |

## Method

1. **Perturbation uncertainty.** The base model runs on 6 versions of each frame: original, horizontal flip, brightness ×1.1, brightness ×0.9, input size 476 and input size 560. Each output gets the base's linear + smooth calibration. Uncertainty per pixel = std / mean of the 6 depths.
2. **Edge-aware refinement.** A guided filter (He et al., 2013) smooths depth using the grey image as guide, so smoothing stops at visible tissue edges such as folds.
3. **Uncertainty-guided blending.** refined = w · guided(D) + (1 − w) · D, with w rising linearly from 0 to 1 between two percentiles of development uncertainty. Confident pixels are left unchanged; only uncertain regions are refined.
4. **Ensemble + uncertainty-guided.** The same blending applied to the mean of the 6 calibrated predictions, which are already computed for the uncertainty.

## Protocol

This follows the group protocol in `common.py` and the main README.

- **Masks.** Mask A (main table): 0 < raw < 65535 and RGB max > 10. Mask B (second table): raw > 0. The whole pipeline, including tuning, runs under each mask.
- **Metrics.** Per frame, then the mean over frames. RMSE = √(mean per-frame MSE).
- **Tuning.** Development data only: 4-block CV with a 2-frame gap, criterion AbsRel. In each fold the base and the uncertainty thresholds come from the fit frames only. Grid: uniform σ ∈ {2, 4, 8, 16}; guided-filter r ∈ {4, 8, 16, 32}, ε ∈ {1e-4, 1e-3, 1e-2}; thresholds p50–p90, p70–p95 or p85–p99.
- **Chosen settings, mask A:** uniform σ = 16; edge-aware r = 4, ε = 0.001; uncertainty-guided r = 32, ε = 0.01, thresholds p70–p95.
- **Chosen settings, mask B:** uniform σ = 16; edge-aware r = 32, ε = 0.01; uncertainty-guided r = 32, ε = 0.01, thresholds p50–p90.
- **Test.** Base and thresholds use all 61 dev frames, and the test run refuses to run a second time once `refinement_test_done.json` exists.
- **Boundary pixels.** The top 10% of ground-truth log-depth gradient on the eroded mask. These are used for evaluation only.

## Does the uncertainty track the error?

| Mask A | Spearman ρ (uncertainty vs AbsRel) | AUSE ↓ | AURG ↑ (> 0 = better than random) |
|---|---:|---:|---:|
| Dev, 4-block CV | 0.40 | 0.080 | 0.057 |
| Test | 0.45 | 0.085 | 0.060 |

See `sparsification.png`. On test, removing the 40% most uncertain pixels lowers AbsRel from 0.280 to 0.195. Beyond about 60%, the remaining "confident" pixels have higher error: the model is consistently wrong in places where every perturbation agrees.

## Findings

1. **Among the filters, only uncertainty-guided refinement improves every metric.** Uniform smoothing lowers AbsRel a little but worsens MAE, RMSE and δ<1.25. A global edge-aware filter changes almost nothing. Restricting refinement to uncertain regions gives a small but consistent gain (AbsRel −1.1%, boundary AbsRel −1.5%).
2. **The gain from filtering alone is small because the error is mostly smooth and large-scale.** Whole regions are offset, rather than locally noisy, and filtering cannot fix region-wide bias.
3. **The perturbation passes are worth more as an ensemble.** Averaging them gives most of the improvement (AbsRel −7.8%), and uncertainty-guided refinement on top adds a further gain (AbsRel −8.8%, δ<1.25 +0.047, boundary AbsRel −5.8%).
4. **The uncertainty is a useful but imperfect reliability signal.** It tracks error only moderately (ρ = 0.45 on test) and misses systematic errors where all perturbations agree. It indicates variance, not bias.
5. **Cost.** Six forward passes at about 84 ms each take about 0.5 s per frame on the Apple M4 Pro, against 64 ms for the base. A smaller perturbation set would trade accuracy for speed.

## Base values

`python results/E/refinement/refine.py --stage reproduce` applies the group notebook's saved parameters from `results/DAv2/final_model_parameters.json`, which were fitted on raw > 0. They give AbsRel 0.2825 under mask B, the notebook's own number, and 0.2853 under mask A. The tables here fit the base under each mask, which gives 0.2805 under mask A.

## Files

| File | Contents |
|---|---|
| `smooth_model.py` | The base model (official DA-V2 code + linear and smooth calibration) |
| `refine.py` | Perturbations, uncertainty, filters, CV tuning, test run, metrics (`--stage predict/reproduce/tune/test`) |
| `improvement_table.py` | With-and-without tables: `improvement_vs_base_{dev_cv,test}.csv` |
| `figures.py` | `sparsification.png`, `examples_test.png` (mask A) |
| `ablation_{dev_cv,test}_summary.csv`, `_per_frame.csv` | Results per split, mask and variant, and per-frame scores |
| `tuning_dev_cv.csv`, `chosen_settings.json` | Every setting tried on dev, and the frozen choice per mask |
| `uncertainty_quality_{dev_cv,test}.json`, `sparsification_{dev_cv,test}.csv` | Uncertainty-against-error statistics per mask |
| `refinement_test_done.json`, `run_info.json` | Record of the single test run, and versions and seed |
| `cache/` | Saved predictions (about 3 GB, ignored by git) |

Setup: official code `git clone https://github.com/DepthAnything/Depth-Anything-V2 ~/Downloads/Depth-Anything-V2` and weights `depth_anything_v2_vits.pth` in its `checkpoints/`, both outside this repo. Set `DA2_ROOT` or `da2_root` in `config.yaml` if you put them elsewhere. Environment: `c3vd-rerun`, the group's `endodepth` plus scikit-learn and timm 0.6.13; seed 42.
