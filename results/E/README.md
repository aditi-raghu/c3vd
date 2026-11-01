# Depth Pro (zero-shot) on C3VD

**Model:** `apple/DepthPro-hf` (Bochkovskii et al., *Depth Pro: Sharp Monocular Metric Depth in Less Than a Second*, 2024), used as released: no training or fine-tuning, no weights changed.
**Input:** raw C3VD frames; only Depth Pro's own preprocessing (resize to 1536×1536, normalise). Depth is returned at 1080×1350 and scored at that size.
**Data:** dev = `trans_t1_a` (61 frames), test = `trans_t2_b` (103 frames).
**Environment:** `c3vd-rerun`, the group's `endodepth` plus scikit-learn and timm 0.6.13 (Python 3.11.17, torch 2.13.0, transformers 5.20.2), Apple M4 Pro (MPS), seed 42. Details are in `run_info_dev.json` and `run_info_test.json`.
**Speed:** about 4.4 to 4.9 s per frame for the whole prediction. This run measured 4.85 s, and the group report's timed run measured 4.36 s.

Run from the repo root: `python results/E/run_depthpro.py --limit 2`, then `--split dev`, then `--split test`. The test run refuses to run a second time once `depthpro_test_done.json` exists. `DepthPro.ipynb` shows the saved tables.

## Protocol

This follows the group protocol in `common.py` and the main README.

- **Masks.** Mask A (main table): 0 < raw < 65535 and RGB max > 10. Mask B (second table): raw > 0.
- **Metrics.** Per frame, then the mean over frames. RMSE = √(mean per-frame MSE).
- **Fitting.** Development data only. Dev scores come from 4-block CV with a 2-frame gap. Test uses fits on all 61 dev frames and runs once.
- **Clipping.** Fitted and oracle predictions are clipped to 0.001–100 mm. The raw metric row is not clipped, to show the raw failure.
- **Oracle rows.** These use the ground truth of the scored frame and are references only, never a selectable method.

## Methods

| Method | What it does | Uses GT of the scored frames? |
|---|---|---|
| Constant median | The median depth of the dev frames, 17.26 mm under mask A | No |
| Raw metric | Depth Pro output in mm, nothing fitted | No |
| Dev scale | × one factor = median(GT/pred) over dev pixels (×0.066 under mask A) | No |
| Linear | a·pred + b, least squares over dev pixels | No |
| Smooth | Linear, plus the mean signed error per 20 mm bin of linear depth (bins with at least 2000 pixels) | No |
| Oracle median | × per-frame median(GT)/median(pred) | Yes |
| Oracle LSQ | per-frame least-squares a·pred + b | Yes |

## Test results, mask A (main table)

`trans_t2_b`, 103 frames.

| Method | AbsRel ↓ | RMSE (mm) ↓ | MAE (mm) ↓ | δ<1.25 ↑ |
|---|---:|---:|---:|---:|
| Constant median | 0.550 | 17.71 | 11.45 | 0.275 |
| Raw metric | 19.624 | 318.86 | 311.65 | 0.000 |
| Dev scale | 0.586 | 13.53 | 9.69 | 0.308 |
| Linear | 0.875 | 14.33 | 11.73 | 0.269 |
| Smooth | 0.826 | 14.08 | 11.28 | 0.286 |
| Oracle median | 0.468 | 14.24 | 9.32 | 0.346 |
| Oracle LSQ | 0.251 | 4.93 | 3.75 | 0.637 |

## Test results, mask B (second table)

| Method | AbsRel ↓ | RMSE (mm) ↓ | MAE (mm) ↓ | δ<1.25 ↑ |
|---|---:|---:|---:|---:|
| Constant median | 0.565 | 21.21 | 12.98 | 0.270 |
| Raw metric | 19.305 | 323.20 | 315.12 | 0.000 |
| Dev scale | 0.601 | 16.11 | 10.87 | 0.298 |
| Linear | 0.933 | 15.95 | 13.04 | 0.249 |
| Smooth | 0.868 | 15.81 | 12.47 | 0.270 |
| Oracle median | 0.480 | 17.10 | 10.53 | 0.341 |
| Oracle LSQ | 0.253 | 5.21 | 3.95 | 0.630 |

## Dev results, 4-block CV, mask A

| Method | AbsRel ↓ | RMSE (mm) ↓ | MAE (mm) ↓ | δ<1.25 ↑ |
|---|---:|---:|---:|---:|
| Constant median | 0.605 | 18.31 | 12.06 | 0.252 |
| Raw metric | 17.108 | 270.87 | 263.09 | 0.000 |
| Dev scale | 0.553 | 14.73 | 10.24 | 0.263 |
| Linear | 0.671 | 13.57 | 10.66 | 0.246 |
| Smooth | 0.667 | 13.67 | 10.64 | 0.253 |
| Oracle median | 0.511 | 14.41 | 9.63 | 0.315 |
| Oracle LSQ | 0.301 | 5.21 | 4.16 | 0.574 |

Mask B dev results are in `depthpro_dev_summary.csv`.

## Findings

1. **Raw metric output fails completely.** On test under mask A, Depth Pro predicts the colon about 15 times too far (mean 335 mm against 23 mm), and no pixel falls within 25% of the ground truth. Its own per-frame estimate of the field of view averages about 51°, far narrower than a wide-angle colonoscope.
2. **Dev-fitted scaling does not beat the constant baseline on AbsRel.** Dev scale improves RMSE, MAE and δ<1.25 but has a worse AbsRel (0.586 against 0.550). Linear and Smooth transfer worse still (0.875 and 0.826), because a shift learnt on one texture does not suit the other.
3. **The shape is good; scale and range are the problem.** With a per-frame scale and shift (oracle), Depth Pro reaches AbsRel 0.251 and δ<1.25 of 0.637, so its relative structure of the tunnel is largely correct. Its predicted near-to-far range is compressed (95th to 5th percentile ratio of about 1.7, against about 8.8 in the ground truth).
4. **It is too slow for the task.** At about 4.4 to 4.9 s per frame it is roughly 70 times slower than Depth Anything V2 Small (64 ms). Together with point 2, this supports the group's choice of Depth Anything V2.

The group report compares every model under the same rules.

## Files

| File | Contents |
|---|---|
| `run_depthpro.py` | Inference, fits, scoring and the test-once check |
| `depthpro_{dev,test}_summary.csv` | Results per split, mask and method |
| `depthpro_{dev,test}_per_frame.csv` | Per-frame scores |
| `DepthPro_summary.csv` | The same results in the group's summary format |
| `depthpro_test_done.json` | Record of the single test run |
| `run_info_{dev,test}.json` | Versions, seed, fitted constants, timing and field of view |
| `DepthPro.ipynb` | Notebook view of the saved tables |
| `cache/` | Saved predictions (about 0.9 GB, ignored by git) |
| `refinement/` | Uncertainty-guided refinement, a separate comparison stream |
