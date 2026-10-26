# Stream E: Depth Pro (zero-shot) on C3VD

**Model:** `apple/DepthPro-hf` (Bochkovskii et al., *Depth Pro: Sharp Monocular Metric Depth in Less Than a Second*, 2024), used as released: no training or fine-tuning, no weights changed.
**Input:** raw C3VD frames; only Depth Pro's own preprocessing (resize to 1536×1536, normalise).
**Data:** dev = trans_t1_a (61 frames), test = trans_t2_b (103 frames). All scaling constants are fitted on dev only; test was run once.
**Environment:** `endodepth` (Python 3.11.17, torch 2.13.0, transformers 5.20.2), Apple M4 Pro (MPS), seed 42. Details in `run_info_dev.json` / `run_info_test.json`.
**Speed:** 3.85 s per frame on test (DA-V2 Small: 0.05 s).

Run: `python results/E/run_depthpro.py --split dev` / `--split test`. Notebook view: `DepthPro.ipynb`.

## Scaling protocols

| Scaling | What it does | Uses GT of the scored frames? |
|---|---|---|
| none (metric) | Depth Pro output in mm, nothing fitted | No |
| dev_scale | × one factor = median(GT/pred) over all dev pixels (×0.067) | No (dev only) |
| dev_lsq_depth | a·pred + b, least squares over all dev pixels (same idea as "Linear" in the DA-V2 stream) | No (dev only) |
| median (oracle) | × per-frame median(GT)/median(pred), as in the ZoeDepth notebook | Yes |
| lsq_depth (oracle) | per-frame least-squares a·pred + b, as in the ZoeDepth notebook | Yes |

## Test results (trans_t2_b), mask = `gt>0` (drops only pixels with no GT; same as the DA-V2 stream's final table)

| Model | Scaling | MAE (mm) ↓ | AbsRel ↓ | RMSE (mm) ↓ | δ<1.25 ↑ |
|---|---|---|---|---|---|
| Constant median (17.58 mm, from dev) | – | 12.98 | **0.566** | 21.21 | 0.270 |
| Depth Pro | none (raw metric) | 315.12 | 19.305 | 319.15 | 0.000 |
| Depth Pro | dev_scale | **10.87** | 0.601 | 16.08 | **0.298** |
| Depth Pro | dev_lsq_depth | 13.06 | 0.933 | **14.79** | 0.249 |
| Depth Pro | median (oracle) | 10.53 | 0.480 | 17.10 | 0.341 |
| Depth Pro | lsq_depth (oracle) | 3.96 | 0.254 | 5.19 | 0.630 |

## Test results, mask = `valid` (also drops 100 mm clamped and near-black pixels; same as ZoeDepth / `env_check.py`)

| Model | Scaling | MAE (mm) ↓ | AbsRel ↓ | RMSE (mm) ↓ | δ<1.25 ↑ |
|---|---|---|---|---|---|
| Constant median (17.26 mm, from dev) | – | 11.45 | **0.550** | 17.69 | 0.275 |
| Depth Pro | none (raw metric) | 311.65 | 19.624 | 314.86 | 0.000 |
| Depth Pro | dev_scale | **9.69** | 0.586 | 13.51 | **0.308** |
| Depth Pro | dev_lsq_depth | 11.73 | 0.875 | **13.34** | 0.269 |
| Depth Pro | median (oracle) | 9.32 | 0.468 | 14.22 | 0.346 |
| Depth Pro | lsq_depth (oracle) | 3.75 | 0.251 | 4.91 | 0.637 |

## Dev results (trans_t1_a), mask = `valid`

| Model | Scaling | MAE (mm) ↓ | AbsRel ↓ | RMSE (mm) ↓ | δ<1.25 ↑ |
|---|---|---|---|---|---|
| Constant median | – | 12.04 | 0.603 | 18.31 | 0.256 |
| Depth Pro | none (raw metric) | 263.09 | 17.108 | 266.75 | 0.000 |
| Depth Pro | dev_scale (in-sample) | 10.01 | 0.532 | 14.55 | 0.274 |
| Depth Pro | dev_lsq_depth (in-sample) | 8.85 | 0.573 | 11.20 | 0.314 |
| Depth Pro | median (oracle) | 9.63 | 0.511 | 14.39 | 0.315 |
| Depth Pro | lsq_depth (oracle) | 4.16 | 0.301 | 5.17 | 0.575 |

(`gt>0` dev numbers are in `depthpro_dev_summary.csv`.)

## Comparison with other streams on test (mask `gt>0`, scaling fitted on dev only)

| Model (stream) | MAE (mm) ↓ | AbsRel ↓ | RMSE (mm) ↓ | δ<1.25 ↑ | s/frame |
|---|---|---|---|---|---|
| Constant median | 12.98 | 0.566 | 21.21 | 0.270 | – |
| Depth Pro + dev_scale (E) | 10.87 | 0.601 | 16.08 | 0.298 | 3.85 |
| Depth Pro + dev_lsq_depth (E) | 13.06 | 0.933 | 14.79 | 0.249 | 3.85 |
| DA-V2 Small + Linear (DA-V2 stream) | 6.40 | 0.370 | 8.71 | 0.416 | 0.05 |
| DA-V2 Large + Linear (DA-V2 stream) | 5.02 | 0.284 | 7.21 | 0.553 | – |
| DA-V2 Small + Smooth Correction (DA-V2 stream) | 4.65 | 0.283 | 5.87 | 0.544 | 0.05 |

DA-V2 rows are copied from `results/Depth Anything V2/final_test_comparison.csv` and `all_models_comparison.csv`. The constant-median row matches theirs exactly (same data, same mask).

## Findings

1. **Raw metric output fails completely.** Depth Pro predicts the colon about 18× too far on test (≈ 315 mm vs ≈ 17 mm), and no pixel falls within 25% of GT. It estimates a ~51° field of view, far narrower than the colonoscope's, and interprets the lumen as a room-scale scene.
2. **With dev-fitted scaling it does not beat the constant baseline on AbsRel.** One dev scale improves MAE, RMSE and δ slightly but gives a worse AbsRel (0.601 vs 0.566). A dev-fitted scale + shift transfers worse still (AbsRel 0.933). The shift learnt on texture t1 does not suit texture t2.
3. **The shape is good; scale and range are the problem.** With a per-frame scale + shift (oracle), Depth Pro reaches AbsRel 0.254 and δ 0.630, so its relative structure of the tunnel is largely correct. Its predicted near-to-far range is compressed (≈1.7× vs ≈8.8× in GT), so one global factor cannot fix it, and the right correction changes from frame to frame.
4. **Too slow for the task.** At 3.85 s/frame it is about 75× slower than DA-V2 Small and below real-time rates needed for endoscopic guidance. Together with point 2, this supports the group's choice of Depth Anything V2.
