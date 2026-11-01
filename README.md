# BN5211 Final Project: Monocular depth estimation on C3VD (Topic 3)

**Final model:** Depth Anything V2 Small + smooth depth-dependent correction (N1, group base) + lumen-aware isotonic calibration (N2).
Test (trans_t2_b, 103 frames, mask A): AbsRel 0.201 (95% CI 0.191-0.210), RMSE 4.72 mm, δ<1.25 0.628.

Team: Sowmya, Aditi, Eshitha, Shravan, Poorvi.

## Setup
1. Install Miniforge, then `conda env create -f environment.yml` and `conda activate endodepth`.
2. Copy `config.example.yaml` to `config.yaml` and set `data_root` to the C3VD folder.
3. `python env_check.py`

Do not commit `config.yaml`, data or model weights. Model predictions (`*.npy`) are cached locally and are not in git.

## Data layout
```
data_root/
    trans_t1_a/0000_color.png, 0000_depth.tiff, ...   (dev, 61 frames)
    trans_t2_b/...                                     (test, 103 frames)
```

## Evaluation protocol (all streams, `common.py`)
- Depth = uint16 / 65535 × 100 mm.
- Mask A (all reported results): 0 < raw < 65535 and RGB max > 10. Mask B: raw > 0 (second table in the result files).
- Metrics per frame, then the mean over frames; RMSE = sqrt(mean per-frame MSE).
- All fitting and selection on dev only (4-block CV with a 2-frame gap, `common.dev_folds`). The test sequence is run once; a `*test_done.json` lock file stops a second run.

## Baseline
The constant-median baseline (median dev depth under mask A, applied to every test pixel) is computed inside each stream with the shared loader in `common.py`. All streams report the same value:
AbsRel 0.5498, RMSE 17.71 mm, MAE 11.45 mm, δ<1.25 0.2752. See `results/ZoeDepth/zoedepth_test_summary.csv`,
`results/E/depthpro_test_summary.csv`, `results/DPT/dpt_results_test_maskA.csv` and `results/MiDaS/midas_summary_mask_A.csv`.

## Stage 1: released models
| Model | Member | Code | Results | Hardware |
|---|---|---|---|---|
| MiDaS DPT-Hybrid | Shravan | `notebooks/BN5211_Depth_Project_MiDaS.ipynb` | `results/MiDaS` | Colab T4 |
| DPT-Large | Poorvi | `notebooks/DPT-Large.py` | `results/DPT` | local |
| ZoeDepth | Aditi | `notebooks/ZoeDepth.ipynb` | `results/ZoeDepth` | M4 Pro (MPS) |
| Depth Pro | Eshitha | `results/E/run_depthpro.py`, `results/E/DepthPro.ipynb` | `results/E` | M4 Pro (MPS) |
| DAv2 Small, Base, Large, Large Indoor, Large Outdoor | Sowmya | `notebooks/DAv2_Model_Comparison.ipynb` | `results/DAv2_MaskA_Verified` | Colab T4 |

## Stage 2: group base (N1)
DAv2 Small 518 + smooth correction. Frozen parameters: `results/DAv2_MaskA_Verified/calibration_parameters_maskA.json`.

## Stage 3: novelties on the group base
| Novelty | Member | Code | Results |
|---|---|---|---|
| N2 Lumen-aware isotonic (final model); ablations: isotonic only, lumen residual | Aditi | `notebooks/LumenAware_on_SmallSmooth.ipynb` | `results/LumenAware_on_SmallSmooth` |
| N3 Uncertainty-guided refinement | Eshitha | `results/E/refinement/refine.py` | `results/E/refinement` |
| N4 Specular-reliability illumination | Shravan | `notebooks/Depth_Anything_V2_C3VD_Specular_Novelty.ipynb` (Colab) | `results/SpecularIllumination_on_SmallSmooth` |
| N5 Geometry consistency | Poorvi | `notebooks/GeoConsistency_on_SmallSmooth.py` | `results/GeoConsistency_on_SmallSmooth` |

Note: the N4 and N5 runs recompute the base in their own environments (test AbsRel 0.285 and 0.279 against 0.2805). Each novelty is compared with the base row of its own run.

## Report figures
`python report/make_figs2.py` reads only the saved result files and writes Figures 1, 3-8, 10 and 14 of the report and the
bootstrap / paired-test tables to `report/figures/`. Figures 2, 9, 11, 12, 13 and 15 come from the stream notebooks.

## Path notes
- `results/E/refinement/smooth_model.py` (DAv2 Small + smooth correction) is shared by N2 and N3. It reads
  `results/DAv2/final_model_parameters.json`, so `results/DAv2` (an earlier DAv2 run) is kept.
- The Colab notebooks (MiDaS, DAv2, N4) have a path cell at the top; change it to your data folder.
- `results/archive` and `notebooks/archive` contain superseded runs (earlier protocol). They are not used in the report and are left out of the submitted code zip (they stay in the GitHub repository).

## External code and models
Depth Anything V2 (github.com/DepthAnything/Depth-Anything-V2), MiDaS (github.com/isl-org/MiDaS), Depth Pro
(github.com/apple/ml-depth-pro). Hugging Face checkpoints: Intel/dpt-large, Intel/dpt-hybrid-midas, Intel/zoedepth-nyu-kitti,
apple/DepthPro-hf, depth-anything/Depth-Anything-V2-*-hf. Dataset: C3VD (github.com/DurrLab/C3VD).
