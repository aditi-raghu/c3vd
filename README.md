# BN5211 Final Project: Monocular depth estimation on C3VD

## Setup
1. Install Miniforge.
2. `conda env create -f environment.yml`
3. `conda activate endodepth`
4. Copy `config.example.yaml` to `config.yaml` and set `data_root`.
5. `python env_check.py`

## Rules
- Do not commit `config.yaml`, data or model weights.
- Do not change `common.py` or `environment.yml` without group agreement.
- Each member works in their own notebook and results folder.

## Data layout
config.yaml → data_root/
    trans_t1_a/0000_color.png, 0000_depth.tiff, ...   (dev, 61 frames)
    trans_t2_b/...                                     (test, 103 frames)

## Evaluation protocol (all streams)
- Depth = uint16 / 65535 × 100 mm. Mask A (main): 0 < raw < 65535 and RGB max > 10. Mask B: raw > 0.
- Metrics per frame, then mean over frames; RMSE = sqrt(mean per-frame MSE).
- All fitting and selection on dev only (4-block CV, 2-frame gap). Test run once.

## Streams
| Stream | Code | Results | Hardware |
|---|---|---|---|
| Constant median | notebooks/baseline_final.py | results/baseline_per_frame.csv | – |
| MiDaS DPT-Hybrid | notebooks/BN5211_Depth_Project_MiDaS.ipynb | results/MiDaS | Colab T4 |
| DPT | notebooks/DPT.py | results/dpt_*.csv | |
| ZoeDepth | notebooks/ZoeDepth.ipynb | results/B | M4 Pro (MPS) |
| Depth Anything V2 (S, L, Indoor, Outdoor) | notebooks/Depth_Anything_V2_C3VD.ipynb | results/DAv2 | Colab T4 |
| Specular illumination | notebooks/Depth_Anything_V2_C3VD_Specular_Weighted.ipynb | results/specular_illumination | Colab T4 |
| Lumen-aware isotonic | notebooks/Final_Lumen_model.ipynb | results/Final_LumenAware_DAv2_Small_exclude_clipped | M4 Pro (MPS) |
| Depth Pro | results/E/run_depthpro.py | results/E | M4 Pro (MPS) |
| Uncertainty refinement | results/E/refinement/refine.py | results/E/refinement | M4 Pro (MPS) |
| Geometry smoothing | notebooks/geo_refine.py | results/geo_*.csv | |
| Report figures | report/make_figs.py | report/figures | – |

## Colab notebooks
The MiDaS, DAv2 and specular notebooks ran on Google Colab (T4). Change the path cell at the top to your data folder.

## External code and models
Depth Anything V2 (github.com/DepthAnything/Depth-Anything-V2), MiDaS (github.com/isl-org/MiDaS),
Hugging Face checkpoints: Intel/dpt-hybrid-midas, Intel/zoedepth-nyu-kitti, apple/DepthPro-hf,
depth-anything/Depth-Anything-V2-*-hf. Dataset: C3VD (github.com/DurrLab/C3VD).
