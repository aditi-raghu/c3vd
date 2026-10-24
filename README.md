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
