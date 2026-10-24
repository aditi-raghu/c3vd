"""Check the endodepth environment on any system.

Usage:
    python env_check.py
    python env_check.py --skip-model     # no internet, or slow connection
"""
import argparse
import platform
import time

import numpy as np


def check_packages():
    import cv2, matplotlib, pandas, scipy, skimage, tifffile, torch, transformers, yaml
    print(f"Python        {platform.python_version()} ({platform.platform()})")
    for name, mod in [("numpy", np), ("scipy", scipy), ("pandas", pandas),
                      ("opencv", cv2), ("tifffile", tifffile), ("scikit-image", skimage),
                      ("matplotlib", matplotlib), ("pyyaml", yaml),
                      ("torch", torch), ("transformers", transformers)]:
        print(f"{name:<13} {mod.__version__}")


def check_model(device):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    from common import sync

    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
    image = Image.fromarray(np.random.randint(0, 255, (1080, 1350, 3), dtype=np.uint8))
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs)
        sync(device)
        t0 = time.perf_counter()
        out = model(**inputs).predicted_depth
        sync(device)
    print(f"DAv2 Small    output {tuple(out.shape)}, "
          f"{(time.perf_counter() - t0) * 1000:.0f} ms per frame")


def check_data(cfg):
    import cv2, tifffile
    from common import sequence_dir

    for split, expected in [("dev", 61), ("test", 103)]:
        try:
            folder = sequence_dir(cfg, split)
        except FileNotFoundError as e:
            print(f"Data ({split})   NOT FOUND: {e}")
            continue
        colors = sorted(folder.glob("*_color.png"))
        depths = sorted(folder.glob("*_depth.tiff"))
        status = "OK" if len(colors) == len(depths) == expected else "CHECK"
        print(f"Data ({split})   {folder.name}: {len(colors)} colour, "
              f"{len(depths)} depth (expected {expected}) {status}")
        if split == "dev" and colors and depths:
            d = tifffile.imread(depths[0])
            rgb = cv2.cvtColor(cv2.imread(str(colors[0])), cv2.COLOR_BGR2RGB)
            assert d.dtype == np.uint16, f"Depth dtype is {d.dtype}, expected uint16."
            assert d.shape[:2] == rgb.shape[:2], "Colour and depth sizes are different."
            valid = (d > 0) & (d < 65535) & (rgb.max(axis=2) > 10)
            mm = d.astype(np.float32) / 65535.0 * 100.0
            print(f"Frame 0       size {d.shape}, valid {valid.mean() * 100:.1f}%, "
                  f"depth {mm[valid].min():.1f} to {mm[valid].max():.1f} mm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()

    check_packages()
    from common import CFG, DEVICE
    print(f"Device        {DEVICE}")
    print(f"Data root     {CFG['data_root']}")
    if not args.skip_model:
        check_model(DEVICE)
    check_data(CFG)
    print("Checks finished. Send this output to the group if a line says NOT FOUND or CHECK.")
