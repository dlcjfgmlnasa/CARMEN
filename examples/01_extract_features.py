# -*- coding:utf-8 -*-
"""01 — Extract features from a pretrained CARMEN checkpoint.

    python examples/01_extract_features.py [path/to/checkpoint.pt]

Loads the pretrained encoder and returns a pooled feature vector per patient that
you can feed to any downstream head.
"""
import os
import sys

from _common import make_batch  # adds the repo root to sys.path

import torch

from wrapper import DownstreamModelWrapper


def main() -> None:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/carmen.pt"
    if not os.path.exists(ckpt):
        print(f"Checkpoint not found: {ckpt}")
        print("Download it (see checkpoints/README.md) or pass a path as argument.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = DownstreamModelWrapper(ckpt, device=device)
    print(f"loaded CARMEN: d_model={wrapper.d_model}, patch_size={wrapper.patch_size}")

    # 30 s of synthetic ECG + PPG + ABP at 100 Hz (replace with your own signals)
    t = torch.linspace(0, 30, 3000)
    ecg = torch.sin(2 * torch.pi * 1.2 * t)
    ppg = torch.sin(2 * torch.pi * 1.2 * t - 0.6)
    abp = 80 + 30 * torch.sin(2 * torch.pi * 1.2 * t - 0.3)
    batch = make_batch(
        [("ecg", ecg), ("ppg", ppg), ("abp", abp)], patch_size=wrapper.patch_size
    )

    # The wrapper moves tensors to its device and mean-pools over valid patches.
    features = wrapper.extract_features(batch)  # (B, d_model)
    print(f"features: {tuple(features.shape)}")


if __name__ == "__main__":
    main()
