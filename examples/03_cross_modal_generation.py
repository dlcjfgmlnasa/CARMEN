# -*- coding:utf-8 -*-
"""03 — Cross-modal generation.

    python examples/03_cross_modal_generation.py [path/to/checkpoint.pt]

Generates a target modality's waveform from other modalities. Here: ECG + PPG -> ABP.
Reliable source/target pairs are listed in ``carmen.CROSS_PRED_ALLOWED_PAIRS``.
"""
import os
import sys

import _common  # noqa: F401 — puts the repo root on sys.path

import torch

from carmen import CARMEN, SIGNAL_KEY_TO_TYPE, make_batch, to_device


def main() -> None:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/carmen.pt"
    if not os.path.exists(ckpt):
        print(f"Checkpoint not found: {ckpt} (see checkpoints/README.md)")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CARMEN.from_pretrained(ckpt, device)

    # Source signals: 30 s of ECG + PPG at 100 Hz
    t = torch.linspace(0, 30, 3000)
    batch = make_batch(
        [
            ("ecg", torch.sin(2 * torch.pi * 1.2 * t)),
            ("ppg", torch.sin(2 * torch.pi * 1.2 * t - 0.6)),
        ],
        patch_size=model.patch_size,
    )
    batch = to_device(batch, device)

    # Generate ABP (signal_type 1) from the source signals
    out = model.generate_cross_modal(batch, target_signal_type=SIGNAL_KEY_TO_TYPE["abp"])
    waveform = out["waveform"]  # (B, N, patch_size) — per-patch generated ABP
    print(f"generated ABP: {tuple(waveform.shape)} (B, num_patches, patch_size)")


if __name__ == "__main__":
    main()
