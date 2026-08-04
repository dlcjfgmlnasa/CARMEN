# -*- coding:utf-8 -*-
"""04 — Waveform forecasting (block next-patch prediction).

    python examples/04_forecasting.py [path/to/checkpoint.pt]

At each position the model predicts the next K patches at once. ``forecast`` returns
the full (B, N, K, patch_size) prediction map; ``generate`` rolls it out autoregressively.
"""
import os
import sys

import _common  # noqa: F401 — puts the repo root on sys.path

import torch

from carmen import CARMEN, make_batch, to_device


def main() -> None:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/carmen.pt"
    if not os.path.exists(ckpt):
        print(f"Checkpoint not found: {ckpt} (see checkpoints/README.md)")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CARMEN.from_pretrained(ckpt, device)

    # A single channel (channel-independent) — 30 s of ECG at 100 Hz
    t = torch.linspace(0, 30, 3000)
    batch = make_batch(
        [("ecg", torch.sin(2 * torch.pi * 1.2 * t))],
        patch_size=model.patch_size,
        collate_mode="ci",
    )
    batch = to_device(batch, device)

    # Block prediction map: at every patch, the next K patches
    pred = model.forecast(batch)  # (B, N, K, patch_size)
    print(f"block forecast map: {tuple(pred.shape)} (B, N, K, patch_size)")

    # Autoregressive roll-out of the next 10 patches
    steps = model.generate(batch, n_steps=10)  # (n_steps, B, patch_size)
    print(f"autoregressive generation: {tuple(steps.shape)} (n_steps, B, patch_size)")


if __name__ == "__main__":
    main()
