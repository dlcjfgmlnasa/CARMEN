# -*- coding:utf-8 -*-
"""00 — Smoke test.

Builds a fresh (randomly initialized) CARMEN from a config — no checkpoint needed —
and runs a forward pass. Use this to confirm the package is wired up correctly.

    python examples/00_smoke_test.py
"""
import _common  # noqa: F401 — puts the repo root on sys.path

import torch

from carmen import CARMEN, ModelConfig, make_batch


def main() -> None:
    # A tiny model on the release model's input path: 0.25 s patches, per-patch
    # conditioning, PPG loc/scale gated. Only the width and depth are smaller.
    cfg = ModelConfig(
        d_model=128, num_layers=2, patch_size=25, num_heads=4, num_signal_types=10,
        cond_trend_mode="patchls", gate_unitless_cond=True, gated_cond_signal_types=[2],
    )
    model = CARMEN.from_config(cfg).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"CARMEN built: {n_params / 1e6:.2f}M params "
        f"(d_model={cfg.d_model}, patch_size={cfg.patch_size})"
    )

    # 10 s of synthetic ECG + PPG + ABP at 100 Hz (1.2 Hz ~ 72 bpm)
    t = torch.linspace(0, 10, 1000)
    ecg = torch.sin(2 * torch.pi * 1.2 * t)
    ppg = torch.sin(2 * torch.pi * 1.2 * t - 0.6)
    abp = 80 + 30 * torch.sin(2 * torch.pi * 1.2 * t - 0.3)
    batch = make_batch(
        [("ecg", ecg), ("ppg", ppg), ("abp", abp)], patch_size=cfg.patch_size
    )

    with torch.no_grad():
        feats = model.extract_features(batch)

    encoded = feats["encoded"]                       # (B, N, d_model)
    mask = feats["patch_mask"].unsqueeze(-1).float()  # (B, N, 1)
    pooled = (encoded * mask).sum(1) / mask.sum(1).clamp(min=1.0)  # (B, d_model)

    print(f"encoded: {tuple(encoded.shape)}  ->  pooled features: {tuple(pooled.shape)}")
    print("OK")


if __name__ == "__main__":
    main()
