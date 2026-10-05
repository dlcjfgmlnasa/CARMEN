# -*- coding:utf-8 -*-
"""02 — Downstream usage: attach a task head to frozen CARMEN features.

    python examples/02_downstream_probe.py [path/to/checkpoint.pt]

Two transfer strategies, one training step each:
  A) Linear probing — freeze the encoder, train a small head on the features.
  B) LoRA fine-tuning — insert low-rank adapters into attention (parameter-efficient).

The labels here are random; use your own.
"""
import os
import sys

import _common  # noqa: F401 — puts the repo root on sys.path

import torch
import torch.nn.functional as F

from carmen import DownstreamModelWrapper, LinearProbe, make_batch


def main() -> None:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/carmen-base.pt"
    if not os.path.exists(ckpt):
        print(f"Checkpoint not found: {ckpt} (see checkpoints/README.md)")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = DownstreamModelWrapper(ckpt, device=device)

    t = torch.linspace(0, 30, 3000)
    batch = make_batch(
        [
            ("ecg", torch.sin(2 * torch.pi * 1.2 * t)),
            ("ppg", torch.sin(2 * torch.pi * 1.2 * t - 0.6)),
        ],
        patch_size=wrapper.patch_size,
    )
    labels = torch.tensor([1], device=device)

    # ── A) Linear probing (encoder frozen) ──
    # Features can be computed once and cached; only the probe is trained.
    probe = LinearProbe(wrapper.d_model, n_classes=2).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    features = wrapper.extract_features(batch)  # (B, d_model), no grad
    loss = F.cross_entropy(probe(features), labels)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"linear probe: features {tuple(features.shape)}, loss {loss.item():.4f}")

    # ── B) LoRA fine-tuning (parameter-efficient) ──
    # extract_features runs under no_grad, so call the encoder directly to let
    # gradients reach the adapters, then mean-pool over valid patches.
    wrapper.inject_lora(rank=8, alpha=16)
    probe = LinearProbe(wrapper.d_model, n_classes=2).to(device)
    opt = torch.optim.Adam(wrapper.lora_parameters() + list(probe.parameters()), lr=1e-4)
    out = wrapper.model(wrapper.batch_to_device(batch), task="masked")
    mask = out["patch_mask"].unsqueeze(-1).float()  # (B, N, 1)
    pooled = (out["encoded"] * mask).sum(1) / mask.sum(1).clamp(min=1.0)  # (B, d_model)
    loss = F.cross_entropy(probe(pooled), labels)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"LoRA: loss {loss.item():.4f}")


if __name__ == "__main__":
    main()
