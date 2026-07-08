# -*- coding:utf-8 -*-
"""02 — Downstream usage: attach a task head to frozen CARMEN features.

    python examples/02_downstream_probe.py [path/to/checkpoint.pt]

Two transfer strategies are shown:
  A) Linear probing — freeze the encoder, train a small head on the features.
  B) LoRA fine-tuning — insert low-rank adapters into attention (parameter-efficient).

The head here is randomly initialized; train it on your own labels.
"""
import os
import sys

from _common import make_batch  # adds the repo root to sys.path

import torch

from wrapper import DownstreamModelWrapper, LinearProbe


def main() -> None:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/carmen.pt"
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

    # ── A) Linear probing (encoder frozen) ──
    probe = LinearProbe(wrapper.d_model, n_classes=2).to(device)
    features = wrapper.extract_features(batch)  # (B, d_model), no grad
    logits = probe(features)                     # (B, 2)
    print("linear-probe logits:", tuple(logits.shape), "->", logits.softmax(-1).tolist())
    # Training loop (sketch):
    #   opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    #   loss = F.cross_entropy(probe(wrapper.extract_features(batch)), labels)

    # ── B) LoRA fine-tuning (parameter-efficient) ──
    # wrapper.inject_lora(rank=8, alpha=16)
    # trainable = wrapper.lora_parameters() + list(probe.parameters())
    # opt = torch.optim.AdamW(trainable, lr=1e-4)
    # ... then backprop through wrapper.extract_features(batch) + probe.


if __name__ == "__main__":
    main()
