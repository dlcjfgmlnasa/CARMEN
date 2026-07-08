# -*- coding:utf-8 -*-
"""Checkpoint save/load utilities.

Stores the model constructor args in ``config`` so the model can be reconstructed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    config: dict[str, Any] | None = None,
    **extra: Any,
) -> None:
    """Save a model checkpoint.

    Parameters
    ----------
    path:
        Save path.
    model:
        Model to save.
    optimizer:
        Optimizer (optional).
    epoch:
        Current epoch number.
    config:
        Constructor args needed to reconstruct the model.
    **extra:
        Additional metadata.
    """
    state: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
    }
    if config is not None:
        state["config"] = config
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    state.update(extra)
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint and apply it to the model (and optimizer).

    Parameters
    ----------
    path:
        Checkpoint path.
    model:
        Model to load the state_dict into.
    optimizer:
        Optimizer to load the state_dict into (optional).
    device:
        Device to load tensors onto.

    Returns
    -------
    dict
        The full state stored in the checkpoint (epoch, config, extra, etc.).
    """
    state = torch.load(path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(
        state["model_state_dict"],
        strict=False,
    )
    if missing:
        print(f"  [checkpoint] Missing keys (newly added): {missing}")
    if unexpected:
        print(f"  [checkpoint] Unexpected keys (removed): {unexpected}")
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state
