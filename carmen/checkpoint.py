# -*- coding:utf-8 -*-
"""Checkpoint save/load utilities.

A checkpoint stores the model weights alongside the ``ModelConfig`` used to build
them, so the architecture can be reconstructed without being specified by hand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
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
    epoch:
        Epoch number the checkpoint was taken at.
    config:
        ``ModelConfig.to_dict()`` — the args needed to reconstruct the model.
    **extra:
        Additional metadata to store alongside.
    """
    state: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
    }
    if config is not None:
        state["config"] = config
    state.update(extra)
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint into ``model``.

    Parameters
    ----------
    path:
        Checkpoint path.
    model:
        Model to load the state_dict into.
    device:
        Device to load tensors onto.

    Returns
    -------
    dict
        The full state stored in the checkpoint (``epoch``, ``config``, extras).
    """
    state = torch.load(path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(
        state["model_state_dict"],
        strict=False,
    )
    if missing:
        print(f"  [checkpoint] Missing keys: {missing}")
    if unexpected:
        print(f"  [checkpoint] Unexpected keys: {unexpected}")
    return state
