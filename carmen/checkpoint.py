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


HF_REPO_ID = "dlcjfgmlnasa28/CARMEN"
PRETRAINED = ("carmen-small", "carmen-base", "carmen-large")
LOCAL_DIRS = (Path("checkpoints"), Path(__file__).resolve().parents[1] / "checkpoints")


def resolve_checkpoint(name_or_path: str | Path) -> Path:
    """Return a local checkpoint path.

    An existing file is returned as is. A released model name (``"carmen-base"``,
    or just ``"base"``) is looked up as ``checkpoints/<name>.pt`` (in the working
    directory, then in the repository), and otherwise downloaded from the Hugging
    Face Hub (``HF_REPO_ID``) and cached; this needs ``pip install huggingface_hub``.
    """
    path = Path(name_or_path)
    if path.is_file():
        return path
    name = str(name_or_path).removesuffix(".pt")
    if not name.startswith("carmen-"):
        name = f"carmen-{name}"
    if name not in PRETRAINED:
        raise FileNotFoundError(
            f"No checkpoint file {name_or_path!r} and not a released model {PRETRAINED}"
        )
    for d in LOCAL_DIRS:
        if (d / f"{name}.pt").is_file():
            return d / f"{name}.pt"
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "Downloading released weights needs `pip install huggingface_hub`"
        ) from e
    return Path(hf_hub_download(HF_REPO_ID, f"{name}.pt"))


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
