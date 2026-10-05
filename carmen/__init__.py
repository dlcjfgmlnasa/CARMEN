# -*- coding:utf-8 -*-
"""CARMEN — a cardiorespiratory foundation model for continuous physiological waveforms.

Quickstart
----------
>>> from carmen import DownstreamModelWrapper, make_batch
>>> wrapper = DownstreamModelWrapper("checkpoints/carmen.pt", device="cpu")
>>> batch = make_batch([("ecg", ecg), ("ppg", ppg)], patch_size=wrapper.patch_size)
>>> features = wrapper.extract_features(batch)   # (B, d_model)

Build the model directly with ``CARMEN.from_config(ModelConfig.from_dict(...))``
when you want the raw encoder instead of the wrapper.
"""
from __future__ import annotations

from carmen.batch import make_batch, to_device
from carmen.checkpoint import load_checkpoint, save_checkpoint
from carmen.config import ModelConfig
from carmen.data import (
    CHANNEL_NAME_TO_SIGNAL_TYPE,
    MECHANISM_GROUP,
    SIGNAL_KEY_TO_TYPE,
    SIGNAL_TYPE_NAMES,
    SIGNAL_TYPE_TO_KEY,
    BiosignalSample,
    PackCollate,
    PackedBatch,
)
from carmen.loss import MaskedPatchLoss
from carmen.model import CARMEN
from carmen.wrapper import DownstreamModelWrapper, LinearProbe, LoRALinear

__version__ = "1.0.0"

__all__ = [
    # model
    "CARMEN",
    "ModelConfig",
    "load_checkpoint",
    "save_checkpoint",
    # downstream
    "DownstreamModelWrapper",
    "LinearProbe",
    "LoRALinear",
    "MaskedPatchLoss",
    # data
    "make_batch",
    "to_device",
    "BiosignalSample",
    "PackCollate",
    "PackedBatch",
    "SIGNAL_TYPE_NAMES",
    "SIGNAL_KEY_TO_TYPE",
    "SIGNAL_TYPE_TO_KEY",
    "CHANNEL_NAME_TO_SIGNAL_TYPE",
    "MECHANISM_GROUP",
]
