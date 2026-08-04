# -*- coding:utf-8 -*-
"""Input plumbing for CARMEN inference.

``BiosignalSample`` is the input unit (one channel, one window); ``PackCollate``
packs a list of them into the ``PackedBatch`` the model consumes. Training-time
datasets and samplers are not part of the release.
"""
from __future__ import annotations

from carmen.data.collate import PackCollate, PackedBatch
from carmen.data.sample import BiosignalSample
from carmen.data.signal_types import (
    CHANNEL_NAME_TO_SIGNAL_TYPE,
    CROSS_PRED_ALLOWED_PAIRS,
    MECHANISM_GROUP,
    SIGNAL_KEY_TO_TYPE,
    SIGNAL_TYPE_NAMES,
    SIGNAL_TYPE_TO_KEY,
)

__all__ = [
    "PackCollate",
    "PackedBatch",
    "BiosignalSample",
    "SIGNAL_TYPE_NAMES",
    "SIGNAL_KEY_TO_TYPE",
    "SIGNAL_TYPE_TO_KEY",
    "CHANNEL_NAME_TO_SIGNAL_TYPE",
    "CROSS_PRED_ALLOWED_PAIRS",
    "MECHANISM_GROUP",
]
