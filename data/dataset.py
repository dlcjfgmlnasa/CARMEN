# -*- coding:utf-8 -*-
"""Input sample container for CARMEN inference.

A single channel-independent time-series segment. Build a list of these and pass
them to ``PackCollate`` to produce a ``PackedBatch`` for the model.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BiosignalSample:
    """A single-channel time-series segment (channel-independent).

    Attributes
    ----------
    values:
        1-D signal tensor. ``(time,)``.
    length:
        Number of samples in ``values``.
    channel_idx:
        Channel index within the source recording.
    recording_idx:
        Index of the source recording.
    sampling_rate:
        Sampling rate in Hz (CARMEN expects 100 Hz).
    n_channels:
        Number of channels in the source recording.
    win_start:
        Start offset (in samples) of this window within the recording.
    signal_type:
        Modality code (0=ECG, 1=ABP, 2=PPG, 3=CVP, 4=CO2, 5=AWP, 6=ICP,
        7=RESP_Impedance, 8=RESP_Flow). See ``data.spatial_map``.
    session_id:
        Session identifier. Samples that share a ``session_id`` and fall in the
        same time slot are grouped together (cross-modal) by ``PackCollate`` in
        ``"any_variate"`` mode. Leave empty ("") for channel-independent use.
    start_sample:
        Absolute start sample of this segment within the session.
    """

    values: torch.Tensor  # (time,)
    length: int
    channel_idx: int
    recording_idx: int
    sampling_rate: float
    n_channels: int
    win_start: int
    signal_type: int
    session_id: str = ""
    start_sample: int = 0
