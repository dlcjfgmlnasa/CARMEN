# -*- coding:utf-8 -*-
"""Turn raw 1-D signals into the ``PackedBatch`` the model consumes."""

from __future__ import annotations

import torch

from carmen.data.collate import PackCollate, PackedBatch
from carmen.data.sample import BiosignalSample
from carmen.data.signal_types import SIGNAL_KEY_TO_TYPE

_BATCH_TENSOR_FIELDS = (
    "values",
    "sample_id",
    "variate_id",
    "signal_types",
    "start_samples",
    "padded_lengths",
    "lengths",
    "sampling_rates",
)


def make_batch(
    signals: list[tuple[str | int, torch.Tensor]],
    patch_size: int,
    sampling_rate: float = 100.0,
    collate_mode: str = "any_variate",
    stride: int | None = None,
    max_length: int | None = None,
) -> PackedBatch:
    """Build a ``PackedBatch`` from raw single-channel signals.

    Parameters
    ----------
    signals:
        List of ``(modality, values)``. ``modality`` is a signal key
        (``"ecg"``, ``"abp"``, ``"ppg"``, ``"cvp"``, ``"co2"``, ``"awp"``,
        ``"icp"``, ``"resp_impedance"``, ``"resp_flow"``, ``"pap"``) or the integer
        signal_type (0-9). ``values`` is a 1-D tensor sampled at
        ``sampling_rate`` Hz. CARMEN was pretrained at 100 Hz — resample first.
    patch_size:
        Model patch size. Use ``wrapper.patch_size`` for a loaded checkpoint.
    sampling_rate:
        Sampling rate of every signal, in Hz.
    collate_mode:
        ``"any_variate"`` groups all signals of one patient so the encoder can
        attend across modalities (cross-modal). ``"ci"`` treats each signal as an
        independent row (channel-independent).
    stride:
        Patch stride. ``None`` means non-overlapping. Must match the model's.
    max_length:
        Row width of the packed tensor. Defaults to fitting all signals.

    Notes
    -----
    In ``"any_variate"`` mode ``PackCollate`` trims all variates of a patient to a
    common length so they pair up across modalities, which can drop signals shorter
    than ``5 * patch_size`` samples. Give each signal enough length, or use
    ``collate_mode="ci"`` for a single short signal.
    """
    samples: list[BiosignalSample] = []
    for rec_idx, (modality, values) in enumerate(signals):
        st = SIGNAL_KEY_TO_TYPE[modality] if isinstance(modality, str) else int(modality)
        v = torch.as_tensor(values, dtype=torch.float32).flatten()
        samples.append(
            BiosignalSample(
                values=v,
                length=int(v.numel()),
                channel_idx=0,
                recording_idx=rec_idx,
                sampling_rate=float(sampling_rate),
                n_channels=1,
                win_start=0,
                signal_type=st,
                session_id="patient",
                start_sample=0,
            )
        )
    if max_length is None:
        max_length = sum(int(s.values.numel()) for s in samples) + patch_size
    collate = PackCollate(
        max_length=max_length,
        collate_mode=collate_mode,
        patch_size=patch_size,
        stride=stride,
    )
    return collate(samples)


def to_device(batch: PackedBatch, device: str | torch.device) -> PackedBatch:
    """Move every tensor field of a ``PackedBatch`` to ``device``.

    ``DownstreamModelWrapper`` does this for you; use it only when calling a
    ``CARMEN`` model directly on GPU.
    """
    for field in _BATCH_TENSOR_FIELDS:
        t = getattr(batch, field, None)
        if isinstance(t, torch.Tensor):
            setattr(batch, field, t.to(device))
    return batch
