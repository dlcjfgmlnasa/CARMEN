# -*- coding:utf-8 -*-
"""Shared helper for the CARMEN examples.

Adds the repo root to ``sys.path`` (so the examples run from anywhere) and provides
``make_batch`` to turn raw single-channel signals into a ``PackedBatch``.
"""
from __future__ import annotations

import pathlib
import sys

# Make `import model`, `import data`, `import wrapper` work no matter the CWD.
_REPO_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from data import PackCollate, PackedBatch, BiosignalSample, SIGNAL_KEY_TO_TYPE  # noqa: E402


def make_batch(
    signals: list[tuple[str | int, torch.Tensor]],
    patch_size: int,
    sampling_rate: float = 100.0,
    collate_mode: str = "any_variate",
    max_length: int | None = None,
) -> PackedBatch:
    """Build a ``PackedBatch`` from raw single-channel signals.

    Parameters
    ----------
    signals:
        List of ``(modality, values)``. ``modality`` is a signal key
        (``"ecg"``, ``"abp"``, ``"ppg"``, ``"cvp"``, ``"co2"``, ``"awp"``,
        ``"icp"``, ``"resp_impedance"``, ``"resp_flow"``) or the integer
        signal_type (0-8). ``values`` is a 1-D tensor sampled at
        ``sampling_rate`` Hz. CARMEN was pretrained at 100 Hz — resample first.
    patch_size:
        Model patch size. Use ``wrapper.patch_size`` for a loaded checkpoint.
    collate_mode:
        ``"any_variate"`` groups all signals of one patient so the encoder can
        attend across modalities (cross-modal). ``"ci"`` treats each signal as an
        independent row (channel-independent).
    max_length:
        Row width of the packed tensor. Defaults to fitting all signals.

    Notes
    -----
    In ``"any_variate"`` mode, signals shorter than ``5 * patch_size`` samples
    (the clinical 10 s floor at 100 Hz / patch_size 200) may be dropped. Give each
    signal enough length, or use ``collate_mode="ci"`` for a single short signal.
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
        max_length=max_length, collate_mode=collate_mode, patch_size=patch_size
    )
    return collate(samples)


def load_model(checkpoint_path: str, device: str | torch.device = "cpu"):
    """Load a pretrained ``CARMEN`` model from a checkpoint (rebuilds from its config)."""
    from model import CARMEN, ModelConfig

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CARMEN.from_config(ModelConfig.from_dict(state["config"]))
    model.load_state_dict(state["model_state_dict"], strict=False)
    return model.to(device).eval()


def to_device(batch: PackedBatch, device: str | torch.device) -> PackedBatch:
    """Move every tensor field of a ``PackedBatch`` to ``device`` (for raw-model calls).

    ``DownstreamModelWrapper`` handles this for you; use this only when calling a
    ``CARMEN`` model directly on GPU.
    """
    for field in (
        "values", "sample_id", "variate_id", "signal_types",
        "start_samples", "padded_lengths", "lengths", "sampling_rates",
    ):
        t = getattr(batch, field, None)
        if isinstance(t, torch.Tensor):
            setattr(batch, field, t.to(device))
    return batch

