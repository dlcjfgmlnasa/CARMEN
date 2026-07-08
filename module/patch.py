# -*- coding:utf-8 -*-
"""Patch-based tokenization of packed biosignal sequences.

Splits a continuous signal into fixed-size patches and converts them into
transformer input tokens. Used together with PackCollate's patch_size/stride
alignment.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualMLP(nn.Module):
    """Residual MLP block for patch embedding (TimesFM style).

    Converts a patch to d_model via an MLP (1 hidden layer) + skip connection.
    The MLP extracts non-linear features while the skip path preserves the
    original information.

    Parameters
    ----------
    in_dim:
        Input dimension (patch_size).
    out_dim:
        Output dimension (d_model).
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        hidden = in_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
        self.skip = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (*, in_dim) -> (*, out_dim)
        return self.mlp(x) + self.skip(x)


class PatchEmbedding(nn.Module):
    """Patch embedding (supports non-overlapping and overlapping).

    Takes a PackedBatch produced by PackCollate(patch_size=P, stride=S), splits
    each variate into patch_size units, and projects them with a Residual MLP.

    Parameters
    ----------
    patch_size:
        Time length of one patch (number of time-steps).
    d_model:
        Output embedding dimension.
    stride:
        Patch stride. ``None`` means equal to ``patch_size`` (non-overlapping).
        ``stride < patch_size`` means overlapping. ``patch_size % stride == 0`` required.
    bias:
        Whether the linear projection uses a bias.
    """

    def __init__(
        self,
        patch_size: int,
        d_model: int,
        stride: int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.d_model = d_model
        assert patch_size % self.stride == 0, (
            f"patch_size({patch_size}) must be a multiple of stride({self.stride})."
        )
        self.proj = ResidualMLP(patch_size, d_model)

    # ── Public API ────────────────────────────────────────────────

    def patchify(
        self,
        values: torch.Tensor,  # (batch, max_length)
        sample_id: torch.Tensor,  # (batch, max_length) long
        variate_id: torch.Tensor,  # (batch, max_length) long
    ) -> tuple[
        torch.Tensor,  # (batch, num_patches, patch_size) — raw patches
        torch.Tensor,  # (batch, num_patches) long — patch-level sample_id
        torch.Tensor,  # (batch, num_patches) long — patch-level variate_id
        torch.Tensor,  # (batch, num_patches) long — patch-level time_id
        torch.Tensor,  # (batch, num_patches) bool — patch_mask (True=valid)
    ]:
        """Extract patches + metadata (no projection applied)."""
        p = self.patch_size
        s = self.stride
        if s == p:
            return self._patchify_non_overlapping(values, sample_id, variate_id)
        else:
            return self._patchify_overlapping(values, sample_id, variate_id)

    def project(
        self,
        patches: torch.Tensor,  # (batch, num_patches, patch_size)
        patch_signal_types: torch.Tensor
        | None = None,  # (batch, num_patches) long — unused, kept for API compat
    ) -> torch.Tensor:  # (batch, num_patches, d_model)
        """Project raw patches to d_model embeddings (Residual MLP)."""
        return self.proj(patches)

    def forward(
        self,
        values: torch.Tensor,  # (batch, max_length)
        sample_id: torch.Tensor,  # (batch, max_length) long
        variate_id: torch.Tensor,  # (batch, max_length) long
        patch_signal_types: torch.Tensor | None = None,  # (batch, num_patches) long
    ) -> tuple[
        torch.Tensor,  # (batch, num_patches, d_model) — patch embeddings
        torch.Tensor,  # (batch, num_patches) long — patch-level sample_id
        torch.Tensor,  # (batch, num_patches) long — patch-level variate_id
        torch.Tensor,  # (batch, num_patches) long — patch-level time_id
        torch.Tensor,  # (batch, num_patches) bool — patch_mask (True=valid)
    ]:
        patches, p_sid, p_vid, time_id, patch_mask = self.patchify(
            values,
            sample_id,
            variate_id,
        )
        embedded = self.project(patches, patch_signal_types)
        return embedded, p_sid, p_vid, time_id, patch_mask

    # ── Internal patchify methods ──────────────────────────────

    def _patchify_non_overlapping(
        self,
        values: torch.Tensor,  # (batch, max_length)
        sample_id: torch.Tensor,  # (batch, max_length) long
        variate_id: torch.Tensor,  # (batch, max_length) long
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.patch_size
        b, l = values.shape
        assert l % p == 0, (
            f"max_length({l}) is not a multiple of patch_size({p}). "
            f"Use PackCollate(patch_size={p})."
        )
        n = l // p

        patches = values.reshape(b, n, p)  # (B, N, P)

        # Downsample metadata
        patch_sample_id = sample_id[:, ::p]  # (B, N)
        patch_variate_id = variate_id[:, ::p]  # (B, N)
        patch_mask = patch_sample_id != 0  # (B, N)

        # time_id
        time_id = self._compute_time_id(patch_sample_id, patch_variate_id)
        time_id[~patch_mask] = 0

        return patches, patch_sample_id, patch_variate_id, time_id, patch_mask

    def _patchify_overlapping(
        self,
        values: torch.Tensor,  # (batch, max_length)
        sample_id: torch.Tensor,  # (batch, max_length) long
        variate_id: torch.Tensor,  # (batch, max_length) long
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.patch_size
        s = self.stride
        b, l = values.shape
        assert l >= p, f"max_length({l}) is smaller than patch_size({p})."
        assert (l - p) % s == 0, (
            f"(max_length({l}) - patch_size({p})) % stride({s}) != 0. "
            f"PackCollate(patch_size={p}, stride={s}) is required."
        )
        n = (l - p) // s + 1

        patches = values.unfold(-1, p, s)  # (B, N, P)

        # Check patch validity via unfold
        sid_unfold = sample_id.unfold(-1, p, s)  # (B, N, P)
        vid_unfold = variate_id.unfold(-1, p, s)  # (B, N, P)

        # Metadata at the first position of each patch
        patch_sample_id = sid_unfold[:, :, 0]  # (B, N)
        patch_variate_id = vid_unfold[:, :, 0]  # (B, N)

        # Patch-valid condition: all p positions share the same (sid, vid) & sid != 0
        sid_ok = (sid_unfold == sid_unfold[:, :, :1]).all(dim=-1)  # (B, N)
        vid_ok = (vid_unfold == vid_unfold[:, :, :1]).all(dim=-1)  # (B, N)
        patch_mask = sid_ok & vid_ok & (patch_sample_id != 0)  # (B, N)

        # time_id
        time_id = self._compute_time_id(patch_sample_id, patch_variate_id)
        time_id[~patch_mask] = 0

        return patches, patch_sample_id, patch_variate_id, time_id, patch_mask

    @staticmethod
    def _compute_time_id(
        sample_id: torch.Tensor,  # (batch, num_patches) long
        variate_id: torch.Tensor,  # (batch, num_patches) long
    ) -> torch.Tensor:  # (batch, num_patches) long
        """Compute the ordinal index of each patch within its variate.

        Assigns a 0-based sequential index to consecutive patches that share the
        same (sample_id, variate_id) combination.
        """
        b, n = sample_id.shape
        device = sample_id.device

        # Unique variate key: combine sample_id and variate_id
        combined = sample_id * (variate_id.max() + 1) + variate_id  # (B, N)

        # Boundary detection: a new variate starts where combined differs from the previous
        boundary = torch.ones(b, n, dtype=torch.bool, device=device)
        boundary[:, 1:] = combined[:, 1:] != combined[:, :-1]

        # Use arange and cummax to compute each position's group-start index
        arange = torch.arange(n, device=device).unsqueeze(0).expand(b, -1)  # (B, N)
        boundary_pos = torch.where(boundary, arange, torch.zeros_like(arange))
        group_start, _ = boundary_pos.cummax(dim=-1)  # (B, N)

        # time_id = current position - group-start position
        time_id = arange - group_start  # (B, N)

        return time_id
