# -*- coding:utf-8 -*-
"""Packed time-series batch-normalization scalers.

Ported from Salesforce uni2ts (Apache 2.0).
scatter_add-based O(L) implementation (replaces the previous O(L^2) pairwise mask).
"""

from __future__ import annotations

import torch
from torch import nn

from ._util import safe_div


class PackedScaler(nn.Module):
    """Base class for a packed-batch normalization scaler."""

    def forward(
        self,
        target: torch.Tensor,  # (*batch, seq_len, #dim)
        observed_mask: torch.Tensor | None = None,  # (*batch, seq_len, #dim) bool
        sample_id: torch.Tensor | None = None,  # (*batch, seq_len) long
        variate_id: torch.Tensor | None = None,  # (*batch, seq_len) long
    ) -> tuple[
        torch.Tensor,  # (*batch, seq_len, #dim) — loc
        torch.Tensor,  # (*batch, seq_len, #dim) — scale
    ]:
        if observed_mask is None:
            observed_mask = torch.ones_like(target, dtype=torch.bool)
        if sample_id is None:
            sample_id = torch.zeros(
                target.shape[:-1], dtype=torch.long, device=target.device
            )
        if variate_id is None:
            variate_id = torch.zeros(
                target.shape[:-1], dtype=torch.long, device=target.device
            )

        loc, scale = self._get_loc_scale(target, observed_mask, sample_id, variate_id)
        return loc, scale

    def _get_loc_scale(
        self,
        target: torch.Tensor,  # (*batch, seq_len, #dim)
        observed_mask: torch.Tensor,  # (*batch, seq_len, #dim) bool
        sample_id: torch.Tensor,  # (*batch, seq_len) long
        variate_id: torch.Tensor,  # (*batch, seq_len) long
    ) -> tuple[
        torch.Tensor,  # (*batch, seq_len, #dim) — loc
        torch.Tensor,  # (*batch, seq_len, #dim) — scale
    ]:
        raise NotImplementedError


def _make_group_key(
    sample_id: torch.Tensor,  # (B, L)
    variate_id: torch.Tensor,  # (B, L)
) -> tuple[torch.Tensor, int]:  # (B, L), n_groups
    """Convert a (sample_id, variate_id) pair into a single integer group key.

    Returns
    -------
    group_key:
        ``(B, L)`` — the group key.
    n_groups:
        Number of groups (size of the scatter_add target tensor).
    """
    max_vid = variate_id.max()  # 0-dim tensor (GPU)
    group_key = sample_id * (max_vid + 1) + variate_id
    # n_groups: needs a Python int for the scatter_add target size — folded into one sync
    n_groups: int = group_key.max().item() + 1
    return group_key, n_groups


class PackedNOPScaler(PackedScaler):
    """No-op scaler: loc=0, scale=1."""

    def _get_loc_scale(
        self,
        target: torch.Tensor,  # (*batch, seq_len, #dim)
        observed_mask: torch.Tensor,  # (*batch, seq_len, #dim) bool
        sample_id: torch.Tensor,  # (*batch, seq_len) long
        variate_id: torch.Tensor,  # (*batch, seq_len) long
    ) -> tuple[
        torch.Tensor,  # (*batch, seq_len, #dim) — loc
        torch.Tensor,  # (*batch, seq_len, #dim) — scale
    ]:
        loc = torch.zeros_like(target, dtype=target.dtype)
        scale = torch.ones_like(target, dtype=target.dtype)
        return loc, scale


class PackedStdScaler(PackedScaler):
    """Z-score normalization (Bessel-corrected, per sample_id/variate_id group).

    scatter_add-based O(L) implementation.

    Parameters
    ----------
    correction:
        Bessel correction used in the variance computation.
    minimum_scale:
        Minimum scale (numerical stability).
    """

    def __init__(self, correction: int = 1, minimum_scale: float = 1e-5):
        super().__init__()
        self.correction = correction
        self.minimum_scale = minimum_scale

    def _get_loc_scale(
        self,
        target: torch.Tensor,  # (B, L, D)
        observed_mask: torch.Tensor,  # (B, L, D) bool
        sample_id: torch.Tensor,  # (B, L) long
        variate_id: torch.Tensor,  # (B, L) long
    ) -> tuple[
        torch.Tensor,  # (B, L, D) — loc
        torch.Tensor,  # (B, L, D) — scale
    ]:
        B, L, D = target.shape
        group_key, n_groups = _make_group_key(sample_id, variate_id)  # (B, L), int

        # Expand group_key to the D dimension
        gk = group_key.unsqueeze(-1).expand(B, L, D)  # (B, L, D)
        obs_float = observed_mask.to(target.dtype)  # (B, L, D)

        # Per-group observation count
        group_count = torch.zeros(
            B, n_groups, D, dtype=target.dtype, device=target.device
        )
        group_count.scatter_add_(1, gk, obs_float)  # (B, n_groups, D)

        # Per-group sum
        group_sum = torch.zeros(
            B, n_groups, D, dtype=target.dtype, device=target.device
        )
        group_sum.scatter_add_(1, gk, target * obs_float)  # (B, n_groups, D)

        # Per-group mean -> per-timestep loc
        group_loc = safe_div(group_sum, group_count)  # (B, n_groups, D)
        loc = group_loc.gather(1, gk)  # (B, L, D)

        # Per-group variance
        diff_sq = ((target - loc) ** 2) * obs_float  # (B, L, D)
        group_var_sum = torch.zeros(
            B, n_groups, D, dtype=target.dtype, device=target.device
        )
        group_var_sum.scatter_add_(1, gk, diff_sq)  # (B, n_groups, D)

        group_var = safe_div(
            group_var_sum, (group_count - self.correction).clamp(min=0)
        )
        group_scale = torch.sqrt(group_var + self.minimum_scale)  # (B, n_groups, D)
        scale = group_scale.gather(1, gk)  # (B, L, D)

        # Reset padding positions (sample_id==0)
        padding = (sample_id == 0).unsqueeze(-1)  # (B, L, 1)
        loc = loc.masked_fill(padding, 0.0)
        scale = scale.masked_fill(padding, 1.0)

        return loc, scale


class PackedAbsMeanScaler(PackedScaler):
    """Absolute-mean scaling (per sample_id/variate_id group).

    scatter_add-based O(L) implementation.

    Parameters
    ----------
    minimum_scale:
        Minimum scale (numerical stability).
    """

    def __init__(self, minimum_scale: float = 1e-5):
        super().__init__()
        self.minimum_scale = minimum_scale

    def _get_loc_scale(
        self,
        target: torch.Tensor,  # (B, L, D)
        observed_mask: torch.Tensor,  # (B, L, D) bool
        sample_id: torch.Tensor,  # (B, L) long
        variate_id: torch.Tensor,  # (B, L) long
    ) -> tuple[
        torch.Tensor,  # (B, L, D) — loc
        torch.Tensor,  # (B, L, D) — scale
    ]:
        B, L, D = target.shape
        group_key, n_groups = _make_group_key(sample_id, variate_id)  # (B, L), int

        gk = group_key.unsqueeze(-1).expand(B, L, D)  # (B, L, D)
        obs_float = observed_mask.to(target.dtype)  # (B, L, D)

        # Per-group observation count
        group_count = torch.zeros(
            B, n_groups, D, dtype=target.dtype, device=target.device
        )
        group_count.scatter_add_(1, gk, obs_float)

        # Per-group sum of absolute values
        group_abs_sum = torch.zeros(
            B, n_groups, D, dtype=target.dtype, device=target.device
        )
        group_abs_sum.scatter_add_(1, gk, target.abs() * obs_float)

        # Per-group absolute mean -> scale
        group_scale = safe_div(group_abs_sum, group_count)
        group_scale = torch.clamp(group_scale, min=self.minimum_scale)
        scale = group_scale.gather(1, gk)  # (B, L, D)

        loc = torch.zeros_like(scale)

        # Reset padding positions
        padding = (sample_id == 0).unsqueeze(-1)
        loc = loc.masked_fill(padding, 0.0)
        scale = scale.masked_fill(padding, 1.0)

        return loc, scale
