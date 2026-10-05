# -*- coding:utf-8 -*-
"""Query/Key projection module (including Rotary Position Encoding).

Ported from Salesforce uni2ts (Apache 2.0).
"""

from __future__ import annotations

import abc
from functools import cached_property
from typing import Any

import torch
from einops import einsum, rearrange, repeat
from torch import nn


class Projection(nn.Module, abc.ABC):
    """Base class for Q/K projection.

    Parameters
    ----------
    proj_width:
        Size of the dimension being projected.
    num_heads:
        Number of attention heads.
    num_groups:
        Number of GQA groups.
    """

    def __init__(self, proj_width: int, num_heads: int, num_groups: int, **kwargs: Any):
        super().__init__()
        self.proj_width = proj_width
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups

    @abc.abstractmethod
    def forward(
        self,
        x: torch.Tensor,  # (*batch, group, hpg, seq, dim)
        seq_id: torch.Tensor | None,  # (*batch, #group, #hpg, seq) long
    ) -> torch.Tensor:  # (*batch, group, hpg, seq, dim)
        ...


class RotaryProjection(Projection):
    """Rotary Position Embedding (RoPE) — for temporal position encoding.

    Parameters
    ----------
    proj_width:
        Size of the dimension being projected (must be even).
    num_heads:
        Number of attention heads.
    num_groups:
        Number of GQA groups.
    max_len:
        Initial maximum sequence length (grows automatically).
    base:
        Frequency base (default 10000).
    """

    def __init__(
        self,
        *,
        proj_width: int,
        num_heads: int,
        num_groups: int,
        max_len: int = 512,
        base: int = 10000,
    ):
        super().__init__(proj_width, num_heads, num_groups)
        assert self.proj_width % 2 == 0, (
            f"proj_width must be even, got {self.proj_width}"
        )
        self.register_buffer(
            "theta",
            1.0
            / torch.pow(
                base,
                torch.arange(0, self.proj_width, 2, dtype=torch.float)
                / self.proj_width,
            ),
            persistent=False,
        )
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)
        self._init_freq(max_len=max_len)

    def _init_freq(self, max_len: int):
        """Initialize or extend the cos/sin cache."""
        if self.cos is None or self.cos.size(-2) < max_len:
            position = torch.arange(
                max_len, device=self.theta.device, dtype=self.theta.dtype
            )
            m_theta = einsum(position, self.theta, "length, width -> length width")
            m_theta = repeat(m_theta, "length width -> length (width 2)")
            self.register_buffer("cos", torch.cos(m_theta), persistent=False)
            self.register_buffer("sin", torch.sin(m_theta), persistent=False)

    @staticmethod
    def _rotate(
        x: torch.Tensor,  # (..., dim)
    ) -> torch.Tensor:  # (..., dim)
        x1, x2 = rearrange(x, "... (dim r) -> r ... dim", r=2)
        return rearrange([-x2, x1], "r ... dim -> ... (dim r)", r=2)  # noqa

    def forward(
        self,
        x: torch.Tensor,  # (*batch, group, hpg, seq, dim)
        seq_id: torch.Tensor | None,  # (*batch, #group, #hpg, seq) long or float
    ) -> torch.Tensor:  # (*batch, group, hpg, seq, dim)
        if torch.is_floating_point(seq_id):
            # Fractional positions (overlapping-stride position interpolation):
            # compute the angles directly instead of indexing the integer cache.
            m_theta = seq_id.unsqueeze(-1).to(self.theta.dtype) * self.theta
            m_theta = repeat(m_theta, "... width -> ... (width 2)")
            rot_cos = torch.cos(m_theta)
            rot_sin = torch.sin(m_theta)
        else:
            self._init_freq(max_len=seq_id.max() + 1)
            rot_cos = self.cos[seq_id]
            rot_sin = self.sin[seq_id]
        return rot_cos * x + rot_sin * self._rotate(x)


class QueryKeyProjection(nn.Module):
    """Apply a projection (e.g. RoPE) to Query/Key.

    With partial_factor, the projection can be applied to only part of head_dim.

    Parameters
    ----------
    dim:
        Input dimension.
    num_heads:
        Number of attention heads.
    num_groups:
        Number of GQA groups.
    proj_layer:
        Query projection layer class.
    kwargs:
        Extra arguments for the Query projection.
    key_proj_layer:
        Key-specific projection layer class. ``None`` means same as Query.
    key_kwargs:
        Extra arguments for the Key projection.
    partial_factor:
        (start, end) range of head_dim to apply the projection to. ``None`` = all.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_groups: int,
        proj_layer: type[Projection],
        kwargs: dict[str, Any] | None = None,
        key_proj_layer: type[Projection] | None = None,
        key_kwargs: dict[str, Any] | None = None,
        partial_factor: tuple[float, float] | None = None,
    ):
        super().__init__()
        if partial_factor is not None:
            assert 0.0 <= partial_factor[0] < partial_factor[1] <= 1.0, (
                f"got {partial_factor[0]}, {partial_factor[1]}"
            )
        assert num_heads > 0 and dim % num_heads == 0
        assert (num_heads % num_groups == 0) and (num_heads >= num_groups)

        self.head_dim = dim // num_heads
        self.partial_factor = partial_factor
        self.query_proj = proj_layer(
            proj_width=self.proj_width,
            num_heads=num_heads,
            num_groups=num_groups,
            **(kwargs or {}),
        )
        if key_proj_layer is None:
            self.key_proj = self.query_proj
        else:
            self.key_proj = key_proj_layer(
                proj_width=self.proj_width,
                num_heads=num_heads,
                num_groups=num_groups,
                **(key_kwargs or {}),
            )

    @cached_property
    def proj_width(self) -> int:
        if self.partial_factor is None:
            return self.head_dim
        return int(self.head_dim * (self.partial_factor[1] - self.partial_factor[0]))

    @cached_property
    def split_sizes(self) -> tuple[int, int, int]:
        if self.partial_factor is None:
            return 0, self.head_dim, 0
        return (
            int(self.partial_factor[0] * self.head_dim),
            self.proj_width,
            int((1.0 - self.partial_factor[1]) * self.head_dim),
        )

    def forward(
        self,
        query: torch.Tensor,  # (*batch, group, hpg, q_len, dim)
        key: torch.Tensor,  # (*batch, group, hpg, kv_len, dim)
        query_id: torch.Tensor | None,  # (*batch, #group, #hpg, q_len) long
        kv_id: torch.Tensor | None,  # (*batch, #group, #hpg, kv_len) long
    ) -> tuple[
        torch.Tensor,  # (*batch, group, hpg, q_len, dim)
        torch.Tensor,  # (*batch, group, hpg, kv_len, dim)
    ]:
        if self.partial_factor is not None:
            queries = list(query.split(self.split_sizes, dim=-1))
            keys = list(key.split(self.split_sizes, dim=-1))
            queries[1] = self.query_proj(queries[1], seq_id=query_id)
            keys[1] = self.key_proj(keys[1], seq_id=kv_id)
            query = torch.cat(queries, dim=-1)
            key = torch.cat(keys, dim=-1)
        else:
            query = self.query_proj(query, seq_id=query_id)
            key = self.key_proj(key, seq_id=kv_id)
        return query, key
