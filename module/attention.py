# -*- coding:utf-8 -*-
"""Grouped Query Attention (supports variate/temporal bias and projection).

Ported from Salesforce uni2ts (Apache 2.0).
Modified to use RMSNorm as the default norm_layer.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from .norm import RMSNorm
from .position import AttentionBias, QueryKeyProjection


def native_scaled_dot_product_attention(
    query: torch.Tensor,  # (*batch, group, hpg, q_len, dim)
    key: torch.Tensor,  # (*batch, group, hpg, kv_len, dim)
    value: torch.Tensor,  # (*batch, group, hpg, kv_len, dim)
    attn_mask: torch.Tensor
    | None = None,  # (*batch, #group, #hpg, q_len, kv_len) bool|float
    dropout_p: float = 0.0,
    scale: float | None = None,
) -> torch.Tensor:  # (*batch, group, hpg, q_len, dim)
    """Fallback scaled dot-product attention (used when FlashAttention is unavailable)."""
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias = torch.zeros_like(attn_weight)
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask
        attn_weight = attn_weight + attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention.

    Q is projected for all heads while K/V are projected per group, so heads
    within a group share K/V. Supports Q/K norm, variate/time bias, and position
    encodings such as RoPE.

    Parameters
    ----------
    dim:
        Input/output dimension.
    num_heads:
        Number of attention heads.
    num_groups:
        Number of K/V groups (``num_heads`` = MHA, ``1`` = MQA).
    bias:
        Whether the linear projections use a bias.
    norm_layer:
        Layer used for Q/K norm. ``None`` disables it.
    softmax_scale:
        Softmax scale factor. ``None`` means ``1/sqrt(head_dim)``.
    attn_dropout_p:
        Attention dropout probability.
    var_attn_bias:
        Variate attention-bias factory.
    time_attn_bias:
        Time attention-bias factory.
    var_qk_proj:
        Variate Q/K projection factory.
    time_qk_proj:
        Time Q/K projection factory (e.g. RoPE).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_groups: int,
        bias: bool = True,
        norm_layer: type[nn.Module] | partial[nn.Module] | None = RMSNorm,
        softmax_scale: float | None = None,
        attn_dropout_p: float = 0.0,
        var_attn_bias: Callable[[], AttentionBias] | None = None,
        time_attn_bias: Callable[[], AttentionBias] | None = None,
        var_qk_proj: Callable[[], QueryKeyProjection] | None = None,
        time_qk_proj: Callable[[], QueryKeyProjection] | None = None,
    ):
        super().__init__()
        assert num_heads > 0 and dim % num_heads == 0
        assert (num_heads % num_groups == 0) and (num_heads >= num_groups)

        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = dim // num_heads
        self.heads_per_group = num_heads // num_groups
        self.var_attn_bias = var_attn_bias() if var_attn_bias is not None else None
        self.time_attn_bias = time_attn_bias() if time_attn_bias is not None else None
        self.var_qk_proj = var_qk_proj() if var_qk_proj is not None else None
        self.time_qk_proj = time_qk_proj() if time_qk_proj is not None else None

        self.softmax_scale = softmax_scale or 1 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.head_dim * num_groups, bias=bias)
        self.v_proj = nn.Linear(dim, self.head_dim * num_groups, bias=bias)
        self.q_norm = (
            norm_layer(self.head_dim) if norm_layer is not None else nn.Identity()
        )
        self.k_norm = (
            norm_layer(self.head_dim) if norm_layer is not None else nn.Identity()
        )
        self.attn_dropout_p = attn_dropout_p
        self.out_proj = nn.Linear(dim, dim, bias=bias)

    def _get_var_id(
        self,
        query: torch.Tensor,  # (*batch, group, hpg, q_len, dim)
        key: torch.Tensor,  # (*batch, group, hpg, kv_len, dim)
        query_var_id: torch.Tensor | None,  # (*batch, q_len) long
        kv_var_id: torch.Tensor | None,  # (*batch, kv_len) long
    ) -> tuple[
        torch.Tensor | None,  # (*batch, #group, #hpg, q_len) long
        torch.Tensor | None,  # (*batch, #group, #hpg, kv_len) long
    ]:
        if self.var_attn_bias is not None or self.var_qk_proj is not None:
            if query_var_id is None:
                query_var_id = repeat(
                    torch.zeros((), device=query.device, dtype=torch.long),
                    f" -> {' '.join(map(str, query.shape[:-4]))} 1 1 {query.shape[-2]}",
                )
            else:
                query_var_id = rearrange(query_var_id, "... q_len -> ... 1 1 q_len")

            if kv_var_id is None:
                kv_var_id = repeat(
                    torch.zeros((), device=key.device, dtype=torch.long),
                    f" -> {' '.join(map(str, key.shape[:-4]))} 1 1 {key.shape[-2]}",
                )
            else:
                kv_var_id = rearrange(kv_var_id, "... kv_len -> ... 1 1 kv_len")

        return query_var_id, kv_var_id

    def _get_time_id(
        self,
        query: torch.Tensor,  # (*batch, group, hpg, q_len, dim)
        key: torch.Tensor,  # (*batch, group, hpg, kv_len, dim)
        query_time_id: torch.Tensor | None,  # (*batch, q_len) long
        kv_time_id: torch.Tensor | None,  # (*batch, kv_len) long
    ) -> tuple[
        torch.Tensor | None,  # (*batch, 1, 1, q_len) long
        torch.Tensor | None,  # (*batch, 1, 1, kv_len) long
    ]:
        if self.time_attn_bias is not None or self.time_qk_proj is not None:
            if query_time_id is None:
                query_time_id = repeat(
                    torch.arange(
                        query.shape[-2], device=query.device, dtype=torch.long
                    ),
                    f"q_len -> {' '.join(map(str, query.shape[:-4]))} 1 1 q_len",
                )
            else:
                query_time_id = rearrange(query_time_id, "... q_len -> ... 1 1 q_len")

            if kv_time_id is None:
                kv_time_id = repeat(
                    torch.arange(key.shape[-2], device=key.device, dtype=torch.long),
                    f"kv_len -> {' '.join(map(str, key.shape[:-4]))} 1 1 kv_len",
                )
            else:
                kv_time_id = rearrange(kv_time_id, "... kv_len-> ... 1 1 kv_len")

        return query_time_id, kv_time_id

    def _update_attn_mask(
        self,
        attn_mask: torch.Tensor | None,  # (*batch, q_len, kv_len) bool
        query: torch.Tensor,  # (*batch, group, hpg, q_len, dim)
        key: torch.Tensor,  # (*batch, group, hpg, kv_len, dim)
        query_var_id: torch.Tensor | None = None,  # (*batch, 1, 1, q_len) long
        kv_var_id: torch.Tensor | None = None,  # (*batch, 1, 1, kv_len) long
        query_time_id: torch.Tensor | None = None,  # (*batch, 1, 1, q_len) long
        kv_time_id: torch.Tensor | None = None,  # (*batch, 1, 1, kv_len) long
    ) -> torch.Tensor | None:  # (*batch, #group, #hpg, q_len, kv_len) bool|float
        if attn_mask is not None:
            attn_mask = rearrange(
                attn_mask,
                "... q_len kv_len -> ... 1 1 q_len kv_len",
            )

        attn_bias = 0
        if self.var_attn_bias is not None:
            attn_bias = attn_bias + self.var_attn_bias(
                query,
                key,
                query_id=query_var_id,
                kv_id=kv_var_id,
            )

        if self.time_attn_bias is not None:
            attn_bias = attn_bias + self.time_attn_bias(
                query,
                key,
                query_id=query_time_id,
                kv_id=kv_time_id,
            )

        attn_mask = (
            attn_mask
            if isinstance(attn_bias, int)
            else (
                attn_bias
                if attn_mask is None
                else attn_bias.masked_fill(attn_mask.logical_not(), float("-inf"))
            )
        )
        return attn_mask

    def _qk_proj(
        self,
        query: torch.Tensor,  # (*batch, group, hpg, q_len, dim)
        key: torch.Tensor,  # (*batch, group, hpg, kv_len, dim)
        query_var_id: torch.Tensor | None,  # (*batch, #group, #hpg, q_len) long
        kv_var_id: torch.Tensor | None,  # (*batch, #group, #hpg, kv_len) long
        query_time_id: torch.Tensor | None,  # (*batch, #group, #hpg, q_len) long
        kv_time_id: torch.Tensor | None,  # (*batch, #group, #hpg, kv_len) long
    ) -> tuple[
        torch.Tensor,  # (*batch, group, hpg, q_len, dim)
        torch.Tensor,  # (*batch, group, hpg, kv_len, dim)
    ]:
        if self.var_qk_proj is not None:
            query, key = self.var_qk_proj(
                query, key, query_id=query_var_id, kv_id=kv_var_id
            )

        if self.time_qk_proj is not None:
            query, key = self.time_qk_proj(
                query, key, query_id=query_time_id, kv_id=kv_time_id
            )

        return query, key

    def forward(
        self,
        query: torch.Tensor,  # (*batch, q_len, dim)
        key: torch.Tensor,  # (*batch, kv_len, dim)
        value: torch.Tensor,  # (*batch, kv_len, dim)
        attn_mask: torch.Tensor | None = None,  # (*batch, q_len, kv_len) bool
        query_var_id: torch.Tensor | None = None,  # (*batch, q_len) long
        kv_var_id: torch.Tensor | None = None,  # (*batch, kv_len) long
        query_time_id: torch.Tensor | None = None,  # (*batch, q_len) long
        kv_time_id: torch.Tensor | None = None,  # (*batch, kv_len) long
    ) -> torch.Tensor:  # (*batch, q_len, dim)
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        query = self.q_norm(
            rearrange(
                query,
                "... q_len (group hpg dim) -> ... group hpg q_len dim",
                group=self.num_groups,
                hpg=self.heads_per_group,
            )
        )
        # Reshape K to (*batch, group, 1, kv_len, dim), apply norm, then broadcast the
        # hpg=1 dim via expand with no physical copy.
        key = self.k_norm(
            rearrange(
                key,
                "... kv_len (group dim) -> ... group 1 kv_len dim",
                group=self.num_groups,
            )
        )
        key = key.expand(
            *key.shape[:-4],
            self.num_groups,
            self.heads_per_group,
            key.shape[-2],
            key.shape[-1],
        )  # (*batch, group, hpg, kv_len, dim)
        # Reshape V to (*batch, group, 1, kv_len, dim) then expand — no copy.
        value = rearrange(
            value,
            "... kv_len (group dim) -> ... group 1 kv_len dim",
            group=self.num_groups,
        )
        value = value.expand(
            *value.shape[:-4],
            self.num_groups,
            self.heads_per_group,
            value.shape[-2],
            value.shape[-1],
        )  # (*batch, group, hpg, kv_len, dim)

        query_var_id, kv_var_id = self._get_var_id(query, key, query_var_id, kv_var_id)
        query_time_id, kv_time_id = self._get_time_id(
            query,
            key,
            query_time_id,
            kv_time_id,
        )

        attn_mask = self._update_attn_mask(
            attn_mask,
            query,
            key,
            query_var_id=query_var_id,
            kv_var_id=kv_var_id,
            query_time_id=query_time_id,
            kv_time_id=kv_time_id,
        )

        query, key = self._qk_proj(
            query,
            key,
            query_var_id=query_var_id,
            kv_var_id=kv_var_id,
            query_time_id=query_time_id,
            kv_time_id=kv_time_id,
        )

        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            scale=self.softmax_scale,
        )
        out = rearrange(out, "... group hpg q_len dim -> ... q_len (group hpg dim)")
        return self.out_proj(out)


class MultiQueryAttention(GroupedQueryAttention):
    """Multi-Query Attention: all heads share a single K/V group."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        bias: bool = True,
        norm_layer: type[nn.Module] | partial[nn.Module] | None = RMSNorm,
        softmax_scale: float | None = None,
        attn_dropout_p: float = 0.0,
        var_attn_bias: Callable[[], AttentionBias] | None = None,
        time_attn_bias: Callable[[], AttentionBias] | None = None,
        var_qk_proj: Callable[[], QueryKeyProjection] | None = None,
        time_qk_proj: Callable[[], QueryKeyProjection] | None = None,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            num_groups=1,
            bias=bias,
            norm_layer=norm_layer,
            softmax_scale=softmax_scale,
            attn_dropout_p=attn_dropout_p,
            var_attn_bias=var_attn_bias,
            time_attn_bias=time_attn_bias,
            var_qk_proj=var_qk_proj,
            time_qk_proj=time_qk_proj,
        )


class MultiHeadAttention(GroupedQueryAttention):
    """Standard Multi-Head Attention: each head has its own K/V."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        bias: bool = True,
        norm_layer: type[nn.Module] | partial[nn.Module] | None = RMSNorm,
        softmax_scale: float | None = None,
        attn_dropout_p: float = 0.0,
        var_attn_bias: Callable[[], AttentionBias] | None = None,
        time_attn_bias: Callable[[], AttentionBias] | None = None,
        var_qk_proj: Callable[[], QueryKeyProjection] | None = None,
        time_qk_proj: Callable[[], QueryKeyProjection] | None = None,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            num_groups=num_heads,
            bias=bias,
            norm_layer=norm_layer,
            softmax_scale=softmax_scale,
            attn_dropout_p=attn_dropout_p,
            var_attn_bias=var_attn_bias,
            time_attn_bias=time_attn_bias,
            var_qk_proj=var_qk_proj,
            time_qk_proj=time_qk_proj,
        )
