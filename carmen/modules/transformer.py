# -*- coding:utf-8 -*-
"""Transformer Encoder (supports GQA, GLU FFN, position encoding).

Ported from Salesforce uni2ts (Apache 2.0).
Modified to use RMSNorm as the default norm_layer.
Note: forward's var_id/time_id correspond to a PackedBatch's sample_id/variate_id.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from .attention import GroupedQueryAttention
from .ffn import FeedForward, GatedLinearUnitFeedForward
from .norm import LSCNorm, RMSNorm
from .position import AttentionBias, QueryKeyProjection


class TransformerEncoderLayer(nn.Module):
    """A single Transformer Encoder layer.

    Supports pre-norm or post-norm, composed of Self-Attention + FFN.

    Parameters
    ----------
    self_attn:
        Self-attention module.
    ffn:
        Feed-forward module.
    norm1:
        First normalization layer.
    norm2:
        Second normalization layer.
    post_attn_dropout_p:
        Dropout probability on the attention output.
    pre_norm:
        ``True`` for Pre-norm, ``False`` for Post-norm.
    """

    def __init__(
        self,
        self_attn: GroupedQueryAttention,
        ffn: FeedForward,
        norm1: nn.Module | None,
        norm2: nn.Module | None,
        post_attn_dropout_p: float = 0.0,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.dropout_p = post_attn_dropout_p

        self.self_attn = self_attn
        self.ffn = ffn
        self.norm1 = norm1 or nn.Identity()
        self.norm2 = norm2 or nn.Identity()
        self.dropout = nn.Dropout(post_attn_dropout_p)

    def _norm(self, n: nn.Module, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        """Pass cond only when it is an LSCNorm; otherwise plain norm."""
        if isinstance(n, LSCNorm):
            assert cond is not None, "LSCNorm requires cond"
            return n(x, cond)
        return n(x)

    def forward(
        self,
        x: torch.Tensor,  # (*batch, time_len, dim)
        attn_mask: torch.Tensor | None = None,  # (*batch, time_len, time_len) bool
        var_id: torch.Tensor | None = None,  # (*batch, time_len) long
        time_id: torch.Tensor | None = None,  # (*batch, time_len) long
        cond: torch.Tensor | None = None,  # (*batch, time_len, d_cond) — AdaLN conditioning
    ) -> torch.Tensor:  # (*batch, time_len, dim)
        if self.pre_norm:
            x = x + self._sa_block(
                self._norm(self.norm1, x, cond), attn_mask, var_id=var_id, time_id=time_id
            )
            x = x + self.ffn(self._norm(self.norm2, x, cond))
        else:
            x = self._norm(
                self.norm1,
                x + self._sa_block(x, attn_mask, var_id=var_id, time_id=time_id),
                cond,
            )
            x = self._norm(self.norm2, x + self.ffn(x), cond)

        return x

    def _sa_block(
        self,
        x: torch.Tensor,  # (*batch, time_len, dim)
        attn_mask: torch.Tensor | None,  # (*batch, time_len, time_len) bool
        var_id: torch.Tensor | None = None,  # (*batch, time_len) long
        time_id: torch.Tensor | None = None,  # (*batch, time_len) long
    ) -> torch.Tensor:  # (*batch, time_len, dim)
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            query_var_id=var_id,
            kv_var_id=var_id,
            query_time_id=time_id,
            kv_time_id=time_id,
        )
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Stacked Transformer Encoder.

    Composes a multi-layer encoder from GQA, GLU FFN, RoPE, variate bias, etc.

    Parameters
    ----------
    d_model:
        Embedding dimension.
    num_layers:
        Number of layers.
    num_heads:
        Number of attention heads. ``None`` means ``d_model // 64``.
    num_groups:
        Number of GQA groups. ``None`` means ``num_heads`` (MHA).
    pre_norm:
        Whether to use pre-norm.
    attn_dropout_p:
        Attention dropout probability.
    dropout_p:
        General dropout probability.
    norm_layer:
        Normalization-layer factory.
    activation:
        FFN activation function.
    use_glu:
        Whether to use GLU FFN.
    use_qk_norm:
        Whether to use Q/K norm.
    d_ff:
        FFN hidden dimension. ``None`` uses the default.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int | None = None,
        num_groups: int | None = None,
        pre_norm: bool = True,
        attn_dropout_p: float = 0.0,
        dropout_p: float = 0.0,
        norm_layer: Callable[[int], nn.Module] | None = RMSNorm,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        use_glu: bool = True,
        use_qk_norm: bool = True,
        var_attn_bias_layer: Callable[[int, int, int], AttentionBias] | None = None,
        time_attn_bias_layer: Callable[[int, int, int], AttentionBias] | None = None,
        var_qk_proj_layer: Callable[[int, int, int], QueryKeyProjection] | None = None,
        time_qk_proj_layer: Callable[[int, int, int], QueryKeyProjection] | None = None,
        shared_var_attn_bias: bool = False,
        shared_time_attn_bias: bool = False,
        shared_var_qk_proj: bool = False,
        shared_time_qk_proj: bool = False,
        d_ff: int | None = None,
        d_cond: int = 16,
    ):
        super().__init__()
        num_heads = num_heads or d_model // 64
        num_groups = num_groups or num_heads  # default MHA

        var_attn_bias = self.get_layer(
            d_model, num_heads, num_groups, var_attn_bias_layer, shared_var_attn_bias
        )
        time_attn_bias = self.get_layer(
            d_model, num_heads, num_groups, time_attn_bias_layer, shared_time_attn_bias
        )
        var_qk_proj = self.get_layer(
            d_model, num_heads, num_groups, var_qk_proj_layer, shared_var_qk_proj
        )
        time_qk_proj = self.get_layer(
            d_model, num_heads, num_groups, time_qk_proj_layer, shared_time_qk_proj
        )

        get_self_attn = partial(
            GroupedQueryAttention,
            dim=d_model,
            num_heads=num_heads,
            num_groups=num_groups,
            bias=False,
            norm_layer=norm_layer if use_qk_norm else None,
            softmax_scale=None,
            attn_dropout_p=attn_dropout_p,
            var_attn_bias=var_attn_bias,
            time_attn_bias=time_attn_bias,
            var_qk_proj=var_qk_proj,
            time_qk_proj=time_qk_proj,
        )
        get_ffn = partial(
            GatedLinearUnitFeedForward if use_glu else FeedForward,
            in_dim=d_model,
            hidden_dim=d_ff,
            out_dim=None,
            activation=activation,
            bias=False,
            ffn_dropout_p=dropout_p,
        )
        assert d_cond > 0, "LSCNorm requires d_cond > 0"
        get_encoder_layer_norm = partial(LSCNorm, d_model, d_cond=d_cond)
        # Apply LSCNorm to the final norm too, so the encoder output depends on cond
        final_norm = LSCNorm(d_model, d_cond=d_cond)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    self_attn=get_self_attn(),
                    ffn=get_ffn(),
                    norm1=get_encoder_layer_norm(),
                    norm2=get_encoder_layer_norm(),
                    pre_norm=pre_norm,
                    post_attn_dropout_p=dropout_p,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = final_norm

    @staticmethod
    def get_layer(
        dim: int,
        num_heads: int,
        num_groups: int,
        layer: Callable | None,
        shared_layer: bool,
    ) -> Callable[[], nn.Module] | None:
        """Return a layer factory. If shared, reuse the same instance."""
        if layer is None:
            return None
        if shared_layer:
            module = layer(dim=dim, num_heads=num_heads, num_groups=num_groups)
            return lambda: module
        return partial(layer, dim=dim, num_heads=num_heads, num_groups=num_groups)

    def forward(
        self,
        x: torch.Tensor,  # (*batch, time_len, dim)
        attn_mask: torch.Tensor | None = None,  # (*batch, time_len, time_len) bool
        var_id: torch.Tensor | None = None,  # (*batch, time_len) long
        time_id: torch.Tensor | None = None,  # (*batch, time_len) long
        cond: torch.Tensor | None = None,  # (*batch, time_len, d_cond) — AdaLN
    ) -> torch.Tensor:  # (*batch, time_len, dim)
        for layer in self.layers:
            x = layer(
                x, attn_mask, var_id=var_id, time_id=time_id, cond=cond,
            )
        if isinstance(self.norm, LSCNorm):
            assert cond is not None
            return self.norm(x, cond)
        return self.norm(x)
