# -*- coding:utf-8 -*-
"""Reusable building blocks of the CARMEN encoder."""
from __future__ import annotations

from .norm import LSCNorm, RMSNorm
from .attention import GroupedQueryAttention, MultiHeadAttention, MultiQueryAttention
from .ffn import FeedForward, GatedLinearUnitFeedForward
from .packed_scaler import (
    PackedScaler,
    PackedNOPScaler,
    PackedStdScaler,
    PackedAbsMeanScaler,
)
from .patch import PatchEmbedding
from .transformer import TransformerEncoderLayer, TransformerEncoder
from .position import (
    AttentionBias,
    BinaryAttentionBias,
    Projection,
    RotaryProjection,
    QueryKeyProjection,
)
