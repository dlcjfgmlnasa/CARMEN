# -*- coding:utf-8 -*-
"""Variate-aware attention-bias module.

Ported from Salesforce uni2ts (Apache 2.0).
"""

from __future__ import annotations

import abc

import torch
from einops import rearrange
from torch import nn


class AttentionBias(nn.Module, abc.ABC):
    """Base class for attention bias.

    Parameters
    ----------
    dim:
        Input dimension.
    num_heads:
        Number of attention heads.
    num_groups:
        Number of GQA groups.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_groups: int,
    ):
        super().__init__()
        assert num_heads > 0 and dim % num_heads == 0
        assert (num_heads % num_groups == 0) and (num_heads >= num_groups)

        self.num_heads = num_heads
        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups
        self.head_dim = dim // num_heads

    @abc.abstractmethod
    def forward(
        self,
        query: torch.Tensor,  # (*batch, group, hpg, q_len, dim)
        key: torch.Tensor,  # (*batch, group, hpg, kv_len, dim)
        query_id: torch.Tensor,  # (*batch, 1, 1, q_len) long
        kv_id: torch.Tensor,  # (*batch, 1, 1, kv_len) long
    ) -> torch.Tensor:  # (*batch, #group, #hpg, q_len, kv_len)
        ...


class BinaryAttentionBias(AttentionBias):
    """Binary attention bias (bias depending on whether IDs match).

    Applies a different learnable bias to tokens of the same variate (sample_id)
    versus tokens of a different variate.

    Parameters
    ----------
    dim:
        Input dimension.
    num_heads:
        Number of attention heads.
    num_groups:
        Number of GQA groups.
    """

    def __init__(self, dim: int, num_heads: int, num_groups: int):
        super().__init__(dim, num_heads, num_groups)
        self.emb = nn.Embedding(num_embeddings=2, embedding_dim=self.num_heads)

    def forward(
        self,
        query: torch.Tensor,  # (*batch, group, hpg, q_len, dim)
        key: torch.Tensor,  # (*batch, group, hpg, kv_len, dim)
        query_id: torch.Tensor,  # (*batch, 1, 1, q_len) long
        kv_id: torch.Tensor,  # (*batch, 1, 1, kv_len) long
    ) -> torch.Tensor:  # (*batch, #group, #hpg, q_len, kv_len)
        ind = torch.eq(query_id.unsqueeze(-1), kv_id.unsqueeze(-2))
        weight = rearrange(self.emb.weight, "two num_heads -> two num_heads 1 1")
        bias = rearrange(
            ~ind * weight[:1] + ind * weight[1:],
            "... 1 (group hpg) q_len kv_len -> ... group hpg q_len kv_len",
            group=self.num_groups,
            hpg=self.heads_per_group,
        )
        return bias
