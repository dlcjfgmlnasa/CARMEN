# -*- coding:utf-8 -*-
"""Feed-Forward Network modules: standard FFN and GLU FFN.

Ported from Salesforce uni2ts (Apache 2.0).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn


class FeedForward(nn.Module):
    """Standard Feed-Forward Network (fc1 -> activation -> fc2).

    Parameters
    ----------
    in_dim:
        Input dimension.
    hidden_dim:
        Hidden dimension. ``None`` means ``4 * in_dim``.
    out_dim:
        Output dimension. ``None`` means ``in_dim``.
    activation:
        Activation function.
    bias:
        Whether the linear layers use a bias.
    ffn_dropout_p:
        Dropout probability.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int | None = None,
        out_dim: int | None = None,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
        bias: bool = True,
        ffn_dropout_p: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or 4 * in_dim
        out_dim = out_dim or in_dim

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.bias = bias
        self.ffn_dropout_p = ffn_dropout_p

        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=bias)
        self.dropout1 = nn.Dropout(ffn_dropout_p)
        self.dropout2 = nn.Dropout(ffn_dropout_p)
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,  # (..., in_dim)
    ) -> torch.Tensor:  # (..., out_dim)
        x = self._in_proj(x)
        return self.dropout2(self.fc2(self.dropout1(x)))

    def _in_proj(
        self,
        x: torch.Tensor,  # (..., in_dim)
    ) -> torch.Tensor:  # (..., out_dim)
        return self.activation(self.fc1(x))


class GatedLinearUnitFeedForward(FeedForward):
    """SiLU-gated FFN (hidden_dim = 2/3 * 4d, rounded to a multiple of 8).

    Parameters
    ----------
    in_dim:
        Input dimension.
    hidden_dim:
        Hidden dimension. ``None`` means ``adjust_hidden_dim(4 * in_dim)``.
    out_dim:
        Output dimension. ``None`` means ``in_dim``.
    activation:
        Gate activation function.
    bias:
        Whether the linear layers use a bias.
    ffn_dropout_p:
        Dropout probability.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int | None = None,
        out_dim: int | None = None,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        bias: bool = True,
        ffn_dropout_p: float = 0.0,
    ):
        super().__init__(
            in_dim,
            hidden_dim=hidden_dim or self.adjust_hidden_dim(4 * in_dim),
            out_dim=out_dim,
            activation=activation,
            bias=bias,
            ffn_dropout_p=ffn_dropout_p,
        )
        self.fc_gate = nn.Linear(self.in_dim, self.hidden_dim, bias=self.bias)

    @staticmethod
    def adjust_hidden_dim(dim: int) -> int:
        return (int(dim * 2 / 3) + 7) // 8 * 8

    def _in_proj(
        self,
        x: torch.Tensor,  # (..., in_dim)
    ) -> torch.Tensor:  # (..., out_dim)
        return self.activation(self.fc_gate(x)) * self.fc1(x)
