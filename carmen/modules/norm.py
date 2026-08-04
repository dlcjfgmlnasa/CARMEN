# -*- coding:utf-8 -*-
from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Parameters
    ----------
    normalized_shape:
        Size of the dimension(s) to normalize over.
    eps:
        Epsilon for numerical stability.
    weight:
        Whether to use a learnable scale parameter (gamma).
    dtype:
        Parameter dtype.
    """

    def __init__(
        self,
        normalized_shape: int | list[int] | torch.Size,
        eps: float = 1e-5,
        weight: bool = True,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.mean_dim = tuple(range(-len(normalized_shape), 0))

        if weight:
            self.weight = torch.nn.Parameter(torch.ones(normalized_shape, dtype=dtype))
        else:
            self.register_parameter("weight", None)

    def forward(
        self,
        x: torch.Tensor,  # (*batch, *normalized_shape)
    ) -> torch.Tensor:  # (*batch, *normalized_shape)
        output = x * torch.rsqrt(
            x.pow(2).mean(dim=self.mean_dim, keepdim=True) + self.eps
        )
        if self.weight is not None:
            return output * self.weight
        return output

    def extra_repr(self) -> str:
        return (
            f"normalized_shape={self.normalized_shape}, "
            f"eps={self.eps}, "
            f"weight={self.weight is not None}"
        )


class LSCNorm(nn.Module):
    """RMSNorm + AdaLN-style affine modulation conditioned on `cond`.

    Output: ``norm(x) * (1 + γ(cond)) + β(cond)`` where γ, β are predicted by
    a Linear layer from the conditioning vector. Zero-init: γ = β = 0 so the
    layer initially behaves identically to plain RMSNorm — safe to swap into
    a pretrained encoder.

    In CARMEN, ``cond`` carries the per-patch (loc, scale) of the signal, so every
    layer sees the absolute level that normalization stripped out. Unlike an
    additive embedding, which a network can learn to ignore, this modulation sits
    on the main path and always affects the output.
    """

    def __init__(
        self,
        normalized_shape: int | list[int] | torch.Size,
        d_cond: int,
        eps: float = 1e-5,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        d_model = normalized_shape[-1]
        # plain RMSNorm without learnable γ — modulation supplies γ
        self.norm = RMSNorm(normalized_shape, eps=eps, weight=False, dtype=dtype)
        self.modulation = nn.Linear(d_cond, 2 * d_model)
        # zero-init → γ=0, β=0 → equivalent to plain RMSNorm at init
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(
        self,
        x: torch.Tensor,  # (*batch, *normalized_shape)
        cond: torch.Tensor,  # (*batch, d_cond) — broadcast over normalized dims
    ) -> torch.Tensor:
        x = self.norm(x)
        gamma_beta = self.modulation(cond)  # (..., 2*d_model)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return x * (1.0 + gamma) + beta
