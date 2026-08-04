# -*- coding:utf-8 -*-
from __future__ import annotations

import torch


def safe_div(
    a: torch.Tensor,  # (*any)
    b: torch.Tensor,  # (*any)
) -> torch.Tensor:  # (*any)
    """Safe division: returns 0 where the denominator is 0."""
    return torch.where(b == 0, torch.zeros_like(a), a / b)
