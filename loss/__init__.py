# -*- coding:utf-8 -*-
"""CARMEN inference-only loss utilities.

The release does not include the pretraining losses (next-pred / contrastive /
combined). It exposes only ``create_patch_mask`` (referenced by the model forward)
and ``MaskedPatchLoss`` (for reconstruction-error scoring).
"""
from __future__ import annotations

from loss.masked_mse_loss import MaskedPatchLoss, create_patch_mask

__all__ = ["MaskedPatchLoss", "create_patch_mask"]
