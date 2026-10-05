# -*- coding:utf-8 -*-
"""Checkpoint loading + encoder freezing + LoRA, for downstream tasks.

Load a pretrained CARMEN, extract features with the encoder frozen, or insert LoRA
adapters for parameter-efficient fine-tuning.

Usage
-----
>>> wrapper = DownstreamModelWrapper("checkpoints/carmen.pt")
>>> features = wrapper.extract_features(batch)  # (B, d_model)
>>> probe = LinearProbe(wrapper.d_model, n_classes=3)
>>>
>>> # LoRA
>>> wrapper.inject_lora(rank=8)
>>> lora_params = wrapper.lora_parameters()
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from carmen.config import ModelConfig
from carmen.data.collate import PackedBatch
from carmen.model import CARMEN


# ── LoRA ──────────────────────────────────────────────────────


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for ``nn.Linear``.

    Freezes the original Linear and adds trainable low-rank A, B matrices::

        output = frozen_linear(x) + (x @ A @ B) * (alpha / rank)
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.original = original
        self.original.requires_grad_(False)

        in_features = original.in_features
        out_features = original.out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

        # Init: A is Kaiming, B is zero -> the initial output equals the original
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frozen_out = self.original(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return frozen_out + lora_out * self.scaling


class DownstreamModelWrapper(nn.Module):
    """Load a pretrained CARMEN, freeze it, and extract features.

    Parameters
    ----------
    checkpoint_path:
        Pretrained checkpoint path (``.pt``).
    device:
        Device to load the model onto.
    patch_stride:
        Override the patch stride for overlapping-patch inference (finer tokens
        from the same weights). Must divide ``patch_size``. ``None`` keeps the
        checkpoint's stride.
    rope_pi:
        With overlapping patches, interpolate RoPE positions to physical spacing.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device = "cuda",
        patch_stride: int | None = None,
        rope_pi: bool = True,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)

        # 1. Load checkpoint -> restore config -> build model
        state = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        if "config" not in state:
            raise ValueError(f"Checkpoint has no 'config' key: {checkpoint_path}")
        config = ModelConfig.from_dict(state["config"])

        if patch_stride is not None:
            config.stride = patch_stride  # not a learned parameter; weights unchanged

        self.model: CARMEN = CARMEN.from_config(config)
        self.model.rope_pi = rope_pi
        self.model.to(self.device)

        # 2. Load state dict
        missing, unexpected = self.model.load_state_dict(
            state["model_state_dict"],
            strict=False,
        )
        if missing:
            print(f"  [CARMEN] Missing keys: {missing}")
        if unexpected:
            print(f"  [CARMEN] Unexpected keys: {unexpected}")

        # 3. Freeze + eval mode
        self.freeze_encoder()
        self.model.eval()

        # 4. Convenience attributes
        self.d_model: int = config.d_model
        self.patch_size: int = config.patch_size
        self.config = config

    def freeze_encoder(self) -> None:
        """Freeze every model parameter (``requires_grad=False``)."""
        self.model.requires_grad_(False)

    def unfreeze_encoder(self) -> None:
        """Unfreeze every model parameter (for full fine-tuning)."""
        self.model.requires_grad_(True)

    def inject_lora(
        self,
        rank: int = 8,
        alpha: float = 16.0,
        dropout_p: float = 0.0,
        target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
    ) -> int:
        """Insert LoRA adapters into the attention layers' target modules.

        Parameters
        ----------
        rank:
            LoRA rank (r).
        alpha:
            LoRA scaling factor.
        dropout_p:
            LoRA dropout.
        target_modules:
            Names of the Linear layers to wrap.

        Returns
        -------
        int
            Number of trainable LoRA parameters inserted.
        """
        self.freeze_encoder()  # freeze everything, train only LoRA

        n_lora_params = 0
        for _name, module in self.model.named_modules():
            for target in target_modules:
                child = getattr(module, target, None)
                if child is not None and isinstance(child, nn.Linear):
                    lora = LoRALinear(
                        child, rank=rank, alpha=alpha, dropout_p=dropout_p
                    )
                    lora = lora.to(self.device)
                    setattr(module, target, lora)
                    n_lora_params += rank * (child.in_features + child.out_features)

        print(f"  [LoRA] rank={rank}, alpha={alpha}, targets={target_modules}")
        print(f"  [LoRA] Trainable params: {n_lora_params:,}")
        return n_lora_params

    def lora_parameters(self) -> list[nn.Parameter]:
        """Return only the trainable parameters of the LoRA adapters."""
        params: list[nn.Parameter] = []
        for module in self.model.modules():
            if isinstance(module, LoRALinear):
                params.extend(module.lora_A.parameters())
                params.extend(module.lora_B.parameters())
        return params

    @torch.no_grad()
    def extract_features(
        self,
        batch: PackedBatch,
        pool: str = "mean",
        gap_mask_patch: torch.Tensor | None = None,
        max_mask_ratio: float | None = 0.5,
        return_validity: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Extract frozen features for a downstream head.

        Parameters
        ----------
        batch:
            PackedBatch produced by PackCollate.
        pool:
            ``"mean"`` = mean-pool over valid patches, ``"none"`` = return
            ``(B, N, d_model)`` unpooled.
        gap_mask_patch:
            ``(B, N)`` bool — True marks a patch that is a data gap and should be
            replaced by the [MASK] token. ``None`` disables gap handling.
        max_mask_ratio:
            Sample-level gate. A sample whose gap ratio among valid patches exceeds
            this threshold is out-of-distribution w.r.t. pretraining and its
            features are zeroed. ``None`` disables the gate.
        return_validity:
            If True, also return a ``(B,)`` bool telling which samples passed the
            gate — useful as a weight when aggregating.

        Returns
        -------
        features:
            ``(B, d_model)`` when ``pool="mean"``, ``(B, N, d_model)`` when
            ``pool="none"``.
        validity:
            ``(B,)`` bool — only when ``return_validity=True``.
        """
        self.model.eval()
        batch = self.batch_to_device(batch)

        out = self.model(batch, task="masked", extra_content_mask=gap_mask_patch)
        encoded = out["encoded"]  # (B, N, d_model)
        patch_mask = out["patch_mask"]  # (B, N) bool — True=valid patch

        validity: torch.Tensor | None = None
        if max_mask_ratio is not None and gap_mask_patch is not None:
            gap_in_valid = (gap_mask_patch & patch_mask).float().sum(dim=1)
            n_valid = patch_mask.float().sum(dim=1).clamp(min=1.0)
            validity = (gap_in_valid / n_valid) <= max_mask_ratio  # (B,) bool

        if pool == "none":
            features = encoded
            if validity is not None:
                features = features * validity.view(-1, 1, 1).float()
            return (features, validity) if return_validity else features

        # Mean pooling over valid patches
        mask_f = patch_mask.unsqueeze(-1).float()  # (B, N, 1)
        pooled = (encoded * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(
            min=1.0
        )  # (B, d_model)

        if validity is not None:
            pooled = pooled * validity.unsqueeze(-1).float()

        return (pooled, validity) if return_validity else pooled

    def batch_to_device(self, batch: PackedBatch) -> PackedBatch:
        """Move the PackedBatch tensors to ``self.device``.

        ``values`` is cast to the model's parameter dtype, so loaders that keep
        windows in fp16 to save memory are promoted right before the forward pass.
        A batch that is already fp32 (the PackCollate default) is unaffected.
        """
        param_dtype = next(self.model.parameters()).dtype
        batch.values = batch.values.to(device=self.device, dtype=param_dtype)
        batch.sample_id = batch.sample_id.to(self.device)
        batch.variate_id = batch.variate_id.to(self.device)
        return batch


class LinearProbe(nn.Module):
    """Lightweight linear head for classification or regression.

    Parameters
    ----------
    d_model:
        Input feature dimension (the foundation model's ``d_model``).
    n_classes:
        Number of output classes. ``1`` means regression (no sigmoid).
    dropout_p:
        Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_classes: int,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout_p),
            nn.Linear(d_model, n_classes),
        )

    def forward(
        self,
        features: torch.Tensor,  # (B, d_model)
    ) -> torch.Tensor:  # (B, n_classes)
        """Features -> logits (or regression value)."""
        return self.head(features)
