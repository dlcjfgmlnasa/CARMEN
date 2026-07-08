# -*- coding:utf-8 -*-
"""Foundation model loading + encoder freeze/unfreeze + LoRA wrapper.

For downstream tasks, load the pretrained model, extract features with the encoder
frozen, or insert LoRA adapters for efficient fine-tuning.

Usage
-----
>>> wrapper = DownstreamModelWrapper("checkpoints/best.pt")
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
import torch.nn.functional as F
from torch import nn

from data.collate import PackedBatch
from model import ModelConfig
from model.biosignal_model import CARMEN


# ── LoRA Layer ────────────────────────────────────────────────


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for nn.Linear.

    Freezes the original Linear and adds trainable low-rank A, B matrices.
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

        # Init: A is Kaiming, B is zero -> initial output equals the original
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frozen_out = self.original(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return frozen_out + lora_out * self.scaling


class DownstreamModelWrapper(nn.Module):
    """Pretrained model loading + encoder freeze + feature extraction wrapper.

    Parameters
    ----------
    checkpoint_path:
        Pretrained checkpoint path (.pt).
    device:
        Device to load the model onto.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)

        # 1. Load checkpoint -> restore config -> build model
        state = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        if "config" in state:
            config = ModelConfig.from_dict(state["config"])
        else:
            raise ValueError("Checkpoint has no 'config' key.")

        model_cls = CARMEN
        self.model: CARMEN = model_cls.from_config(config)
        self.model.to(self.device)

        # 2. Load state dict
        missing, unexpected = self.model.load_state_dict(
            state["model_state_dict"],
            strict=False,
        )
        if missing:
            print(f"  [model_wrapper] Missing keys: {missing}")
        if unexpected:
            print(f"  [model_wrapper] Unexpected keys: {unexpected}")

        # 3. Freeze encoder + eval mode
        self.freeze_encoder()
        self.model.eval()

        # 4. Convenience attributes
        self.d_model: int = config.d_model
        self.patch_size: int = config.patch_size
        self.config = config

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters (requires_grad=False).

        Freeze scope: scaler, patch_embed, encoder, signal_type_embed, cond_proj.
        (v2: spatial_id_embed removed — single modality embedding.)
        """
        self.model.requires_grad_(False)

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder parameters (for fine-tuning)."""
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
        rank: LoRA rank (r).
        alpha: LoRA scaling factor.
        dropout_p: LoRA dropout.
        target_modules: names of the Linear layers to apply LoRA to.

        Returns
        -------
        Number of inserted LoRA parameters.
        """
        self.freeze_encoder()  # freeze everything, train only LoRA

        n_lora_params = 0
        for name, module in self.model.named_modules():
            for target in target_modules:
                child = getattr(module, target, None)
                if child is not None and isinstance(child, nn.Linear):
                    lora = LoRALinear(child, rank=rank, alpha=alpha, dropout_p=dropout_p)
                    lora = lora.to(self.device)
                    setattr(module, target, lora)
                    n_lora_params += rank * (child.in_features + child.out_features)

        print(f"  [LoRA] rank={rank}, alpha={alpha}, targets={target_modules}")
        print(f"  [LoRA] Trainable params: {n_lora_params:,}")
        return n_lora_params

    def lora_parameters(self) -> list[nn.Parameter]:
        """Return only the trainable parameters of the LoRA adapters."""
        params = []
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
        """Feature extraction for downstream tasks.

        Parameters
        ----------
        batch:
            PackedBatch produced by PackCollate.
        pool:
            Pooling mode. ``"mean"`` = patch_mask-based mean pool,
            ``"none"`` = return (B, N, d_model) as-is.
        gap_mask_patch:
            ``(B, N)`` bool — True=gap (patch to replace with mask_token).
            Produced by ``downstream._gap_mask.sample_to_patch_mask``.
            None means no gap handling (original behavior).
        max_mask_ratio:
            Option B input-level mask gate. A sample whose gap ratio among valid
            patches exceeds this threshold is deemed untrustworthy (OOD w.r.t. the
            pretrain mask distribution). Default 0.5 (50%). Set None to disable so
            all samples pass. The features of invalid samples are filled with zero.
        return_validity:
            If True, return (features, validity). validity[b]=True means the sample
            passed the mask gate (trustworthy). Can be used as a weight during
            aggregation.

        Returns
        -------
        features: ``(B, d_model)`` (pool="mean") or ``(B, N, d_model)`` (pool="none").
        validity: ``(B,)`` bool — when return_validity=True.
        """
        self.model.eval()
        batch = self.batch_to_device(batch)

        out = self.model(
            batch, task="masked", extra_content_mask=gap_mask_patch,
        )
        encoded = out["encoded"]  # (B, N, d_model)
        patch_mask = out["patch_mask"]  # (B, N) bool — True=valid patch

        # Option B — Input-level mask gate:
        # If the gap ratio among valid patches > max_mask_ratio, the sample is untrustworthy.
        # Flag it with validity; the features of invalid samples are zeroed out.
        validity: torch.Tensor | None = None
        if max_mask_ratio is not None and gap_mask_patch is not None:
            # Gap ratio within valid_patches
            gap_in_valid = (gap_mask_patch & patch_mask).float().sum(dim=1)
            n_valid = patch_mask.float().sum(dim=1).clamp(min=1.0)
            mask_ratio_per_sample = gap_in_valid / n_valid  # (B,)
            validity = mask_ratio_per_sample <= max_mask_ratio  # (B,) bool

        if pool == "none":
            features = encoded
            if validity is not None:
                # Features of invalid samples = 0
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

    @torch.no_grad()
    def forward_masked(
        self,
        batch: PackedBatch,
    ) -> dict[str, torch.Tensor]:
        """Wrap the existing forward(task="masked").

        Returns
        -------
        dict with keys: ``reconstructed``, ``cross_pred``, ``encoded``,
        ``patch_mask``, ``loc``, ``scale``, etc.
        """
        self.model.eval()
        batch = self.batch_to_device(batch)
        return self.model(batch, task="masked")

    @torch.no_grad()
    def get_reconstruction_loss(
        self,
        batch: PackedBatch,
        mask: torch.Tensor,  # (B, N) bool — patches to reconstruct
    ) -> torch.Tensor:
        """Masked reconstruction MSE loss (for anomaly scoring).

        Parameters
        ----------
        batch:
            PackedBatch produced by PackCollate.
        mask:
            ``(B, N)`` bool — compute MSE on patches where this is ``True``.

        Returns
        -------
        ``()`` — per-window mean MSE scalar.
        """
        self.model.eval()
        batch = self.batch_to_device(batch)

        out = self.model(batch, task="masked")
        reconstructed = out["reconstructed"]  # (B, N, patch_size)

        # Extract original patches (normalized values)
        normalized = ((batch.values.unsqueeze(-1) - out["loc"]) / out["scale"]).squeeze(
            -1
        )  # (b, l)
        b, l = normalized.shape
        p = self.patch_size
        n = l // p
        original_patches = normalized.reshape(b, n, p)  # (b, n, p)

        mask = mask.to(self.device)
        if not mask.any():
            return reconstructed.new_tensor(0.0)

        loss = F.mse_loss(
            reconstructed[mask],  # (M, patch_size)
            original_patches[mask],  # (M, patch_size)
        )
        return loss

    def batch_to_device(self, batch: PackedBatch) -> PackedBatch:
        """Move the PackedBatch tensors to self.device.

        ``values`` is cast to the model parameter dtype. If the downstream loader
        kept fp16 windows to save memory (run.py), this is the only place they are
        promoted to the model dtype right before forward, preventing a dtype
        mismatch. For a batch that is already fp32 (default PackCollate output) this
        is a no-op, so the pretrain/existing path is unchanged.
        """
        param_dtype = next(self.model.parameters()).dtype
        batch.values = batch.values.to(device=self.device, dtype=param_dtype)
        batch.sample_id = batch.sample_id.to(self.device)
        batch.variate_id = batch.variate_id.to(self.device)
        return batch


class LinearProbe(nn.Module):
    """Lightweight linear head for classification/regression tasks.

    Parameters
    ----------
    d_model:
        Input feature dimension (foundation model d_model).
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
        """Feature -> logits (or regression value)."""
        return self.head(features)
