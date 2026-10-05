# -*- coding:utf-8 -*-
"""CARMEN — cardiorespiratory foundation model for continuous physiological waveforms.

Encoding pipeline::

    Scaler -> Patchify -> Project -> ModalityEmbed -> LocScale(AdaLN) -> TransformerEncoder -> Head

One Transformer encoder serves every modality. ``task="masked"`` runs bidirectional
attention (feature extraction); ``task="next_pred"`` runs causal attention through the
next-patch head used in pretraining.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import fields
from functools import partial
from pathlib import Path

import torch
from torch import nn

from carmen.config import ModelConfig
from carmen.data.collate import PackedBatch
from carmen.modules.packed_scaler import PackedScaler, PackedStdScaler
from carmen.modules.patch import PatchEmbedding
from carmen.modules.position import (
    BinaryAttentionBias,
    QueryKeyProjection,
    RotaryProjection,
)
from carmen.modules.transformer import TransformerEncoder


class BlockNextHead(nn.Module):
    """Shared trunk + K horizon-specific heads for block next-patch prediction.

    Each position's encoded vector is transformed by a shared non-linear trunk,
    then K independent Linear heads predict the future patch per horizon.

    Input:  ``(B, N, d_model)``
    Output: ``(B, N, K, patch_size)`` — the k-th head predicts the t+k patch.

    Parameters
    ----------
    d_model:
        Input dimension.
    patch_size:
        Output patch size (number of samples).
    block_size:
        K — number of future patches to predict.
    d_inner:
        Trunk inner dimension. ``None`` means ``d_model``.
    """

    def __init__(
        self,
        d_model: int,
        patch_size: int,
        block_size: int,
        d_inner: int | None = None,
    ) -> None:
        super().__init__()
        d_inner = d_inner if d_inner is not None else d_model
        self.block_size = block_size
        self.patch_size = patch_size

        self.trunk = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.GELU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(d_inner, patch_size) for _ in range(block_size)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d_model)
        h = self.trunk(x)  # (B, N, d_inner)
        outs = [head(h) for head in self.heads]  # list of (B, N, patch_size)
        return torch.stack(outs, dim=2)  # (B, N, K, patch_size)


class CARMEN(nn.Module):
    """Cardiorespiratory foundation model. Raw-patch reconstruction for all signals.

    Every signal type goes through the same raw-patch pipeline. ``_encode()``
    factors out the common encoding stages so subclasses can extend it.

    Parameters
    ----------
    d_model:
        Transformer embedding dimension.
    num_layers:
        Number of transformer encoder layers.
    patch_size:
        Patch size (number of time-steps).
    stride:
        Patch stride (for overlapping patches). ``None`` means equal to ``patch_size``.
    num_heads:
        Number of attention heads. ``None`` means ``d_model // 64``.
    num_groups:
        Number of GQA groups. ``None`` means ``num_heads`` (MHA).
    use_glu:
        Whether to use a Gated Linear Unit FFN.
    use_rope:
        Whether to use Rotary Position Embedding.
    use_var_attn_bias:
        Whether to use BinaryAttentionBias (inter-variate bias).
    scaler:
        Input normalization scaler. ``None`` means ``PackedStdScaler``.
    dropout_p:
        Dropout probability.
    num_signal_types:
        Number of modalities: 10 — ECG(0), ABP(1), PPG(2), CVP(3), CO2(4), AWP(5),
        ICP(6), RESP_Impedance(7), RESP_Flow(8), PAP(9).
    use_modality_embed:
        Whether to add the per-modality (signal_type) embedding to each token.
    next_block_size:
        Number of future patches (K) each position predicts in parallel. At position
        n, the raw patches at n+1 ... n+K are predicted simultaneously (not
        autoregressively) from ``encoded[n]``.
    next_head_d_inner:
        Inner dimension of ``BlockNextHead``'s trunk. ``None`` means ``d_model``.
    contrastive_proj_dim:
        Output dim of the pretraining contrastive projection head. 0 disables it.
        Inference never uses it; it is kept so pretrained checkpoints load with no
        missing/unexpected keys.
    d_cond:
        Width of the conditioning vector fed to every LSCNorm.
    cond_trend_mode:
        ``"none"`` — the conditioning input is the window ``[loc, scale]``.
        ``"patchls"`` — additionally each patch's own mean and std, in units of the
        window scale, which gives the conditioning path time resolution.
    mask_cond_trend:
        Zero the per-patch statistics at [MASK]-replaced positions; they describe
        the hidden patch itself and would otherwise leak its content.
    gate_unitless_cond:
        Block the conditioning of the modalities in ``gated_cond_signal_types``.
        Their absolute amplitude is set by device gain, not physiology.
    gated_cond_signal_types:
        Modalities to gate. ``None`` means ECG(0), PPG(2), RESP_Impedance(7).
    gate_absolute_only:
        Gate only the window ``[loc, scale]`` columns and keep the per-patch
        (relative, gain-invariant) columns. Otherwise the whole conditioning output
        is zeroed for gated modalities.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        patch_size: int,
        stride: int | None = None,
        num_heads: int | None = None,
        num_groups: int | None = None,
        use_glu: bool = True,
        use_rope: bool = True,
        use_var_attn_bias: bool = True,
        scaler: PackedScaler | None = None,
        dropout_p: float = 0.0,
        num_signal_types: int = 10,
        use_modality_embed: bool = True,
        next_block_size: int = 4,
        next_head_d_inner: int | None = None,
        contrastive_proj_dim: int = 0,
        d_cond: int = 16,
        cond_trend_mode: str = "none",
        mask_cond_trend: bool = True,
        gate_unitless_cond: bool = False,
        gated_cond_signal_types: list[int] | tuple[int, ...] | None = None,
        gate_absolute_only: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.num_signal_types = num_signal_types
        self.d_cond = d_cond
        if cond_trend_mode not in ("none", "patchls"):
            raise ValueError(f"unsupported cond_trend_mode: {cond_trend_mode!r}")
        self.cond_trend_mode = cond_trend_mode
        self.mask_cond_trend = mask_cond_trend
        self.gate_unitless_cond = gate_unitless_cond
        self.gate_absolute_only = gate_absolute_only
        self._gated_signal_types = (
            tuple(gated_cond_signal_types)
            if gated_cond_signal_types is not None
            else (0, 2, 7)
        )

        # 1. Scaler (point-level)
        self.scaler = scaler or PackedStdScaler()

        # 2. Patch embedding
        self.patch_embed = PatchEmbedding(
            patch_size=patch_size,
            d_model=d_model,
            stride=stride,
        )
        # RoPE position interpolation for overlapping-stride inference: positions
        # become time_id * stride / patch_size, the physical spacing seen in
        # training. No effect when stride == patch_size.
        self.rope_pi = True

        # 3. Transformer encoder
        num_heads = num_heads or d_model // 64

        var_attn_bias_layer: Callable | None = None
        if use_var_attn_bias:
            var_attn_bias_layer = partial(BinaryAttentionBias)

        time_qk_proj_layer: Callable | None = None
        if use_rope:
            time_qk_proj_layer = partial(
                QueryKeyProjection,
                proj_layer=partial(RotaryProjection),
            )

        self.encoder = TransformerEncoder(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            num_groups=num_groups,
            use_glu=use_glu,
            var_attn_bias_layer=var_attn_bias_layer,
            time_qk_proj_layer=time_qk_proj_layer,
            dropout_p=dropout_p,
            d_cond=self.d_cond,
        )

        # 4. Modality embedding (one embedding per signal_type)
        self.use_modality_embed = use_modality_embed
        if use_modality_embed:
            self.signal_type_embed = nn.Embedding(num_signal_types, d_model)

        # 5. Loc/scale AdaLN conditioning (preserves per-patient absolute-level info).
        # (loc, scale[, patch mean, patch std]) -> d_cond vector -> LSCNorm
        # modulation input of every encoder layer. The non-linearity is what gives
        # the conditioning its expressiveness.
        cond_in_dim = 2 + (2 if cond_trend_mode == "patchls" else 0)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_in_dim, self.d_cond),
            nn.SiLU(),
            nn.Linear(self.d_cond, self.d_cond),
        )

        # 6. Reconstruction head (reconstruct own variate)
        self.head = nn.Linear(d_model, patch_size)

        # 7. Block next-patch prediction head (shared trunk + K horizon heads).
        # A non-linear trunk plus per-horizon specialization predicts long horizons
        # better than a single Linear(d_model, K * patch_size).
        self.next_block_size = next_block_size
        self.next_head = BlockNextHead(
            d_model=d_model,
            patch_size=patch_size,
            block_size=next_block_size,
            d_inner=next_head_d_inner,
        )

        # 8. Cross-modal prediction heads (one Linear per target signal type)
        self.cross_heads = nn.ModuleDict({
            str(st): nn.Linear(d_model, patch_size)
            for st in range(num_signal_types)
        })

        # 9. Contrastive projection head (pretraining only; see the class docstring)
        self.contrastive_proj_dim = contrastive_proj_dim
        if contrastive_proj_dim > 0:
            self.contrastive_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, contrastive_proj_dim),
            )

        # 10. Learnable [MASK] token
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    @classmethod
    def from_config(cls, config: ModelConfig) -> CARMEN:
        """Create a model instance from a ModelConfig."""
        valid_params = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
        kwargs = {
            f.name: getattr(config, f.name)
            for f in fields(config)
            if f.name in valid_params
        }
        return cls(**kwargs)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
    ) -> CARMEN:
        """Load a pretrained CARMEN from a checkpoint, in eval mode.

        The checkpoint embeds its own ``ModelConfig``, so the architecture is
        rebuilt automatically. Use ``DownstreamModelWrapper`` instead when you also
        want freezing, pooling, or LoRA.
        """
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "config" not in state:
            raise ValueError(f"Checkpoint has no 'config' key: {checkpoint_path}")
        model = cls.from_config(ModelConfig.from_dict(state["config"]))
        missing, unexpected = model.load_state_dict(
            state["model_state_dict"], strict=False
        )
        if missing:
            print(f"  [CARMEN] Missing keys: {missing}")
        if unexpected:
            print(f"  [CARMEN] Unexpected keys: {unexpected}")
        return model.to(device).eval()

    # ── Encode pipeline ────────────────────────────────────────────

    def _encode(
        self,
        batch: PackedBatch,
        task: str = "masked",
        extra_content_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Common encoding pipeline.

        Parameters
        ----------
        batch:
            PackedBatch produced by PackCollate.
        task:
            ``"masked"`` for bidirectional attention, ``"next_pred"`` for causal.
        extra_content_mask:
            ``(B, N)`` bool — patch positions whose content is replaced by the
            learned [MASK] token. Used to blank out data gaps (NaN-filled regions)
            while keeping their modality / loc / scale conditioning intact.

        Returns
        -------
        dict with keys:
            ``encoded``: ``(B, N, d_model)`` — encoded patch representations.
            ``patches``: ``(B, N, patch_size)`` — raw (normalized) patches.
            ``patch_signal_types``: ``(B, N)`` — per-patch signal type.
            ``loc``: ``(B, L, 1)`` — per-variate location.
            ``scale``: ``(B, L, 1)`` — per-variate scale.
            ``patch_mask``: ``(B, N)`` — valid-patch mask.
            ``patch_sample_id``: ``(B, N)`` — per-patch sample_id.
            ``patch_variate_id``: ``(B, N)`` — per-patch variate_id.
            ``time_id``: ``(B, N)`` — patch index within its variate (drives RoPE).
            ``abs_time_id``: ``(B, N)`` — absolute-time bucket, shared by patches of
            different variates that cover the same physical time.
        """
        # 1. Scaler: point-level normalization
        values = batch.values.unsqueeze(-1)  # (B, L, 1)
        loc, scale = self.scaler(
            values,
            sample_id=batch.sample_id,
            variate_id=batch.variate_id,
        )
        normalized = ((values - loc) / scale.clamp(min=1e-8)).squeeze(-1)  # (B, L)

        # 2. Patchify (extract raw patches before projection)
        patches, p_sid, p_vid, time_id, patch_mask = self.patch_embed.patchify(
            normalized, batch.sample_id, batch.variate_id
        )
        # patches: (B, N, patch_size)

        b = patches.shape[0]
        device = patches.device

        # 3. Per-patch signal type + absolute time id
        patch_signal_types: torch.Tensor | None = None
        abs_time_id = time_id  # fallback when signal types are unavailable

        if batch.signal_types is not None:
            per_row_max_var = p_vid.max(dim=-1).values  # (B,)
            var_offsets = torch.zeros(b, dtype=torch.long, device=device)
            if b > 1:
                var_offsets[1:] = per_row_max_var[:-1].cumsum(dim=0)
            global_var_idx = var_offsets.unsqueeze(-1) + (p_vid - 1)  # (B, N)
            global_var_idx = global_var_idx.clamp(min=0)

            patch_signal_types = batch.signal_types.to(device)[global_var_idx]  # (B, N)

            # abs_time_id: quantize absolute time by patch_size so patches of
            # different variates covering the same physical time share an id.
            # time_id stays variate-relative because RoPE needs it that way.
            if getattr(batch, "start_samples", None) is not None:
                patch_start = batch.start_samples.to(device)[global_var_idx]  # (B, N)
                abs_time = patch_start + time_id * self.patch_size  # (B, N)
                abs_time_id = abs_time // self.patch_size  # (B, N)
                abs_time_id[~patch_mask] = 0

        # 4. Projection — patch content representation only
        patch_embed = self.patch_embed.project(patches)  # (B, N, d_model)

        # Padding mask (p_vid == 0 is a padding token)
        valid_token = (p_vid > 0).unsqueeze(-1)  # (B, N, 1)

        # 5. Conditioning, computed separately from patch content so that masked
        # positions still carry their own modality / level information after the
        # [MASK] token overwrites the content.
        cond = torch.zeros_like(patch_embed)
        if self.use_modality_embed and patch_signal_types is not None:
            cond = cond + self.signal_type_embed(patch_signal_types)  # (B, N, d_model)

        n = patch_embed.shape[1]
        stride = self.patch_embed.stride
        patch_starts = torch.arange(n, device=device) * stride  # (N,)
        patch_starts = patch_starts.clamp(max=loc.shape[1] - 1)
        patch_loc = loc[:, patch_starts, :]  # (B, N, 1)
        patch_scale = scale[:, patch_starts, :]  # (B, N, 1)

        # AdaLN input: window [loc, scale], plus per-patch [mean, std] for "patchls".
        # Patches are already window-normalized, so their mean/std are relative to
        # the window loc/scale (dimensionless, invariant to device gain).
        cond_stats = torch.cat([patch_loc, patch_scale], dim=-1)  # (B, N, 2)
        n_abs = cond_stats.shape[-1]  # window-level columns
        if self.cond_trend_mode == "patchls":
            cond_stats = torch.cat([
                cond_stats,
                patches.mean(dim=-1, keepdim=True).to(cond_stats.dtype),
                patches.std(dim=-1, keepdim=True).to(cond_stats.dtype),
            ], dim=-1)  # (B, N, 4)

        gated: torch.Tensor | None = None  # (B, N) — patches whose cond is gated
        if self.gate_unitless_cond and patch_signal_types is not None:
            gated = torch.zeros_like(patch_signal_types, dtype=torch.bool)
            for st in self._gated_signal_types:
                gated |= patch_signal_types == st
            if self.gate_absolute_only:
                # Zero only the window-level columns; the relative patch columns stay.
                keep = torch.ones_like(cond_stats)
                keep[..., :n_abs] = (~gated).unsqueeze(-1).to(cond_stats.dtype)
                cond_stats = cond_stats * keep

        def _build_ada(cs: torch.Tensor) -> torch.Tensor:
            """(B, N, C) conditioning statistics -> (B, N, d_cond) AdaLN vector."""
            ada = self.cond_proj(cs)  # (B, N, d_cond)
            if gated is not None and not self.gate_absolute_only:
                # Gate the output, not the input: an input-side zero would still
                # pass cond_proj's bias through.
                ada = ada * (~gated).unsqueeze(-1)
            return ada * valid_token  # padding positions -> 0

        ada_cond = _build_ada(cond_stats)
        cond = cond * valid_token

        # Per-patch statistics describe the patch itself, so at [MASK]-replaced
        # positions they would hand the hidden content back. Zero them there.
        if (
            task == "masked"
            and self.mask_cond_trend
            and cond_stats.shape[-1] > n_abs
            and extra_content_mask is not None
        ):
            cs = cond_stats.clone()
            cs[..., n_abs:] = torch.where(
                extra_content_mask.unsqueeze(-1),
                torch.zeros_like(cs[..., n_abs:]),
                cs[..., n_abs:],
            )
            ada_cond = _build_ada(cs)

        # 6. Base attention mask: attend only within the same sample, valid patches only
        attn_mask = (
            (p_sid.unsqueeze(-1) == p_sid.unsqueeze(-2))
            & patch_mask.unsqueeze(-2)
            & patch_mask.unsqueeze(-1)
        )  # (B, N, N)
        if task == "next_pred":
            # Causal over physical time (abs_time_id), not packed index: with several
            # variates in a row, a later-packed variate must not see an earlier one's
            # future. Same-time cross-modal attention is allowed.
            attn_mask = attn_mask & (
                abs_time_id.unsqueeze(-1) >= abs_time_id.unsqueeze(-2)
            )

        # 7. Encoder input: [MASK]-replace the requested positions, then add conditioning
        x = patch_embed
        if extra_content_mask is not None:
            mask_token = self.mask_token.expand_as(patch_embed)
            x = torch.where(extra_content_mask.unsqueeze(-1), mask_token, patch_embed)
        x = x + cond

        # RoPE uses the variate-relative index, interpolated to physical spacing
        # when patches overlap.
        if self.patch_embed.stride < self.patch_size and self.rope_pi:
            rope_time_id = time_id.to(torch.float32) * (
                self.patch_embed.stride / self.patch_size
            )
        else:
            rope_time_id = time_id

        # One variate per sample (collate_mode="ci"): the variate bias is a constant
        # within every attended block, so the encoder skips it.
        single_variate = not bool((p_vid > 1).any())
        encoded = self.encoder(
            x,
            attn_mask=attn_mask,
            var_id=None if single_variate else p_vid,
            time_id=rope_time_id,
            cond=ada_cond,
        )

        return {
            "encoded": encoded,
            "patches": patches,
            "patch_signal_types": patch_signal_types,
            "loc": loc,
            "scale": scale,
            "patch_mask": patch_mask,
            "patch_sample_id": p_sid,
            "patch_variate_id": p_vid,
            "time_id": time_id,
            "abs_time_id": abs_time_id,
        }

    # ── Forward ────────────────────────────────────────────────────

    def forward(
        self,
        batch: PackedBatch,
        task: str = "masked",
        extra_content_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the encoder and the heads selected by ``task``.

        Parameters
        ----------
        batch:
            PackedBatch produced by PackCollate.
        task:
            ``"masked"`` — bidirectional attention; adds ``reconstructed`` and
            ``cross_pred_per_type``.
            ``"next_pred"`` — causal attention; adds ``next_pred``.
        extra_content_mask:
            ``(B, N)`` bool — patch positions to replace with the [MASK] token.

        Returns
        -------
        dict
            Everything ``_encode`` returns — except that ``time_id`` here carries
            ``abs_time_id`` (the absolute-time bucket used to pair modalities) —
            plus the task-specific head outputs.
        """
        enc = self._encode(batch, task=task, extra_content_mask=extra_content_mask)
        encoded = enc["encoded"]  # (B, N, d_model)

        out_dict: dict[str, torch.Tensor] = {
            "encoded": encoded,
            "patches": enc["patches"],
            "patch_signal_types": enc["patch_signal_types"],
            "loc": enc["loc"],
            "scale": enc["scale"],
            "patch_mask": enc["patch_mask"],
            "patch_sample_id": enc["patch_sample_id"],
            "patch_variate_id": enc["patch_variate_id"],
            "time_id": enc["abs_time_id"],  # absolute time, for cross-modal matching
        }

        if task == "masked":
            # ── Masked reconstruction ──
            out_dict["reconstructed"] = self.head(encoded)  # (B, N, patch_size)
            # Cross-modal prediction, one head per target type
            out_dict["cross_pred_per_type"] = torch.stack([
                self.cross_heads[str(st)](encoded)
                for st in range(self.num_signal_types)
            ], dim=2)  # (B, N, num_signal_types, patch_size)
            if self.contrastive_proj_dim > 0:
                out_dict["contrastive_z"] = self.contrastive_proj(encoded)
        elif task == "next_pred":
            # ── Block next-patch prediction ──
            # encoded[n] -> the K future patches n+1 ... n+K, predicted in parallel.
            out_dict["next_pred"] = self.next_head(encoded)  # (B, N, K, patch_size)
        else:
            raise ValueError(f'task must be "masked" or "next_pred", got {task!r}')

        return out_dict

    # ── Inference API ──────────────────────────────────────────────

    @torch.no_grad()
    def extract_features(self, batch: PackedBatch) -> dict[str, torch.Tensor]:
        """Extract features for downstream tasks (bidirectional attention).

        Parameters
        ----------
        batch:
            PackedBatch produced by PackCollate.

        Returns
        -------
        dict with keys:
            ``encoded``, ``patch_mask``, ``loc``, ``scale``,
            ``patch_sample_id``, ``patch_variate_id``.
        """
        self.eval()
        out = self.forward(batch, task="masked")
        out.pop("reconstructed", None)
        out.pop("cross_pred_per_type", None)
        return out
