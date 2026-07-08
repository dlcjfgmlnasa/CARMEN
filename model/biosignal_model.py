# -*- coding:utf-8 -*-
"""Biosignal Foundation Model.

Pipeline: Scaler -> PatchEmbedding -> SpatialEmbedding -> TransformerEncoder -> Head.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from functools import partial

import torch
from torch import nn

from data.collate import PackedBatch
from loss.masked_mse_loss import create_patch_mask
from model._config import ModelConfig
from module.packed_scaler import PackedStdScaler, PackedScaler
from module.patch import PatchEmbedding
from module.position import BinaryAttentionBias, QueryKeyProjection, RotaryProjection
from module.transformer import TransformerEncoder


class BlockNextHead(nn.Module):
    """Shared trunk + K horizon-specific heads for Block Next Prediction.

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
    """CARMEN — Cardiorespiratory foundation model. Raw-patch reconstruction for all signals.

    Performs raw patch reconstruction identically for every signal type.
    ``_encode()`` factors out the common encoding pipeline (Scaler -> Patchify ->
    Project -> SpatialEmbed -> LocScale -> Encoder) so subclasses can extend it.

    Parameters
    ----------
    d_model:
        Transformer embedding dimension.
    num_layers:
        Number of transformer encoder layers.
    patch_size:
        Patch size (number of time-steps).
    stride:
        Patch stride (for overlapping). ``None`` means equal to ``patch_size``.
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
        Number of signal types (modalities). v2: 9 (contiguous numbering after
        PAP removal on 2026-06-23) (ecg=0, abp=1, ppg=2, cvp=3, co2=4, awp=5,
        icp=6, resp_impedance=7, resp_flow=8).
    use_spatial_embed:
        Whether to use the single modality (signal_type) embedding.
        (The name is kept for backward compatibility — in v2 its meaning is
        redefined as "modality embedding". The fine-grained spatial_id embedding
        has been removed.)
    next_block_size:
        Number of future patches (K) each position predicts in parallel for
        Block Next Prediction. At each position n, from encoded_causal[n] the
        raw patches at n+1, n+2, ..., n+K are predicted non-autoregressively and
        simultaneously.
    contrastive_proj_dim:
        Output dim of the contrastive projection head. 0 disables it.
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
        # v2 single modality embedding: ECG0, ABP1, PPG2, CVP3, CO24, AWP5, ICP6,
        # RESP_Impedance7, RESP_Flow8 (9 contiguous types after PAP removal 2026-06-23).
        num_signal_types: int = 9,
        use_spatial_embed: bool = True,
        next_block_size: int = 4,
        next_head_d_inner: int | None = None,
        contrastive_proj_dim: int = 0,
        d_cond: int = 16,
        use_lscnorm: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.num_signal_types = num_signal_types
        # d_cond: AdaLN modulation input dim (overridable hyperparameter).
        self.d_cond = d_cond

        # 1. Scaler (point-level)
        self.scaler = scaler or PackedStdScaler()

        # 2. Patch Embedding
        self.patch_embed = PatchEmbedding(
            patch_size=patch_size,
            d_model=d_model,
            stride=stride,
        )

        # 3. Transformer Encoder
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

        # 4. Modality Embedding (v2: single signal_type embedding)
        # The fine-grained spatial_id embedding is removed — only a single per-modality
        # (signal_type) embedding is used.
        # (The use_spatial_embed name is kept for backward compatibility; its meaning is
        # redefined as modality embedding.)
        self.use_spatial_embed = use_spatial_embed
        if use_spatial_embed:
            self.signal_type_embed = nn.Embedding(num_signal_types, d_model)

        # 5. Loc/Scale AdaLN Conditioning (preserve per-patient absolute-level info)
        # (loc, scale) 2D scalar -> d_cond conditioning vector -> LSCNorm modulation
        # input of every encoder layer. MLP(2 -> d_cond -> d_cond) — a non-linearity
        # gives it expressiveness.
        self.cond_proj = nn.Sequential(
            nn.Linear(2, self.d_cond),
            nn.SiLU(),
            nn.Linear(self.d_cond, self.d_cond),
        )

        # 6. Reconstruction Head (reconstruct own variate)
        self.head = nn.Linear(d_model, patch_size)

        # 7. Block Next-Patch Prediction Head (shared trunk + K horizon-specific heads)
        # - trunk: non-linear transform shared across all horizons (Linear+GELU)
        # - heads: K independent Linear projections (one per horizon)
        # A non-linearity + per-horizon specialization improves long-range prediction
        # quality over a single Linear(d, K*P).
        self.next_block_size = next_block_size
        self.next_head = BlockNextHead(
            d_model=d_model,
            patch_size=patch_size,
            block_size=next_block_size,
            d_inner=next_head_d_inner,
        )

        # 8. Cross-Modal Prediction Heads (independent Linear per target signal type)
        self.cross_heads = nn.ModuleDict({
            str(st): nn.Linear(d_model, patch_size)
            for st in range(num_signal_types)
        })

        # 9. Contrastive Projection Head (SimCLR-style 2-layer MLP)
        self.contrastive_proj_dim = contrastive_proj_dim
        if contrastive_proj_dim > 0:
            self.contrastive_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, contrastive_proj_dim),
            )

        # 10. Learnable [MASK] Token
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # 11. (Ablation) Disable LSCNorm — zero-freeze cond_proj and every
        # LSCNorm.modulation so the forward pass matches plain RMSNorm (gamma=0, beta=0).
        # The model structure is left intact; only parameters are frozen, keeping
        # checkpoint compatibility.
        self.use_lscnorm = use_lscnorm
        if not use_lscnorm:
            self._disable_lscnorm_modulation()

    def _disable_lscnorm_modulation(self) -> None:
        """Ablation: fix cond_proj and every LSCNorm.modulation to 0.

        Result: encoder LSCNorm output = norm(x) * (1+0) + 0 = norm(x) = plain RMSNorm.
        """
        from module.norm import LSCNorm

        # Force cond_proj output to always be 0 (Linear(0)=bias=0, SiLU(0)=0, Linear(0)=bias=0)
        for p in self.cond_proj.parameters():
            p.data.zero_()
            p.requires_grad = False

        # Freeze every LSCNorm.modulation to 0
        for m in self.modules():
            if isinstance(m, LSCNorm):
                m.modulation.weight.data.zero_()
                m.modulation.bias.data.zero_()
                m.modulation.weight.requires_grad = False
                m.modulation.bias.requires_grad = False

    @staticmethod
    def _sample_variate_drop(
        p_sid: torch.Tensor,  # (B, N)
        p_vid: torch.Tensor,  # (B, N)
        patch_mask: torch.Tensor,  # (B, N)
        drop_prob: float,
    ) -> torch.Tensor | None:
        """Complete Variate Dropout: fully remove one variate per row from attention.

        Returns a (B, N) bool mask — True = dropped from attention.
        Only acts on rows with 2+ variates. None if nothing was dropped.
        """
        b, n = p_vid.shape
        drop_mask = torch.zeros(b, n, dtype=torch.bool, device=p_vid.device)
        any_dropped = False
        for bi in range(b):
            if torch.rand(1).item() >= drop_prob:
                continue
            valid = patch_mask[bi]
            valid_vids = p_vid[bi][valid]
            unique_vids = valid_vids[valid_vids > 0].unique()
            if len(unique_vids) < 2:
                continue  # single variate -> dropout not possible
            # Pick one at random
            chosen = unique_vids[torch.randint(len(unique_vids), (1,))]
            drop_mask[bi] = (p_vid[bi] == chosen) & valid
            any_dropped = True
        return drop_mask if any_dropped else None

    @classmethod
    def from_config(cls, config: ModelConfig) -> CARMEN:
        """Create a model instance from a ModelConfig."""
        import inspect

        valid_params = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
        kwargs = {
            f.name: getattr(config, f.name)
            for f in fields(config)
            if f.name in valid_params
        }
        return cls(**kwargs)

    # ── Encode Pipeline ────────────────────────────────────────────

    def _encode(
        self,
        batch: PackedBatch,
        task: str = "masked",
        mask_ratio: float = 0.0,
        block_mask: bool = False,
        block_size_min: int = 3,
        block_size_max: int = 8,
        variate_mask_prob: float = 0.0,
        variate_drop_prob: float = 0.0,
        extra_content_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Common encoding pipeline: Scaler -> Patchify -> Project -> SpatialEmbed -> LocScale -> Encoder.

        Parameters
        ----------
        batch:
            PackedBatch produced by PackCollate.
        task:
            ``"masked"``: bidirectional attention.
            ``"next_pred"``: causal attention.
            ``"both"``: bidirectional + causal at once (encoder called twice,
            compatible with a single DDP forward).

        Returns
        -------
        dict with keys:
            ``encoded``: ``(B, N, d_model)`` — bidirectionally encoded patch reps (task="both"/"masked").
            ``encoded_causal``: ``(B, N, d_model)`` — causal encoding (only when task="both").
            ``patches``: ``(B, N, patch_size)`` — raw patches.
            ``patch_signal_types``: ``(B, N)`` — per-patch signal type.
            ``loc``: ``(B, L, 1)`` — per-variate location.
            ``scale``: ``(B, L, 1)`` — per-variate scale.
            ``patch_mask``: ``(B, N)`` — valid-patch mask.
            ``patch_sample_id``: ``(B, N)`` — per-patch sample_id.
            ``patch_variate_id``: ``(B, N)`` — per-patch variate_id.
            ``time_id``: ``(B, N)`` — per-patch time index.
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

        # 3. Compute global_var_idx — reused by both the CNN stem and modality embedding
        patch_signal_types: torch.Tensor | None = None

        if batch.signal_types is not None:
            per_row_max_var = p_vid.max(dim=-1).values  # (B,)
            var_offsets = torch.zeros(b, dtype=torch.long, device=device)
            if b > 1:
                var_offsets[1:] = per_row_max_var[:-1].cumsum(dim=0)
            global_var_idx = var_offsets.unsqueeze(-1) + (p_vid - 1)  # (B, N)
            global_var_idx = global_var_idx.clamp(min=0)

            patch_signal_types = batch.signal_types.to(device)[global_var_idx]  # (B, N)

            # Compute absolute-time-based abs_time_id (for cross-modal matching only).
            # time_id (relative) is kept for RoPE; abs_time_id is for the cross-modal loss.
            #
            # Within the same sample_id, subtract the minimum absolute time to get a
            # bucket-relative offset -> quantize by patch_size.
            # -> patches of different variates at the same physical time get the same abs_time_id.
            abs_time_id = time_id  # fallback
            if (
                hasattr(batch, "start_samples")
                and batch.start_samples is not None
            ):
                patch_start = batch.start_samples.to(device)[global_var_idx]  # (B, N)
                abs_time = patch_start + time_id * self.patch_size  # (B, N)
                # Quantize by patch_size — patches at the same physical time match exactly
                abs_time_id = abs_time // self.patch_size  # (B, N)
                abs_time_id[~patch_mask] = 0

        # 4. Projection (linear or CNN stem) — produces only the patch-content representation
        patch_embed = self.patch_embed.project(patches, patch_signal_types)
        # patch_embed: (B, N, d_model)

        # Padding mask (p_vid==0 is a padding token)
        valid_token = (p_vid > 0).unsqueeze(-1)  # (B, N, 1)

        # 5-6. Compute conditioning embedding (signal_type modality + loc + scale)
        # Computed separately so it survives even if mask_token overwrites patch content.
        # After applying the mask it is added back, so masked positions still keep their
        # own signal-type/level information.
        cond = torch.zeros_like(patch_embed)
        if self.use_spatial_embed and patch_signal_types is not None:
            # v2: add only the single modality (signal_type) embedding. spatial_id embedding removed.
            sig_emb = self.signal_type_embed(patch_signal_types)  # (B, N, d_model)
            cond = cond + sig_emb

        n = patch_embed.shape[1]
        stride = self.patch_embed.stride
        patch_starts = torch.arange(n, device=device) * stride  # (N,)
        patch_starts = patch_starts.clamp(max=loc.shape[1] - 1)
        patch_loc = loc[:, patch_starts, :]  # (B, N, 1)
        patch_scale = scale[:, patch_starts, :]  # (B, N, 1)

        # AdaLN: loc/scale -> cond_proj -> LSCNorm modulation input of every encoder layer
        loc_scale = torch.cat([patch_loc, patch_scale], dim=-1)  # (B, N, 2)
        ada_cond = self.cond_proj(loc_scale)  # (B, N, d_cond)
        ada_cond = ada_cond * valid_token  # padding positions -> 0
        cond = cond * valid_token  # only signal_type + spatial_id added to the token

        # 7. Build Pred Mask (random/block/variate-level)
        pred_mask: torch.Tensor | None = None
        if mask_ratio > 0 and task in ("masked", "both"):
            pred_mask = create_patch_mask(
                patch_mask,
                mask_ratio=mask_ratio,
                patch_variate_id=p_vid if variate_mask_prob > 0 else None,
                variate_mask_prob=variate_mask_prob,
                block_mask=block_mask,
                block_size_min=block_size_min,
                block_size_max=block_size_max,
            )

        # 8. Base Attention Mask: attend only within the same sample + only valid patches
        base_attn_mask = (
            (p_sid.unsqueeze(-1) == p_sid.unsqueeze(-2))
            & patch_mask.unsqueeze(-2)
            & patch_mask.unsqueeze(-1)
        )  # (B, N, n)

        # 8.5. Complete Variate Dropout: physically remove a variate from attention
        # -> during training the model experiences the "cross-pred without this variate" scenario
        # -> closes the train-inference gap for zero-shot cross-modal generation
        drop_mask: torch.Tensor | None = None
        if variate_drop_prob > 0 and self.training and task in ("masked", "both"):
            drop_mask = self._sample_variate_drop(
                p_sid, p_vid, patch_mask, variate_drop_prob
            )  # (B, N) bool — True = removed from attention
            if drop_mask is not None:
                keep = ~drop_mask  # (B, N)
                # Remove from attention: dropped tokens can neither attend nor be attended to
                base_attn_mask = base_attn_mask & keep.unsqueeze(-1) & keep.unsqueeze(-2)

        # 9. Encoder-input builder helper
        # Replace patch content with mask_token (at content_mask positions) -> add conditioning.
        # This keeps signal_type/spatial/loc/scale info at masked/dropped positions.
        def _make_input(content_mask: torch.Tensor | None) -> torch.Tensor:
            if content_mask is None:
                x = patch_embed
            else:
                mt = self.mask_token.expand_as(patch_embed)
                x = torch.where(content_mask.unsqueeze(-1), mt, patch_embed)
            return x + cond

        # 10. Encoder call depending on task
        result: dict[str, torch.Tensor] = {
            "patches": patches,
            "patch_signal_types": patch_signal_types,
            "loc": loc,
            "scale": scale,
            "patch_mask": patch_mask,
            "patch_sample_id": p_sid,
            "patch_variate_id": p_vid,
            "time_id": time_id,          # relative (for RoPE)
            "abs_time_id": abs_time_id,  # absolute (for cross-modal matching)
            "pred_mask": pred_mask,
        }

        encoder_kwargs = dict(
            var_id=p_vid, time_id=time_id, cond=ada_cond,
        )  # RoPE uses relative time_id; cond is for AdaLN (ignored if None)
        use_causal = task in ("next_pred", "both")

        # causal mask (shared by next_pred and both)
        if use_causal:
            causal_tri = torch.tril(torch.ones(n, n, dtype=torch.bool, device=device))
            causal_mask = base_attn_mask & causal_tri.unsqueeze(0)  # (B, N, N)

        # bidirectional input: replace pred_mask | drop_mask positions with mask_token
        bi_content_mask = drop_mask
        if pred_mask is not None:
            bi_content_mask = (
                pred_mask if bi_content_mask is None else (pred_mask | bi_content_mask)
            )
        # Downstream gap masking: replace patch positions that were NaN->0 filled during
        # data prep with mask_token (a downstream-finetune-only path).
        if extra_content_mask is not None:
            bi_content_mask = (
                extra_content_mask if bi_content_mask is None
                else (extra_content_mask | bi_content_mask)
            )

        if task == "both":
            result["encoded"] = self.encoder(
                _make_input(bi_content_mask),
                attn_mask=base_attn_mask,
                **encoder_kwargs,
            )
            # causal: apply drop_mask only (causal attention already blocks future info,
            # so pred_mask is unnecessary).
            result["encoded_causal"] = self.encoder(
                _make_input(drop_mask),
                attn_mask=causal_mask,
                **encoder_kwargs,
            )
        elif task == "next_pred":
            result["encoded"] = self.encoder(
                _make_input(drop_mask),
                attn_mask=causal_mask,
                **encoder_kwargs,
            )
        else:  # "masked"
            result["encoded"] = self.encoder(
                _make_input(bi_content_mask),
                attn_mask=base_attn_mask,
                **encoder_kwargs,
            )

        return result

    # ── Forward ────────────────────────────────────────────────────

    def forward(
        self,
        batch: PackedBatch,
        task: str = "masked",  # "masked" or "next_pred"
        mask_ratio: float = 0.0,
        block_mask: bool = False,
        block_size_min: int = 3,
        block_size_max: int = 8,
        variate_mask_prob: float = 0.0,
        variate_drop_prob: float = 0.0,
        extra_content_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        enc = self._encode(
            batch,
            task=task,
            mask_ratio=mask_ratio,
            block_mask=block_mask,
            block_size_min=block_size_min,
            block_size_max=block_size_max,
            variate_mask_prob=variate_mask_prob,
            variate_drop_prob=variate_drop_prob,
            extra_content_mask=extra_content_mask,
        )

        encoded = enc["encoded"]  # bidirectional (or sole encoding for single-task)
        patch_signal_types = enc["patch_signal_types"]  # (B, N) or None

        out_dict: dict[str, torch.Tensor] = {
            "encoded": encoded,
            "patches": enc["patches"],
            "patch_signal_types": patch_signal_types,
            "loc": enc["loc"],
            "scale": enc["scale"],
            "patch_mask": enc["patch_mask"],
            "patch_sample_id": enc["patch_sample_id"],
            "patch_variate_id": enc["patch_variate_id"],
            "time_id": enc["abs_time_id"],  # for cross-modal matching (absolute time)
            "pred_mask": enc["pred_mask"],
        }

        # ── Masked Reconstruction ──
        if task in ("masked", "both"):
            out_dict["reconstructed"] = self.head(encoded)  # (B, N, patch_size)
            # Per-target-type cross-modal prediction (separate heads)
            cross_pred_per_type = torch.stack([
                self.cross_heads[str(st)](encoded)
                for st in range(self.num_signal_types)
            ], dim=2)  # (B, N, num_signal_types, patch_size)
            out_dict["cross_pred_per_type"] = cross_pred_per_type
            if self.contrastive_proj_dim > 0:
                out_dict["contrastive_z"] = self.contrastive_proj(
                    encoded
                )  # (B, N, proj_dim)

        # ── Block Next-Patch Prediction ──
        # encoded_causal[n] -> K future raw patches (n+1, ..., n+K) predicted in parallel.
        # BlockNextHead (shared trunk + K heads) directly returns (B, N, K, P).
        if task in ("next_pred", "both"):
            encoded_for_next = enc.get("encoded_causal", encoded)  # (B, N, d_model)
            out_dict["next_pred"] = self.next_head(encoded_for_next)  # (B, N, K, P)

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

    @torch.no_grad()
    def generate_cross_modal(
        self,
        batch: PackedBatch,
        target_signal_type: int,
        denormalize: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Zero-shot cross-modal waveform generation (Virtual Token Injection).

        Generate the waveform of ``target_signal_type`` from the source signals in
        ``batch``. A [MASK] virtual token is injected at the target variate,
        reproducing the same situation as variate dropout during training.

        Parameters
        ----------
        batch:
            PackedBatch containing only source signals.
        target_signal_type:
            Target signal type to generate (0=ECG, 1=ABP, 2=PPG, ...).
        denormalize:
            If ``True``, denormalize with the source loc/scale (approximate).

        Returns
        -------
        dict with keys:
            ``waveform``: ``(B, N, patch_size)`` — generated target waveform.
            ``patch_mask``: ``(B, N)`` — valid-patch mask.
        """
        self.eval()

        # Forward (mask_ratio=0 -> use pure source info with no masking)
        out = self.forward(batch, task="masked", mask_ratio=0.0)

        cross_pred_per_type = out["cross_pred_per_type"]  # (B, N, T, P)
        target_pred = cross_pred_per_type[:, :, target_signal_type, :]  # (B, N, P)

        if denormalize:
            loc = out["loc"]  # (B, L, 1)
            scale = out["scale"]  # (B, L, 1)
            p = self.patch_size
            stride = self.patch_embed.stride
            n = target_pred.shape[1]
            patch_starts = torch.arange(n, device=loc.device) * stride
            patch_starts = patch_starts.clamp(max=loc.shape[1] - 1)
            patch_loc = loc[:, patch_starts, :]  # (B, N, 1)
            patch_scale = scale[:, patch_starts, :]  # (B, N, 1)
            target_pred = target_pred * patch_scale + patch_loc

        return {
            "waveform": target_pred,
            "patch_mask": out["patch_mask"],
        }

    @torch.no_grad()
    def forecast(
        self,
        batch: PackedBatch,
        denormalize: bool = True,
    ) -> torch.Tensor:
        """Block next-patch prediction (non-autoregressive).

        At each position n, predict the next K patches simultaneously.

        Parameters
        ----------
        batch:
            PackedBatch produced by PackCollate.
        denormalize:
            If ``True``, restore the original scale with the scaler's loc/scale.

        Returns
        -------
        torch.Tensor
            ``(B, N, K, patch_size)`` block prediction map.
        """
        self.eval()
        out = self.forward(batch, task="next_pred")
        pred = out["next_pred"]  # (B, N, K, patch_size)

        if denormalize:
            loc = out["loc"]  # (B, L, 1)
            scale = out["scale"]  # (B, L, 1)
            p = self.patch_size
            patch_loc = loc[:, ::p, :]  # (B, N_approx, 1)
            patch_scale = scale[:, ::p, :]  # (B, N_approx, 1)
            n = pred.shape[1]
            patch_loc = patch_loc[:, :n, :]  # (B, N, 1)
            patch_scale = patch_scale[:, :n, :]  # (B, N, 1)
            # Broadcast over K dimension
            pred = pred * patch_scale.unsqueeze(2) + patch_loc.unsqueeze(2)

        return pred

    @torch.no_grad()
    def generate(
        self,
        batch: PackedBatch,
        n_steps: int,
        denormalize: bool = True,
    ) -> torch.Tensor:
        """Block-autoregressive multi-step generation.

        The Block Next Prediction head emits K patches in one shot, so each forward
        takes all K, appends them to the input, forwards again, and repeats.
        Assumes ``collate_mode="ci"`` (single-variate-per-row).

        Parameters
        ----------
        batch:
            PackedBatch produced by PackCollate.
        n_steps:
            Number of patches to generate.
        denormalize:
            If ``True``, restore the final output to the original scale.

        Returns
        -------
        torch.Tensor
            ``(n_steps, B, patch_size)`` generated patches.
        """
        self.eval()
        p = self.patch_size
        k = self.next_block_size

        out = self.forward(batch, task="next_pred")
        loc = out["loc"]  # (B, L, 1)
        scale = out["scale"]  # (B, L, 1)
        cached_loc = loc[:, 0:1, :]  # (B, 1, 1)
        cached_scale = scale[:, 0:1, :]  # (B, 1, 1)

        generated: list[torch.Tensor] = []
        pred = out["next_pred"]  # (B, N, K, patch_size)
        patch_mask = out["patch_mask"]  # (B, N)
        b = pred.shape[0]
        last_valid_idx = patch_mask.sum(dim=-1) - 1  # (B,)
        last_valid_idx = last_valid_idx.clamp(min=0)
        arange_b = torch.arange(b, device=pred.device)
        block = pred[arange_b, last_valid_idx]  # (B, K, patch_size)

        # Append the K patches from one forward in order.
        for j in range(k):
            if len(generated) >= n_steps:
                break
            generated.append(block[:, j, :])  # (B, patch_size)

        while len(generated) < n_steps:
            # Append all K patches of the block to the input -> new prediction next forward.
            for j in range(k):
                batch = _append_patch_to_batch(batch, block[:, j, :], p)

            out = self.forward(batch, task="next_pred")
            pred = out["next_pred"]  # (B, N, K, patch_size)
            patch_mask = out["patch_mask"]
            last_valid_idx = patch_mask.sum(dim=-1) - 1
            last_valid_idx = last_valid_idx.clamp(min=0)
            block = pred[arange_b, last_valid_idx]  # (B, K, patch_size)
            for j in range(k):
                if len(generated) >= n_steps:
                    break
                generated.append(block[:, j, :])

        result = torch.stack(generated[:n_steps], dim=0)  # (n_steps, B, patch_size)

        if denormalize:
            dl = cached_loc.squeeze(-1).permute(1, 0)  # (1, B)
            ds = cached_scale.squeeze(-1).permute(1, 0)  # (1, B)
            result = result * ds.unsqueeze(-1) + dl.unsqueeze(-1)

        return result


def _append_patch_to_batch(
    batch: PackedBatch,
    new_patch: torch.Tensor,  # (B, patch_size)
    patch_size: int,
) -> PackedBatch:
    """Append a new patch to a PackedBatch.

    Assumes single-variate-per-row. Extends right padding if max_length is exceeded.

    Parameters
    ----------
    batch:
        The existing PackedBatch.
    new_patch:
        The patch to append. ``(B, patch_size)``.
    patch_size:
        Patch size.

    Returns
    -------
    PackedBatch
        The PackedBatch with the new patch appended.
    """
    b, l = batch.values.shape
    device = batch.values.device

    valid_mask = batch.sample_id > 0  # (B, L)
    valid_lengths = valid_mask.sum(dim=-1)  # (B,)

    new_end = valid_lengths + patch_size  # (B,)
    max_new_end = new_end.max().item()

    if max_new_end > l:
        pad_size = max_new_end - l
        batch = PackedBatch(
            values=torch.cat(
                [batch.values, torch.zeros(b, pad_size, device=device)], dim=-1
            ),
            sample_id=torch.cat(
                [
                    batch.sample_id,
                    torch.zeros(b, pad_size, dtype=torch.long, device=device),
                ],
                dim=-1,
            ),
            variate_id=torch.cat(
                [
                    batch.variate_id,
                    torch.zeros(b, pad_size, dtype=torch.long, device=device),
                ],
                dim=-1,
            ),
            lengths=batch.lengths,
            sampling_rates=batch.sampling_rates,
            signal_types=batch.signal_types,
            padded_lengths=batch.padded_lengths,
        )

    for i in range(b):
        start = valid_lengths[i].item()
        end = start + patch_size
        batch.values[i, start:end] = new_patch[i]
        batch.sample_id[i, start:end] = (
            batch.sample_id[i, start - 1] if start > 0 else 1
        )
        batch.variate_id[i, start:end] = (
            batch.variate_id[i, start - 1] if start > 0 else 1
        )

    return batch
