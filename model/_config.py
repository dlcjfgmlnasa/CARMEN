# -*- coding:utf-8 -*-
"""Model configuration dataclass.

Consolidates every architectural parameter of ``CARMEN`` into a
single dataclass to guarantee experiment reproducibility and checkpoint
serialization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass
class ModelConfig:
    """Architecture configuration for ``CARMEN``.

    Parameters
    ----------
    d_model:
        Transformer embedding dimension.
    num_layers:
        Number of transformer encoder layers.
    patch_size:
        Patch size (number of time-steps).
    stride:
        Patch stride. ``None`` means equal to ``patch_size`` (non-overlapping).
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
    use_spatial_embed:
        Whether to use the single modality (signal_type) embedding.
        (The name is kept for backward compatibility — in v2 its meaning is
        redefined as "modality embedding". The fine-grained spatial_id embedding
        has been removed.)
    dropout_p:
        Dropout probability.
    num_signal_types:
        Number of signal types (modalities). v2: 9 (contiguous numbering after
        PAP removal on 2026-06-23) (ECG0, ABP1, PPG2, CVP3, CO24, AWP5, ICP6,
        RESP_Impedance7, RESP_Flow8).
    next_block_size:
        Number of future patches (K) each position predicts in parallel for
        Block Next Prediction.
    """

    # Architecture
    d_model: int = 64
    num_layers: int = 2
    patch_size: int = 100
    stride: int | None = None
    num_heads: int | None = None
    num_groups: int | None = None

    # Features
    use_glu: bool = True
    use_rope: bool = True
    use_var_attn_bias: bool = True
    use_spatial_embed: bool = True
    dropout_p: float = 0.0

    # Signal types (v2: single modality embedding)
    # ECG0, ABP1, PPG2, CVP3, CO24, AWP5, ICP6,
    # RESP_Impedance7, RESP_Flow8 — 9 contiguous types after PAP removal on 2026-06-23.
    num_signal_types: int = 9
    # NOTE: num_spatial_ids was removed in v2 (fine-grained spatial_id embedding dropped).
    # A leftover num_spatial_ids key in old yaml/checkpoints is ignored by from_dict.

    # Task
    next_block_size: int = 4  # Block Next Prediction (K future patches per position)
    next_head_d_inner: int | None = None  # BlockNextHead inner dim. None -> use d_model

    # Contrastive
    contrastive_proj_dim: int = 0  # 0=disabled, >0=projection head output dim

    # AdaLN conditioning (inject loc/scale into every layer as a multiplicative gate)
    # use_lscnorm=True: LSCNorm (RMSNorm + AdaLN modulation, default).
    # use_lscnorm=False: zero-freeze cond_proj / modulation so the forward pass matches
    #   plain RMSNorm exactly (for ablation; the model structure is preserved).
    use_lscnorm: bool = True
    d_cond: int = 16  # AdaLN cond vector dim (16 chosen in ablation, overridable)

    def to_dict(self) -> dict:
        """Serialize for checkpoint saving."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        """Restore a ModelConfig from a dict.

        Unknown keys are ignored for compatibility with older checkpoints.
        """
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})
