# -*- coding:utf-8 -*-
"""Model configuration dataclass.

Consolidates every architectural parameter of ``CARMEN`` into a single dataclass.
Checkpoints embed a serialized ``ModelConfig``, so the architecture is restored
automatically at load time — you never specify it by hand.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields


# Config keys that were renamed. Old checkpoints still carry the old name.
_LEGACY_KEYS: dict[str, str] = {
    "use_spatial_embed": "use_modality_embed",
}


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
    use_modality_embed:
        Whether to add the per-modality (signal_type) embedding to each token.
    dropout_p:
        Dropout probability.
    num_signal_types:
        Number of modalities: 9 — ECG(0), ABP(1), PPG(2), CVP(3), CO2(4), AWP(5),
        ICP(6), RESP_Impedance(7), RESP_Flow(8).
    next_block_size:
        Number of future patches (K) each position predicts in parallel.
    next_head_d_inner:
        Inner dimension of ``BlockNextHead``'s trunk. ``None`` means ``d_model``.
    contrastive_proj_dim:
        Output dim of the pretraining contrastive projection head. 0 disables it.
    d_cond:
        Width of the (loc, scale) AdaLN conditioning vector.
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
    use_modality_embed: bool = True
    dropout_p: float = 0.0

    # Modalities
    num_signal_types: int = 9

    # Heads
    next_block_size: int = 4
    next_head_d_inner: int | None = None
    contrastive_proj_dim: int = 0

    # AdaLN conditioning (loc/scale injected into every layer's LSCNorm)
    d_cond: int = 16

    def to_dict(self) -> dict:
        """Serialize for checkpoint saving."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        """Restore a ModelConfig from a dict.

        Renamed keys are migrated; unknown keys are ignored so that checkpoints
        written by older or training-side code still load.
        """
        valid_keys = {f.name for f in fields(cls)}
        migrated = {_LEGACY_KEYS.get(k, k): v for k, v in d.items()}
        return cls(**{k: v for k, v in migrated.items() if k in valid_keys})
