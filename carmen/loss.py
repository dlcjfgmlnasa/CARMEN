# -*- coding:utf-8 -*-
"""Masked patch reconstruction loss.

Scores how well the model reconstructs selected patches. The pretraining objective
combined this with next-patch, cross-modal and contrastive terms; the release keeps
only the reconstruction term, which is what inference-time uses need (anomaly
scoring, reconstruction quality, fine-tuning a generation head).
"""

from __future__ import annotations

import torch
from torch import nn


def _multi_resolution_stft_loss(
    pred: torch.Tensor,  # (M, P)
    target: torch.Tensor,  # (M, P)
    n_ffts: tuple[int, ...] = (16, 32, 64),
) -> torch.Tensor:
    """Multi-resolution STFT loss.

    Runs an STFT at several ``n_fft`` sizes to compare time-frequency structure at
    multiple scales. Sums log-magnitude L1 + spectral convergence at each scale.

    Parameters
    ----------
    pred:
        Predicted patch. ``(M, P)``.
    target:
        Original patch. ``(M, P)``.
    n_ffts:
        STFT window sizes. ``hop_length = n_fft // 4``.
    """
    loss = pred.new_tensor(0.0)
    # cuFFT half precision only supports powers of 2 -> cast to float32
    pred_f = pred.float()
    target_f = target.float()

    patch_len = pred_f.shape[-1]
    # Skip any n_fft larger than the patch
    valid_ffts = [n for n in n_ffts if n <= patch_len]
    if not valid_ffts:
        return loss

    for n_fft in valid_ffts:
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=pred.device)
        pred_stft = torch.stft(
            pred_f,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )  # (M, n_fft//2+1, T)
        target_stft = torch.stft(
            target_f,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )  # (M, n_fft//2+1, T)

        pred_mag = pred_stft.abs()  # (M, F, T)
        target_mag = target_stft.abs()  # (M, F, T)

        # Spectral convergence: Frobenius norm ratio
        sc = torch.norm(target_mag - pred_mag, p="fro") / (
            torch.norm(target_mag, p="fro") + 1e-8
        )

        # Log-magnitude L1
        log_mag = (torch.log1p(pred_mag) - torch.log1p(target_mag)).abs().mean()

        loss = loss + sc + log_mag

    return loss / len(valid_ffts)


def compute_peak_weighted_mse(
    pred: torch.Tensor,  # (M, P)
    target: torch.Tensor,  # (M, P)
    peak_alpha: float = 0.0,
) -> torch.Tensor:
    """Peak-weighted MSE.

    Weights high-amplitude samples more heavily, which emphasizes clinically
    important peaks such as R-peaks and systolic peaks. ``peak_alpha=0`` is plain MSE.

    Parameters
    ----------
    pred:
        Predicted patch. ``(M, P)``.
    target:
        Original patch. ``(M, P)``.
    peak_alpha:
        Peak-weight strength. 0 means plain MSE.
    """
    if peak_alpha > 0:
        abs_target = target.abs()  # (M, P)
        max_abs = abs_target.amax(dim=-1, keepdim=True).clamp(min=1e-8)  # (M, 1)
        weight = 1.0 + peak_alpha * (abs_target / max_abs)  # (M, P)
        return (weight * (pred - target) ** 2).mean()
    return ((pred - target) ** 2).mean()


def compute_patch_loss(
    pred: torch.Tensor,  # (M, P)
    target: torch.Tensor,  # (M, P)
    peak_alpha: float = 0.0,
    lambda_spec: float = 0.0,
    spec_n_ffts: tuple[int, ...] = (16, 32, 64),
) -> dict[str, torch.Tensor]:
    """Peak-weighted MSE + multi-resolution STFT loss.

    Parameters
    ----------
    pred:
        Predicted patch. ``(M, P)``.
    target:
        Original patch. ``(M, P)``.
    peak_alpha:
        Peak-weight strength. 0 means plain MSE.
    lambda_spec:
        STFT loss weight. 0 disables it.
    spec_n_ffts:
        STFT window sizes.

    Returns
    -------
    dict with keys: ``mse``, ``spec``, ``total``.
    """
    mse = compute_peak_weighted_mse(pred, target, peak_alpha)

    if lambda_spec > 0:
        spec_loss = _multi_resolution_stft_loss(pred, target, spec_n_ffts)
        total = mse + lambda_spec * spec_loss
    else:
        spec_loss = mse.new_tensor(0.0)
        total = mse

    return {"mse": mse, "spec": spec_loss, "total": total}


class MaskedPatchLoss(nn.Module):
    """Reconstruction loss evaluated only at the selected patch positions.

    Returns the loss over patches where ``pred_mask`` is True, or 0 if none are
    selected.

    Parameters
    ----------
    peak_alpha:
        Peak-weight strength. 0 means plain MSE; higher focuses more on peaks.
    lambda_spec:
        Multi-resolution STFT loss weight. 0 disables it.
    spec_n_ffts:
        STFT window sizes.
    """

    def __init__(
        self,
        peak_alpha: float = 0.0,
        lambda_spec: float = 0.0,
        spec_n_ffts: tuple[int, ...] = (16, 32, 64),
    ) -> None:
        super().__init__()
        self.peak_alpha = peak_alpha
        self.lambda_spec = lambda_spec
        self.spec_n_ffts = spec_n_ffts

    def forward(
        self,
        reconstructed: torch.Tensor,  # (B, N, P)
        original_patches: torch.Tensor,  # (B, N, P)
        pred_mask: torch.Tensor,  # (B, N) bool
    ) -> dict[str, torch.Tensor]:
        n_masked = pred_mask.float().sum()
        if n_masked == 0:
            zero = reconstructed.new_tensor(0.0)
            return {"mse": zero, "spec": zero, "total": zero}

        pred_m = reconstructed[pred_mask]  # (M, P)
        target_m = original_patches[pred_mask]  # (M, P)

        return compute_patch_loss(
            pred_m,
            target_m,
            peak_alpha=self.peak_alpha,
            lambda_spec=self.lambda_spec,
            spec_n_ffts=self.spec_n_ffts,
        )
