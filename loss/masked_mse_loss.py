# -*- coding:utf-8 -*-
from __future__ import annotations

"""Masked Patch Modeling Loss.

Phase 1 (CI): random patch masking -> reconstruct morphology within the same variate.
Phase 2 (Any-variate): variate-level masking -> reconstruct from another modality (Virtual Sensing).
"""
import torch
from torch import nn


def _multi_resolution_stft_loss(
    pred: torch.Tensor,  # (M, P)
    target: torch.Tensor,  # (M, P)
    n_ffts: tuple[int, ...] = (16, 32, 64),
) -> torch.Tensor:
    """Multi-Resolution STFT Loss.

    Runs STFT at several n_fft sizes to compare time-frequency structure at
    multiple scales. Sums log-magnitude L1 + spectral convergence at each scale.

    Parameters
    ----------
    pred:
        Predicted patch. (M, P).
    target:
        Original patch. (M, P).
    n_ffts:
        STFT window sizes. hop_length = n_fft // 4.
    """
    loss = pred.new_tensor(0.0)
    # cuFFT half precision only supports powers of 2 -> cast to float32
    pred_f = pred.float()
    target_f = target.float()

    patch_len = pred_f.shape[-1]
    # Skip n_fft larger than patch_size
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

        # Spectral Convergence: Frobenius norm ratio
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
    """Compute Peak-Weighted MSE.

    Automatically gives higher weight to high-amplitude samples, strengthening
    reconstruction of clinically important peaks such as R-peaks and systolic
    peaks. peak_alpha=0 is equivalent to plain MSE.

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
    """Compute Peak-Weighted MSE + Multi-Resolution STFT Loss.

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
    """Loss that computes Peak-Weighted MSE only at masked patch positions.

    Automatically gives higher weight to high-amplitude samples, strengthening
    reconstruction of clinically important peaks such as R-peaks and systolic
    peaks. peak_alpha=0 is equivalent to plain MSE.

    Returns the loss over patches where pred_mask=True.
    Returns 0 if there are no masked positions.

    Parameters
    ----------
    peak_alpha:
        Peak-weight strength. 0 means plain MSE. Higher focuses more on peaks.
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


def create_patch_mask(
    patch_mask: torch.Tensor,  # (B, N) — valid patches (True=valid)
    mask_ratio: float = 0.15,
    patch_variate_id: torch.Tensor | None = None,  # (B, N)
    variate_mask_prob: float = 0.0,  # Phase 2: probability of masking a whole variate
    block_mask: bool = False,  # True = contiguous-block masking
    block_size_min: int = 3,  # minimum block size (in patches)
    block_size_max: int = 8,  # maximum block size (in patches)
) -> torch.Tensor:  # (B, N) bool — mask target (True=masked)
    """Create a patch mask.

    Parameters
    ----------
    patch_mask:
        Valid-patch mask. True=valid patch.
    mask_ratio:
        Random patch masking ratio.
    patch_variate_id:
        Per-patch variate_id. Required for variate-level masking.
    variate_mask_prob:
        Variate-level masking probability. 0 disables it (Phase 1 behavior).
        > 0 selects a random variate with this probability and masks all its patches.
    block_mask:
        If True, mask in contiguous blocks. Prevents interpolation-based
        reconstruction and forces learning long-range temporal dependencies.
    block_size_min:
        Minimum block size (in patches). Default 3 (3 seconds).
    block_size_max:
        Maximum block size (in patches). Default 8 (8 seconds).

    Returns
    -------
    torch.Tensor
        (B, N) bool. True=mask target.
    """
    b, n = patch_mask.shape
    device = patch_mask.device

    pred_mask = torch.zeros(b, n, dtype=torch.bool, device=device)

    for bi in range(b):
        valid_idx = patch_mask[bi].nonzero(as_tuple=True)[0]  # valid patch indices
        if len(valid_idx) == 0:
            continue

        # variate-level masking (Phase 2)
        if (
            variate_mask_prob > 0
            and patch_variate_id is not None
            and torch.rand(1).item() < variate_mask_prob
        ):
            valid_var_ids = patch_variate_id[bi, valid_idx]
            unique_vars = valid_var_ids[valid_var_ids > 0].unique()
            if len(unique_vars) > 1:
                chosen_var = unique_vars[torch.randint(len(unique_vars), (1,)).item()]
                var_mask = patch_variate_id[bi] == chosen_var
                pred_mask[bi] = var_mask & patch_mask[bi]
                continue

        n_valid = len(valid_idx)
        n_mask = max(1, int(n_valid * mask_ratio))

        if block_mask and n_valid >= block_size_min:
            # ── Block Masking ──
            # Find contiguous runs per variate and mask block by block.
            # After placing a block, exclude that region from the run to prevent
            # overlapping/adjacent placement.
            masked_count = 0
            # valid_idx is sorted, so extract contiguous runs
            runs = _find_contiguous_runs(valid_idx)

            while masked_count < n_mask and runs:
                # Filter to runs that can hold a block
                eligible = [
                    (i, s, l) for i, (s, l) in enumerate(runs) if l >= block_size_min
                ]
                if not eligible:
                    break

                # Pick a random run
                pick = torch.randint(0, len(eligible), (1,)).item()
                _, run_start, run_len = eligible[pick]

                bs = torch.randint(
                    block_size_min, min(block_size_max, run_len) + 1, (1,)
                ).item()
                bs = min(bs, n_mask - masked_count)  # avoid exceeding the target
                if bs < 1:
                    break

                max_start = run_len - bs
                offset = torch.randint(0, max_start + 1, (1,)).item()
                start_idx = run_start + offset
                pred_mask[bi, start_idx : start_idx + bs] = True
                masked_count += bs

                # Remove the placed block region + a 1-patch gap on each side from the run
                ri = eligible[pick][0]
                old_start, old_len = runs[ri]
                old_end = old_start + old_len
                new_runs: list[tuple[int, int]] = []
                # Left sub-run (keep a 1-patch gap)
                left_len = start_idx - old_start - 1
                if left_len > 0:
                    new_runs.append((old_start, left_len))
                # Right sub-run (keep a 1-patch gap)
                right_start = start_idx + bs + 1
                right_len = old_end - right_start
                if right_len > 0:
                    new_runs.append((right_start, right_len))
                # Replace the old run (runs shorter than block_size_min are kept too —
                # the eligible filter screens them out)
                runs = runs[:ri] + new_runs + runs[ri + 1 :]

            # If the target is not met, fill the remainder randomly
            if masked_count < n_mask:
                remaining = valid_idx[~pred_mask[bi, valid_idx]]
                if len(remaining) > 0:
                    extra = min(n_mask - masked_count, len(remaining))
                    perm = torch.randperm(len(remaining), device=device)[:extra]
                    pred_mask[bi, remaining[perm]] = True
        else:
            # ── Random Masking (default) ──
            perm = torch.randperm(n_valid, device=device)[:n_mask]
            pred_mask[bi, valid_idx[perm]] = True

    return pred_mask


def _find_contiguous_runs(
    indices: torch.Tensor,  # (K,) sorted indices
) -> list[tuple[int, int]]:
    """Return a list of contiguous runs (start, length) from sorted indices."""
    if len(indices) == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = indices[0].item()
    prev = start
    for i in range(1, len(indices)):
        cur = indices[i].item()
        if cur == prev + 1:
            prev = cur
        else:
            runs.append((start, prev - start + 1))
            start = cur
            prev = cur
    runs.append((start, prev - start + 1))
    return runs
