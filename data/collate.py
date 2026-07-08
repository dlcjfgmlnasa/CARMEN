# -*- coding:utf-8 -*-
from __future__ import annotations

import heapq
import math
import random
from collections import defaultdict
from dataclasses import dataclass

import torch

from data.dataset import BiosignalSample


@dataclass
class PackedBatch:
    """Output of PackCollate.

    Attributes
    ----------
    values:
        Packed signal values. ``(batch, max_length)``.
        Leftover positions are filled with 0.
    sample_id:
        ID indicating which original sample each time-step belongs to.
        1-based within a row; 0 means padding. ``(batch, max_length)``.
    variate_id:
        The variate ID each time-step belongs to (1-based, 0=padding).
        ``(batch, max_length)``.
    lengths:
        Per-variate original length. ``(total_variates,)``.
    sampling_rates:
        Per-variate sampling rate (Hz). ``(total_variates,)``.
    signal_types:
        Per-variate signal type. ``(total_variates,)``.
    padded_lengths:
        Per-variate patch-aligned length. ``(total_variates,)``.
        Present only when patching is configured; each variate's length is
        rounded up to a multiple of ``patch_size``. ``None`` if unused.
    """

    values: torch.Tensor  # (batch, max_length)
    sample_id: torch.Tensor  # (batch, max_length) long
    variate_id: torch.Tensor  # (batch, max_length) long
    lengths: torch.Tensor  # (total_variates,) long
    sampling_rates: torch.Tensor  # (total_variates,) float
    signal_types: torch.Tensor  # (total_variates,) long
    padded_lengths: torch.Tensor | None = None  # (total_variates,) long
    start_samples: torch.Tensor | None = None  # (total_variates,) long — absolute start sample


# ── Internal data structures ────────────────────────────────────────────


@dataclass
class _PackUnit:
    """Unit of FFD packing. Variates of the same group concatenated into one sequence."""

    values: torch.Tensor  # (time,)
    total_length: int
    channel_spans: list[tuple[int, int, int]]  # [(ch_idx, start, end), ...]
    variate_rates: list[float]
    variate_types: list[int]
    variate_lengths: list[int]
    padded_variate_lengths: list[int]
    variate_start_samples: list[int]


class PackCollate:
    """Bin-packing collate: pack variable-length time series with no gaps.

    Groups channels of the same ``(recording_idx, win_start)`` or
    ``(session_id, physical_time)`` and packs them into rows via the FFD algorithm.

    Parameters
    ----------
    max_length:
        Fixed row width of the output tensor. Samples longer than this are cut.
    collate_mode:
        Grouping mode. ``"any_variate"`` (default) or ``"ci"`` (channel-independent).
    patch_size:
        Fixed patch size.
    stride:
        Patch stride (supports overlapping). Used together with ``patch_size``.

    Packing strategy: First-Fit Decreasing (FFD)
    """

    def __init__(
        self,
        max_length: int,
        collate_mode: str = "any_variate",
        patch_size: int | None = None,
        stride: int | None = None,
        slot_size: int = 60000,
        min_patches: int = 5,
    ) -> None:
        self.patch_size = patch_size
        # Slot size for cross-modal grouping (same slot = same sample_id)
        self._slot_size = slot_size
        # Minimum variate length (in patches) for cross-modal matching in any_variate mode
        # Clinical criterion 10s (patch_size=200 * 5 / 100Hz)
        self._min_patches = min_patches

        if patch_size is not None:
            self.stride = stride if stride is not None else patch_size
            assert patch_size % self.stride == 0, (
                f"patch_size({patch_size}) must be a multiple of stride({self.stride})."
            )
            max_length = max(patch_size, -(-max_length // self.stride) * self.stride)
        else:
            self.stride = None

        self.max_length = max_length
        self.collate_mode = collate_mode

    def __call__(self, samples: list[BiosignalSample]) -> PackedBatch:
        # 1. Grouping: key determined by collate_mode
        groups: dict[tuple, list[BiosignalSample]] = defaultdict(list)
        for i, s in enumerate(samples):
            if self.collate_mode == "ci":
                key = (i,)  # unique key -> no cross-channel grouping
            elif s.session_id:
                # Group by session_id + time slot
                # Different signal types in the same slot -> same sample_id -> cross-modal pair
                abs_sample = s.start_sample + s.win_start
                slot = abs_sample // self._slot_size
                key = (s.session_id, slot)
            else:
                key = (s.recording_idx, s.win_start)
            groups[key].append(s)

        # 2. Sort each group then concatenate into one PackUnit
        units: list[_PackUnit] = []
        for _key, group_samples in groups.items():
            group_samples.sort(key=lambda s: (s.signal_type, s.channel_idx))

            # Any-Variate mode: Multi-tier length truncate
            # - variates shorter than 10s are clinically meaningless -> removed from the group
            # - among the rest, randomly pick a valid tier (keeps >=2 variates)
            # - keep only variates at least as long as the chosen tier and truncate to tier length
            # - result: all variates in a row have the same length -> perfect cross-modal pairing
            # CI mode is unaffected (each group has 1 variate, so the condition never triggers)
            group_limit: int | None = None
            if (
                self.collate_mode == "any_variate"
                and self.patch_size is not None
                and len(group_samples) >= 2
            ):
                ps = self.patch_size
                min_required = self._min_patches * ps

                # Compute each variate's post-trim effective length
                sample_effs: list[tuple[BiosignalSample, int]] = []
                for s in group_samples:
                    abs_start = s.start_sample + s.win_start
                    remainder = abs_start % ps
                    trim = (ps - remainder) if remainder > 0 else 0
                    eff_len = s.values.shape[0] - trim
                    if eff_len >= min_required:
                        sample_effs.append((s, eff_len))

                if len(sample_effs) >= 2:
                    # Unique lengths aligned to patch multiples
                    candidate_lengths = sorted(
                        {(eff // ps) * ps for _, eff in sample_effs}
                    )
                    # Valid tier: a tier with >=2 variates at least that long
                    valid_tiers = [
                        L
                        for L in candidate_lengths
                        if sum(1 for _, eff in sample_effs if eff >= L) >= 2
                    ]
                    if valid_tiers:
                        # Sqrt-length-weighted selection — slightly prefer longer tiers
                        # Validated on real VitalDB data (crop ON):
                        # sqrt gives mean ~5min, median 5min, 48% concentrated in 300-600s
                        # -> no extreme short/long, distribution centered on clinical context
                        chosen = random.choices(
                            valid_tiers,
                            weights=[math.sqrt(L) for L in valid_tiers],
                            k=1,
                        )[0]
                        group_samples = [
                            s for s, eff in sample_effs if eff >= chosen
                        ]
                        group_limit = chosen
                    else:
                        group_samples = [s for s, _ in sample_effs]
                elif len(sample_effs) == 1:
                    # Only 1 left -> cross-modal impossible, pack as a single variate
                    group_samples = [s for s, _ in sample_effs]
                else:
                    # All shorter than 10s -> exclude the group entirely
                    continue

            channel_values: list[torch.Tensor] = []  # each (time,)
            channel_spans: list[tuple[int, int, int]] = []
            variate_rates: list[float] = []
            variate_types: list[int] = []
            variate_lengths: list[int] = []
            padded_variate_lengths: list[int] = []
            variate_start_samples: list[int] = []
            offset = 0

            # [Optimization] Check length before concat and drop the overflow
            for s in group_samples:
                remaining = self.max_length - offset
                if remaining <= 0:
                    break  # already reached max_length

                # Absolute start sample
                abs_start = s.start_sample + s.win_start

                # Align to a common time grid: round abs_start up to a multiple of patch_size
                # Trim the front so every variate's patch boundary aligns at the same absolute time
                trim = 0
                if self.patch_size is not None:
                    remainder = abs_start % self.patch_size
                    if remainder > 0:
                        trim = self.patch_size - remainder
                        abs_start += trim  # round up to a multiple of patch_size

                values = s.values[trim:]  # trim the front
                seg_len = min(values.shape[0], remaining)
                # Multi-tier truncate: cap at tier length in any_variate mode
                if group_limit is not None:
                    seg_len = min(seg_len, group_limit)

                if seg_len <= 0:
                    continue

                # Determine per-variate patch parameters
                var_p: int | None = None
                var_s: int | None = None
                if self.patch_size is not None:
                    var_p = self.patch_size
                    var_s = self.stride

                if var_p is not None:
                    # Give up this variate if not even 1 patch fits
                    if remaining < var_p:
                        break
                    if seg_len < var_p:
                        continue

                    # FLOOR seg_len to a valid patch length (var_p + k*var_s)
                    # — avoids zero-padding of partial patches (removes viz/train artifacts)
                    seg_len = var_p + ((seg_len - var_p) // var_s) * var_s
                    # Also floor the remaining-based cap (remaining is already stride-aligned)
                    remaining_valid = var_p + ((remaining - var_p) // var_s) * var_s
                    seg_len = min(seg_len, remaining_valid)

                    padded_seg_len = seg_len       # no more zero-padding
                    v = values[:seg_len]           # pure real signal
                    effective_len = padded_seg_len
                else:
                    v = values[:seg_len]
                    effective_len = seg_len
                    padded_seg_len = seg_len

                channel_spans.append((s.channel_idx, offset, offset + effective_len))
                channel_values.append(v)
                variate_rates.append(s.sampling_rate)
                variate_types.append(s.signal_type)
                variate_lengths.append(seg_len)
                padded_variate_lengths.append(padded_seg_len)
                variate_start_samples.append(abs_start)
                offset += effective_len

            # Concat is already <= max_length, so no re-trim needed
            if channel_values:  # empty-group check
                concat = torch.cat(channel_values)

                units.append(
                    _PackUnit(
                        values=concat,
                        total_length=concat.shape[0],
                        channel_spans=channel_spans,
                        variate_rates=variate_rates,
                        variate_types=variate_types,
                        variate_lengths=variate_lengths,
                        padded_variate_lengths=padded_variate_lengths,
                        variate_start_samples=variate_start_samples,
                    )
                )

        # 3. FFD packing
        bins = self._ffd_pack(units)

        # 4. Build tensors — allocate at final size directly (no intermediate tensors)
        n_rows = len(bins)

        padded_values = torch.zeros(n_rows, self.max_length)
        padded_ids = torch.zeros(n_rows, self.max_length, dtype=torch.long)
        padded_var_ids = torch.zeros(n_rows, self.max_length, dtype=torch.long)

        all_lengths: list[int] = []
        all_padded_lengths: list[int] = []
        all_rates: list[float] = []
        all_types: list[int] = []
        all_start_samples: list[int] = []

        for row_idx, contents in enumerate(bins):
            row_len = min(
                sum(u.total_length for u in contents),
                self.max_length,
            )
            row_offset = 0
            for local_id, unit in enumerate(contents, start=1):
                seg_len = unit.total_length
                end_offset = min(row_offset + seg_len, row_len)
                actual_seg_len = end_offset - row_offset

                if actual_seg_len > 0:
                    padded_values[row_idx, row_offset:end_offset] = unit.values[
                        :actual_seg_len
                    ]
                    padded_ids[row_idx, row_offset:end_offset] = local_id

                    # Assign variate_id
                    for var_id, (_ch_idx, start, end) in enumerate(
                        unit.channel_spans, start=1
                    ):
                        var_start = max(row_offset, row_offset + start)
                        var_end = min(end_offset, row_offset + end)
                        if var_end > var_start:
                            padded_var_ids[row_idx, var_start:var_end] = var_id

                # Collect metadata
                if actual_seg_len == seg_len:  # whole unit included
                    all_lengths.extend(unit.variate_lengths)
                    all_padded_lengths.extend(unit.padded_variate_lengths)
                    all_rates.extend(unit.variate_rates)
                    all_types.extend(unit.variate_types)
                    all_start_samples.extend(unit.variate_start_samples)
                else:
                    # Truncated unit: collect only the included variates.
                    # span (start, end) is in unit-relative coordinates (0-based).
                    # The cut-off is unit-relative actual_seg_len.
                    # included = min(end, actual_seg_len) - start  (unit-relative).
                    # NOTE: compare in unit-relative coords without adding row_offset, so
                    # `included` is not miscomputed when a unit with row_offset > 0 is cut.
                    for var_id, (_ch_idx, start, end) in enumerate(unit.channel_spans):
                        seg_end_in_unit = min(end, actual_seg_len)
                        if seg_end_in_unit > start:
                            included = seg_end_in_unit - start
                            all_lengths.append(
                                min(included, unit.variate_lengths[var_id])
                            )
                            all_padded_lengths.append(included)
                            all_rates.append(unit.variate_rates[var_id])
                            all_types.append(unit.variate_types[var_id])
                            all_start_samples.append(unit.variate_start_samples[var_id])

                row_offset += actual_seg_len
                if row_offset >= row_len:
                    break

        padded_lengths_tensor = (
            torch.tensor(all_padded_lengths, dtype=torch.long)
            if self.patch_size is not None
            else None
        )

        return PackedBatch(
            values=padded_values,
            sample_id=padded_ids,
            variate_id=padded_var_ids,
            lengths=torch.tensor(all_lengths, dtype=torch.long),
            sampling_rates=torch.tensor(all_rates, dtype=torch.float32),
            signal_types=torch.tensor(all_types, dtype=torch.long),
            padded_lengths=padded_lengths_tensor,
            start_samples=torch.tensor(all_start_samples, dtype=torch.long),
        )

    # ── FFD packing ─────────────────────────────────────────────────

    def _ffd_pack(self, units: list[_PackUnit]) -> list[list[_PackUnit]]:
        """First-Fit Decreasing bin-packing (optimized). Place the longest unit first."""
        sorted_units = sorted(units, key=lambda u: u.total_length, reverse=True)

        # [Optimization] Use a version number to discard stale entries
        heap: list[tuple[int, int, int]] = []  # (-remaining, bin_idx, version)
        bin_remaining: list[int] = []
        bin_version: list[int] = []  # current version of each bin
        bin_contents: list[list[_PackUnit]] = []

        for unit in sorted_units:
            placed = False

            # [Optimization] Purge stale entries in one pass
            while heap and heap[0][2] != bin_version[heap[0][1]]:
                heapq.heappop(heap)

            if heap:
                neg_rem, bin_idx, ver = heap[0]
                remaining = -neg_rem
                if remaining >= unit.total_length:
                    heapq.heappop(heap)
                    bin_contents[bin_idx].append(unit)
                    new_remaining = remaining - unit.total_length
                    bin_remaining[bin_idx] = new_remaining
                    bin_version[bin_idx] += 1
                    if new_remaining > 0:
                        heapq.heappush(
                            heap, (-new_remaining, bin_idx, bin_version[bin_idx])
                        )
                    placed = True

            if not placed:
                bi = len(bin_contents)
                rem = self.max_length - unit.total_length
                bin_contents.append([unit])
                bin_remaining.append(rem)
                bin_version.append(1)
                if rem > 0:
                    heapq.heappush(heap, (-rem, bi, 1))

        return bin_contents
