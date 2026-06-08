from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def phase_align_from_native_support(
    native_time_s: Any,
    native_smoothed_mT: Any,
    output_time_s: np.ndarray,
    active_mask: np.ndarray,
    *,
    phase_alignment_shift_s: float,
    tail_off_active_only: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Align measured field from native actual-drive support, not trimmed output support."""
    native_time = pd.to_numeric(pd.Series(native_time_s), errors="coerce").to_numpy(dtype=float)
    native_values = pd.to_numeric(pd.Series(native_smoothed_mT), errors="coerce").to_numpy(dtype=float)
    output_time = np.asarray(output_time_s, dtype=float)
    active = np.asarray(active_mask, dtype=bool) & np.isfinite(output_time)
    shift = float(phase_alignment_shift_s)
    source_time = output_time + shift
    aligned = _interp(native_time, native_values, source_time)

    native_finite = np.isfinite(native_time) & np.isfinite(native_values)
    source_min = float(np.nanmin(native_time[native_finite])) if native_finite.any() else np.nan
    source_max = float(np.nanmax(native_time[native_finite])) if native_finite.any() else np.nan
    active_source = source_time[active]
    required_start = float(np.nanmin(active_source)) if active_source.size else np.nan
    required_end = float(np.nanmax(active_source)) if active_source.size else np.nan
    covers_start = bool(np.isfinite(source_min) and np.isfinite(required_start) and source_min <= required_start + 1e-12)
    covers_end = bool(np.isfinite(source_max) and np.isfinite(required_end) and source_max >= required_end - 1e-12)
    active_finite = active & np.isfinite(aligned)
    finite_ratio = float(active_finite.sum() / max(int(active.sum()), 1))
    status = "ok" if covers_start and covers_end and finite_ratio >= 0.999 else "insufficient"
    if active.any() and not np.isfinite(aligned[np.flatnonzero(active)[-1]]):
        status = "insufficient_active_end"
    return aligned, {
        "measurement_support_grid_separate_from_output_grid": True,
        "measured_alignment_source": "native_smoothed_actual_drive",
        "output_command_grid_tail_off_active_only": bool(tail_off_active_only),
        "aligned_source_time_min_s": source_min,
        "aligned_source_time_max_s": source_max,
        "required_aligned_source_start_s": required_start,
        "required_aligned_source_end_s": required_end,
        "required_phase_aligned_source_end_s": required_end,
        "actual_drive_source_time_end_s": source_max,
        "aligned_source_range_status": status,
        "phase_aligned_active_support_status": status,
        "aligned_measured_active_finite_ratio": finite_ratio,
    }


def protect_active_unit_delta(
    unit_delta_v: np.ndarray,
    residual_mT: np.ndarray,
    active_mask: np.ndarray,
    correction_mask: np.ndarray,
    time_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    unit = np.asarray(unit_delta_v, dtype=float).copy()
    residual = np.asarray(residual_mT, dtype=float)
    active = np.asarray(active_mask, dtype=bool)
    correction = np.asarray(correction_mask, dtype=bool)
    invalid_active = active & (~np.isfinite(residual) | ~np.isfinite(unit))
    unit[~correction] = 0.0
    unit[(~active) & correction & ~np.isfinite(unit)] = 0.0
    ranges = _time_ranges(time_s, invalid_active)
    active_indices = np.flatnonzero(active)
    end_valid = True
    if active_indices.size:
        end_index = int(active_indices[-1])
        end_valid = bool(np.isfinite(residual[end_index]) and np.isfinite(unit[end_index]))
    return unit, invalid_active, {
        "active_residual_invalid_detected": bool(invalid_active.any()),
        "active_residual_invalid_time_ranges": ranges,
        "active_unit_delta_zero_fill_used": False,
        "nonfinite_active_residual_policy": "preserve_nan_and_warn",
        "active_end_residual_support_status": "ok" if end_valid else "missing",
    }


def diagnose_active_end_kink(
    time_s: np.ndarray,
    active_mask: np.ndarray,
    residual_mT: np.ndarray,
    unit_delta_v: np.ndarray,
    correction_delta_v: np.ndarray,
    second_limited_voltage_v: np.ndarray,
    *,
    active_invalid_mask: np.ndarray,
    phase_support_status: str,
    threshold_v: float = 0.75,
) -> dict[str, Any]:
    time = np.asarray(time_s, dtype=float)
    active = np.asarray(active_mask, dtype=bool)
    indices = np.flatnonzero(active)
    if indices.size < 2:
        return _kink_meta(False, None, "none", 0.0, 0.0, 0.0)
    end = int(indices[-1])
    prev = int(indices[-2])
    delta = np.asarray(correction_delta_v, dtype=float)
    second = np.asarray(second_limited_voltage_v, dtype=float)
    residual = np.asarray(residual_mT, dtype=float)
    unit = np.asarray(unit_delta_v, dtype=float)
    active_invalid = np.asarray(active_invalid_mask, dtype=bool)
    delta_step = _abs_step(delta, prev, end)
    second_step = _abs_step(second, prev, end)
    residual_step = _abs_step(residual, prev, end)
    source = "unknown"
    detected = bool(max(delta_step, second_step) > float(threshold_v))
    if end < active_invalid.size and active_invalid[end]:
        detected = True
        source = "residual_nan_to_zero"
    elif str(phase_support_status).startswith("insufficient"):
        detected = detected or not (np.isfinite(unit[end]) and np.isfinite(residual[end]))
        source = "phase_support_missing" if detected else "none"
    elif detected:
        source = "active_end_step"
    else:
        source = "none"
    return _kink_meta(detected, float(time[end]) if end < time.size else None, source, delta_step, second_step, residual_step)


def _kink_meta(detected: bool, time_s: float | None, source: str, delta_step: float, second_step: float, residual_step: float) -> dict[str, Any]:
    return {
        "active_end_kink_detected": bool(detected),
        "active_end_kink_time_s": time_s,
        "active_end_kink_source": source,
        "active_end_delta_step_v": float(delta_step),
        "active_end_second_voltage_step_v": float(second_step),
        "active_end_residual_step_mT": float(residual_step),
    }


def _abs_step(values: np.ndarray, prev: int, current: int) -> float:
    if current >= values.size or prev >= values.size:
        return 0.0
    if not (np.isfinite(values[current]) and np.isfinite(values[prev])):
        return float("inf")
    return float(abs(values[current] - values[prev]))


def _interp(source_time: np.ndarray, source_values: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    finite = np.isfinite(source_time) & np.isfinite(source_values)
    if finite.sum() == 0:
        return np.full(len(target_time), np.nan)
    order = np.argsort(source_time[finite])
    return np.interp(target_time, source_time[finite][order], source_values[finite][order], left=np.nan, right=np.nan)


def _time_ranges(time_s: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    time = np.asarray(time_s, dtype=float)
    flags = np.asarray(mask, dtype=bool)
    ranges: list[tuple[float, float]] = []
    start: int | None = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        if (not flag or index == len(flags) - 1) and start is not None:
            end = index if flag and index == len(flags) - 1 else index - 1
            if start < time.size and end < time.size and np.isfinite(time[start]) and np.isfinite(time[end]):
                ranges.append((float(time[start]), float(time[end])))
            start = None
    return ranges


__all__ = ["diagnose_active_end_kink", "phase_align_from_native_support", "protect_active_unit_delta"]
