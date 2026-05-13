"""Smoothing, alignment, and correction stabilization helpers for second modeling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .finite_actual_drive_normalization import peak_abs


def smooth_measured_field_for_second_modeling(
    source_time: Any,
    source_values: Any,
    target_time: np.ndarray,
    active_mask: np.ndarray,
    *,
    freq_hz: float,
    cycle_count: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    source_t = pd.to_numeric(pd.Series(source_time), errors="coerce").to_numpy(dtype=float)
    source_y = pd.to_numeric(pd.Series(source_values), errors="coerce").to_numpy(dtype=float)
    target_t = np.asarray(target_time, dtype=float)
    finite = np.isfinite(source_t) & np.isfinite(source_y)
    active_target = np.asarray(active_mask, dtype=bool) & np.isfinite(target_t)
    active_duration = float(cycle_count) / max(float(freq_hz), 1e-12)
    tail_margin_s = max(0.15 * active_duration, 0.0)
    active_start = float(np.nanmin(target_t[active_target])) if active_target.any() else 0.0
    active_end = float(np.nanmax(target_t[active_target])) if active_target.any() else active_start
    smooth_mask = finite & (source_t >= active_start - tail_margin_s) & (source_t <= active_end + tail_margin_s)
    active_count = int(smooth_mask.sum())
    base_meta: dict[str, Any] = {
        "measured_field_smoothing_enabled": True,
        "measured_field_smoothing_method": "median_then_rolling",
        "measured_field_smoothing_polyorder": None,
        "measured_field_smoothing_scope": "full_native_window_with_tail_margin",
        "measured_field_smoothing_tail_margin_s": float(tail_margin_s),
        "measured_field_smoothing_padding_mode": "edge",
        "residual_source_for_second_modeling": "smoothed_measured_field",
        "correction_delta_source": "first_model_residual_smoothed_mT",
    }
    if active_count < 3:
        return _interp(source_t, source_y, target_t), {
            **base_meta,
            "measured_field_smoothing_status": "unavailable_too_few_active_samples",
            "measured_field_smoothing_window_samples": 1,
            "measured_field_smoothing_median_window_samples": 1,
        }

    median_window = _odd_window(min(max(5, active_count // 25), 9), active_count)
    smooth_window = _odd_window(min(max(7, int(round(active_count * 0.05))), 101), active_count)
    smooth_t = source_t[smooth_mask]
    series = pd.Series(source_y[smooth_mask]).interpolate(limit_direction="both").ffill().bfill()
    rolled = series.rolling(window=median_window, center=True, min_periods=1).median()
    rolled = rolled.rolling(window=smooth_window, center=True, min_periods=1).mean()
    smoothed_target = _interp(smooth_t, rolled.to_numpy(dtype=float), target_t)
    fallback = _interp(source_t, source_y, target_t)
    return np.where(np.isfinite(smoothed_target), smoothed_target, fallback), {
        **base_meta,
        "measured_field_smoothing_status": "ok",
        "measured_field_smoothing_window_samples": int(smooth_window),
        "measured_field_smoothing_median_window_samples": int(median_window),
    }


def align_measured_field_for_residual(
    time_s: np.ndarray,
    target: np.ndarray,
    measured_smoothed: np.ndarray,
    active_mask: np.ndarray,
    *,
    freq_hz: float,
    residual_alignment_mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    measured = np.asarray(measured_smoothed, dtype=float)
    if residual_alignment_mode == "pointwise":
        return measured.copy(), {
            "residual_alignment_mode": "pointwise",
            "phase_alignment_enabled": False,
            "phase_alignment_method": "none",
            "target_first_peak_time_s": None,
            "measured_first_peak_time_s": None,
            "phase_alignment_shift_s": 0.0,
            "phase_alignment_shift_cycles": 0.0,
            "phase_alignment_status": "disabled_pointwise",
            "residual_source_for_second_modeling": "smoothed_measured_field",
            "correction_delta_source": "first_model_residual_for_second_mT",
            "residual_alignment_interpolation_status": "not_applied",
        }

    time = np.asarray(time_s, dtype=float)
    target_values = np.asarray(target, dtype=float)
    active = np.asarray(active_mask, dtype=bool) & np.isfinite(time) & np.isfinite(target_values) & np.isfinite(measured)
    target_peak = _first_positive_peak_time(time, target_values, active, freq_hz=freq_hz, allow_active_fallback=False)
    measured_peak = _first_positive_peak_time(time, measured, active, freq_hz=freq_hz, allow_active_fallback=True)
    base_meta: dict[str, Any] = {
        "residual_alignment_mode": residual_alignment_mode,
        "phase_alignment_enabled": True,
        "phase_alignment_method": "first_positive_peak",
        "target_first_peak_time_s": target_peak,
        "measured_first_peak_time_s": measured_peak,
        "residual_source_for_second_modeling": "first_peak_aligned_smoothed_measured_field",
        "correction_delta_source": "first_model_residual_for_second_mT",
    }
    if target_peak is None or measured_peak is None:
        return measured.copy(), _alignment_fallback_meta(base_meta, "peak_detection_failed")
    shift_s = float(measured_peak - target_peak)
    shift_cycles = shift_s * max(float(freq_hz), 0.0)
    if abs(shift_cycles) > 0.35:
        return measured.copy(), {
            **_alignment_fallback_meta(base_meta, "shift_too_large"),
            "phase_alignment_shift_s": shift_s,
            "phase_alignment_shift_cycles": shift_cycles,
        }
    aligned = _interp(time, measured, time + shift_s)
    return aligned, {
        **base_meta,
        "phase_alignment_shift_s": shift_s,
        "phase_alignment_shift_cycles": shift_cycles,
        "phase_alignment_status": "ok",
        "residual_alignment_interpolation_status": "ok_no_extrapolation",
    }


def stabilize_correction_delta(
    raw_delta: np.ndarray,
    first_voltage: np.ndarray,
    time_s: np.ndarray,
    active_mask: np.ndarray,
    *,
    freq_hz: float,
    cycle_count: float,
    enabled: bool,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    raw = np.asarray(raw_delta, dtype=float)
    active = np.asarray(active_mask, dtype=bool) & np.isfinite(raw)
    envelope = np.where(active, 1.0, 0.0)
    if not enabled or active.sum() < 2:
        return raw.copy(), _stabilization_meta(False), envelope

    time = np.asarray(time_s, dtype=float)
    active_indices = np.flatnonzero(active)
    start_index = int(active_indices[0])
    end_index = int(active_indices[-1])
    active_duration = float(cycle_count) / max(float(freq_hz), 1e-12)
    ramp_duration = min(0.12 * active_duration, 0.15 / max(float(freq_hz), 1e-12))
    taper_duration = min(0.10 * active_duration, 0.15 / max(float(freq_hz), 1e-12))
    active_time = time[active]
    start_time = float(time[start_index])
    end_time = float(time[end_index])
    envelope = np.zeros_like(raw)
    envelope[active] = _smoothstep((active_time - start_time) / max(ramp_duration, 1e-12)) * _smoothstep(
        (end_time - active_time) / max(taper_duration, 1e-12)
    )
    anchored = raw.copy()
    anchored[active] = anchored[active] - raw[start_index]
    anchored[~active | ~np.isfinite(anchored)] = 0.0
    window = _odd_window(min(max(5, int(round(active.sum() * 0.04))), 41), int(active.sum()))
    smoothed = _smooth(anchored * envelope, window=window)
    smoothed[~active] = 0.0
    smoothed[start_index] = 0.0
    smoothed[end_index] = 0.0
    return smoothed, {
        **_stabilization_meta(True),
        "correction_ramp_in_duration_s": float(ramp_duration),
        "correction_taper_out_duration_s": float(taper_duration),
        "correction_delta_smoothing_window_samples": int(window),
    }, envelope


def apply_polarity_guard(
    second_voltage: np.ndarray,
    first_voltage: np.ndarray,
    *,
    enabled: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    guarded = np.asarray(second_voltage, dtype=float).copy()
    first = np.asarray(first_voltage, dtype=float)
    threshold = max(0.05, float(peak_abs(first)) * 0.02)
    modified = 0
    if enabled:
        before = guarded.copy()
        guarded[first >= threshold] = np.maximum(guarded[first >= threshold], 0.0)
        guarded[first <= -threshold] = np.minimum(guarded[first <= -threshold], 0.0)
        modified = int(np.sum(np.abs(guarded - before) > 1e-12))
    return guarded, {
        "correction_polarity_guard_enabled": bool(enabled),
        "polarity_guard_status": "ok" if enabled else "not_applied",
        "polarity_guard_threshold_v": float(threshold),
        "polarity_guard_modified_sample_count": modified,
    }


def _alignment_fallback_meta(base_meta: dict[str, Any], status: str) -> dict[str, Any]:
    return {
        **base_meta,
        "phase_alignment_shift_s": 0.0,
        "phase_alignment_shift_cycles": 0.0,
        "phase_alignment_status": status,
        "residual_alignment_interpolation_status": "fallback_pointwise",
    }


def _stabilization_meta(enabled: bool) -> dict[str, Any]:
    return {
        "correction_stabilization_enabled": bool(enabled),
        "correction_zero_start_guard_enabled": bool(enabled),
        "correction_ramp_in_enabled": bool(enabled),
        "correction_taper_out_enabled": bool(enabled),
        "correction_start_anchor_mode": "force_start_zero_with_ramp" if enabled else "none",
        "correction_envelope_type": "smoothstep" if enabled else "none",
        "correction_envelope_applied": bool(enabled),
        "correction_delta_smoothing_enabled": bool(enabled),
        "correction_delta_smoothing_window_samples": 1,
        "correction_delta_smoothing_status": "ok" if enabled else "not_applied",
        "correction_slew_rate_limit_enabled": False,
    }


def _first_positive_peak_time(
    time: np.ndarray,
    values: np.ndarray,
    active_mask: np.ndarray,
    *,
    freq_hz: float,
    allow_active_fallback: bool,
) -> float | None:
    finite_active = np.asarray(active_mask, dtype=bool) & np.isfinite(time) & np.isfinite(values)
    if finite_active.sum() < 3:
        return None
    active_time = time[finite_active]
    active_start = float(np.nanmin(active_time))
    active_end = float(np.nanmax(active_time))
    first_lobe_end = min(active_start + 0.5 / max(float(freq_hz), 1e-12), active_end)
    lobe = finite_active & (time >= active_start - 1e-12) & (time <= first_lobe_end + 1e-12)
    peak_time = _peak_time_from_mask(time, values, lobe, allow_edge_peak=not allow_active_fallback)
    if peak_time is not None:
        return peak_time
    if allow_active_fallback:
        return _peak_time_from_mask(time, values, finite_active, allow_edge_peak=True)
    return None


def _peak_time_from_mask(time: np.ndarray, values: np.ndarray, mask: np.ndarray, *, allow_edge_peak: bool) -> float | None:
    candidates = np.where(mask & np.isfinite(values) & np.isfinite(time))[0]
    if candidates.size == 0:
        return None
    candidate_values = values[candidates]
    max_value = float(np.nanmax(candidate_values))
    if not np.isfinite(max_value) or max_value <= 1e-9:
        return None
    max_index = int(candidates[int(np.nanargmax(candidate_values))])
    if not allow_edge_peak and (max_index == int(candidates[0]) or max_index == int(candidates[-1])):
        return None
    return float(time[max_index])


def _interp(source_time: Any, source_values: Any, target_time: Any) -> np.ndarray:
    x = pd.to_numeric(pd.Series(source_time), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(source_values), errors="coerce").to_numpy(dtype=float)
    t = pd.to_numeric(pd.Series(target_time), errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() == 0:
        return np.full(len(t), np.nan)
    order = np.argsort(x[finite])
    return np.interp(t, x[finite][order], y[finite][order], left=np.nan, right=np.nan)


def _smooth(values: np.ndarray, window: int = 7) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rolling(window=window, center=True, min_periods=1).mean().to_numpy(dtype=float)


def _smoothstep(values: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _odd_window(value: int, max_size: int) -> int:
    size = max(1, min(int(value), int(max_size)))
    if size % 2 == 0:
        size = max(1, size - 1)
    return size
