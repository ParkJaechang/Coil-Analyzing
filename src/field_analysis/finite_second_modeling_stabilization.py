"""Smoothing, alignment, and correction stabilization helpers for second modeling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .finite_actual_drive_normalization import peak_abs
from .voltage_policy import COMMAND_VOLTAGE_LIMIT_V


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
    freq = max(float(freq_hz), 1e-12)
    pre_margin_s = 0.10 / freq
    post_margin_s = 0.5 / freq
    alignment_margin_s = 0.35 / freq
    active_start = float(np.nanmin(target_t[active_target])) if active_target.any() else 0.0
    active_end = float(np.nanmax(target_t[active_target])) if active_target.any() else active_start
    target_end = float(np.nanmax(target_t[np.isfinite(target_t)])) if np.isfinite(target_t).any() else active_end
    smooth_start = active_start - pre_margin_s
    smooth_end = target_end + post_margin_s + alignment_margin_s
    smooth_mask = finite & (source_t >= smooth_start) & (source_t <= smooth_end)
    active_count = int(smooth_mask.sum())
    base_meta: dict[str, Any] = {
        "measured_field_smoothing_enabled": True,
        "measured_field_smoothing_method": "median_then_rolling",
        "measured_field_smoothing_polyorder": None,
        "measured_field_smoothing_scope": "native_window_until_zero_return",
        "measured_field_smoothing_pre_margin_s": float(pre_margin_s),
        "measured_field_smoothing_post_margin_s": float(post_margin_s),
        "measured_field_smoothing_tail_margin_s": float(post_margin_s),
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


def measured_support_metadata(
    source_time: Any,
    *,
    freq_hz: float,
    cycle_count: float,
    tail_cycle_count: float,
    phase_alignment_shift_s: float,
    tail_enabled: bool,
    measured_support_end_s: float | None = None,
    measured_support_end_mode: str | None = None,
) -> dict[str, Any]:
    freq = max(float(freq_hz), 1e-12)
    active_duration = float(cycle_count) / freq
    tail_cycles = float(np.clip(tail_cycle_count, 0.0, 0.5)) if tail_enabled else 0.0
    tail_duration = tail_cycles / freq
    extra_cycles = 0.5
    extra_duration = extra_cycles / freq
    support_end = float(measured_support_end_s) if measured_support_end_s is not None else active_duration + tail_duration + extra_duration
    required_end = support_end + abs(float(phase_alignment_shift_s))
    source = pd.to_numeric(pd.Series(source_time), errors="coerce").to_numpy(dtype=float)
    source_end = float(np.nanmax(source[np.isfinite(source)])) if np.isfinite(source).any() else np.nan
    covers = bool(np.isfinite(source_end) and source_end >= required_end - 1e-12)
    status = "ok" if covers else "insufficient_for_zero_return"
    return {
        "measured_support_extra_cycle_count": extra_cycles,
        "measured_support_extra_duration_s": float(extra_duration),
        "measured_support_end_s": support_end,
        "measured_support_end_mode": measured_support_end_mode or "extra_cycle_margin",
        "required_measured_support_end_s": float(required_end),
        "required_source_end_s": float(required_end),
        "actual_drive_source_time_end_s": source_end,
        "measured_support_coverage_status": status,
        "measured_support_covers_aligned_tail": covers,
        "measured_support_covers_zero_return": covers,
        "aligned_measured_source_range_status": status,
        "measured_source_switch_at_active_end": False,
    }


def detect_measured_zero_return_support(
    source_time: Any,
    source_values: Any,
    *,
    freq_hz: float,
    cycle_count: float,
    requested_tail_cycle_count: float,
    tail_enabled: bool,
) -> dict[str, Any]:
    freq = max(float(freq_hz), 1e-12)
    active_end = float(cycle_count) / freq
    requested_tail = float(max(requested_tail_cycle_count, 0.0)) if tail_enabled else 0.0
    min_return_time = active_end + min(0.25 / freq, max(requested_tail / freq, 0.0))
    max_return_time = active_end + 1.0 / freq
    threshold = 2.5
    time = pd.to_numeric(pd.Series(source_time), errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(pd.Series(source_values), errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(time) & np.isfinite(values)
    source_end = float(np.nanmax(time[finite])) if finite.any() else active_end
    search_end = min(max_return_time, source_end)
    order = np.argsort(time[finite]) if finite.any() else np.array([], dtype=int)
    t = time[finite][order] if finite.any() else np.array([], dtype=float)
    y = values[finite][order] if finite.any() else np.array([], dtype=float)
    smoothed = pd.Series(y).rolling(window=7, center=True, min_periods=1).median().rolling(window=9, center=True, min_periods=1).mean().to_numpy(dtype=float) if y.size else y
    search = (t >= min_return_time - 1e-12) & (t <= search_end + 1e-12) & (np.abs(smoothed) <= threshold)
    if np.any(search):
        zero_time = float(t[np.flatnonzero(search)[0]])
        status = "detected_zero_return"
        mode = "detected_zero_return"
    elif source_end >= min_return_time:
        zero_time = float(source_end)
        status = "zero_return_not_detected_using_data_end"
        mode = "data_end"
    else:
        zero_time = active_end + requested_tail / freq
        status = "insufficient_for_zero_return"
        mode = "insufficient"
    effective_tail_cycles = max(requested_tail, max(0.0, zero_time - active_end) * freq) if tail_enabled else 0.0
    return {
        "zero_return_detection_enabled": True,
        "zero_return_threshold_mT": threshold,
        "zero_return_slope_threshold": None,
        "measured_zero_return_time_s": zero_time,
        "measured_zero_return_status": status,
        "measured_support_end_s": zero_time,
        "measured_support_end_mode": mode,
        "post_cycle_zero_tail_cycle_count_effective": float(effective_tail_cycles),
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
    tail_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    raw = np.array(raw_delta, dtype=float, copy=True)
    active_only = np.asarray(active_mask, dtype=bool)
    tail = np.asarray(tail_mask, dtype=bool) if tail_mask is not None else np.zeros_like(active_only, dtype=bool)
    active = (active_only | tail) & np.isfinite(raw)
    active_float = active.astype(float)
    arrays = {
        "smoothed_correction_delta_v": raw.copy(),
        "start_gate": active_float.copy(),
        "taper_gate": active_float.copy(),
        "correction_envelope": active_float.copy(),
        "stabilized_correction_delta_v": raw.copy(),
        "correction_nan_mask": ~np.isfinite(raw),
        "correction_active_mask": active.copy(),
        "source_range_valid_mask": np.isfinite(raw),
        "correction_invalid_mask": ~np.isfinite(raw),
        "correction_zero_flat_segment_mask": np.zeros_like(active, dtype=bool),
    }
    if not enabled or active.sum() < 2:
        delta = raw.copy()
        delta[~active | ~np.isfinite(delta)] = 0.0
        arrays["smoothed_correction_delta_v"] = delta.copy()
        arrays["stabilized_correction_delta_v"] = delta.copy()
        diagnostic_active = active & (np.abs(raw) > 1e-12)
        arrays["correction_zero_flat_segment_mask"] = _zero_flat_segment_mask(delta, diagnostic_active, arrays["start_gate"], arrays["taper_gate"])
        return delta, _stabilization_meta(False), arrays

    time = np.asarray(time_s, dtype=float)
    active_indices = np.flatnonzero(active)
    tail_indices = np.flatnonzero(tail & np.isfinite(raw))
    start_index = int(active_indices[0])
    freq = max(float(freq_hz), 1e-12)
    start_gate_duration = 0.25 / freq
    taper_duration = 0.10 / freq
    active_time = time[active]
    start_time = float(time[start_index])

    start_gate = np.zeros_like(raw)
    start_gate[active] = _smoothstep((active_time - start_time) / max(start_gate_duration, 1e-12))
    taper_gate = np.zeros_like(raw)
    taper_gate[active] = 1.0
    if tail_indices.size:
        tail_start_time = float(time[tail_indices[0]])
        tail_end_time = float(time[tail_indices[-1]])
        taper_start = max(tail_start_time, tail_end_time - taper_duration)
        taper_x = np.clip((time[tail_indices] - taper_start) / max(tail_end_time - taper_start, 1e-12), 0.0, 1.0)
        taper_gate[tail_indices] = 1.0 - _smoothstep(taper_x)
    else:
        active_end_time = float(time[active_indices[-1]])
        active_taper_start = max(start_time, active_end_time - start_gate_duration)
        active_taper_x = np.clip(
            (time[active_indices] - active_taper_start) / max(active_end_time - active_taper_start, 1e-12),
            0.0,
            1.0,
        )
        taper_gate[active_indices] = 1.0 - _smoothstep(active_taper_x)
    envelope = start_gate * taper_gate

    cleaned = raw.copy()
    cleaned[~active | ~np.isfinite(cleaned)] = 0.0
    window = _odd_window(min(max(5, int(round(active.sum() * 0.04))), 41), int(active.sum()))
    smoothed = np.array(_smooth(cleaned, window=window), dtype=float, copy=True)
    smoothed = np.where(active, smoothed, 0.0)
    stabilized = smoothed * envelope
    stabilized = np.where(active, stabilized, 0.0)
    stabilized[start_index] = 0.0
    if tail_indices.size:
        stabilized[int(tail_indices[-1])] = 0.0
    else:
        stabilized[int(active_indices[-1])] = 0.0
    arrays = {
        "smoothed_correction_delta_v": smoothed,
        "start_gate": start_gate,
        "taper_gate": taper_gate,
        "correction_envelope": envelope,
        "stabilized_correction_delta_v": stabilized,
        "correction_nan_mask": ~np.isfinite(raw),
        "correction_active_mask": active.copy(),
        "source_range_valid_mask": np.isfinite(raw),
        "correction_invalid_mask": ~np.isfinite(raw),
        "correction_zero_flat_segment_mask": _zero_flat_segment_mask(stabilized, active & (np.abs(raw) > 1e-12), start_gate, taper_gate),
    }
    zero_flat_detected = bool(np.any(arrays["correction_zero_flat_segment_mask"]))
    return stabilized, {
        **_stabilization_meta(True),
        "correction_start_gate_enabled": True,
        "correction_start_gate_cycle_fraction": 0.25,
        "correction_start_gate_duration_s": float(start_gate_duration),
        "correction_start_gate_type": "smoothstep",
        "correction_start_gate_applied_only_to_initial_segment": True,
        "correction_ramp_in_duration_s": float(start_gate_duration),
        "active_taper_out_enabled": not bool(tail_indices.size),
        "active_end_correction_preserved": bool(tail_indices.size),
        "active_end_correction_zero_taper_enabled": not bool(tail_indices.size),
        "correction_end_gate_cycle_fraction": 0.25 if not tail_indices.size else 0.0,
        "correction_end_gate_duration_s": float(start_gate_duration if not tail_indices.size else 0.0),
        "correction_end_gate_type": "smoothstep" if not tail_indices.size else "not_applied",
        "correction_tail_taper_enabled": bool(tail_indices.size),
        "correction_tail_taper_gate": "tail_segment_only" if tail_indices.size else "not_applied",
        "correction_taper_out_cycle_fraction": 0.10,
        "correction_taper_out_duration_s": float(taper_duration),
        "correction_taper_out_type": "smoothstep",
        "tail_end_taper_out_enabled": bool(tail_indices.size),
        "tail_end_taper_duration_s": float(taper_duration if tail_indices.size else 0.0),
        "tail_end_taper_cycle_fraction": 0.10 if tail_indices.size else 0.0,
        "global_start_offset_subtraction_used": False,
        "correction_zero_flat_segment_detected": zero_flat_detected,
        "correction_zero_flat_segment_time_ranges": _zero_flat_time_ranges(time, arrays["correction_zero_flat_segment_mask"]),
        "correction_zero_flat_segment_source": "mask_or_guard" if zero_flat_detected else "none",
        "correction_invalid_policy": "invalid_values_excluded_from_extended_correction_mask",
        "correction_delta_smoothing_window_samples": int(window),
    }, arrays


def apply_polarity_guard(
    second_voltage: np.ndarray,
    first_voltage: np.ndarray,
    start_gate: np.ndarray | None = None,
    *,
    enabled: bool,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    guarded = np.asarray(second_voltage, dtype=float).copy()
    first = np.asarray(first_voltage, dtype=float)
    threshold = max(0.05, float(peak_abs(first)) * 0.02)
    modified = 0
    applied_mask = np.zeros_like(guarded, dtype=bool)
    if enabled:
        before = guarded.copy()
        gate = np.asarray(start_gate, dtype=float) if start_gate is not None else np.ones_like(guarded)
        start_segment = np.isfinite(gate) & (gate > 0.0) & (gate < 0.999)
        positive = start_segment & (first >= threshold) & (guarded < 0.0)
        negative = start_segment & (first <= -threshold) & (guarded > 0.0)
        guarded[positive] = 0.0
        guarded[negative] = 0.0
        applied_mask = np.abs(guarded - before) > 1e-12
        modified = int(np.sum(applied_mask))
    flat_zero_segment = _has_flat_zero_segment(guarded, applied_mask)
    return guarded, {
        "correction_polarity_guard_enabled": bool(enabled),
        "polarity_guard_status": "ok" if enabled else "not_applied",
        "polarity_guard_mode": "start_segment_only" if enabled else "none",
        "polarity_guard_threshold_v": float(threshold),
        "polarity_guard_modified_sample_count": modified,
        "polarity_guard_discontinuity_risk": "low_start_segment_only" if enabled else "none",
        "polarity_guard_flat_zero_segment_detected": flat_zero_segment,
    }, applied_mask


def diagnose_correction_discontinuity(
    time_s: np.ndarray,
    stabilized_delta: np.ndarray,
    second_limited_voltage: np.ndarray,
    arrays: dict[str, np.ndarray],
    polarity_guard_mask: np.ndarray,
    *,
    threshold_v: float = 0.75,
) -> dict[str, Any]:
    time = np.asarray(time_s, dtype=float)
    delta = np.asarray(stabilized_delta, dtype=float)
    second = np.asarray(second_limited_voltage, dtype=float)
    delta_step = np.abs(np.diff(delta))
    second_step = np.abs(np.diff(second))
    max_delta_step = float(np.nanmax(delta_step)) if delta_step.size else 0.0
    max_second_step = float(np.nanmax(second_step)) if second_step.size else 0.0
    combined = np.maximum(
        np.nan_to_num(delta_step, nan=0.0),
        np.nan_to_num(second_step, nan=0.0),
    )
    if combined.size == 0 or float(np.nanmax(combined)) <= threshold_v:
        return {
            "correction_discontinuity_detected": False,
            "correction_discontinuity_time_s": None,
            "correction_discontinuity_source": "none",
            "max_abs_delta_step_v": max_delta_step,
            "max_abs_second_voltage_step_v": max_second_step,
            "discontinuity_threshold_v": float(threshold_v),
        }
    index = int(np.nanargmax(combined)) + 1
    source = _discontinuity_source(index, arrays, polarity_guard_mask, second)
    return {
        "correction_discontinuity_detected": True,
        "correction_discontinuity_time_s": float(time[index]) if index < time.size and np.isfinite(time[index]) else None,
        "correction_discontinuity_source": source,
        "max_abs_delta_step_v": max_delta_step,
        "max_abs_second_voltage_step_v": max_second_step,
        "discontinuity_threshold_v": float(threshold_v),
    }


def _discontinuity_source(
    index: int,
    arrays: dict[str, np.ndarray],
    polarity_guard_mask: np.ndarray,
    second_limited_voltage: np.ndarray,
) -> str:
    def changed(name: str) -> bool:
        values = np.asarray(arrays.get(name, []))
        return index > 0 and index < values.size and bool(values[index] != values[index - 1])

    if changed("tail_window_mask"):
        return "active_to_tail_transition"
    if changed("correction_active_mask"):
        return "active_mask_transition"
    if changed("source_range_valid_mask"):
        return "source_time_range_nan_to_valid"
    if changed("correction_nan_mask"):
        return "interpolation_nan"
    if changed("start_gate"):
        return "start_gate_boundary"
    if changed("taper_gate"):
        return "taper_gate_boundary"
    guard = np.asarray(polarity_guard_mask, dtype=bool)
    if index < guard.size and guard[index]:
        return "polarity_guard_start_segment"
    if index > 0 and index < second_limited_voltage.size and (
        abs(second_limited_voltage[index]) >= COMMAND_VOLTAGE_LIMIT_V - 1e-9
        or abs(second_limited_voltage[index - 1]) >= COMMAND_VOLTAGE_LIMIT_V - 1e-9
    ):
        return "voltage_clip"
    return "unknown"


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
        "correction_taper_out_enabled": False,
        "correction_start_anchor_mode": "start_gate_only_no_global_offset" if enabled else "none",
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


def _has_flat_zero_segment(values: np.ndarray, mask: np.ndarray, min_run: int = 4) -> bool:
    zeroed = np.asarray(mask, dtype=bool) & np.isclose(np.asarray(values, dtype=float), 0.0, atol=1e-12)
    run = 0
    for flag in zeroed:
        run = run + 1 if bool(flag) else 0
        if run >= int(min_run):
            return True
    return False


def _zero_flat_segment_mask(values: np.ndarray, active_mask: np.ndarray, start_gate: np.ndarray, taper_gate: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    active = np.asarray(active_mask, dtype=bool)
    start = np.asarray(start_gate, dtype=float)
    taper = np.asarray(taper_gate, dtype=float)
    interior = active & (start >= 0.999) & (taper >= 0.999)
    zero = interior & np.isclose(values, 0.0, atol=1e-12)
    result = np.zeros_like(zero, dtype=bool)
    run_start: int | None = None
    for index, flag in enumerate(zero):
        if flag and run_start is None:
            run_start = index
        if (not flag or index == len(zero) - 1) and run_start is not None:
            run_end = index if flag and index == len(zero) - 1 else index - 1
            if run_end - run_start + 1 >= 4:
                result[run_start : run_end + 1] = True
            run_start = None
    return result


def _zero_flat_time_ranges(time_s: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    time = np.asarray(time_s, dtype=float)
    flags = np.asarray(mask, dtype=bool)
    ranges: list[tuple[float, float]] = []
    run_start: int | None = None
    for index, flag in enumerate(flags):
        if flag and run_start is None:
            run_start = index
        if (not flag or index == len(flags) - 1) and run_start is not None:
            run_end = index if flag and index == len(flags) - 1 else index - 1
            if run_start < time.size and run_end < time.size:
                ranges.append((float(time[run_start]), float(time[run_end])))
            run_start = None
    return ranges


def _smoothstep(values: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _odd_window(value: int, max_size: int) -> int:
    size = max(1, min(int(value), int(max_size)))
    if size % 2 == 0:
        size = max(1, size - 1)
    return size
