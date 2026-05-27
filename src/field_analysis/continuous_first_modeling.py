from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from field_analysis.continuous_phase_support import continuous_peak_alignment_metadata, interpolate_signal
from field_analysis.finite_second_modeling_stabilization import smooth_measured_field_for_second_modeling
from field_analysis.finite_second_modeling_stabilization import stabilize_correction_delta
from field_analysis.finite_second_modeling_tail import compute_second_modeling_gain


def build_continuous_phase_aligned_command_profile(
    steady_state_one_cycle_frame: pd.DataFrame,
    *,
    support_frame: pd.DataFrame | None = None,
    freq_hz: float,
    waveform_type: str | None = None,
    correction_gain: float = 0.25,
    correction_gain_mode: str = "auto",
    voltage_limit_v: float = 5.0,
    base_voltage_peak_v: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = steady_state_one_cycle_frame.copy()
    if frame.empty:
        return frame, {
            "continuous_first_modeling_available": False,
            "continuous_first_modeling_status": "empty_steady_state_window",
        }
    time_s = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
    time_origin = float(np.nanmin(time_s[np.isfinite(time_s)])) if np.isfinite(time_s).any() else 0.0
    time_s = time_s - time_origin
    support = support_frame.copy() if isinstance(support_frame, pd.DataFrame) and not support_frame.empty else frame.copy()
    measured = _first_numeric_column(frame, ("measured_field_normalized_mT", "normalized_measured_field_mT")).to_numpy(dtype=float)
    target = _first_numeric_column(frame, ("normalized_physical_target_output_mT", "physical_target_output_mT")).to_numpy(dtype=float)
    source_voltage_series, source_voltage_column = _continuous_first_voltage(frame)
    source_voltage = source_voltage_series.to_numpy(dtype=float)
    first_voltage = source_voltage.copy()
    voltage_norm_meta: dict[str, Any] = {
        "continuous_base_voltage_peak_v": _peak_abs(first_voltage),
        "continuous_final_voltage_limit_v": float(voltage_limit_v),
        "source_voltage_raw_peak_v": _peak_abs(source_voltage),
        "source_voltage_base_normalized_peak_v": _peak_abs(first_voltage),
        "source_voltage_base_normalization_scale": 1.0,
        "continuous_base_voltage_headroom_v": float(max(abs(voltage_limit_v) - _peak_abs(first_voltage), 0.0)),
        "continuous_voltage_normalization_mode": "raw_source_voltage_field_scale_pending",
    }
    active_mask = np.isfinite(time_s) & np.isfinite(measured) & np.isfinite(target)
    target_peak_reference_mT = _peak_abs(target[active_mask])
    if not np.isfinite(target_peak_reference_mT) or target_peak_reference_mT <= 1e-12:
        target_peak_reference_mT = 50.0
    support_time = pd.to_numeric(support["time_s"], errors="coerce").to_numpy(dtype=float) - time_origin
    support_measured = _first_numeric_column(support, ("measured_field_normalized_mT", "normalized_measured_field_mT")).to_numpy(dtype=float)
    support_voltage_series, support_voltage_column = _continuous_first_voltage(support)
    support_voltage_source = support_voltage_series.to_numpy(dtype=float)
    support_voltage = support_voltage_source.copy()
    support_mask = np.isfinite(support_time) & np.isfinite(support_measured)
    support_smoothed, smoothing_meta = smooth_measured_field_for_second_modeling(
        support_time,
        support_measured,
        support_time,
        support_mask,
        freq_hz=float(freq_hz),
        cycle_count=1.0,
    )
    measured_smoothed = interpolate_signal(time_s, support_time, support_smoothed)
    period_s = 1.0 / max(float(freq_hz), 1e-12)
    peak_meta = continuous_peak_alignment_metadata(support_time, support_voltage, support_smoothed, period_s=period_s)
    delay_s = float(peak_meta.get("continuous_phase_delay_s") or 0.0)
    aligned_source_time = time_s + delay_s
    measured_aligned = interpolate_signal(aligned_source_time, support_time, support_smoothed)
    aligned_finite = np.isfinite(measured_aligned)
    aligned_finite_ratio = float(np.mean(aligned_finite)) if aligned_finite.size else 0.0
    support_start_s = float(np.nanmin(support_time[np.isfinite(support_time)])) if np.isfinite(support_time).any() else float("nan")
    support_end_s = float(np.nanmax(support_time[np.isfinite(support_time)])) if np.isfinite(support_time).any() else float("nan")
    output_start_s = float(np.nanmin(time_s[np.isfinite(time_s)])) if np.isfinite(time_s).any() else 0.0
    output_end_s = float(np.nanmax(time_s[np.isfinite(time_s)])) if np.isfinite(time_s).any() else period_s
    support_status = "ok" if aligned_finite_ratio >= 0.999999 else "incomplete"
    alignment_meta = {
        "phase_alignment_status": "ok" if support_status == "ok" else "unavailable_phase_support_incomplete",
        "continuous_phase_alignment_method": "field_peak_to_voltage_peak",
        "continuous_first_modeling_phase_reference": "voltage_peak",
        "phase_alignment_fallback_used": False,
        "measurement_support_grid_separate_from_output_grid": True,
        "continuous_output_grid_start_s": output_start_s,
        "continuous_output_grid_end_s": output_end_s,
        "continuous_output_period_s": period_s,
        "continuous_support_grid_start_s": support_start_s,
        "continuous_support_grid_end_s": support_end_s,
        "measured_support_start_s": support_start_s,
        "measured_support_end_s": support_end_s,
        "aligned_source_time_min_s": float(np.nanmin(aligned_source_time[np.isfinite(aligned_source_time)])) if np.isfinite(aligned_source_time).any() else None,
        "aligned_source_time_max_s": float(np.nanmax(aligned_source_time[np.isfinite(aligned_source_time)])) if np.isfinite(aligned_source_time).any() else None,
        "aligned_measured_finite_ratio": aligned_finite_ratio,
        "aligned_measured_support_status": support_status,
        "continuous_residual_invalid_detected": support_status != "ok",
        "continuous_residual_invalid_time_ranges": _invalid_ranges(time_s, ~aligned_finite),
        "continuous_nan_to_zero_used": False,
        "continuous_phase_support_incomplete_blocked": support_status != "ok",
        **peak_meta,
    }
    if alignment_meta["phase_alignment_status"] != "ok":
        return pd.DataFrame(), {
            **smoothing_meta,
            **alignment_meta,
            **voltage_norm_meta,
            "continuous_first_modeling_available": False,
            "continuous_first_modeling_status": "unavailable_phase_support_incomplete",
            "continuous_first_modeling_uses_phase_aligned_kernel": True,
            "continuous_first_modeling_tail_disabled": True,
            "continuous_first_modeling_cycle_count": 1.0,
            "continuous_loop_output": True,
            "continuous_modeling_kernel_source": "finite_second_modeling_shared_kernel",
            "continuous_target_shape": "fixed_rounded_triangle",
            "continuous_target_cycle_count": 1.0,
            "continuous_export_cycle_count": 1.0,
            "loop_endpoint_policy": "period_exclusive",
        }
    measured_for_modeling = measured_aligned
    residual = target - measured_for_modeling
    measured_peak = _peak_abs(measured_for_modeling[active_mask])
    field_scale = _frame_first_numeric(frame, "measured_field_normalization_scale_to_target_mT")
    field_scale_source = "extraction_field_normalization_scale_to_target"
    if field_scale is None or not np.isfinite(field_scale) or field_scale <= 1e-12:
        field_scale = target_peak_reference_mT / measured_peak if measured_peak > 1e-12 else 1.0
        field_scale_source = "aligned_measured_peak_fallback"
    first_voltage = source_voltage * field_scale
    voltage_norm_meta.update(
        {
            "continuous_base_voltage_peak_v": _peak_abs(first_voltage),
            "source_voltage_base_normalized_peak_v": _peak_abs(first_voltage),
            "source_voltage_base_normalization_scale": float(field_scale),
            "continuous_base_voltage_headroom_v": float(max(abs(voltage_limit_v) - _peak_abs(first_voltage), 0.0)),
            "continuous_input_voltage_field_scale": float(field_scale),
            "continuous_voltage_normalization_mode": "raw_source_voltage_scaled_by_target_field_peak",
            "continuous_input_voltage_field_scale_source": field_scale_source,
            "continuous_source_voltage_column": source_voltage_column,
            "continuous_support_voltage_column": support_voltage_column,
            "continuous_base_voltage_source": "raw_input_voltage_scaled_by_field_peak",
        }
    )
    unit_delta = residual / target_peak_reference_mT * float(voltage_limit_v)
    gain, gain_meta = compute_second_modeling_gain(
        unit_delta,
        first_voltage,
        active_mask,
        manual_gain=float(correction_gain),
        gain_mode=correction_gain_mode,
        voltage_limit_v=float(voltage_limit_v),
        tail_mask=np.zeros_like(active_mask, dtype=bool),
    )
    if str(correction_gain_mode).lower() == "manual":
        continuous_gain_meta = _continuous_gain_meta(float(gain), first_voltage, unit_delta, voltage_limit_v=float(voltage_limit_v), clamped=False)
    else:
        gain, continuous_gain_meta = _apply_headroom_gain_limit(
            gain,
            unit_delta,
            first_voltage,
            voltage_limit_v=float(voltage_limit_v),
            safety_factor=0.7,
        )
    raw_delta = unit_delta * float(gain)
    correction_delta, stabilization_meta, arrays = stabilize_correction_delta(
        raw_delta,
        first_voltage,
        time_s,
        active_mask,
        freq_hz=float(freq_hz),
        cycle_count=1.0,
        enabled=True,
        tail_mask=np.zeros_like(active_mask, dtype=bool),
    )
    first_modeled = first_voltage + correction_delta
    limited = np.clip(first_modeled, -float(voltage_limit_v), float(voltage_limit_v))
    clipping_meta = _clipping_metadata(first_modeled, limited, voltage_limit_v=float(voltage_limit_v))
    command = pd.DataFrame(
        {
            "time_s": time_s,
            "base_voltage_v": first_voltage,
            "source_voltage_v": source_voltage,
            "first_modeled_voltage_v": first_modeled,
            "limited_voltage_v": limited,
            "correction_delta_v": correction_delta,
            "raw_correction_delta_v": raw_delta,
            "smoothed_correction_delta_v": arrays.get("smoothed_correction_delta_v", correction_delta),
            "measured_field_smoothed_mT": measured_smoothed,
            "measured_field_aligned_mT": measured_aligned,
            "target_field_1cycle_mT": target,
            "residual_for_modeling_mT": residual,
            "continuous_loop_output": True,
            "loop_endpoint_policy": "period_exclusive",
            "continuous_export_cycle_count": 1.0,
            "freq_hz": float(freq_hz),
            "waveform_type": waveform_type,
        }
    )
    metadata = {
        **smoothing_meta,
        **alignment_meta,
        **gain_meta,
        **continuous_gain_meta,
        "correction_gain_used": float(gain),
        **stabilization_meta,
        **voltage_norm_meta,
        **clipping_meta,
        "continuous_first_modeling_available": True,
        "continuous_first_modeling_uses_phase_aligned_kernel": True,
        "continuous_first_modeling_tail_disabled": True,
        "continuous_first_modeling_cycle_count": 1.0,
        "continuous_loop_output": True,
        "continuous_modeling_kernel_source": "finite_second_modeling_shared_kernel",
        "continuous_target_shape": "fixed_rounded_triangle",
        "continuous_target_cycle_count": 1.0,
        "field_modeling_normalization_reference_mT": target_peak_reference_mT,
        "continuous_residual_unit_delta_reference_mT": target_peak_reference_mT,
        "continuous_export_cycle_count": 1.0,
        "continuous_final_voltage_limit_v": float(voltage_limit_v),
        "loop_endpoint_policy": "period_exclusive",
        "fourier_resynthesis_involved": False,
        "harmonic_export_involved": False,
    }
    return command.reset_index(drop=True), metadata


def _first_numeric_column(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    raise ValueError(f"missing one of required columns: {columns}")


def _continuous_first_voltage(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    # Continuous first modeling must start from the actual source input voltage.
    # Normalized/limited columns are fallbacks only for legacy frames that no
    # longer carry raw upload voltage.
    for column in (
        "raw_voltage_v",
        "Voltage1_V",
        "command_voltage_v",
        "actual_drive_voltage_v",
        "voltage_normalized_v",
        "limited_voltage_v",
    ):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce"), column
    return pd.Series(np.zeros(len(frame), dtype=float)), "missing_voltage_fallback_zero"


def _peak_abs(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.nanmax(np.abs(finite))) if finite.size else 0.0


def _frame_first_numeric(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[0])


def _normalize_base_voltage(values: np.ndarray, *, base_peak_v: float, final_limit_v: float) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.asarray(values, dtype=float)
    finite = source[np.isfinite(source)]
    raw_peak = float(np.nanmax(np.abs(finite))) if finite.size else 0.0
    target_peak = float(min(abs(base_peak_v), abs(final_limit_v)))
    scale = target_peak / raw_peak if raw_peak > 1e-12 else 1.0
    base = source * scale
    base_peak = float(np.nanmax(np.abs(base[np.isfinite(base)]))) if np.isfinite(base).any() else 0.0
    return base, {
        "continuous_base_voltage_peak_v": target_peak,
        "continuous_final_voltage_limit_v": float(final_limit_v),
        "source_voltage_raw_peak_v": raw_peak,
        "source_voltage_base_normalized_peak_v": base_peak,
        "source_voltage_base_normalization_scale": float(scale),
        "continuous_base_voltage_headroom_v": float(max(abs(final_limit_v) - base_peak, 0.0)),
        "continuous_voltage_normalization_mode": "base_peak_to_configured_peak",
    }


def _apply_headroom_gain_limit(
    gain: float,
    unit_delta: np.ndarray,
    base_voltage: np.ndarray,
    *,
    voltage_limit_v: float,
    safety_factor: float,
) -> tuple[float, dict[str, Any]]:
    base = np.asarray(base_voltage, dtype=float)
    unit = np.asarray(unit_delta, dtype=float)
    active = np.isfinite(base) & np.isfinite(unit)
    headroom = np.maximum(float(voltage_limit_v) - np.abs(base[active]), 0.0)
    headroom_safe = float(np.nanpercentile(headroom, 20)) if headroom.size else 0.0
    unit_peak = float(np.nanpercentile(np.abs(unit[active]), 95)) if active.any() else 0.0
    target_delta_peak = headroom_safe * float(safety_factor)
    headroom_gain = target_delta_peak / max(unit_peak, 1e-12) if unit_peak > 1e-12 else float(gain)
    used = float(min(float(gain), headroom_gain))
    return used, {
        "continuous_correction_gain_mode": "auto_headroom_limited",
        "continuous_correction_gain_used": used,
        "continuous_gain_headroom_safe_v": headroom_safe,
        "continuous_gain_target_delta_peak_v": target_delta_peak,
        "continuous_auto_gain_clamped_by_headroom": bool(used < float(gain) - 1e-12),
    }


def _continuous_gain_meta(gain: float, base_voltage: np.ndarray, unit_delta: np.ndarray, *, voltage_limit_v: float, clamped: bool) -> dict[str, Any]:
    active = np.isfinite(base_voltage) & np.isfinite(unit_delta)
    headroom = np.maximum(float(voltage_limit_v) - np.abs(base_voltage[active]), 0.0)
    unit_peak = float(np.nanpercentile(np.abs(unit_delta[active]), 95)) if active.any() else 0.0
    headroom_safe = float(np.nanpercentile(headroom, 20)) if headroom.size else 0.0
    return {
        "continuous_correction_gain_mode": "manual",
        "continuous_correction_gain_used": float(gain),
        "continuous_gain_headroom_safe_v": headroom_safe,
        "continuous_gain_target_delta_peak_v": unit_peak * float(gain),
        "continuous_auto_gain_clamped_by_headroom": bool(clamped),
    }


def _clipping_metadata(unclipped: np.ndarray, limited: np.ndarray, *, voltage_limit_v: float) -> dict[str, Any]:
    raw = np.asarray(unclipped, dtype=float)
    clipped = np.abs(raw) > float(voltage_limit_v) + 1e-12
    count = int(np.sum(clipped))
    fraction = float(count / max(raw.size, 1))
    status = "severe" if fraction >= 0.05 else ("warning" if count else "ok")
    warning = (
        "Continuous 1차 command가 전압 제한에 걸립니다. base voltage peak 또는 gain을 낮추십시오."
        if status in {"warning", "severe"}
        else None
    )
    return {
        "continuous_unclipped_voltage_peak_v": float(np.nanmax(np.abs(raw[np.isfinite(raw)]))) if np.isfinite(raw).any() else 0.0,
        "continuous_clipping_fraction": fraction,
        "continuous_clipping_warning": warning,
        "continuous_voltage_clip_sample_count": count,
        "continuous_voltage_clip_fraction": fraction,
        "continuous_voltage_clip_status": status,
    }


def _invalid_ranges(time_s: np.ndarray, invalid_mask: np.ndarray) -> list[dict[str, float]]:
    time = np.asarray(time_s, dtype=float)
    invalid = np.asarray(invalid_mask, dtype=bool) & np.isfinite(time)
    ranges: list[dict[str, float]] = []
    indices = np.flatnonzero(invalid)
    if indices.size == 0:
        return ranges
    start = int(indices[0])
    previous = int(indices[0])
    for index in indices[1:]:
        idx = int(index)
        if idx != previous + 1:
            ranges.append({"start_s": float(time[start]), "end_s": float(time[previous])})
            start = idx
        previous = idx
    ranges.append({"start_s": float(time[start]), "end_s": float(time[previous])})
    return ranges
