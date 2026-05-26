from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .finite_second_modeling_stabilization import smooth_measured_field_for_second_modeling, stabilize_correction_delta
from .finite_second_modeling_tail import compute_second_modeling_gain
from .finite_first_normalization import coerce_measured_field_centered, normalize_smoothed_field_to_pm50
from .finite_phase_sync_support import native_measured_support_source


def apply_finite_first_phase_sync_modeling(
    command_profile: pd.DataFrame,
    *,
    freq_hz: float,
    cycle_count: float,
    mode: str = "phase_synced",
    voltage_limit_v: float = 5.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if str(mode) == "legacy_delay_preserving":
        return command_profile.copy(deep=True), {
            "finite_first_modeling_mode": "legacy_delay_preserving",
            "finite_first_modeling_mode_default": "phase_synced",
            "finite_first_modeling_review_only": True,
            "finite_first_modeling_phase_sync_enabled": False,
            "finite_first_modeling_legacy_delay_preserving": True,
        }
    source_attrs = dict(getattr(command_profile, "attrs", {}) or {})
    frame = command_profile.copy(deep=True).reset_index(drop=True)
    frame.attrs.update(source_attrs)
    if frame.empty or "time_s" not in frame.columns:
        return frame, {
            "finite_first_modeling_mode": "phase_synced",
            "finite_first_modeling_phase_sync_enabled": False,
            "finite_first_modeling_status": "empty_command_profile",
        }
    time_s = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
    active_duration_s = float(cycle_count) / max(float(freq_hz), 1e-12)
    output_start_s = float(np.nanmin(time_s[np.isfinite(time_s)])) if np.isfinite(time_s).any() else 0.0
    output_end_s = output_start_s + active_duration_s
    active_mask = np.isfinite(time_s) & (time_s >= output_start_s - 1e-12) & (time_s <= output_end_s + 1e-12)
    target_column = _first_existing_column(
        frame,
        ("physical_target_output_mT", "target_field_mT", "target_output", "aligned_target_output"),
    )
    measured_column = _first_existing_column(
        frame,
        (
            "finite_first_actual_measured_field_mT",
            "measured_field_effective_mT",
            "measured_field_normalized_mT",
            "normalized_measured_field_mT",
            "raw_hallbz_mT",
            "HallBz",
            "HallZ",
            "bz_mT",
            "Bz_mT",
        ),
    )
    voltage_column = _first_existing_column(
        frame,
        ("limited_voltage_v", "feedback_corrected_limited_voltage_v", "recommended_voltage_v", "first_modeled_voltage_v"),
    )
    if measured_column is None:
        return frame, {
            "finite_first_modeling_mode": "phase_synced",
            "finite_first_modeling_phase_sync_enabled": False,
            "finite_first_modeling_status": "missing_actual_measured_field",
            "finite_first_measured_source_is_actual_measured": False,
            "finite_first_rejected_reference_field_source": _has_reference_like_field(frame),
            "finite_first_source_data_origin": "finite_lut_measured",
            "finite_first_uses_continuous_source": False,
            "finite_first_uses_support_reference_as_measured": False,
            "finite_first_uses_target_as_measured": False,
            "phase_kernel_source_data_origin": "finite_lut_measured",
            "phase_kernel_source_validation_status": "missing_actual_measured_field",
            "phase_kernel_reference_as_measured_allowed": False,
        }
    if target_column is None or voltage_column is None:
        return frame, {
            "finite_first_modeling_mode": "phase_synced",
            "finite_first_modeling_phase_sync_enabled": False,
            "finite_first_modeling_status": "missing_required_columns",
            "finite_first_modeling_missing_target": target_column is None,
            "finite_first_modeling_missing_measured": False,
            "finite_first_modeling_missing_voltage": voltage_column is None,
        }
    target = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(dtype=float)
    measured_raw = pd.to_numeric(frame[measured_column], errors="coerce").to_numpy(dtype=float)
    native_time_s, native_measured_raw, measured_alignment_source = native_measured_support_source(
        frame,
        fallback_time_s=time_s,
        fallback_measured=measured_raw,
    )
    source_active_start_s = _source_active_start_s(frame, native_time_s, output_start_s)
    source_time_for_output = source_active_start_s + (time_s - output_start_s)
    source_active_end_s = source_active_start_s + active_duration_s
    native_active_mask = (
        np.isfinite(native_time_s)
        & (native_time_s >= source_active_start_s - 1e-12 if np.any(active_mask) else True)
        & (native_time_s <= source_active_end_s + 1e-12)
    )
    measured_centered, measured_source_type, raw_hallbz_available, measured_center_meta = coerce_measured_field_centered(
        native_measured_raw,
        measured_column,
        native_active_mask,
    )
    base_voltage = pd.to_numeric(frame[voltage_column], errors="coerce").to_numpy(dtype=float)
    native_smoothed_unscaled, smoothing_meta = smooth_measured_field_for_second_modeling(
        native_time_s,
        measured_centered,
        native_time_s,
        native_active_mask,
        freq_hz=float(freq_hz),
        cycle_count=float(cycle_count),
    )
    native_smoothed, measured_norm_meta = normalize_smoothed_field_to_pm50(
        native_measured_raw,
        native_smoothed_unscaled,
        native_active_mask,
        measured_center_meta,
    )
    peak_detection_signal, peak_detection_meta = _phase_peak_detection_signal(native_smoothed_unscaled, native_active_mask)
    measured_peak_time, measured_peak_polarity, measured_peak_value = _dominant_peak_time(
        native_time_s,
        peak_detection_signal,
        native_active_mask,
    )
    measured_peak_rel_for_voltage = (
        float(measured_peak_time - source_active_start_s)
        if measured_peak_time is not None and np.isfinite(source_active_start_s)
        else None
    )
    voltage_peak_time, voltage_peak_polarity, voltage_peak_value = _reference_voltage_peak_for_measured_peak(
        time_s,
        base_voltage,
        active_mask,
        preferred_polarity=measured_peak_polarity,
        measured_peak_rel_s=measured_peak_rel_for_voltage,
        output_start_s=output_start_s,
    )
    phase_delay_s = 0.0
    measured_peak_plot_time = measured_peak_time
    measured_peak_source_time = measured_peak_time
    alignment_status = "peak_detection_failed"
    if voltage_peak_time is not None and measured_peak_time is not None:
        voltage_peak_rel = float(voltage_peak_time - output_start_s)
        measured_peak_rel = float(measured_peak_time - source_active_start_s)
        phase_delay_s = float(measured_peak_rel - voltage_peak_rel)
        measured_peak_plot_time = float(output_start_s + measured_peak_rel)
        measured_peak_source_time = float(measured_peak_time)
        alignment_status = "ok"
    support_margin_s = _phase_support_margin_s(time_s)
    required_end = source_active_end_s + max(phase_delay_s, 0.0) + support_margin_s
    source_end = float(np.nanmax(native_time_s[np.isfinite(native_time_s)])) if np.isfinite(native_time_s).any() else float("nan")
    support_ok = bool(np.isfinite(source_end) and source_end >= required_end - 1e-12)
    smoothed = _interp(native_time_s, native_smoothed, source_time_for_output)
    aligned = _interp(native_time_s, native_smoothed, source_time_for_output + phase_delay_s)
    active_aligned_ok = bool(np.asarray(active_mask, dtype=bool).sum() > 0 and np.isfinite(aligned[np.asarray(active_mask, dtype=bool)]).all())
    if not support_ok or not active_aligned_ok:
        invalid_residual = target - aligned
        invalid_active = np.asarray(active_mask, dtype=bool)
        invalid_count = int(invalid_active.sum())
        invalid_ratio = (
            float(np.isfinite(invalid_residual[invalid_active]).sum() / invalid_count) if invalid_count else 0.0
        )
        frame["measured_field_smoothed_mT"] = smoothed
        frame["measured_field_aligned_mT"] = aligned
        frame["residual_for_modeling_mT"] = invalid_residual
        frame["finite_first_measured_source_column"] = measured_column
        frame["finite_first_measured_source_is_actual_measured"] = True
        return frame.loc[active_mask].copy().reset_index(drop=True), {
            **smoothing_meta,
            **peak_detection_meta,
            "finite_first_modeling_status": "insufficient_phase_sync_support",
            "finite_first_modeling_mode": "phase_synced",
            "finite_first_modeling_phase_sync_enabled": False,
            "finite_first_modeling_kernel": "shared_phase_aligned",
            "finite_first_measured_source_column": measured_column,
            "finite_first_measured_source_is_actual_measured": True,
            "finite_first_source_data_origin": "finite_lut_measured",
            "finite_first_uses_continuous_source": False,
            "finite_first_uses_support_reference_as_measured": False,
            "finite_first_uses_target_as_measured": False,
            "phase_sync_required_source_end_s": required_end,
            "phase_sync_actual_source_end_s": source_end,
            "phase_sync_support_status": "insufficient",
            "measured_alignment_source": measured_alignment_source,
            "measurement_support_grid_separate_from_output_grid": measured_alignment_source.startswith("selected_support_source_native"),
            "measurement_support_source_sample_count": int(np.isfinite(native_time_s).sum()),
            "required_phase_aligned_source_end_s": required_end,
            "actual_source_time_end_s": source_end,
            "phase_support_status": "insufficient",
            "phase_sync_peak_reference": "dominant_absolute_peak",
            "phase_sync_voltage_reference": "nearest_same_polarity_peak_to_measured_peak",
            "phase_sync_peak_polarity": measured_peak_polarity,
            "voltage_first_peak_time_s": voltage_peak_time,
            "measured_first_peak_time_s": measured_peak_plot_time,
            "measured_first_peak_source_time_s": measured_peak_source_time,
            "voltage_peak_value_v": voltage_peak_value,
            "measured_peak_value_mT": measured_peak_value,
            "phase_delay_s": phase_delay_s,
            "phase_delay_cycles": phase_delay_s * float(freq_hz),
            "phase_sync_support_margin_s": support_margin_s,
            "active_residual_finite_through_end": False,
            "active_residual_finite_ratio": invalid_ratio,
            "active_end_kink_detected": True,
            "nonfinite_active_residual_policy": "block_or_warning",
            "phase_kernel_source_data_origin": "finite_lut_measured",
            "phase_kernel_source_validation_status": "insufficient_phase_sync_support",
            "phase_kernel_reference_as_measured_allowed": False,
            "phase_sync_source_active_start_s": source_active_start_s,
            "phase_sync_source_active_end_s": source_active_end_s,
        }
    finite_active = active_mask & np.isfinite(aligned) & np.isfinite(target)
    residual = target - aligned
    active_residual_valid = np.asarray(active_mask, dtype=bool) & np.isfinite(residual)
    active_count = int(np.asarray(active_mask, dtype=bool).sum())
    active_residual_finite_ratio = float(active_residual_valid.sum() / active_count) if active_count else 0.0
    measured_on_output = _interp(native_time_s, native_smoothed, source_time_for_output)
    identity_meta = _merge_identity_metadata(
        _measured_target_identity_metadata(target, measured_on_output, active_mask),
        _measured_target_identity_metadata(target, aligned, finite_active),
    )
    unit_delta = residual / 50.0 * float(voltage_limit_v)
    unit_delta[~finite_active] = np.nan
    gain, gain_meta = compute_second_modeling_gain(
        unit_delta,
        base_voltage,
        finite_active,
        manual_gain=0.25,
        gain_mode="auto",
        voltage_limit_v=float(voltage_limit_v),
        tail_mask=np.zeros_like(finite_active, dtype=bool),
    )
    raw_delta = unit_delta * float(gain)
    correction_delta, stabilization_meta, arrays = stabilize_correction_delta(
        raw_delta,
        base_voltage,
        time_s,
        finite_active,
        freq_hz=float(freq_hz),
        cycle_count=float(cycle_count),
        enabled=True,
        tail_mask=np.zeros_like(finite_active, dtype=bool),
    )
    modeled = base_voltage + correction_delta
    limited = np.clip(modeled, -float(voltage_limit_v), float(voltage_limit_v))
    frame["finite_first_base_voltage_v"] = base_voltage
    frame["first_modeled_voltage_v"] = modeled
    frame["limited_voltage_v"] = limited
    frame["correction_delta_v"] = correction_delta
    frame["raw_correction_delta_v"] = raw_delta
    frame["smoothed_correction_delta_v"] = arrays.get("smoothed_correction_delta_v", correction_delta)
    frame["measured_field_smoothed_mT"] = smoothed
    frame["measured_field_aligned_mT"] = aligned
    frame["residual_for_modeling_mT"] = residual
    frame["phase_delay_s"] = phase_delay_s
    frame["phase_delay_cycles"] = phase_delay_s * float(freq_hz)
    frame["phase_sync_enabled"] = True
    frame["phase_sync_method"] = "field_peak_to_voltage_peak"
    frame["finite_first_modeling_kernel"] = "shared_phase_aligned"
    frame["finite_first_modeling_mode"] = "phase_synced"
    frame["finite_first_measured_source_column"] = measured_column
    frame["finite_first_measured_source_is_actual_measured"] = True
    output_frame = frame.loc[active_mask].copy().reset_index(drop=True)
    metadata = {
        **smoothing_meta,
        **peak_detection_meta,
        **gain_meta,
        **stabilization_meta,
        **identity_meta,
        "finite_first_modeling_status": "ok" if support_ok and alignment_status == "ok" else alignment_status,
        "finite_first_modeling_mode": "phase_synced",
        "finite_first_modeling_mode_default": "phase_synced",
        "finite_first_modeling_phase_sync_enabled": True,
        "finite_first_modeling_source_waveform_family": str(frame.get("waveform_type", pd.Series(["triangle"])).iloc[0]),
        "finite_first_modeling_target_shape": "fixed_rounded_triangle",
        "finite_first_modeling_kernel": "shared_phase_aligned",
        "finite_first_modeling_legacy_delay_preserving": False,
        "finite_first_legacy_trace_hidden_by_default": True,
        "finite_first_modeling_cycle_count": float(cycle_count),
        "finite_first_measured_source_type": measured_source_type,
        "finite_first_measured_source_file": _first_text_value(frame, "finite_first_measured_source_file"),
        "finite_first_measured_source_label": _first_text_value(frame, "finite_first_measured_source_label") or _first_text_value(frame, "source_file"),
        "finite_first_measured_source_column": measured_column,
        "finite_first_measured_source_is_actual_measured": True,
        "finite_first_rejected_reference_field_source": False,
        "finite_first_source_data_origin": "finite_lut_measured",
        "finite_first_uses_continuous_source": False,
        "finite_first_uses_support_reference_as_measured": False,
        "finite_first_uses_target_as_measured": False,
        "raw_hallbz_available": raw_hallbz_available,
        "phase_kernel_source_data_origin": "finite_lut_measured",
        "phase_kernel_source_validation_status": "ok",
        "phase_kernel_reference_as_measured_allowed": False,
        "phase_sync_enabled": True,
        "phase_sync_method": "field_peak_to_voltage_peak",
        "phase_sync_peak_reference": "dominant_absolute_peak",
        "phase_sync_voltage_reference": "nearest_same_polarity_peak_to_measured_peak",
        "phase_sync_peak_polarity": measured_peak_polarity,
        "voltage_first_peak_time_s": voltage_peak_time,
        "measured_first_peak_time_s": measured_peak_plot_time,
        "measured_first_peak_source_time_s": measured_peak_source_time,
        "voltage_peak_polarity": voltage_peak_polarity,
        "voltage_peak_value_v": voltage_peak_value,
        "measured_peak_value_mT": measured_peak_value,
        "phase_delay_s": phase_delay_s,
        "phase_delay_cycles": phase_delay_s * float(freq_hz),
        "residual_for_modeling_source": "phase_aligned_measured",
        "modeling_kernel": "shared_phase_aligned",
        "phase_sync_required_source_end_s": required_end,
        "phase_sync_actual_source_end_s": source_end,
        "phase_sync_support_status": "ok" if support_ok else "insufficient",
        "phase_sync_source_active_start_s": source_active_start_s,
        "phase_sync_source_active_end_s": source_active_end_s,
        "measured_alignment_source": "native_smoothed_source",
        "measurement_support_grid_separate_from_output_grid": measured_alignment_source.startswith("selected_support_source_native"),
        "measurement_support_source": measured_alignment_source,
        "measurement_support_source_sample_count": int(np.isfinite(native_time_s).sum()),
        "phase_sync_support_margin_s": support_margin_s,
        **measured_norm_meta,
        "measured_aligned_normalized_peak_mT": _peak_abs(aligned[active_mask]),
        "residual_gain_field_scale_applied": True,
        "harmonic_inverse_field_scale_applied_or_not_used": "harmonic_inverse_not_used_for_final_export",
        "source_voltage_raw_peak_v": _peak_abs(base_voltage),
        "source_voltage_base_normalized_peak_v": _peak_abs(base_voltage),
        "base_voltage_peak_setting_v": _peak_abs(base_voltage),
        "final_voltage_limit_v": float(voltage_limit_v),
        "voltage_headroom_v": max(float(voltage_limit_v) - _peak_abs(base_voltage), 0.0),
        "correction_delta_peak_v": _peak_abs(correction_delta),
        "clipping_fraction": float(np.mean(np.abs(modeled) > float(voltage_limit_v) + 1e-12)) if len(modeled) else 0.0,
        "clipping_status": "warning" if np.any(np.abs(modeled) > float(voltage_limit_v) + 1e-12) else "ok",
        "phase_alignment_support_window_s": max(required_end - float(np.nanmin(time_s[np.isfinite(time_s)])), 0.0)
        if np.isfinite(time_s).any()
        else float("nan"),
        "required_phase_aligned_source_end_s": required_end,
        "actual_source_time_end_s": source_end,
        "phase_support_status": "ok" if support_ok and active_residual_finite_ratio >= 0.999 else "insufficient",
        "active_residual_finite_through_end": bool(np.isfinite(residual[active_mask]).all()) if np.any(active_mask) else False,
        "active_residual_finite_ratio": active_residual_finite_ratio,
        "active_end_kink_detected": _active_end_kink_detected(limited, residual, active_mask),
        "nonfinite_active_residual_policy": "block_or_warning",
        "command_voltage_scaling_mode": "normalized_modeling_with_field_scale",
        "absolute_voltage_calibration_available": False,
        "calibration_gain_available": False,
        "normalized_modeling_voltage_v": _peak_abs(limited),
    }
    for key in (
        "target_template_type",
        "target_template_ripple_check_passed",
        "target_linear_segment_deviation_max_mT",
        "target_peak_positive_mT",
        "target_peak_negative_mT",
    ):
        if key in frame.columns:
            metadata[key] = _first_text_value(frame, key) if key == "target_template_type" else first_numeric(frame[key])
    return output_frame, metadata


def _first_existing_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in frame.columns and pd.to_numeric(frame[column], errors="coerce").notna().any():
            return column
    return None


def _has_reference_like_field(frame: pd.DataFrame) -> bool:
    return any(
        column in frame.columns and pd.to_numeric(frame[column], errors="coerce").notna().any()
        for column in (
            "target_field_mT",
            "physical_target_output_mT",
            "support_reference_output_mT",
            "target_aligned_support_reference_mT",
            "predicted_field_mT",
            "expected_field_mT",
            "expected_output",
            "modeled_output",
        )
    )


def _phase_peak_detection_signal(values: np.ndarray, active_mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    active_count = int(np.asarray(active_mask, dtype=bool).sum())
    window = _odd_window(min(max(7, active_count // 8), 101), max(active_count, 1))
    signal = pd.Series(np.asarray(values, dtype=float)).rolling(window=window, center=True, min_periods=1).median()
    signal = signal.rolling(window=window, center=True, min_periods=1).mean().to_numpy(dtype=float)
    return signal, {
        "phase_peak_detection_signal": "smoothed_normalized_measured_field",
        "phase_peak_detection_window_samples": int(window),
    }


def _odd_window(value: int, max_size: int) -> int:
    size = max(1, min(int(value), int(max_size)))
    if size % 2 == 0:
        size -= 1
    return max(size, 1)


def _measured_target_identity_metadata(target: np.ndarray, measured: np.ndarray, active_mask: np.ndarray) -> dict[str, Any]:
    active = np.asarray(active_mask, dtype=bool) & np.isfinite(target) & np.isfinite(measured)
    if active.sum() < 3:
        return {
            "measured_target_nearly_identical_detected": False,
            "measured_target_identity_risk": "unknown",
            "measured_target_rmse_mT": float("nan"),
            "measured_target_corr": float("nan"),
            "measured_source_suspicious_reference_like": False,
        }
    residual = np.asarray(target, dtype=float)[active] - np.asarray(measured, dtype=float)[active]
    rmse = float(np.sqrt(np.nanmean(residual**2)))
    target_active = np.asarray(target, dtype=float)[active]
    measured_active = np.asarray(measured, dtype=float)[active]
    corr = float(np.corrcoef(target_active, measured_active)[0, 1]) if np.nanstd(target_active) > 0 and np.nanstd(measured_active) > 0 else float("nan")
    nearly = bool(rmse <= 1e-6 or (np.isfinite(corr) and corr > 0.999 and rmse <= 1.0))
    return {
        "measured_target_nearly_identical_detected": nearly,
        "measured_target_identity_risk": "warning" if nearly else "ok",
        "measured_target_rmse_mT": rmse,
        "measured_target_corr": corr,
        "measured_source_suspicious_reference_like": nearly,
    }


def _merge_identity_metadata(raw_meta: dict[str, Any], aligned_meta: dict[str, Any]) -> dict[str, Any]:
    if raw_meta.get("measured_target_nearly_identical_detected"):
        return raw_meta
    return aligned_meta


def _first_text_value(frame: pd.DataFrame, column: str) -> str | None:
    if column not in frame.columns:
        return None
    series = frame[column].dropna()
    if series.empty:
        return None
    return str(series.iloc[0])


def _dominant_peak_time(
    time_s: np.ndarray,
    values: np.ndarray,
    active_mask: np.ndarray,
    *,
    preferred_polarity: str | None = None,
) -> tuple[float | None, str | None, float | None]:
    peaks = _peak_candidates(time_s, values, active_mask)
    if not peaks:
        return None, None, None
    if preferred_polarity in {"positive", "negative"}:
        preferred_sign = 1.0 if preferred_polarity == "positive" else -1.0
        same_polarity = [peak for peak in peaks if np.sign(peak[1]) == preferred_sign]
        if same_polarity:
            peaks = same_polarity
    peak_time, selected_value = max(peaks, key=lambda item: abs(float(item[1])))
    polarity = "positive" if float(selected_value) >= 0.0 else "negative"
    return float(peak_time), polarity, float(selected_value)


def _reference_voltage_peak_for_measured_peak(
    time_s: np.ndarray,
    values: np.ndarray,
    active_mask: np.ndarray,
    *,
    preferred_polarity: str | None,
    measured_peak_rel_s: float | None,
    output_start_s: float,
) -> tuple[float | None, str | None, float | None]:
    peaks = _peak_candidates(time_s, values, active_mask)
    if not peaks:
        return None, None, None
    if preferred_polarity in {"positive", "negative"}:
        preferred_sign = 1.0 if preferred_polarity == "positive" else -1.0
        same_polarity = [peak for peak in peaks if np.sign(peak[1]) == preferred_sign]
        if same_polarity:
            peaks = same_polarity
    if measured_peak_rel_s is not None and np.isfinite(measured_peak_rel_s):
        measured_peak_plot_time = float(output_start_s) + float(measured_peak_rel_s)
        selected_time, selected_value = min(peaks, key=lambda item: abs(float(item[0]) - measured_peak_plot_time))
    else:
        selected_time, selected_value = min(peaks, key=lambda item: float(item[0]))
    polarity = "positive" if float(selected_value) >= 0.0 else "negative"
    return float(selected_time), polarity, float(selected_value)


def _peak_candidates(time_s: np.ndarray, values: np.ndarray, active_mask: np.ndarray) -> list[tuple[float, float]]:
    time_arr = np.asarray(time_s, dtype=float)
    value_arr = np.asarray(values, dtype=float)
    active = np.asarray(active_mask, dtype=bool) & np.isfinite(time_arr) & np.isfinite(value_arr)
    if active.sum() < 3:
        return []
    indices = np.flatnonzero(active)
    local = value_arr[indices]
    peaks: list[tuple[float, float]] = []
    for local_index in range(1, len(local) - 1):
        value = float(local[local_index])
        is_max = value >= float(local[local_index - 1]) and value >= float(local[local_index + 1])
        is_min = value <= float(local[local_index - 1]) and value <= float(local[local_index + 1])
        if is_max or is_min:
            source_index = int(indices[int(local_index)])
            peaks.append((float(time_arr[source_index]), value))
    if not peaks:
        selected = int(np.nanargmax(np.abs(local)))
        source_index = int(indices[selected])
        peaks = [(float(time_arr[source_index]), float(local[selected]))]
    return peaks


def _phase_support_margin_s(time_s: np.ndarray) -> float:
    finite = np.asarray(time_s, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 3:
        return 0.0
    diffs = np.diff(np.sort(finite))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 0.0
    return float(2.0 * np.nanmedian(diffs))


def _interp(source_time: np.ndarray, source_values: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    source_t = np.asarray(source_time, dtype=float)
    source_y = np.asarray(source_values, dtype=float)
    target_t = np.asarray(target_time, dtype=float)
    finite = np.isfinite(source_t) & np.isfinite(source_y)
    if finite.sum() < 2:
        return np.full_like(target_t, np.nan, dtype=float)
    order = np.argsort(source_t[finite])
    sorted_t = source_t[finite][order]
    sorted_y = source_y[finite][order]
    out = np.interp(target_t, sorted_t, sorted_y)
    out[(target_t < sorted_t[0]) | (target_t > sorted_t[-1])] = np.nan
    return out


def _peak_abs(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.nanmax(np.abs(finite))) if finite.size else 0.0


def first_numeric(values: pd.Series) -> float | bool | None:
    if values.dtype == bool:
        return bool(values.dropna().iloc[0]) if not values.dropna().empty else None
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.iloc[0]) if not numeric.empty else None


def _source_active_start_s(frame: pd.DataFrame, native_time_s: np.ndarray, output_start_s: float) -> float:
    for column in (
        "selected_support_original_nonzero_start_s",
        "support_reference_source_window_start_s",
        "selected_support_source_window_start_s",
    ):
        if column in frame.columns:
            value = first_numeric(frame[column])
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                return float(value)
    finite = np.asarray(native_time_s, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size:
        return float(np.nanmin(finite))
    return float(output_start_s)


def _active_end_kink_detected(voltage: np.ndarray, residual: np.ndarray, active_mask: np.ndarray) -> bool:
    active_indices = np.flatnonzero(np.asarray(active_mask, dtype=bool) & np.isfinite(voltage) & np.isfinite(residual))
    if active_indices.size < 4:
        return False
    tail = active_indices[-4:]
    voltage_step = float(np.nanmax(np.abs(np.diff(np.asarray(voltage, dtype=float)[tail]))))
    residual_step = float(np.nanmax(np.abs(np.diff(np.asarray(residual, dtype=float)[tail]))))
    return bool(voltage_step > 1.0 and residual_step > 5.0)
