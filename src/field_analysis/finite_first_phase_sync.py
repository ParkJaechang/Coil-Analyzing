from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .finite_second_modeling_stabilization import smooth_measured_field_for_second_modeling, stabilize_correction_delta
from .finite_second_modeling_tail import compute_second_modeling_gain


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
    frame = command_profile.copy(deep=True).reset_index(drop=True)
    if frame.empty or "time_s" not in frame.columns:
        return frame, {
            "finite_first_modeling_mode": "phase_synced",
            "finite_first_modeling_phase_sync_enabled": False,
            "finite_first_modeling_status": "empty_command_profile",
        }
    time_s = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
    active_end_s = float(cycle_count) / max(float(freq_hz), 1e-12)
    active_mask = np.isfinite(time_s) & (time_s <= active_end_s + 1e-12)
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
    native_time_s, native_measured_raw, measured_alignment_source = _native_measured_support_source(
        frame,
        fallback_time_s=time_s,
        fallback_measured=measured_raw,
    )
    native_active_mask = (
        np.isfinite(native_time_s)
        & (native_time_s >= float(np.nanmin(time_s[active_mask])) - 1e-12 if np.any(active_mask) else True)
        & (native_time_s <= active_end_s + 1e-12)
    )
    measured, measured_source_type, raw_hallbz_available, measured_norm_meta = _coerce_measured_field(
        native_measured_raw,
        measured_column,
        native_active_mask,
    )
    base_voltage = pd.to_numeric(frame[voltage_column], errors="coerce").to_numpy(dtype=float)
    native_smoothed, smoothing_meta = smooth_measured_field_for_second_modeling(
        native_time_s,
        measured,
        native_time_s,
        native_active_mask,
        freq_hz=float(freq_hz),
        cycle_count=float(cycle_count),
    )
    measured_peak_time, measured_peak_polarity, measured_peak_value = _dominant_peak_time(
        native_time_s,
        native_smoothed,
        native_active_mask,
    )
    voltage_peak_time, voltage_peak_polarity, voltage_peak_value = _dominant_peak_time(
        time_s,
        base_voltage,
        active_mask,
        preferred_polarity=measured_peak_polarity,
    )
    phase_delay_s = 0.0
    alignment_status = "peak_detection_failed"
    if voltage_peak_time is not None and measured_peak_time is not None:
        phase_delay_s = float(measured_peak_time - voltage_peak_time)
        alignment_status = "ok"
    support_margin_s = _phase_support_margin_s(time_s)
    required_end = active_end_s + max(phase_delay_s, 0.0) + support_margin_s
    source_end = float(np.nanmax(native_time_s[np.isfinite(native_time_s)])) if np.isfinite(native_time_s).any() else float("nan")
    support_ok = bool(np.isfinite(source_end) and source_end >= required_end - 1e-12)
    smoothed = _interp(native_time_s, native_smoothed, time_s)
    aligned = _interp(native_time_s, native_smoothed, time_s + phase_delay_s)
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
            "measurement_support_grid_separate_from_output_grid": measured_alignment_source == "selected_support_source_native",
            "measurement_support_source_sample_count": int(np.isfinite(native_time_s).sum()),
            "required_phase_aligned_source_end_s": required_end,
            "actual_source_time_end_s": source_end,
            "phase_support_status": "insufficient",
            "phase_sync_peak_reference": "dominant_absolute_peak",
            "phase_sync_peak_polarity": measured_peak_polarity,
            "voltage_first_peak_time_s": voltage_peak_time,
            "measured_first_peak_time_s": measured_peak_time,
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
        }
    finite_active = active_mask & np.isfinite(aligned) & np.isfinite(target)
    residual = target - aligned
    active_residual_valid = np.asarray(active_mask, dtype=bool) & np.isfinite(residual)
    active_count = int(np.asarray(active_mask, dtype=bool).sum())
    active_residual_finite_ratio = float(active_residual_valid.sum() / active_count) if active_count else 0.0
    measured_on_output = _interp(native_time_s, measured, time_s)
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
        "phase_sync_peak_polarity": measured_peak_polarity,
        "voltage_first_peak_time_s": voltage_peak_time,
        "measured_first_peak_time_s": measured_peak_time,
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
        "measured_alignment_source": "native_smoothed_source",
        "measurement_support_grid_separate_from_output_grid": measured_alignment_source == "selected_support_source_native",
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


def _coerce_measured_field(values: np.ndarray, column: str, active_mask: np.ndarray) -> tuple[np.ndarray, str, bool, dict[str, Any]]:
    raw = np.asarray(values, dtype=float)
    active = np.asarray(active_mask, dtype=bool) & np.isfinite(raw)
    if column in {"HallBz", "HallZ", "raw_hallbz_mT"}:
        effective = -raw
        active_effective = np.asarray(active_mask, dtype=bool) & np.isfinite(effective)
        baseline = float(np.nanmedian(effective[active_effective])) if np.any(active_effective) else 0.0
        centered = effective - baseline
        centered_active = np.asarray(active_mask, dtype=bool) & np.isfinite(centered)
        peak = float(np.nanmax(np.abs(centered[centered_active]))) if np.any(centered_active) else 0.0
        scale = 50.0 / peak if peak > 1e-12 else 1.0
        normalized = centered * scale
        return normalized, "raw_hallbz_effective_normalized", True, _measured_normalization_metadata(
            raw_peak=_peak_abs(raw[active]) if np.any(active) else 0.0,
            effective_peak=peak,
            scale=scale,
            status="ok" if peak > 1e-12 else "zero_peak",
        )
    baseline = float(np.nanmedian(raw[active])) if np.any(active) else 0.0
    centered = raw - baseline
    centered_active = np.asarray(active_mask, dtype=bool) & np.isfinite(centered)
    peak = float(np.nanmax(np.abs(centered[centered_active]))) if np.any(centered_active) else 0.0
    scale = 50.0 / peak if peak > 1e-12 else 1.0
    normalized = centered * scale
    return normalized, "actual_measured_field_normalized", False, _measured_normalization_metadata(
        raw_peak=_peak_abs(raw[active]) if np.any(active) else 0.0,
        effective_peak=peak,
        scale=scale,
        status="ok" if peak > 1e-12 else "zero_peak",
    )


def _measured_normalization_metadata(*, raw_peak: float, effective_peak: float, scale: float, status: str) -> dict[str, Any]:
    return {
        "measured_abs_peak_raw_mT": float(raw_peak),
        "measured_abs_peak_effective_mT": float(effective_peak),
        "measured_field_scale_to_50mT": float(scale),
        "measured_field_normalization_status": status,
    }


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


def _native_measured_support_source(
    frame: pd.DataFrame,
    *,
    fallback_time_s: np.ndarray,
    fallback_measured: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    source_time = _first_sequence_value(frame, "selected_support_source_time_s")
    source_measured = _first_sequence_value(frame, "selected_support_source_mT")
    if source_time is not None and source_measured is not None:
        source_time_arr = np.asarray(source_time, dtype=float)
        source_measured_arr = np.asarray(source_measured, dtype=float)
        if source_time_arr.size == source_measured_arr.size and source_time_arr.size >= 3:
            finite = np.isfinite(source_time_arr) & np.isfinite(source_measured_arr)
            if finite.sum() >= 3:
                return source_time_arr, source_measured_arr, "selected_support_source_native"
    return np.asarray(fallback_time_s, dtype=float), np.asarray(fallback_measured, dtype=float), "output_command_grid_fallback"


def _first_sequence_value(frame: pd.DataFrame, column: str) -> list[float] | tuple[float, ...] | np.ndarray | None:
    if column not in frame.columns or frame.empty:
        return None
    for value in frame[column]:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, (list, tuple)):
            return value
        if isinstance(value, str):
            parsed = _parse_sequence_string(value)
            if parsed is not None:
                return parsed
    return None


def _parse_sequence_string(value: str) -> list[float] | None:
    text = value.strip()
    if not text or text.lower() in {"none", "nan"}:
        return None
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    parts = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    if len(parts) < 3:
        return None
    try:
        return [float(part) for part in parts]
    except ValueError:
        return None


def _dominant_peak_time(
    time_s: np.ndarray,
    values: np.ndarray,
    active_mask: np.ndarray,
    *,
    preferred_polarity: str | None = None,
) -> tuple[float | None, str | None, float | None]:
    active = np.asarray(active_mask, dtype=bool) & np.isfinite(time_s) & np.isfinite(values)
    if active.sum() < 3:
        return None, None, None
    indices = np.flatnonzero(active)
    local = np.asarray(values, dtype=float)[indices]
    peak_candidates: list[tuple[int, float]] = []
    for local_index in range(1, len(local) - 1):
        value = float(local[local_index])
        is_max = value >= float(local[local_index - 1]) and value >= float(local[local_index + 1])
        is_min = value <= float(local[local_index - 1]) and value <= float(local[local_index + 1])
        if is_max or is_min:
            peak_candidates.append((local_index, value))
    if not peak_candidates:
        selected = int(np.nanargmax(np.abs(local)))
        peak_candidates = [(selected, float(local[selected]))]
    if preferred_polarity in {"positive", "negative"}:
        preferred_sign = 1.0 if preferred_polarity == "positive" else -1.0
        same_polarity = [(idx, value) for idx, value in peak_candidates if np.sign(value) == preferred_sign]
        if same_polarity:
            peak_candidates = same_polarity
    selected_index, selected_value = max(peak_candidates, key=lambda item: abs(float(item[1])))
    peak_index = int(indices[int(selected_index)])
    polarity = "positive" if float(selected_value) >= 0.0 else "negative"
    return float(np.asarray(time_s, dtype=float)[peak_index]), polarity, float(selected_value)


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


def _active_end_kink_detected(voltage: np.ndarray, residual: np.ndarray, active_mask: np.ndarray) -> bool:
    active_indices = np.flatnonzero(np.asarray(active_mask, dtype=bool) & np.isfinite(voltage) & np.isfinite(residual))
    if active_indices.size < 4:
        return False
    tail = active_indices[-4:]
    voltage_step = float(np.nanmax(np.abs(np.diff(np.asarray(voltage, dtype=float)[tail]))))
    residual_step = float(np.nanmax(np.abs(np.diff(np.asarray(residual, dtype=float)[tail]))))
    return bool(voltage_step > 1.0 and residual_step > 5.0)
