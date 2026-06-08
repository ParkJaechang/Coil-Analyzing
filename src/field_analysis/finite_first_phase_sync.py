from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .finite_second_modeling_stabilization import smooth_measured_field_for_second_modeling, stabilize_correction_delta
from .finite_second_modeling_tail import compute_second_modeling_gain
from .finite_first_normalization import (
    coerce_measured_field_centered,
    normalize_smoothed_field_to_pm50,
    scale_target_field_columns_to_peak,
)
from .first_modeling_voltage_response import (
    field_per_volt_response_metadata,
    residual_to_voltage_delta_from_field_per_volt,
)
from .finite_phase_sync_support import (
    active_end_kink_detected,
    native_measured_support_source,
    native_voltage_support_source,
    source_active_start_s as resolve_source_active_start_s,
)
from .finite_phase_sync_math import (
    dominant_peak_time as _dominant_peak_time,
    interp_no_extrapolation as _interp,
    midpoint_between_peak_pair as _midpoint_between_peak_pair,
    nearest_zero_crossing_time as _nearest_zero_crossing_time,
    peak_abs as _peak_abs,
    phase_peak_detection_signal as _phase_peak_detection_signal,
    phase_support_margin_s as _phase_support_margin_s,
    reference_voltage_peak_for_measured_peak as _reference_voltage_peak_for_measured_peak,
)
from .modeling_error_metrics import peak_error_metrics
from .voltage_policy import COMMAND_VOLTAGE_LIMIT_V


def apply_finite_first_phase_sync_modeling(
    command_profile: pd.DataFrame,
    *,
    freq_hz: float,
    cycle_count: float,
    mode: str = "phase_synced",
    voltage_limit_v: float = COMMAND_VOLTAGE_LIMIT_V,
    target_peak_field_mT: float | None = None,
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
    explicit_target_peak = float(target_peak_field_mT) if target_peak_field_mT is not None else float("nan")
    raw_target = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(dtype=float)
    raw_target_peak_mT = _peak_abs(raw_target[active_mask])
    target_peak_reference_mT = explicit_target_peak if np.isfinite(explicit_target_peak) and explicit_target_peak > 1e-12 else raw_target_peak_mT
    if not np.isfinite(target_peak_reference_mT) or target_peak_reference_mT <= 1e-12:
        target_peak_reference_mT = 50.0
    target_scale_meta = scale_target_field_columns_to_peak(
        frame,
        active_mask,
        target_peak_mT=target_peak_reference_mT,
        target_peak_source="ui_user_selection" if np.isfinite(explicit_target_peak) and explicit_target_peak > 1e-12 else "target_trace_peak",
    )
    target = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(dtype=float)
    measured_raw = pd.to_numeric(frame[measured_column], errors="coerce").to_numpy(dtype=float)
    native_time_s, native_measured_raw, measured_alignment_source = native_measured_support_source(
        frame,
        fallback_time_s=time_s,
        fallback_measured=measured_raw,
    )
    base_voltage = pd.to_numeric(frame[voltage_column], errors="coerce").to_numpy(dtype=float)
    native_voltage_time_s, native_input_voltage, input_voltage_source = native_voltage_support_source(
        frame,
        fallback_time_s=time_s,
        fallback_voltage=base_voltage,
    )
    source_active_start_s = resolve_source_active_start_s(
        frame,
        native_time_s,
        output_start_s,
        native_voltage_v=native_input_voltage
        if native_voltage_time_s.size == native_time_s.size
        and str(input_voltage_source).startswith("selected_support_source_voltage_v")
        else None,
        native_measured_mT=native_measured_raw
        if str(measured_alignment_source).startswith("selected_support_source_native")
        else None,
    )
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
    if str(input_voltage_source).startswith("selected_support_source_voltage_v"):
        input_lut_voltage = _interp(native_voltage_time_s, native_input_voltage, source_time_for_output)
    else:
        input_lut_voltage = base_voltage.copy()
    voltage_for_peak_reference = np.where(np.isfinite(input_lut_voltage), input_lut_voltage, base_voltage)
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
        target_peak_mT=target_peak_reference_mT,
    )
    peak_detection_signal, peak_detection_meta = _phase_peak_detection_signal(native_smoothed_unscaled, native_active_mask)
    preferred_peak_polarity = "positive" if float(cycle_count) <= 1.0 + 1e-9 else "negative"
    peak_reference_label = "first_positive_peak" if preferred_peak_polarity == "positive" else "dominant_negative_peak"
    measured_peak_time, measured_peak_polarity, measured_peak_value = _dominant_peak_time(
        native_time_s,
        peak_detection_signal,
        native_active_mask,
        preferred_polarity=preferred_peak_polarity,
    )
    measured_peak_rel_for_voltage = (
        float(measured_peak_time - source_active_start_s)
        if measured_peak_time is not None and np.isfinite(source_active_start_s)
        else None
    )
    if preferred_peak_polarity == "positive":
        measured_peak_rel_for_voltage = None
    voltage_peak_time, voltage_peak_polarity, voltage_peak_value = _reference_voltage_peak_for_measured_peak(
        time_s,
        voltage_for_peak_reference,
        active_mask,
        preferred_polarity=measured_peak_polarity,
        measured_peak_rel_s=measured_peak_rel_for_voltage,
        output_start_s=output_start_s,
    )
    midpoint_time_s, midpoint_polarity, midpoint_left_s, midpoint_right_s = _midpoint_between_peak_pair(
        native_time_s,
        peak_detection_signal,
        native_active_mask,
        pair_start_number=2,
    )
    target_zero_time_s = _nearest_zero_crossing_time(
        time_s,
        target,
        active_mask,
        reference_time_s=(
            output_start_s + float(midpoint_time_s - source_active_start_s)
            if midpoint_time_s is not None and np.isfinite(source_active_start_s)
            else None
        ),
    )
    phase_delay_s = 0.0
    measured_peak_plot_time = measured_peak_time
    measured_peak_source_time = measured_peak_time
    alignment_status = "peak_detection_failed"
    phase_sync_method = "field_peak_to_voltage_peak"
    if midpoint_time_s is not None and target_zero_time_s is not None:
        midpoint_rel = float(midpoint_time_s - source_active_start_s)
        target_zero_rel = float(target_zero_time_s - output_start_s)
        phase_delay_s = float(midpoint_rel - target_zero_rel)
        measured_peak_plot_time = float(output_start_s + midpoint_rel)
        measured_peak_source_time = float(midpoint_time_s)
        alignment_status = "ok"
        phase_sync_method = "peak_pair_midpoint_to_target_zero_crossing"
        peak_reference_label = "second_to_third_peak_midpoint"
    elif voltage_peak_time is not None and measured_peak_time is not None:
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
    field_scale_to_target_mT = float(
        measured_norm_meta.get("measured_field_scale_to_target_mT")
        or measured_norm_meta.get("measured_field_scale_to_target_peak_mT")
        or measured_norm_meta.get("measured_field_scale_to_50mT")
        or 1.0
    )
    input_lut_voltage_normalized = input_lut_voltage * field_scale_to_target_mT
    base_voltage = np.where(np.isfinite(input_lut_voltage_normalized), input_lut_voltage_normalized, base_voltage)
    field_response_meta = field_per_volt_response_metadata(
        input_lut_voltage,
        active_mask,
        target_peak_mT=target_peak_reference_mT,
        field_scale_to_target=field_scale_to_target_mT,
        prefix="finite_first",
    )
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
        frame["finite_first_input_lut_voltage_v"] = input_lut_voltage
        frame["finite_first_input_lut_voltage_normalized_v"] = input_lut_voltage_normalized
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
            "phase_sync_peak_reference": peak_reference_label,
            "phase_sync_voltage_reference": "target_zero_crossing" if phase_sync_method == "peak_pair_midpoint_to_target_zero_crossing" else "nearest_same_polarity_peak_to_measured_peak",
            "phase_sync_midpoint_left_peak_time_s": midpoint_left_s,
            "phase_sync_midpoint_right_peak_time_s": midpoint_right_s,
            "phase_sync_midpoint_time_s": midpoint_time_s,
            "phase_sync_midpoint_polarity": midpoint_polarity,
            "phase_sync_target_zero_crossing_time_s": target_zero_time_s,
            "phase_sync_alignment_anchor": phase_sync_method,
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
            "finite_first_input_voltage_source": input_voltage_source,
            "measured_field_scale_to_target_mT": field_scale_to_target_mT,
            "finite_first_input_voltage_normalization_scale": field_scale_to_target_mT,
            **field_response_meta,
            "finite_first_base_voltage_source": "field_scale_normalized_input_lut_voltage"
            if input_voltage_source == "selected_support_source_voltage_v"
            else input_voltage_source,
            **target_scale_meta,
        }
    finite_active = active_mask & np.isfinite(aligned) & np.isfinite(target)
    residual = target - aligned
    eval_start_cycle, eval_end_cycle = _evaluation_cycle_window(float(cycle_count))
    evaluation_start_s = output_start_s + eval_start_cycle / max(float(freq_hz), 1e-12)
    evaluation_end_s = output_start_s + eval_end_cycle / max(float(freq_hz), 1e-12)
    evaluation_mask = (
        finite_active
        & (time_s >= evaluation_start_s - 1e-12)
        & (time_s <= evaluation_end_s + 1e-12)
    )
    active_residual_valid = np.asarray(active_mask, dtype=bool) & np.isfinite(residual)
    evaluation_residual_valid = np.asarray(evaluation_mask, dtype=bool) & np.isfinite(residual)
    active_count = int(np.asarray(active_mask, dtype=bool).sum())
    active_residual_finite_ratio = float(active_residual_valid.sum() / active_count) if active_count else 0.0
    evaluation_count = int(np.asarray(evaluation_mask, dtype=bool).sum())
    evaluation_residual_finite_ratio = float(evaluation_residual_valid.sum() / evaluation_count) if evaluation_count else 0.0
    measured_on_output = _interp(native_time_s, native_smoothed, source_time_for_output)
    identity_meta = _merge_identity_metadata(
        _measured_target_identity_metadata(target, measured_on_output, active_mask),
        _measured_target_identity_metadata(target, aligned, finite_active),
    )
    unit_delta = residual_to_voltage_delta_from_field_per_volt(
        residual,
        float(field_response_meta.get("field_per_volt_mT_per_v") or float("nan")),
    )
    unit_delta_for_gain = unit_delta.copy()
    unit_delta_for_gain[~evaluation_mask] = np.nan
    unit_delta_for_correction = unit_delta.copy()
    unit_delta_for_correction[~finite_active] = np.nan
    gain, gain_meta = compute_second_modeling_gain(
        unit_delta_for_gain,
        base_voltage,
        evaluation_mask,
        manual_gain=0.25,
        gain_mode="auto",
        voltage_limit_v=float(voltage_limit_v),
        tail_mask=np.zeros_like(finite_active, dtype=bool),
        error_ratio_for_gain=np.abs(residual) / max(float(target_peak_reference_mT), 1e-12),
    )
    raw_delta = unit_delta_for_correction * float(gain)
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
    frame["finite_first_input_lut_voltage_v"] = input_lut_voltage
    frame["finite_first_input_lut_voltage_normalized_v"] = input_lut_voltage_normalized
    frame["first_modeled_voltage_v"] = modeled
    frame["limited_voltage_v"] = limited
    frame["correction_delta_v"] = correction_delta
    frame["raw_correction_delta_v"] = raw_delta
    frame["smoothed_correction_delta_v"] = arrays.get("smoothed_correction_delta_v", correction_delta)
    frame["measured_field_smoothed_mT"] = smoothed
    frame["measured_field_aligned_mT"] = aligned
    frame["residual_for_modeling_mT"] = residual
    frame["modeling_error_evaluation_mask"] = evaluation_mask
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
        "phase_sync_method": phase_sync_method,
        "phase_sync_peak_reference": peak_reference_label,
        "phase_sync_voltage_reference": "target_zero_crossing" if phase_sync_method == "peak_pair_midpoint_to_target_zero_crossing" else "nearest_same_polarity_peak_to_measured_peak",
        "phase_sync_midpoint_left_peak_time_s": midpoint_left_s,
        "phase_sync_midpoint_right_peak_time_s": midpoint_right_s,
        "phase_sync_midpoint_time_s": midpoint_time_s,
        "phase_sync_midpoint_polarity": midpoint_polarity,
        "phase_sync_target_zero_crossing_time_s": target_zero_time_s,
        "phase_sync_alignment_anchor": phase_sync_method,
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
        "field_modeling_normalization_reference_mT": target_peak_reference_mT,
        "user_target_peak_field_mT": target_peak_reference_mT,
        "target_peak_field_source": "ui_user_selection" if np.isfinite(explicit_target_peak) and explicit_target_peak > 1e-12 else "target_trace_peak",
        **target_scale_meta,
        "correction_delta_v_target_peak_reference_mT": target_peak_reference_mT,
        "correction_delta_v_uses_scaled_target_field": True,
        "legacy_residual_over_target_peak_voltage_limit": False,
        "residual_to_voltage_conversion_basis": "measured_field_per_input_volt",
        "harmonic_inverse_field_scale_applied_or_not_used": "harmonic_inverse_not_used_for_final_export",
        "source_voltage_raw_peak_v": _peak_abs(base_voltage),
        "source_voltage_base_normalized_peak_v": _peak_abs(base_voltage),
        "base_voltage_peak_setting_v": _peak_abs(base_voltage),
        "finite_first_input_voltage_source": input_voltage_source,
        "measured_field_scale_to_target_mT": field_scale_to_target_mT,
        "finite_first_input_voltage_normalization_scale": field_scale_to_target_mT,
        **field_response_meta,
        "finite_first_base_voltage_source": "field_scale_normalized_input_lut_voltage"
        if input_voltage_source == "selected_support_source_voltage_v"
        else input_voltage_source,
        "finite_first_input_lut_voltage_peak_v": _peak_abs(input_lut_voltage),
        "finite_first_input_lut_voltage_normalized_peak_v": _peak_abs(input_lut_voltage_normalized),
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
        "error_evaluation_start_cycle": eval_start_cycle,
        "error_evaluation_cycle_count": eval_end_cycle,
        "error_evaluation_end_cycle": eval_end_cycle,
        "error_evaluation_start_s": evaluation_start_s,
        "error_evaluation_end_s": evaluation_end_s,
        "error_evaluation_policy": "evaluate_after_startup_from_0.25cycle_to_0.75cycle_or_1.25cycle",
        "error_evaluation_sample_count": evaluation_count,
        "error_evaluation_finite_ratio": evaluation_residual_finite_ratio,
        **peak_error_metrics(
            target,
            aligned,
            evaluation_mask,
            reference_peak_mT=target_peak_reference_mT,
        ),
        "active_end_kink_detected": active_end_kink_detected(limited, residual, active_mask),
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


def _evaluation_cycle_window(cycle_count: float) -> tuple[float, float]:
    if float(cycle_count) <= 1.0 + 1e-9:
        return 0.25, 0.75
    if float(cycle_count) <= 1.5 + 1e-9:
        return 0.25, 1.25
    return 0.25, max(float(cycle_count) - 0.25, 0.25)


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


def first_numeric(values: pd.Series) -> float | bool | None:
    if values.dtype == bool:
        return bool(values.dropna().iloc[0]) if not values.dropna().empty else None
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.iloc[0]) if not numeric.empty else None




