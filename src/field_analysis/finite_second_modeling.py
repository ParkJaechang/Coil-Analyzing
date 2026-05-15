from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .finite_actual_drive import ActualDriveRecord
from .finite_actual_drive import build_actual_drive_review_case
from .finite_actual_drive import read_actual_drive_result
from .finite_actual_drive_normalization import peak_abs
from .finite_second_modeling_stabilization import align_measured_field_for_residual
from .finite_second_modeling_stabilization import apply_polarity_guard
from .finite_second_modeling_stabilization import diagnose_correction_discontinuity
from .finite_second_modeling_stabilization import detect_measured_zero_return_support
from .finite_second_modeling_stabilization import measured_support_metadata
from .finite_second_modeling_stabilization import smooth_measured_field_for_second_modeling
from .finite_second_modeling_stabilization import stabilize_correction_delta
from .finite_second_modeling_tail import compute_second_modeling_gain as _compute_second_modeling_gain
from .finite_second_modeling_tail import extend_profile_for_zero_tail as _tail_extend_profile_for_zero_tail
from .finite_second_modeling_tail import tail_mask as _tail_mask_values
from .finite_second_modeling_tail_controller import apply_finite_time_zero_return_tail
from .finite_second_modeling_tail_controller import fill_tail_measured_field as _fill_tail_measured_field
from .finite_second_modeling_tail_controller import missing_time_ranges as _missing_time_ranges
from .finite_second_modeling_tail_controller import normalize_tail_duration_mode as _normalize_tail_duration_mode
from .finite_second_modeling_tail_controller import normalize_tail_return_mode as _normalize_tail_return_mode
from .finite_second_modeling_tail_controller import tail_cycle_count_from_duration as _tail_cycle_count_from_duration
from .finite_second_modeling_tail_controller import unified_tail_diagnostics as _unified_tail_diagnostics

SUPPORTED_SECOND_MODELING_CYCLES = (1.0, 1.5)
UNSUPPORTED_SECOND_MODELING_CYCLES = (1.25, 1.75, 2.0)
PRODUCTION_CYCLE_POLICY = "1p0_1p5_cycles"
UNSUPPORTED_CYCLE_STATUS = "unsupported_cycle_policy_1p0_1p5_only"


def generate_second_modeled_voltage_lut(
    first_command_profile: pd.DataFrame,
    actual_drive_source: str | Path | ActualDriveRecord,
    *,
    freq_hz: float,
    cycle_count: float,
    waveform_type: str | None = None,
    correction_gain: float = 0.25,
    correction_gain_mode: str = "auto",
    voltage_limit_v: float = 5.0,
    residual_alignment_mode: str = "first_peak_aligned_stabilized",
    post_cycle_zero_tail_enabled: bool = True,
    post_cycle_zero_tail_cycle_count: float = 0.25,
    post_cycle_zero_tail_duration_s: float = 0.25,
    tail_return_mode: str = "finite_time_zero_return",
    tail_duration_mode: str = "seconds",
    tail_controller_scaling_mode: str = "auto",
    tail_controller_gain: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    profile = first_command_profile.copy()
    residual_alignment_mode = _normalize_residual_alignment_mode(residual_alignment_mode)
    tail_return_mode = _normalize_tail_return_mode(tail_return_mode)
    tail_duration_mode = _normalize_tail_duration_mode(tail_duration_mode)
    requested_tail_cycle_count = _tail_cycle_count_from_duration(
        freq_hz=float(freq_hz),
        tail_duration_mode=tail_duration_mode,
        tail_cycle_count=float(post_cycle_zero_tail_cycle_count),
        tail_duration_s=float(post_cycle_zero_tail_duration_s),
    )
    alignment_mode = (
        "pointwise" if residual_alignment_mode == "pointwise" else "first_peak_aligned"
    )
    stabilization_enabled = residual_alignment_mode == "first_peak_aligned_stabilized"
    base_metadata = {
        "second_modeling_method": "residual_proportional_feedback",
        "correction_gain": float(correction_gain),
        "production_cycle_policy": PRODUCTION_CYCLE_POLICY,
        "supported_production_cycles": list(SUPPORTED_SECOND_MODELING_CYCLES),
        "unsupported_cycles": list(UNSUPPORTED_SECOND_MODELING_CYCLES),
        "fourier_resynthesis_involved": False,
        "harmonic_export_involved": False,
        "target_unchanged": True,
        "residual_alignment_mode": residual_alignment_mode,
    }
    if not _is_supported_cycle(cycle_count):
        return profile, {
            **base_metadata,
            "second_modeling_available": False,
            "second_modeling_status": UNSUPPORTED_CYCLE_STATUS,
            "second_correction_delta_v_generated": False,
            "second_voltage_v_generated": False,
            "second_lut_generated": False,
        }
    missing = sorted({"time_s"}.difference(profile.columns))
    if missing:
        return profile, {
            **base_metadata,
            "second_modeling_available": False,
            "second_modeling_status": "missing_first_command_columns",
            "second_modeling_unavailable_reason": ",".join(missing),
        }
    record = (
        actual_drive_source
        if isinstance(actual_drive_source, ActualDriveRecord)
        else read_actual_drive_result(
            actual_drive_source,
            waveform_type=waveform_type,
            freq_hz=freq_hz if waveform_type is not None else None,
            cycle_count=cycle_count if waveform_type is not None else None,
        )
    )
    review, review_meta = build_actual_drive_review_case(record)
    if str(review_meta.get("timebase_status", "ok")) != "ok":
        return profile, {
            **base_metadata,
            **_review_diagnostic_metadata(review_meta),
            "second_modeling_available": False,
            "second_modeling_status": "actual_drive_timebase_not_ok",
            "second_modeling_unavailable_reason": str(review_meta.get("timebase_status", "unknown")),
            "second_correction_delta_v_generated": False,
            "second_voltage_v_generated": False,
            "second_lut_generated": False,
        }
    if abs(float(record.freq_hz) - float(freq_hz)) > 1e-9 or abs(float(record.cycle_count) - float(cycle_count)) > 1e-9:
        return profile, {
            **base_metadata,
            **_review_diagnostic_metadata(review_meta),
            "second_modeling_available": False,
            "second_modeling_status": "actual_drive_target_mismatch",
            "second_modeling_unavailable_reason": "actual_drive_file_freq_or_cycle_mismatch",
            "target_freq_hz": float(freq_hz),
            "target_cycle_count": float(cycle_count),
            "file_freq_hz": float(record.freq_hz),
            "file_cycle_count": float(record.cycle_count),
            "second_correction_delta_v_generated": False,
            "second_voltage_v_generated": False,
            "second_lut_generated": False,
        }
    zero_return_meta = detect_measured_zero_return_support(
        review["time_s"],
        review["normalized_measured_field_mT"],
        freq_hz=float(freq_hz),
        cycle_count=float(cycle_count),
        requested_tail_cycle_count=requested_tail_cycle_count,
        tail_enabled=bool(post_cycle_zero_tail_enabled),
    )
    effective_tail_cycle_count = requested_tail_cycle_count
    time_s = pd.to_numeric(profile["time_s"], errors="coerce").to_numpy(dtype=float)
    first_voltage = _first_voltage(profile)
    first_limited_voltage = _first_limited_voltage(profile, first_voltage)
    first_recommended_voltage = _optional_voltage(profile, "recommended_voltage_v")
    active_mask = _active_mask(time_s, freq_hz=freq_hz, cycle_count=cycle_count)
    tail_mask = _tail_mask_values(time_s, freq_hz=freq_hz, cycle_count=cycle_count, tail_cycle_count=effective_tail_cycle_count, enabled=bool(post_cycle_zero_tail_enabled))
    if not _source_covers_target_active_window(review["time_s"], time_s[active_mask]):
        return profile, {
            **base_metadata,
            **_review_diagnostic_metadata(review_meta),
            "second_modeling_available": False,
            "second_modeling_status": "actual_drive_time_range_insufficient",
            "second_modeling_unavailable_reason": "actual_drive_source_time_does_not_cover_target_active_window",
            "interpolation_status": "source_time_range_insufficient_no_extrapolation",
            "second_correction_delta_v_generated": False,
            "second_voltage_v_generated": False,
            "second_lut_generated": False,
        }
    profile, tail_setup_meta = _tail_extend_profile_for_zero_tail(
        profile,
        freq_hz=float(freq_hz),
        cycle_count=float(cycle_count),
        enabled=bool(post_cycle_zero_tail_enabled),
        tail_cycle_count=effective_tail_cycle_count,
    )
    time_s = pd.to_numeric(profile["time_s"], errors="coerce").to_numpy(dtype=float)
    first_voltage = _first_voltage(profile)
    first_limited_voltage = _first_limited_voltage(profile, first_voltage)
    first_recommended_voltage = _optional_voltage(profile, "recommended_voltage_v")
    active_mask = _active_mask(time_s, freq_hz=freq_hz, cycle_count=cycle_count)
    tail_mask = _tail_mask_values(
        time_s,
        freq_hz=freq_hz,
        cycle_count=cycle_count,
        tail_cycle_count=effective_tail_cycle_count,
        enabled=bool(post_cycle_zero_tail_enabled),
    )
    target = _target(profile, review, time_s)
    measured = _interp(review["time_s"], review["normalized_measured_field_mT"], time_s)
    measured_smoothed, smoothing_meta = smooth_measured_field_for_second_modeling(
        review["time_s"],
        review["normalized_measured_field_mT"],
        time_s,
        active_mask,
        freq_hz=float(freq_hz),
        cycle_count=float(cycle_count),
    )
    actual_voltage = _interp(review["time_s"], review["normalized_actual_drive_voltage_v"], time_s)
    raw_hallbz = _interp(review["time_s"], review["raw_hallbz_mT"], time_s)
    effective_field = _interp(review["time_s"], review["measured_field_effective_mT"], time_s)
    baseline_removed_effective = _interp(review["time_s"], review["baseline_removed_effective_field_mT"], time_s)
    normalized_field = _interp(review["time_s"], review["normalized_measured_field_mT"], time_s)
    residual_raw = target - measured
    residual_pointwise = target - measured_smoothed
    measured_aligned, residual_alignment_meta = align_measured_field_for_residual(
        time_s,
        target,
        measured_smoothed,
        active_mask,
        freq_hz=float(freq_hz),
        residual_alignment_mode=alignment_mode,
    )
    residual_aligned = target - measured_aligned
    support_meta = measured_support_metadata(
        review["time_s"],
        freq_hz=float(freq_hz),
        cycle_count=float(cycle_count),
        tail_cycle_count=effective_tail_cycle_count,
        phase_alignment_shift_s=float(residual_alignment_meta.get("phase_alignment_shift_s") or 0.0),
        tail_enabled=bool(post_cycle_zero_tail_enabled),
        measured_support_end_s=float(zero_return_meta["measured_support_end_s"]),
        measured_support_end_mode=str(zero_return_meta["measured_support_end_mode"]),
    )
    target_for_second = target.copy()
    target_for_second[tail_mask] = 0.0
    measured_for_second = measured_aligned.copy() if residual_alignment_meta.get("phase_alignment_status") == "ok" else measured_smoothed.copy()
    measured_for_second, post_active_measured_available = _fill_tail_measured_field(
        measured_for_second,
        measured_smoothed,
        active_mask,
        tail_mask,
    )
    measured_support_valid_mask = np.isfinite(measured_for_second)
    missing_tail_measured = bool(np.any(tail_mask & ~measured_support_valid_mask))
    if tail_return_mode == "residual" and bool(post_cycle_zero_tail_enabled) and missing_tail_measured:
        return profile, {
            **base_metadata,
            **_review_diagnostic_metadata(review_meta),
            **zero_return_meta,
            **support_meta,
            "tail_return_mode": "residual",
            "tail_duration_mode": tail_duration_mode,
            "zero_return_duration_s": float(effective_tail_cycle_count) / max(float(freq_hz), 1e-12),
            "tail_duration_s": float(effective_tail_cycle_count) / max(float(freq_hz), 1e-12),
            "tail_cycle_count": float(effective_tail_cycle_count),
            "tail_cycle_count_equivalent": float(effective_tail_cycle_count),
            "second_modeling_available": False,
            "second_modeling_status": "residual_tail_measured_data_unavailable",
            "second_modeling_unavailable_reason": "phase_shifted_tail_source_range_insufficient",
            "residual_tail_available": False,
            "residual_tail_unavailable_reason": "phase_shifted_tail_source_range_insufficient",
            "measured_tail_synthetic_fill_used": False,
            "measured_tail_last_value_hold_used": False,
            "measured_tail_fake_decay_used": False,
            "measured_tail_actual_data_only": True,
            "measured_support_valid_mask_available": True,
            "second_correction_delta_v_generated": False,
            "second_voltage_v_generated": False,
            "second_lut_generated": False,
        }
    residual_for_second = target_for_second - measured_for_second
    residual_for_second[~(active_mask | tail_mask)] = 0.0
    residual_pointwise_diag = residual_pointwise.copy()
    residual_aligned_diag = residual_aligned.copy()
    residual_pointwise_diag[tail_mask] = residual_for_second[tail_mask]
    residual_aligned_diag[tail_mask] = residual_for_second[tail_mask]
    unit_delta = (residual_for_second / 50.0) * float(voltage_limit_v)
    correction_mask = active_mask | tail_mask
    unit_delta[~correction_mask | ~np.isfinite(unit_delta)] = 0.0
    gain_used, gain_meta = _compute_second_modeling_gain(
        unit_delta,
        first_voltage,
        correction_mask,
        manual_gain=float(correction_gain),
        gain_mode=str(correction_gain_mode),
        voltage_limit_v=float(voltage_limit_v),
        tail_mask=tail_mask,
    )
    raw_delta = unit_delta * float(gain_used)
    delta, stabilization_meta, stabilization_arrays = stabilize_correction_delta(
        raw_delta,
        first_voltage,
        time_s,
        active_mask,
        freq_hz=float(freq_hz),
        cycle_count=float(cycle_count),
        enabled=stabilization_enabled,
        tail_mask=tail_mask,
    )
    smoothed_delta = stabilization_arrays["smoothed_correction_delta_v"]
    stabilized_delta = stabilization_arrays["stabilized_correction_delta_v"]
    second_voltage_before_guard = first_voltage + stabilized_delta
    second_voltage, polarity_meta, polarity_mask = apply_polarity_guard(
        second_voltage_before_guard,
        first_voltage,
        stabilization_arrays["start_gate"],
        enabled=stabilization_enabled,
    )
    second_limited = np.clip(second_voltage, -abs(float(voltage_limit_v)), abs(float(voltage_limit_v)))
    tail_controller_arrays: dict[str, np.ndarray] = {}
    tail_controller_meta: dict[str, Any] = {"tail_return_mode": tail_return_mode}
    if tail_return_mode == "finite_time_zero_return" and bool(post_cycle_zero_tail_enabled):
        tail_controller_arrays, tail_controller_meta = apply_finite_time_zero_return_tail(
            time_s=time_s,
            active_mask=active_mask,
            tail_mask=tail_mask,
            measured_field_for_second_mT=measured_for_second,
            native_measured_field_mT=measured_smoothed,
            actual_drive_voltage_v=actual_voltage,
            first_voltage_v=first_voltage,
            correction_delta_v=stabilized_delta,
            second_voltage_v=second_voltage,
            second_limited_voltage_v=second_limited,
            freq_hz=float(freq_hz),
            voltage_limit_v=float(voltage_limit_v),
            tail_controller_scaling_mode=tail_controller_scaling_mode,
            tail_controller_gain=float(tail_controller_gain),
        )
        if "correction_delta_v" in tail_controller_arrays:
            stabilized_delta = tail_controller_arrays["correction_delta_v"]
            second_voltage = tail_controller_arrays["second_voltage_before_clip_v"]
            second_limited = tail_controller_arrays["second_limited_voltage_v"]
            second_voltage_before_guard = second_voltage.copy()
    stabilization_arrays["tail_window_mask"] = tail_mask
    tail_arrays, tail_meta = _unified_tail_diagnostics(
        time_s,
        target_for_second,
        measured_for_second,
        raw_delta,
        stabilized_delta,
        second_limited,
        active_mask,
        tail_mask,
        freq_hz=float(freq_hz),
        tail_cycle_count=effective_tail_cycle_count,
        post_active_measured_available=post_active_measured_available,
    )
    discontinuity_meta = diagnose_correction_discontinuity(
        time_s,
        stabilized_delta,
        second_limited,
        stabilization_arrays,
        polarity_mask,
    )
    final_voltage = second_limited.copy()

    result = pd.DataFrame(
        {
            "time_s": time_s,
            "physical_target_output_mT": target,
            "target_field_for_second_mT": target_for_second,
            "target_field_active_mT": np.where(active_mask, target_for_second, np.nan),
            "target_field_tail_mT": np.where(tail_mask, target_for_second, np.nan),
            "first_modeled_voltage_v": first_voltage,
            "first_limited_voltage_v": first_limited_voltage,
            "actual_drive_voltage_v": actual_voltage,
            "actual_drive_voltage_normalized_v": actual_voltage,
            "raw_hallbz_mT": raw_hallbz,
            "hallbz_raw_mT": raw_hallbz,
            "measured_field_raw_mT": raw_hallbz,
            "measured_field_effective_mT": effective_field,
            "baseline_removed_effective_field_mT": baseline_removed_effective,
            "measured_field_baseline_removed_mT": baseline_removed_effective,
            "measured_field_normalized_mT": normalized_field,
            "normalized_effective_field_mT": normalized_field,
            "measured_field_smoothed_mT": measured_smoothed,
            "measured_field_smoothed_full_mT": measured_smoothed,
            "measured_field_smoothed_native_mT": measured_smoothed,
            "measured_field_aligned_mT": measured_aligned,
            "measured_field_aligned_native_mT": measured_aligned,
            "measured_field_for_second_mT": measured_for_second,
            "measured_field_tail_source_mT": np.where(tail_mask, measured_for_second, np.nan),
            "second_modeling_measured_field_mT": measured_for_second,
            "first_model_residual_raw_mT": residual_raw,
            "first_model_residual_smoothed_mT": residual_pointwise_diag,
            "first_model_residual_pointwise_mT": residual_pointwise_diag,
            "first_model_residual_aligned_mT": residual_aligned_diag,
            "first_model_residual_for_second_mT": residual_for_second,
            "first_model_residual_mT": residual_for_second,
            "residual_for_second_mT": residual_for_second,
            "residual_active_mT": np.where(active_mask, residual_for_second, np.nan),
            "residual_tail_mT": np.where(tail_mask, residual_for_second, np.nan),
            "unit_delta_v": unit_delta,
            "correction_delta_v_raw": raw_delta,
            "correction_delta_v_smoothed": smoothed_delta,
            "correction_start_gate": stabilization_arrays["start_gate"],
            "correction_tail_taper_gate": stabilization_arrays["taper_gate"],
            "correction_delta_v": stabilized_delta,
            "second_voltage_before_clip_v": second_voltage,
            "raw_correction_delta_v": raw_delta,
            "smoothed_correction_delta_v": smoothed_delta,
            "stabilized_correction_delta_v": stabilized_delta,
            "raw_active_correction_delta_v": raw_delta,
            "smoothed_active_correction_delta_v": smoothed_delta,
            "stabilized_active_correction_delta_v": stabilized_delta,
            "raw_second_correction_delta_v": raw_delta,
            "second_correction_delta_v_smooth": smoothed_delta,
            "second_correction_delta_v": stabilized_delta,
            "start_gate": stabilization_arrays["start_gate"],
            "taper_gate": stabilization_arrays["taper_gate"],
            "correction_envelope": stabilization_arrays["correction_envelope"],
            "second_voltage_before_polarity_guard_v": second_voltage_before_guard,
            "second_voltage_after_polarity_guard_v": second_voltage,
            "second_modeled_voltage_v": second_voltage,
            "second_limited_voltage_v": second_limited,
            "active_correction_delta_v": stabilized_delta,
            "tail_target_field_mT": tail_arrays["tail_target_field_mT"],
            "tail_residual_mT": tail_arrays["tail_residual_mT"],
            "raw_tail_correction_delta_v": tail_arrays["raw_tail_correction_delta_v"],
            "tail_correction_delta_v": tail_arrays["tail_correction_delta_v"],
            "stabilized_tail_voltage_v": tail_arrays["stabilized_tail_voltage_v"],
            "tail_voltage_v": tail_arrays["tail_voltage_v"],
            "tail_B0_mT": tail_controller_arrays.get("tail_B0_mT", np.full(len(time_s), np.nan)),
            "tail_dB0_dt_mT_s": tail_controller_arrays.get("tail_dB0_dt_mT_s", np.full(len(time_s), np.nan)),
            "tail_B_ref_mT": tail_controller_arrays.get("tail_B_ref_mT", np.full(len(time_s), np.nan)),
            "tail_dB_ref_dt_mT_s": tail_controller_arrays.get("tail_dB_ref_dt_mT_s", np.full(len(time_s), np.nan)),
            "tail_model_voltage_v": tail_controller_arrays.get("tail_model_voltage_v", np.full(len(time_s), np.nan)),
            "tail_voltage_before_clip_v": tail_controller_arrays.get("tail_voltage_before_clip_v", np.full(len(time_s), np.nan)),
            "tail_start_voltage_v": tail_arrays["tail_start_voltage_v"],
            "tail_window_mask": tail_mask,
            "post_command_zero_mask": np.zeros(len(time_s), dtype=bool),
            "post_cycle_zero_tail_enabled": bool(post_cycle_zero_tail_enabled),
            "post_cycle_zero_tail_cycle_count": float(np.clip(effective_tail_cycle_count, 0.0, 1.0)),
            "post_cycle_zero_tail_duration_s": tail_setup_meta["post_cycle_zero_tail_duration_s"],
            "correction_nan_mask": stabilization_arrays["correction_nan_mask"],
            "correction_active_mask": stabilization_arrays["correction_active_mask"],
            "source_range_valid_mask": stabilization_arrays["source_range_valid_mask"],
            "measured_support_valid_mask": measured_support_valid_mask,
            "correction_invalid_mask": stabilization_arrays["correction_invalid_mask"],
            "correction_zero_flat_segment_mask": stabilization_arrays["correction_zero_flat_segment_mask"],
            "polarity_guard_applied_mask": polarity_mask,
            "limited_voltage_v": first_limited_voltage,
            "final_voltage_v": final_voltage,
            "active_window_mask": active_mask,
            "production_cycle_policy": PRODUCTION_CYCLE_POLICY,
            "modeling_stage": "second_model",
            "target_unchanged": True,
        }
    )
    if first_recommended_voltage is not None:
        result["recommended_voltage_v"] = first_recommended_voltage
    metadata = {
        **base_metadata,
        "second_modeling_available": True,
        "second_modeling_status": "ok",
        "first_model_source": "command_profile",
        "actual_drive_source_file": record.source_file,
        "hallbz_sign_applied": True,
        "field_normalization_mode": "peak_to_50mT",
        "voltage_normalization_mode": "peak_to_5V_or_limit",
        **_review_diagnostic_metadata(review_meta),
        "interpolation_status": "ok" if np.isfinite(measured).any() else "unavailable",
        "second_command_synthesis_mode": "active_residual_with_finite_time_zero_return_tail"
        if tail_return_mode == "finite_time_zero_return"
        else "unified_active_tail_residual",
        "tail_voltage_overlay_used": False,
        "raw_delta_zeroed_outside_active": False,
        "correction_valid_mask_source": "active_plus_tail_measured_support",
        "measured_field_source_for_second": "aligned_smoothed" if residual_alignment_meta.get("phase_alignment_status") == "ok" else "smoothed",
        "measured_field_source_switch_at_active_end": False,
        "post_active_measured_available": bool(post_active_measured_available),
        "measured_tail_synthetic_fill_used": False,
        "measured_tail_last_value_hold_used": False,
        "measured_tail_fake_decay_used": False,
        "measured_tail_actual_data_only": True,
        "measured_support_missing_time_ranges": _missing_time_ranges(time_s, ~measured_support_valid_mask & (active_mask | tail_mask)),
        "residual_tail_available": not missing_tail_measured,
        "residual_tail_unavailable_reason": None if not missing_tail_measured else "phase_shifted_tail_source_range_insufficient",
        "double_sign_flip_detected": False,
        "field_convention": "raw_hallbz -> effective=-raw -> baseline_removed -> normalized",
        "correction_delta_peak_v": peak_abs(stabilized_delta),
        "voltage_limit_status": "clamped" if np.any(np.abs(second_voltage - second_limited) > 1e-9) else "ok",
        "final_export_voltage_source_column": "second_limited_voltage_v",
        "second_correction_delta_v_generated": True,
        "second_voltage_v_generated": True,
        "second_lut_generated": False,
        "raw_peak_values_informational_only": True,
        "user_decides_final_suitability_from_graphs": True,
        "automatic_pass_fail_judgement": False,
        "field_normalization_scale_factor": review_meta.get("field_normalization_scale_factor"),
        "voltage_normalization_scale_factor": review_meta.get("voltage_normalization_scale_factor"),
        **smoothing_meta,
        **residual_alignment_meta,
        **zero_return_meta,
        **support_meta,
        **gain_meta,
        **stabilization_meta,
        **polarity_meta,
        **discontinuity_meta,
        **tail_setup_meta,
        **tail_meta,
        **tail_controller_meta,
        "tail_duration_mode": tail_duration_mode,
        "zero_return_duration_s": float(effective_tail_cycle_count) / max(float(freq_hz), 1e-12),
        "tail_duration_s": float(effective_tail_cycle_count) / max(float(freq_hz), 1e-12),
        "tail_cycle_count": float(effective_tail_cycle_count),
        "tail_cycle_count_equivalent": float(effective_tail_cycle_count),
        "residual_alignment_mode": residual_alignment_mode,
    }
    return result, metadata


def _review_diagnostic_metadata(review_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline_source": review_meta.get("baseline_source"),
        "baseline_mT": review_meta.get("baseline_mT"),
        "actual_drive_time_unit": review_meta.get("actual_drive_time_unit"),
        "actual_drive_time_unit_detected": review_meta.get("actual_drive_time_unit_detected"),
        "selected_time_unit_reason": review_meta.get("selected_time_unit_reason"),
        "dt_median_s": review_meta.get("dt_median_s"),
        "voltage_nonzero_duration_s": review_meta.get("voltage_nonzero_duration_s"),
        "actual_voltage_active_duration_s": review_meta.get("actual_voltage_active_duration_s"),
        "expected_active_duration_s": review_meta.get("expected_active_duration_s"),
        "active_duration_ratio": review_meta.get("active_duration_ratio"),
        "timebase_status": review_meta.get("timebase_status"),
        "source_time_monotonic": review_meta.get("source_time_monotonic"),
        "duplicate_time_count": review_meta.get("duplicate_time_count"),
        "field_convention": review_meta.get("field_convention"),
    }


def _is_supported_cycle(cycle_count: float) -> bool:
    return any(abs(float(cycle_count) - cycle) <= 1e-9 for cycle in SUPPORTED_SECOND_MODELING_CYCLES)


def _normalize_residual_alignment_mode(value: str) -> str:
    if value == "pointwise":
        return "pointwise"
    if value == "first_peak_aligned":
        return "first_peak_aligned"
    return "first_peak_aligned_stabilized"


def _first_voltage(profile: pd.DataFrame) -> np.ndarray:
    for column in (
        "first_modeled_voltage_v",
        "baseline_limited_voltage_v",
        "feedback_corrected_limited_voltage_v",
        "limited_voltage_v",
        "recommended_voltage_v",
    ):
        if column in profile.columns:
            return pd.to_numeric(profile[column], errors="coerce").to_numpy(dtype=float)
    return np.zeros(len(profile), dtype=float)


def _first_limited_voltage(profile: pd.DataFrame, fallback: np.ndarray) -> np.ndarray:
    if "limited_voltage_v" in profile.columns:
        return pd.to_numeric(profile["limited_voltage_v"], errors="coerce").to_numpy(dtype=float)
    return np.asarray(fallback, dtype=float).copy()


def _optional_voltage(profile: pd.DataFrame, column: str) -> np.ndarray | None:
    if column not in profile.columns:
        return None
    return pd.to_numeric(profile[column], errors="coerce").to_numpy(dtype=float)


def _target(profile: pd.DataFrame, review: pd.DataFrame, time_s: np.ndarray) -> np.ndarray:
    if "physical_target_output_mT" in profile.columns:
        return pd.to_numeric(profile["physical_target_output_mT"], errors="coerce").to_numpy(dtype=float)
    return _interp(review["time_s"], review["normalized_physical_target_output_mT"], time_s)


def _active_mask(time_s: np.ndarray, *, freq_hz: float, cycle_count: float) -> np.ndarray:
    duration = float(cycle_count) / max(float(freq_hz), 1e-12)
    values = np.asarray(time_s, dtype=float)
    return np.isfinite(values) & (values >= -1e-12) & (values <= duration + 1e-12)


def _interp(source_time: Any, source_values: Any, target_time: Any) -> np.ndarray:
    x = pd.to_numeric(pd.Series(source_time), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(source_values), errors="coerce").to_numpy(dtype=float)
    t = pd.to_numeric(pd.Series(target_time), errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() == 0:
        return np.full(len(t), np.nan)
    order = np.argsort(x[finite])
    return np.interp(t, x[finite][order], y[finite][order], left=np.nan, right=np.nan)


def _source_covers_target_active_window(source_time: Any, target_active_time: np.ndarray) -> bool:
    source = pd.to_numeric(pd.Series(source_time), errors="coerce").to_numpy(dtype=float)
    target = np.asarray(target_active_time, dtype=float)
    source = source[np.isfinite(source)]
    target = target[np.isfinite(target)]
    if source.size < 2 or target.size < 1:
        return False
    return bool(np.nanmin(source) <= np.nanmin(target) + 1e-12 and np.nanmax(source) >= np.nanmax(target) - 1e-12)


__all__ = ["generate_second_modeled_voltage_lut"]
