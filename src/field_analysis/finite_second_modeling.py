from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .finite_actual_drive import ActualDriveRecord
from .finite_actual_drive import build_actual_drive_review_case
from .finite_actual_drive import read_actual_drive_result
from .finite_actual_drive_normalization import peak_abs

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
    voltage_limit_v: float = 5.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    profile = first_command_profile.copy()
    base_metadata = {
        "second_modeling_method": "residual_proportional_feedback",
        "correction_gain": float(correction_gain),
        "production_cycle_policy": PRODUCTION_CYCLE_POLICY,
        "supported_production_cycles": list(SUPPORTED_SECOND_MODELING_CYCLES),
        "unsupported_cycles": list(UNSUPPORTED_SECOND_MODELING_CYCLES),
        "fourier_resynthesis_involved": False,
        "harmonic_export_involved": False,
        "target_unchanged": True,
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
    time_s = pd.to_numeric(profile["time_s"], errors="coerce").to_numpy(dtype=float)
    first_voltage = _first_voltage(profile)
    active_mask = _active_mask(time_s, freq_hz=freq_hz, cycle_count=cycle_count)
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
    target = _target(profile, review, time_s)
    measured = _interp(review["time_s"], review["normalized_measured_field_mT"], time_s)
    actual_voltage = _interp(review["time_s"], review["normalized_actual_drive_voltage_v"], time_s)
    raw_hallbz = _interp(review["time_s"], review["raw_hallbz_mT"], time_s)
    effective_field = _interp(review["time_s"], review["measured_field_effective_mT"], time_s)
    baseline_removed_effective = _interp(review["time_s"], review["baseline_removed_effective_field_mT"], time_s)
    normalized_field = _interp(review["time_s"], review["normalized_measured_field_mT"], time_s)
    residual = target - measured
    delta = (residual / 50.0) * float(voltage_limit_v) * float(correction_gain)
    delta[~active_mask | ~np.isfinite(delta)] = 0.0
    delta = _smooth(delta)
    second_voltage = first_voltage + delta
    second_limited = np.clip(second_voltage, -abs(float(voltage_limit_v)), abs(float(voltage_limit_v)))
    final_voltage = second_limited.copy()

    result = pd.DataFrame(
        {
            "time_s": time_s,
            "physical_target_output_mT": target,
            "first_modeled_voltage_v": first_voltage,
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
            "first_model_residual_mT": residual,
            "second_correction_delta_v": delta,
            "second_modeled_voltage_v": second_voltage,
            "second_limited_voltage_v": second_limited,
            "limited_voltage_v": second_limited,
            "final_voltage_v": final_voltage,
            "active_window_mask": active_mask,
            "production_cycle_policy": PRODUCTION_CYCLE_POLICY,
            "modeling_stage": "second_model",
            "target_unchanged": True,
        }
    )
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
        "double_sign_flip_detected": False,
        "field_convention": "raw_hallbz -> effective=-raw -> baseline_removed -> normalized",
        "correction_delta_peak_v": peak_abs(delta),
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


def _first_voltage(profile: pd.DataFrame) -> np.ndarray:
    for column in ("final_voltage_v", "feedback_corrected_limited_voltage_v", "limited_voltage_v", "recommended_voltage_v"):
        if column in profile.columns:
            return pd.to_numeric(profile[column], errors="coerce").to_numpy(dtype=float)
    return np.zeros(len(profile), dtype=float)


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


def _smooth(values: np.ndarray, window: int = 7) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rolling(window=window, center=True, min_periods=1).mean().to_numpy(dtype=float)


__all__ = ["generate_second_modeled_voltage_lut"]
