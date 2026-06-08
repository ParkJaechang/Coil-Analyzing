from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .finite_actual_drive import ActualDriveRecord
from .finite_actual_drive import build_actual_drive_review_case
from .finite_actual_drive import read_actual_drive_result
from .finite_actual_drive_normalization import normalize_peak_to_limit
from .finite_actual_drive_normalization import peak_abs
from .voltage_policy import COMMAND_VOLTAGE_LIMIT_V, COMMAND_VOLTAGE_NORMALIZATION_OR_LIMIT_MODE


PRODUCTION_FEEDBACK_PEAK_CYCLES = (1.0, 1.5)
REFERENCE_FEEDBACK_PEAK_CYCLES: tuple[float, ...] = ()
UNSUPPORTED_FEEDBACK_PEAK_CYCLES = (1.25, 1.75, 2.0)
SUPPORTED_FEEDBACK_PEAK_CYCLES = PRODUCTION_FEEDBACK_PEAK_CYCLES
SUGGESTED_REPLACEMENT_CYCLES = {1.25: 1.5, 1.75: 1.5, 2.0: 1.5}
PRODUCTION_CYCLE_POLICY = "1p0_1p5_cycles"
UNSUPPORTED_CYCLE_STATUS = "unsupported_cycle_policy_1p0_1p5_only"
UNSUPPORTED_CYCLE_REASON = "cycle_not_in_1p0_1p5_production_policy"
FEEDBACK_ROUTE_NAME = "finite_feedback_symmetric_peak_correction"


def apply_finite_feedback_peak_correction(
    command_profile: pd.DataFrame,
    feedback_source: str | Path | ActualDriveRecord | dict[str, Any],
    *,
    waveform_type: str,
    freq_hz: float,
    cycle_count: float,
    voltage_limit_v: float = COMMAND_VOLTAGE_LIMIT_V,
    correction_gain: float = 0.25,
    forward_model: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply Quick LUT finite feedback peak correction using actual-drive result data.

    This helper is intentionally command-profile based: it does not alter the
    physical target, and it never treats Raw Waveforms as the modeling route.
    """

    profile = command_profile.copy()
    base_metadata = _base_metadata(
        feedback_source=feedback_source,
        freq_hz=freq_hz,
        cycle_count=cycle_count,
    )
    if not _cycle_supported(cycle_count):
        return profile, {
            **base_metadata,
            "feedback_correction_available": False,
            "feedback_correction_status": UNSUPPORTED_CYCLE_STATUS,
            "feedback_correction_unavailable_reason": UNSUPPORTED_CYCLE_REASON,
            "feedback_used_for_correction": False,
            "target_unchanged": True,
            "correction_delta_v_generated": False,
            "second_voltage_v_generated": False,
            "second_lut_generated": False,
        }

    required = {"time_s", "limited_voltage_v"}
    missing = sorted(required - set(profile.columns))
    if missing:
        return profile, {
            **base_metadata,
            "feedback_correction_available": False,
            "feedback_correction_status": "missing_command_columns",
            "feedback_correction_unavailable_reason": ",".join(missing),
            "feedback_used_for_correction": False,
            "target_unchanged": True,
        }

    record = _load_feedback_record(feedback_source)
    review_frame, review_metadata = build_actual_drive_review_case(record)
    target = _target_array(profile, review_frame)
    time_s = pd.to_numeric(profile["time_s"], errors="coerce").to_numpy(dtype=float)
    measured = _interp(review_frame["time_s"], review_frame["normalized_measured_field_mT"], time_s)
    actual_voltage = _interp(review_frame["time_s"], review_frame["normalized_actual_drive_voltage_v"], time_s)
    raw_hallbz = _interp(record.frame["time_s_abs"] - review_metadata["command_start_s"], record.frame["hallbz_raw_mT"], time_s)
    signed_field = -raw_hallbz
    signed_normalized, field_norm_meta = normalize_peak_to_limit(
        signed_field,
        _finite_mask(time_s, freq_hz=freq_hz, cycle_count=cycle_count),
        limit=50.0,
        unavailable_status="unavailable_zero_peak",
    )
    residual = target - measured
    active_mask = _finite_mask(time_s, freq_hz=freq_hz, cycle_count=cycle_count)
    positive_mask = active_mask & (target > 1e-9)
    negative_mask = active_mask & (target < -1e-9)

    if not positive_mask.any() or not negative_mask.any():
        return profile, {
            **base_metadata,
            "feedback_correction_available": False,
            "feedback_correction_status": "unavailable_missing_lobe",
            "feedback_correction_unavailable_reason": "missing_positive_or_negative_lobe",
            "feedback_used_for_correction": False,
            "target_unchanged": True,
        }

    baseline_limited = pd.to_numeric(profile["limited_voltage_v"], errors="coerce").to_numpy(dtype=float)
    baseline_recommended = (
        pd.to_numeric(profile["recommended_voltage_v"], errors="coerce").to_numpy(dtype=float)
        if "recommended_voltage_v" in profile.columns
        else baseline_limited.copy()
    )
    raw_delta = correction_gain * (residual / 50.0) * float(voltage_limit_v)
    raw_delta[~active_mask | ~np.isfinite(raw_delta)] = 0.0
    correction_delta = _smooth(raw_delta)
    corrected_recommended = baseline_recommended + correction_delta
    corrected_limited = np.clip(corrected_recommended, -abs(float(voltage_limit_v)), abs(float(voltage_limit_v)))
    clipped = bool(np.any(np.abs(corrected_recommended - corrected_limited) > 1e-9))

    profile["measured_field_raw_mT"] = raw_hallbz
    profile["measured_field_signed_mT"] = signed_field
    profile["measured_field_normalized_mT"] = signed_normalized
    profile["measured_residual_mT"] = target - signed_normalized
    profile["actual_first_voltage_v"] = actual_voltage
    profile["baseline_recommended_voltage_v"] = baseline_recommended
    profile["baseline_limited_voltage_v"] = baseline_limited
    profile["feedback_correction_delta_v"] = correction_delta
    profile["feedback_corrected_recommended_voltage_v"] = corrected_recommended
    profile["feedback_corrected_limited_voltage_v"] = corrected_limited
    profile["active_limited_voltage_v"] = corrected_limited
    profile["limited_voltage_v"] = corrected_limited
    profile["active_command_source"] = "feedback_corrected_limited_voltage_v"
    profile["plotted_command_source"] = "feedback_corrected_limited_voltage_v"
    profile["exported_voltage_source_column"] = "feedback_corrected_limited_voltage_v"
    profile["run_waveform_voltage_source"] = "feedback_corrected_limited_voltage_v"
    profile["positive_lobe_mask"] = positive_mask
    profile["negative_lobe_mask"] = negative_mask
    profile["feedback_correction_status"] = "ok"
    profile["feedback_correction_available"] = True
    profile["feedback_route"] = FEEDBACK_ROUTE_NAME
    profile["feedback_used_for_correction"] = True
    profile["correction_method"] = "residual_proportional_feedback"

    prediction_available = forward_model is not None
    if forward_model is not None:
        predicted = np.asarray(forward_model(time_s, corrected_limited), dtype=float)
        if len(predicted) == len(profile):
            profile["feedback_corrected_predicted_field_mT"] = predicted
            profile["displayed_predicted_field_mT"] = predicted
        else:
            prediction_available = False
    profile["predicted_from_plotted_command"] = bool(prediction_available)
    profile["displayed_predicted_valid"] = bool(prediction_available)
    profile["plotted_predicted_source"] = (
        "feedback_corrected_predicted_field_mT" if prediction_available else "unavailable"
    )
    profile["command_prediction_consistency_status"] = (
        "ok" if prediction_available else "forward_prediction_unavailable_for_feedback_corrected_command"
    )

    before_pos = _peak_error(target, signed_normalized, positive_mask)
    before_neg = _peak_error(target, signed_normalized, negative_mask)
    after_pos = float("nan")
    after_neg = float("nan")
    after_symmetry = float("nan")
    if prediction_available and "feedback_corrected_predicted_field_mT" in profile.columns:
        predicted_values = profile["feedback_corrected_predicted_field_mT"].to_numpy(dtype=float)
        after_pos = _peak_error(target, predicted_values, positive_mask)
        after_neg = _peak_error(target, predicted_values, negative_mask)
        after_symmetry = _symmetry_error(predicted_values, positive_mask, negative_mask)

    metadata = {
        **base_metadata,
        **_startup_diagnostics(
            time_s,
            signed_field,
            signed_normalized,
            active_mask,
            freq_hz=freq_hz,
            cycle_count=cycle_count,
        ),
        "feedback_correction_available": True,
        "feedback_correction_status": "ok",
        "feedback_route": FEEDBACK_ROUTE_NAME,
        "correction_method": "residual_proportional_feedback",
        "feedback_source_file": record.source_file,
        "feedback_run_label": _run_label(record.source_file),
        "feedback_schema_status": "ok",
        "feedback_alignment_status": str(review_metadata.get("alignment_status") or "ok"),
        "feedback_used_for_correction": True,
        "hallbz_sign_applied": True,
        "hallbz_effective_convention": "effective_field_mT=-HallBz_raw",
        "field_normalization_mode": "peak_to_50mT",
        "voltage_normalization_mode": COMMAND_VOLTAGE_NORMALIZATION_OR_LIMIT_MODE,
        "field_normalization_scale_factor": field_norm_meta["scale_factor"],
        "voltage_normalization_scale_factor": review_metadata.get("voltage_normalization_scale_factor"),
        "raw_field_peak_mT": peak_abs(raw_hallbz[active_mask]),
        "raw_peak_values_informational_only": True,
        "normalized_field_peak_mT": peak_abs(signed_normalized[active_mask]),
        "raw_voltage_peak_v": review_metadata.get("raw_voltage_peak_v"),
        "normalized_voltage_peak_v": review_metadata.get("normalized_voltage_peak_v"),
        "alignment_status": str(review_metadata.get("alignment_status") or "ok"),
        "alignment_time_shift_s": float(review_metadata.get("alignment_offset_s") or 0.0),
        "alignment_confidence": review_metadata.get("alignment_confidence"),
        "positive_peak_error_before_mT": before_pos,
        "negative_peak_error_before_mT": before_neg,
        "positive_peak_error_after_mT": after_pos,
        "negative_peak_error_after_mT": after_neg,
        "peak_symmetry_error_before_mT": _symmetry_error(signed_normalized, positive_mask, negative_mask),
        "peak_symmetry_error_after_mT": after_symmetry,
        "correction_delta_peak_v": peak_abs(correction_delta),
        "voltage_limit_status": "clamped" if clipped else "ok",
        "target_unchanged": True,
        "active_command_source": "feedback_corrected_limited_voltage_v",
        "plotted_command_source": "feedback_corrected_limited_voltage_v",
        "exported_voltage_source_column": "feedback_corrected_limited_voltage_v",
        "run_waveform_voltage_source": "feedback_corrected_limited_voltage_v",
        "forward_prediction_available": bool(prediction_available),
        "predicted_from_plotted_command": bool(prediction_available),
        "displayed_predicted_valid": bool(prediction_available),
        "command_prediction_consistency_status": (
            "ok" if prediction_available else "forward_prediction_unavailable_for_feedback_corrected_command"
        ),
        "plotted_predicted_source": (
            "feedback_corrected_predicted_field_mT" if prediction_available else "unavailable"
        ),
        "correction_delta_v_generated": True,
        "second_voltage_v_generated": False,
        "second_lut_generated": False,
    }
    return profile, metadata


def _load_feedback_record(source: str | Path | ActualDriveRecord | dict[str, Any]) -> ActualDriveRecord:
    if isinstance(source, ActualDriveRecord):
        return source
    if isinstance(source, dict) and isinstance(source.get("record"), ActualDriveRecord):
        return source["record"]
    return read_actual_drive_result(Path(source))


def _base_metadata(*, feedback_source: object, freq_hz: float, cycle_count: float) -> dict[str, Any]:
    return {
        "feedback_route": FEEDBACK_ROUTE_NAME,
        "feedback_source_file": Path(str(feedback_source)).name if isinstance(feedback_source, (str, Path)) else None,
        "feedback_run_label": _run_label(Path(str(feedback_source)).name) if isinstance(feedback_source, (str, Path)) else "unknown",
        "freq_hz": float(freq_hz),
        "cycle_count": float(cycle_count),
        "supported_feedback_peak_cycles": list(SUPPORTED_FEEDBACK_PEAK_CYCLES),
        "production_cycle_policy": PRODUCTION_CYCLE_POLICY,
        **_cycle_policy_metadata(cycle_count),
    }


def _cycle_supported(cycle_count: float) -> bool:
    return any(abs(float(cycle_count) - supported) <= 1e-9 for supported in SUPPORTED_FEEDBACK_PEAK_CYCLES)


def _cycle_policy_metadata(cycle_count: float) -> dict[str, Any]:
    return {
        "production_supported_cycles": list(PRODUCTION_FEEDBACK_PEAK_CYCLES),
        "reference_supported_cycles": list(REFERENCE_FEEDBACK_PEAK_CYCLES),
        "unsupported_cycles": list(UNSUPPORTED_FEEDBACK_PEAK_CYCLES),
        "cycle_policy": _cycle_policy(cycle_count),
        "suggested_replacement_cycle": _suggested_replacement_cycle(cycle_count),
    }


def _cycle_policy(cycle_count: float) -> str:
    value = float(cycle_count)
    if any(abs(value - cycle) <= 1e-9 for cycle in PRODUCTION_FEEDBACK_PEAK_CYCLES):
        return "production_supported"
    if any(abs(value - cycle) <= 1e-9 for cycle in REFERENCE_FEEDBACK_PEAK_CYCLES):
        return "reference_supported"
    if any(abs(value - cycle) <= 1e-9 for cycle in UNSUPPORTED_FEEDBACK_PEAK_CYCLES):
        return UNSUPPORTED_CYCLE_STATUS
    return "unsupported_unknown_cycle"


def _suggested_replacement_cycle(cycle_count: float) -> float | None:
    value = float(cycle_count)
    for cycle, replacement in SUGGESTED_REPLACEMENT_CYCLES.items():
        if abs(value - cycle) <= 1e-9:
            return replacement
    return None


def _target_array(profile: pd.DataFrame, review_frame: pd.DataFrame) -> np.ndarray:
    if "physical_target_output_mT" in profile.columns:
        return pd.to_numeric(profile["physical_target_output_mT"], errors="coerce").to_numpy(dtype=float)
    return _interp(review_frame["time_s"], review_frame["normalized_physical_target_output_mT"], profile["time_s"])


def _finite_mask(time_s: np.ndarray, *, freq_hz: float, cycle_count: float) -> np.ndarray:
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
    x_valid = x[finite][order]
    y_valid = y[finite][order]
    return np.interp(t, x_valid, y_valid, left=np.nan, right=np.nan)


def _smooth(values: np.ndarray, window: int = 7) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    return series.rolling(window=window, center=True, min_periods=1).mean().to_numpy(dtype=float)


def _peak_error(target: np.ndarray, measured: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    target_peak = _signed_peak(target[mask])
    measured_peak = _signed_peak(measured[mask])
    return float(target_peak - measured_peak)


def _signed_peak(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    max_abs_index = int(np.nanargmax(np.abs(finite)))
    return float(finite[max_abs_index])


def _symmetry_error(values: np.ndarray, positive_mask: np.ndarray, negative_mask: np.ndarray) -> float:
    pos = peak_abs(values[positive_mask])
    neg = peak_abs(values[negative_mask])
    return float(abs(pos - neg)) if np.isfinite(pos) and np.isfinite(neg) else float("nan")


def _startup_diagnostics(
    time_s: np.ndarray,
    signed_field: np.ndarray,
    normalized_field: np.ndarray,
    active_mask: np.ndarray,
    *,
    freq_hz: float,
    cycle_count: float,
) -> dict[str, Any]:
    duration_s = float(cycle_count) / max(float(freq_hz), 1e-12)
    if not active_mask.any() or not np.isfinite(duration_s) or duration_s <= 0:
        return _startup_unavailable("unavailable_no_active_window")

    startup_duration_s = min(duration_s, max(duration_s * 0.2, 0.25 / max(float(freq_hz), 1e-12)))
    startup_start_s = 0.0
    startup_end_s = float(startup_duration_s)
    startup_mask = active_mask & (time_s >= startup_start_s - 1e-12) & (time_s <= startup_end_s + 1e-12)
    steady_mask = active_mask & (time_s >= startup_end_s - 1e-12)
    if startup_mask.sum() < 3 or steady_mask.sum() < 3:
        return _startup_unavailable("insufficient_window", start_s=startup_start_s, end_s=startup_end_s)

    raw_offset = _mean_delta(signed_field, startup_mask, steady_mask)
    normalized_offset = _mean_delta(normalized_field, startup_mask, steady_mask)
    slope = _linear_slope(time_s[startup_mask], normalized_field[startup_mask])
    return {
        "startup_offset_mT": raw_offset,
        "startup_offset_normalized_mT": normalized_offset,
        "startup_offset_ratio": abs(normalized_offset) / 50.0 if np.isfinite(normalized_offset) else float("nan"),
        "startup_decay_slope": slope,
        "startup_window_start_s": startup_start_s,
        "startup_window_end_s": startup_end_s,
        "startup_offset_status": "ok",
    }


def _startup_unavailable(reason: str, *, start_s: float = float("nan"), end_s: float = float("nan")) -> dict[str, Any]:
    return {
        "startup_offset_mT": float("nan"),
        "startup_offset_normalized_mT": float("nan"),
        "startup_offset_ratio": float("nan"),
        "startup_decay_slope": float("nan"),
        "startup_window_start_s": start_s,
        "startup_window_end_s": end_s,
        "startup_offset_status": reason,
    }


def _mean_delta(values: np.ndarray, first_mask: np.ndarray, second_mask: np.ndarray) -> float:
    first = np.asarray(values, dtype=float)[first_mask]
    second = np.asarray(values, dtype=float)[second_mask]
    if first.size == 0 or second.size == 0:
        return float("nan")
    return float(np.nanmean(first) - np.nanmean(second))


def _linear_slope(time_s: np.ndarray, values: np.ndarray) -> float:
    finite = np.isfinite(time_s) & np.isfinite(values)
    if finite.sum() < 2:
        return float("nan")
    x = np.asarray(time_s, dtype=float)[finite]
    y = np.asarray(values, dtype=float)[finite]
    return float(np.polyfit(x - x[0], y, deg=1)[0])


def _run_label(source_file: str | None) -> str:
    text = (source_file or "").lower()
    if "second" in text or "2nd" in text:
        return "second_run"
    if "first" in text or "1st" in text:
        return "first_run"
    return "unknown"
