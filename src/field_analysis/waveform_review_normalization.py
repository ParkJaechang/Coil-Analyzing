from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .finite_actual_drive_normalization import normalize_peak_to_limit
from .finite_actual_drive_normalization import peak_abs


PRODUCTION_FINITE_SYMMETRIC_CYCLES = [1.5, 2.0]
REFERENCE_FINITE_SYMMETRIC_CYCLES = [1.0]
UNSUPPORTED_FINITE_SYMMETRIC_CYCLES = [1.25, 1.75]
SUPPORTED_FINITE_SYMMETRIC_CYCLES = [*REFERENCE_FINITE_SYMMETRIC_CYCLES, *PRODUCTION_FINITE_SYMMETRIC_CYCLES]
FIELD_REVIEW_LIMIT_MT = 50.0
VOLTAGE_REVIEW_LIMIT_V = 5.0


def normalize_raw_waveform_frame(
    frame: pd.DataFrame,
    *,
    source_type: str,
    freq_hz: float,
    cycle_count: float,
    field_column: str | None = None,
    voltage_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if str(source_type).replace("_", "-") == "finite-cycle":
        return normalize_finite_waveform_frame(
            frame,
            freq_hz=freq_hz,
            cycle_count=cycle_count,
            field_column=field_column,
            voltage_column=voltage_column,
        )
    return normalize_continuous_waveform_frame(
        frame,
        freq_hz=freq_hz,
        field_column=field_column,
    )


def normalize_continuous_waveform_frame(
    frame: pd.DataFrame,
    *,
    freq_hz: float,
    field_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = frame.copy()
    field_name = field_column or _preferred_field_column(result)
    if field_name is None or "time_s" not in result.columns:
        return result, _unavailable_continuous_metadata("missing_key_columns")

    time_s = pd.to_numeric(result["time_s"], errors="coerce").to_numpy(dtype=float)
    raw_field = pd.to_numeric(result[field_name], errors="coerce").to_numpy(dtype=float)
    startup_end = _first_scalar(result, ("startup_window_end_s", "steady_state_start_s"))
    steady_start = _first_scalar(result, ("steady_state_start_s",))
    steady_end = _first_scalar(result, ("steady_state_end_s",))
    if not np.isfinite(steady_start):
        period_s = 1.0 / float(freq_hz) if np.isfinite(freq_hz) and freq_hz > 0 else 0.0
        steady_start = float(np.nanmin(time_s) + period_s) if period_s > 0 else float(np.nanmin(time_s))
    if not np.isfinite(startup_end):
        startup_end = steady_start
    if not np.isfinite(steady_end):
        steady_end = float(np.nanmax(time_s))
    steady_mask = np.isfinite(time_s) & (time_s >= steady_start - 1e-12) & (time_s <= steady_end + 1e-12)
    baseline = _pre_window_baseline(time_s, raw_field, steady_start)
    baseline_removed = raw_field - baseline
    normalized, norm_meta = normalize_peak_to_limit(
        baseline_removed,
        steady_mask,
        limit=FIELD_REVIEW_LIMIT_MT,
        unavailable_status="unavailable_zero_peak",
    )
    result["raw_continuous_field_mT"] = raw_field
    result["normalized_continuous_field_mT"] = normalized
    metadata = {
        "waveform_normalization_enabled": norm_meta["status"] == "ok",
        "waveform_normalization_mode": "steady_state_peak_to_50mT",
        "waveform_normalization_window": "steady_state",
        "waveform_normalization_source_peak_mT": norm_meta["source_peak"],
        "waveform_normalization_scale_factor": norm_meta["scale_factor"],
        "waveform_normalization_status": norm_meta["status"],
        "raw_field_peak_mT": peak_abs(raw_field[steady_mask]),
        "normalized_field_peak_mT": peak_abs(normalized[steady_mask]),
        "startup_excluded": bool(np.isfinite(startup_end) and np.isfinite(steady_start) and steady_start >= startup_end - 1e-12),
        "startup_window_start_s": float(np.nanmin(time_s)) if np.isfinite(time_s).any() else float("nan"),
        "startup_window_end_s": float(startup_end),
        "steady_state_start_s": float(steady_start),
        "steady_state_end_s": float(steady_end),
        "steady_state_duration_s": float(max(steady_end - steady_start, 0.0)),
        "continuous_evaluation_window": "steady_state",
        "steady_state_window_source": "metadata_or_last_cycles",
        "steady_state_window_reason": "startup_excluded_for_shape_review",
    }
    _attach_metadata_columns(result, metadata)
    return result, metadata


def normalize_finite_waveform_frame(
    frame: pd.DataFrame,
    *,
    freq_hz: float,
    cycle_count: float,
    field_column: str | None = None,
    voltage_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = frame.copy()
    field_name = field_column or _preferred_field_column(result)
    if field_name is None or "time_s" not in result.columns:
        return result, _unavailable_finite_metadata("missing_key_columns")
    time_s = pd.to_numeric(result["time_s"], errors="coerce").to_numpy(dtype=float)
    raw_field = pd.to_numeric(result[field_name], errors="coerce").to_numpy(dtype=float)
    voltage_name = voltage_column or _preferred_voltage_column(result)
    voltage = pd.to_numeric(result[voltage_name], errors="coerce").to_numpy(dtype=float) if voltage_name else None
    start_s = _active_start(time_s, voltage, raw_field)
    expected_duration = _expected_duration(freq_hz=freq_hz, cycle_count=cycle_count)
    end_s = start_s + expected_duration if np.isfinite(start_s) and np.isfinite(expected_duration) else float("nan")
    active_mask = np.isfinite(time_s) & (time_s >= start_s - 1e-12) & (time_s <= end_s + 1e-12)
    baseline = _pre_window_baseline(time_s, raw_field, start_s)
    baseline_removed = raw_field - baseline
    normalized, norm_meta = normalize_peak_to_limit(
        baseline_removed,
        active_mask,
        limit=FIELD_REVIEW_LIMIT_MT,
        unavailable_status="unavailable_zero_peak",
    )
    positive_raw = _positive_peak(baseline_removed[active_mask])
    negative_raw = _negative_peak(baseline_removed[active_mask])
    positive_norm = _positive_peak(normalized[active_mask])
    negative_norm = _negative_peak(normalized[active_mask])
    result["raw_finite_field_mT"] = raw_field
    result["normalized_finite_field_mT"] = normalized
    result["active_window_mask"] = active_mask
    metadata = {
        "finite_normalization_enabled": norm_meta["status"] == "ok",
        "finite_normalization_mode": "active_peak_to_50mT",
        "finite_normalization_status": norm_meta["status"],
        "finite_active_window_start_s": float(start_s),
        "finite_active_window_end_s": float(end_s),
        "finite_active_window_duration_s": float(expected_duration),
        "finite_positive_peak_raw_mT": positive_raw,
        "finite_negative_peak_raw_mT": negative_raw,
        "finite_positive_peak_normalized_mT": positive_norm,
        "finite_negative_peak_normalized_mT": negative_norm,
        "finite_normalization_scale_factor": norm_meta["scale_factor"],
        "finite_normalization_source_peak_mT": norm_meta["source_peak"],
        "raw_field_peak_mT": peak_abs(raw_field[active_mask]),
        "normalized_field_peak_mT": peak_abs(normalized[active_mask]),
        "source_pre_baseline_excluded_from_reference": bool(np.isfinite(start_s) and start_s > np.nanmin(time_s)),
        "source_tail_excluded_from_reference": bool(np.isfinite(end_s) and end_s < np.nanmax(time_s)),
    }
    _attach_metadata_columns(result, metadata)
    return result, metadata


def build_finite_symmetric_peak_review(
    frame: pd.DataFrame,
    *,
    freq_hz: float,
    cycle_count: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not _cycle_supported(cycle_count):
        return frame, _symmetric_metadata(status="unsupported_cycle", enabled=False, cycle_count=cycle_count)
    if "normalized_finite_field_mT" not in frame.columns or "time_s" not in frame.columns:
        return frame, _symmetric_metadata(status="unavailable_no_active_window", enabled=False, cycle_count=cycle_count)
    result = frame.copy()
    time_s = pd.to_numeric(result["time_s"], errors="coerce").to_numpy(dtype=float)
    field = pd.to_numeric(result["normalized_finite_field_mT"], errors="coerce").to_numpy(dtype=float)
    raw_field = pd.to_numeric(result.get("raw_finite_field_mT", result["normalized_finite_field_mT"]), errors="coerce").to_numpy(dtype=float)
    active_mask = _active_mask_from_frame(result, time_s)
    if not active_mask.any():
        return frame, _symmetric_metadata(status="unavailable_no_active_window", enabled=False, cycle_count=cycle_count)
    if _source_quality_bad(raw_field):
        return result, _symmetric_metadata(status="unavailable_bad_source_quality", enabled=False, cycle_count=cycle_count)
    pos_mask = active_mask & (field > 0.0)
    neg_mask = active_mask & (field < 0.0)
    if not pos_mask.any() or not neg_mask.any():
        return result, _symmetric_metadata(status="unavailable_missing_lobe", enabled=False, cycle_count=cycle_count)
    pos_peak = _positive_peak(field[active_mask])
    neg_peak = _negative_peak(field[active_mask])
    pos_abs = abs(pos_peak)
    neg_abs = abs(neg_peak)
    if pos_abs <= 1e-12 or neg_abs <= 1e-12:
        return result, _symmetric_metadata(status="unavailable_missing_lobe", enabled=False, cycle_count=cycle_count)
    voltage_name = _preferred_voltage_column(result)
    baseline_voltage = (
        pd.to_numeric(result[voltage_name], errors="coerce").to_numpy(dtype=float)
        if voltage_name
        else np.zeros(len(result), dtype=float)
    )
    baseline_voltage, voltage_norm = normalize_peak_to_limit(
        baseline_voltage,
        np.isfinite(baseline_voltage),
        limit=VOLTAGE_REVIEW_LIMIT_V,
        unavailable_status="unavailable_zero_peak",
    )
    pos_gain = FIELD_REVIEW_LIMIT_MT / pos_abs
    neg_gain = FIELD_REVIEW_LIMIT_MT / neg_abs
    lobe_scale = np.ones(len(result), dtype=float)
    lobe_scale[pos_mask] = pos_gain
    lobe_scale[neg_mask] = neg_gain
    symmetric_voltage = baseline_voltage * lobe_scale
    symmetric_voltage, limit_applied = _limit_voltage(symmetric_voltage, VOLTAGE_REVIEW_LIMIT_V)
    result["baseline_recommended_voltage_v"] = baseline_voltage
    result["symmetric_peak_recommended_voltage_v"] = symmetric_voltage
    result["symmetric_peak_command_delta_v"] = symmetric_voltage - baseline_voltage
    result["positive_lobe_mask"] = pos_mask
    result["negative_lobe_mask"] = neg_mask
    result["symmetric_peak_predicted_field_mT"] = field
    metrics = _shape_metrics(field[active_mask])
    metadata = {
        **_symmetric_metadata(status="ok", enabled=True, cycle_count=cycle_count),
        "positive_peak_mT": pos_peak,
        "negative_peak_mT": neg_peak,
        "peak_symmetry_error_mT": abs(pos_abs - neg_abs),
        "peak_symmetry_ratio": min(pos_abs, neg_abs) / max(pos_abs, neg_abs),
        "normalized_peak_target_mT": FIELD_REVIEW_LIMIT_MT,
        "active_shape_corr": metrics["active_shape_corr"],
        "active_nrmse": metrics["active_nrmse"],
        "zero_crossing_time_error_s": float("nan"),
        "terminal_tail_residual": _tail_residual(field, active_mask),
        "command_voltage_peak_v": peak_abs(symmetric_voltage),
        "command_voltage_limit_status": "ok",
        "lobe_gain_positive": pos_gain,
        "lobe_gain_negative": neg_gain,
        "lobe_balance_applied": True,
        "lobe_balance_scale_positive": pos_gain,
        "lobe_balance_scale_negative": neg_gain,
        "lobe_balance_smoothing_applied": False,
        "voltage_limit_applied": bool(limit_applied or voltage_norm["status"] == "ok"),
    }
    _attach_metadata_columns(result, metadata)
    return result, metadata


def _symmetric_metadata(*, status: str, enabled: bool, cycle_count: float) -> dict[str, Any]:
    return {
        "finite_symmetric_peak_modeling_enabled": enabled,
        "finite_symmetric_peak_cycle_supported": _cycle_supported(cycle_count),
        "finite_symmetric_peak_status": status,
        "supported_finite_symmetric_cycles": list(SUPPORTED_FINITE_SYMMETRIC_CYCLES),
        "production_supported_finite_symmetric_cycles": list(PRODUCTION_FINITE_SYMMETRIC_CYCLES),
        "reference_supported_finite_symmetric_cycles": list(REFERENCE_FINITE_SYMMETRIC_CYCLES),
        "unsupported_finite_symmetric_cycles": list(UNSUPPORTED_FINITE_SYMMETRIC_CYCLES),
        "finite_symmetric_peak_cycle_role": _cycle_role(cycle_count),
    }


def _cycle_supported(cycle_count: float) -> bool:
    return any(abs(float(cycle_count) - supported) <= 1e-9 for supported in SUPPORTED_FINITE_SYMMETRIC_CYCLES)


def _cycle_role(cycle_count: float) -> str:
    value = float(cycle_count)
    if any(abs(value - cycle) <= 1e-9 for cycle in PRODUCTION_FINITE_SYMMETRIC_CYCLES):
        return "production"
    if any(abs(value - cycle) <= 1e-9 for cycle in REFERENCE_FINITE_SYMMETRIC_CYCLES):
        return "reference_legacy"
    if any(abs(value - cycle) <= 1e-9 for cycle in UNSUPPORTED_FINITE_SYMMETRIC_CYCLES):
        return "unsupported_review_only"
    return "unsupported_unknown"


def _preferred_field_column(frame: pd.DataFrame) -> str | None:
    for column in ("bz_mT", "bmag_mT", "bx_mT", "by_mT", "measured_field_mT", "HallBz"):
        if column in frame.columns:
            return column
    return None


def _preferred_voltage_column(frame: pd.DataFrame) -> str | None:
    for column in (
        "daq_input_v",
        "limited_voltage_v",
        "recommended_voltage_v",
        "first_voltage_v",
        "Voltage1_V",
    ):
        if column in frame.columns:
            return column
    return None


def _expected_duration(*, freq_hz: float, cycle_count: float) -> float:
    return float(cycle_count / freq_hz) if np.isfinite(freq_hz) and freq_hz > 0 and np.isfinite(cycle_count) else float("nan")


def _active_start(time_s: np.ndarray, voltage: np.ndarray | None, field: np.ndarray) -> float:
    if voltage is not None:
        start = _motion_start(time_s, voltage)
        if np.isfinite(start):
            return start
    start = _motion_start(time_s, field)
    if np.isfinite(start):
        return start
    return float(np.nanmin(time_s)) if np.isfinite(time_s).any() else float("nan")


def _motion_start(time_s: np.ndarray, values: np.ndarray) -> float:
    finite = np.isfinite(time_s) & np.isfinite(values)
    if not finite.any():
        return float("nan")
    finite_values = values[finite]
    peak = peak_abs(finite_values)
    if not np.isfinite(peak) or peak <= 1e-12:
        return float("nan")
    threshold = max(peak * 0.02, 1e-9)
    active = finite & (np.abs(values) > threshold)
    if not active.any():
        return float("nan")
    return float(time_s[active][0])


def _pre_window_baseline(time_s: np.ndarray, values: np.ndarray, start_s: float) -> float:
    pre_mask = np.isfinite(time_s) & np.isfinite(values) & (time_s < start_s - 1e-12)
    if pre_mask.any():
        return float(np.nanmedian(values[pre_mask]))
    finite = values[np.isfinite(values)]
    return float(np.nanmedian(finite)) if finite.size else 0.0


def _first_scalar(frame: pd.DataFrame, columns: tuple[str, ...]) -> float:
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if not values.empty:
            return float(values.iloc[0])
    return float("nan")


def _positive_peak(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.nanmax(finite)) if finite.size else float("nan")


def _negative_peak(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.nanmin(finite)) if finite.size else float("nan")


def _active_mask_from_frame(frame: pd.DataFrame, time_s: np.ndarray) -> np.ndarray:
    if "active_window_mask" in frame.columns:
        return frame["active_window_mask"].astype(bool).to_numpy()
    start = _first_scalar(frame, ("finite_active_window_start_s",))
    end = _first_scalar(frame, ("finite_active_window_end_s",))
    if np.isfinite(start) and np.isfinite(end):
        return np.isfinite(time_s) & (time_s >= start - 1e-12) & (time_s <= end + 1e-12)
    return np.isfinite(time_s)


def _source_quality_bad(values: np.ndarray) -> bool:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 5:
        return False
    pp = float(np.nanmax(finite) - np.nanmin(finite))
    if pp <= 0.0:
        return True
    diffs = np.abs(np.diff(finite))
    if diffs.size == 0:
        return False
    median_jump = float(np.nanmedian(diffs))
    max_jump = float(np.nanmax(diffs))
    return bool(max_jump > max(0.6 * pp, 8.0 * median_jump, 1e-9))


def _limit_voltage(voltage: np.ndarray, limit: float) -> tuple[np.ndarray, bool]:
    peak = peak_abs(voltage)
    if not np.isfinite(peak) or peak <= limit + 1e-12:
        return voltage, False
    return voltage * (float(limit) / peak), True


def _shape_metrics(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return {"active_shape_corr": float("nan"), "active_nrmse": float("nan")}
    centered = finite - float(np.nanmean(finite))
    reference = np.linspace(-1.0, 1.0, finite.size)
    corr = float(np.corrcoef(reference, centered)[0, 1]) if np.nanstd(centered) > 0 else float("nan")
    nrmse = float(np.sqrt(np.nanmean(np.square(centered))) / max(peak_abs(centered), 1e-9))
    return {"active_shape_corr": corr, "active_nrmse": nrmse}


def _tail_residual(values: np.ndarray, active_mask: np.ndarray) -> float:
    tail = np.asarray(values, dtype=float)[~np.asarray(active_mask, dtype=bool)]
    return peak_abs(tail) / max(peak_abs(values), 1e-9) if tail.size else 0.0


def _attach_metadata_columns(frame: pd.DataFrame, metadata: dict[str, Any]) -> None:
    for key, value in metadata.items():
        if isinstance(value, (list, tuple, dict)):
            continue
        frame[key] = value


def _unavailable_continuous_metadata(reason: str) -> dict[str, Any]:
    return {
        "waveform_normalization_enabled": False,
        "waveform_normalization_status": reason,
        "waveform_normalization_mode": "unavailable",
        "waveform_normalization_window": "unavailable",
    }


def _unavailable_finite_metadata(reason: str) -> dict[str, Any]:
    return {
        "finite_normalization_enabled": False,
        "finite_normalization_status": reason,
        "finite_normalization_mode": "unavailable",
    }
