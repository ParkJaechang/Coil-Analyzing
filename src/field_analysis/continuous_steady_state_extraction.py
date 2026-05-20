from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from field_analysis.compensation import FIELD_ROUTE_NORMALIZED_TARGET_PP, _finite_target_template
from field_analysis.continuous_cycle_boundaries import detect_cycle_boundaries
from field_analysis.continuous_steady_state_schema import adapt_continuous_source_frame
from field_analysis.finite_actual_drive_normalization import normalize_peak_to_limit
from field_analysis.finite_second_modeling_stabilization import align_measured_field_for_residual
from field_analysis.finite_second_modeling_stabilization import smooth_measured_field_for_second_modeling
from field_analysis.finite_second_modeling_stabilization import stabilize_correction_delta
from field_analysis.finite_second_modeling_tail import compute_second_modeling_gain

CONTINUOUS_PRODUCTION_CYCLE_COUNT = 1.0
DEFAULT_MIN_DISCARD_CYCLES = 2
DEFAULT_STABILITY_WINDOW_CYCLES = 3


def extract_steady_state_one_cycle_window(
    frame: pd.DataFrame,
    *,
    waveform_type: str,
    freq_hz: float,
    min_discard_cycles: int = DEFAULT_MIN_DISCARD_CYCLES,
    representative_cycle_mode: str = "last_stable_cycle",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = _coerce_continuous_source_frame(frame)
    source_attrs = dict(getattr(source, "attrs", {}) or {})
    source_time = pd.to_numeric(source["time_s_abs"], errors="coerce").to_numpy(dtype=float)
    finite_time = source_time[np.isfinite(source_time)]
    source_duration_s = float(np.nanmax(finite_time) - np.nanmin(finite_time)) if finite_time.size else 0.0
    period_s = 1.0 / max(float(freq_hz), 1e-12)
    source_freq_hz = _safe_float(source_attrs.get("continuous_source_freq_hz"))
    freq_match_status = "unknown_source_frequency"
    if np.isfinite(source_freq_hz):
        freq_error = abs(source_freq_hz - float(freq_hz)) / max(abs(float(freq_hz)), 1e-12)
        freq_match_status = "ok" if freq_error <= 0.02 else "mismatch"
        if freq_match_status != "ok":
            return pd.DataFrame(), _base_metadata(freq_hz=freq_hz) | {
                "steady_state_extraction_status": "unavailable_frequency_mismatch",
                "extraction_blocked_reason": "frequency_mismatch",
                "continuous_source_file": source_attrs.get("continuous_source_file"),
                "continuous_source_freq_hz": source_freq_hz,
                "quick_lut_target_freq_hz": float(freq_hz),
                "source_freq_hz": source_freq_hz,
                "target_freq_hz": float(freq_hz),
                "frequency_error_pct": freq_error * 100.0,
                "frequency_match_status": "mismatch",
                "frequency_mismatch_blocked": True,
                "continuous_source_freq_source": source_attrs.get("continuous_source_freq_source"),
                "expected_period_s": period_s,
                "continuous_source_duration_s": source_duration_s,
                "continuous_estimated_cycles": source_duration_s * float(freq_hz),
            }
    estimated_cycles = source_duration_s * float(freq_hz)
    if estimated_cycles < 3.0:
        return pd.DataFrame(), _base_metadata(freq_hz=freq_hz) | {
            "steady_state_extraction_status": "unavailable_timebase_invalid",
            "extraction_blocked_reason": "continuous_duration_too_short",
            "continuous_timebase_status": "too_short_for_continuous",
            "expected_period_s": period_s,
            "continuous_source_duration_s": source_duration_s,
            "continuous_estimated_cycles": estimated_cycles,
        }
    metrics, stability_meta = evaluate_cycle_stability(source, freq_hz=freq_hz)
    selected_cycle_index = select_representative_steady_cycle(
        metrics,
        min_discard_cycles=min_discard_cycles,
        mode=representative_cycle_mode,
    )
    cycle_starts = list(stability_meta.get("cycle_start_times_s") or [])
    if 0 <= int(selected_cycle_index) < max(len(cycle_starts) - 1, 0):
        start_s = float(cycle_starts[int(selected_cycle_index)])
        end_s = float(cycle_starts[int(selected_cycle_index) + 1])
    else:
        start_s = float(selected_cycle_index) * period_s
        end_s = start_s + period_s
    selected_duration_s = float(end_s - start_s)
    duration_ratio = selected_duration_s / max(period_s, 1e-12)
    dt_s = float(np.nanmedian(np.diff(finite_time))) if finite_time.size > 1 else period_s / 100.0
    support_ok = bool(finite_time.size and np.nanmin(finite_time) <= start_s + max(dt_s, 1e-12) and np.nanmax(finite_time) >= end_s - max(2.0 * dt_s, 0.02 * period_s))
    if 0.45 <= duration_ratio <= 0.55:
        duration_status = "rejected_half_cycle_window"
    elif abs(duration_ratio - 1.0) > 0.10:
        duration_status = "rejected_invalid_duration"
    elif not support_ok:
        duration_status = "rejected_incomplete_one_cycle_support"
    else:
        duration_status = "ok"
    duration_meta = {
        "continuous_source_file": source_attrs.get("continuous_source_file"),
        "continuous_source_freq_hz": source_freq_hz if np.isfinite(source_freq_hz) else None,
        "quick_lut_target_freq_hz": float(freq_hz),
        "source_freq_hz": source_freq_hz if np.isfinite(source_freq_hz) else None,
        "target_freq_hz": float(freq_hz),
        "frequency_error_pct": (
            abs(source_freq_hz - float(freq_hz)) / max(abs(float(freq_hz)), 1e-12) * 100.0
            if np.isfinite(source_freq_hz)
            else None
        ),
        "frequency_match_status": freq_match_status,
        "frequency_mismatch_blocked": False,
        "continuous_source_freq_source": source_attrs.get("continuous_source_freq_source"),
        "expected_period_s": period_s,
        "selected_cycle_start_s": start_s,
        "selected_cycle_end_s": end_s,
        "selected_cycle_duration_s": selected_duration_s,
        "selected_cycle_duration_ratio": duration_ratio,
        "selected_cycle_duration_status": duration_status,
        "selected_cycle_duration_tolerance": 0.05,
        "selected_cycle_duration_error_pct": abs(duration_ratio - 1.0) * 100.0,
        "continuous_time_column": source.attrs.get("continuous_schema_time_column"),
        "continuous_time_unit_detected": "milliseconds" if source.attrs.get("continuous_schema_time_column") in {"TimeMs", "Time_ms"} else "seconds",
        "continuous_time_scale_to_seconds": 0.001 if source.attrs.get("continuous_schema_time_column") in {"TimeMs", "Time_ms"} else 1.0,
        "continuous_source_duration_s": source_duration_s,
        "continuous_estimated_cycles": estimated_cycles,
        "continuous_timebase_status": "ok",
    }
    if duration_status != "ok":
        return pd.DataFrame(), _base_metadata(freq_hz=freq_hz) | {
            **stability_meta,
            **duration_meta,
            "steady_state_extraction_status": "unavailable_invalid_cycle_duration",
            "steady_state_support_status": "unavailable",
            "extraction_blocked_reason": duration_status,
            "selected_cycle_index": int(selected_cycle_index),
        }
    mask = (source["time_s_abs"] >= start_s) & (source["time_s_abs"] < end_s)
    if not bool(mask.any()):
        empty = pd.DataFrame()
        return empty, _base_metadata(freq_hz=freq_hz) | {
            **stability_meta,
            **duration_meta,
            "steady_state_extraction_status": "unavailable_missing_one_cycle_support",
            "steady_state_support_status": "unavailable",
            "selected_cycle_index": selected_cycle_index,
        }

    window = source.loc[mask].copy()
    window["cycle_index"] = selected_cycle_index
    window["cycle_phase_s"] = window["time_s_abs"] - start_s
    window["time_s"] = window["cycle_phase_s"]
    active_mask = np.ones(len(window), dtype=bool)
    effective = pd.to_numeric(window["measured_field_effective_mT"], errors="coerce").to_numpy(dtype=float)
    baseline = float(np.nanmedian(effective)) if np.isfinite(effective).any() else 0.0
    baseline_removed = effective - baseline
    normalized_field, field_meta = normalize_peak_to_limit(
        baseline_removed,
        active_mask,
        limit=50.0,
        unavailable_status="unavailable_zero_peak",
    )
    voltage = pd.to_numeric(window["raw_voltage_v"], errors="coerce").to_numpy(dtype=float)
    normalized_voltage, voltage_meta = normalize_peak_to_limit(
        voltage,
        np.isfinite(voltage),
        limit=5.0,
        unavailable_status="unavailable_zero_peak",
    )
    target = _finite_target_template(
        window["time_s"].to_numpy(dtype=float),
        waveform_type=waveform_type,
        freq_hz=float(freq_hz),
        target_cycle_count=1.0,
        target_output_pp=float(FIELD_ROUTE_NORMALIZED_TARGET_PP),
        force_rounded_triangle=True,
    )
    normalized_target, target_meta = normalize_peak_to_limit(
        target,
        active_mask,
        limit=50.0,
        unavailable_status="unavailable_zero_peak",
    )
    window["measured_field_baseline_removed_mT"] = baseline_removed
    window["measured_field_normalized_mT"] = normalized_field
    window["voltage_normalized_v"] = normalized_voltage
    window["physical_target_output_mT"] = target
    window["normalized_physical_target_output_mT"] = normalized_target
    window["steady_state_selected_mask"] = True
    window["steady_state_cycle_index"] = selected_cycle_index
    output_columns = [
        "time_s",
        "time_s_abs",
        "cycle_index",
        "cycle_phase_s",
        "raw_hallbz_mT",
        "measured_field_effective_mT",
        "measured_field_baseline_removed_mT",
        "measured_field_normalized_mT",
        "raw_voltage_v",
        "voltage_normalized_v",
        "physical_target_output_mT",
        "normalized_physical_target_output_mT",
        "steady_state_selected_mask",
        "steady_state_cycle_index",
    ]
    metadata = _base_metadata(freq_hz=freq_hz) | {
        **stability_meta,
        **duration_meta,
        "steady_state_extraction_status": "ok",
        "steady_state_support_status": "ok",
        "selected_cycle_index": int(selected_cycle_index),
        "selected_cycle_count": 1.0,
        "discarded_startup_cycles": int(min_discard_cycles),
        "representative_cycle_mode": representative_cycle_mode,
        "representative_cycle_indices": [int(selected_cycle_index)],
        "representative_cycle_average_used": False,
        "representative_cycle_source": "actual_measured_cycle",
        "continuous_window_cycle_count": 1.0,
        "finite_like_window_from_continuous": True,
        "continuous_selected_steady_state_start_s": start_s,
        "continuous_selected_steady_state_end_s": end_s,
        "continuous_window_duration_s": period_s,
        "relative_time_zero_source": "selected_steady_cycle_start",
        "continuous_steady_state_window_support_status": "ok",
        "field_normalization_mode": "peak_to_50mT",
        "field_normalization_status": field_meta["status"],
        "voltage_normalization_mode": "peak_to_5V",
        "voltage_normalization_status": voltage_meta["status"],
        "target_normalization_status": target_meta["status"],
    }
    output = window.loc[:, output_columns].reset_index(drop=True)
    output.attrs["cycle_stability_metrics"] = metrics.copy(deep=True)
    return output, metadata


def build_continuous_steady_state_modeling_case(
    frame: pd.DataFrame,
    *,
    waveform_type: str,
    freq_hz: float,
    min_discard_cycles: int = DEFAULT_MIN_DISCARD_CYCLES,
) -> dict[str, Any]:
    window, metadata = extract_steady_state_one_cycle_window(
        frame,
        waveform_type=waveform_type,
        freq_hz=freq_hz,
        min_discard_cycles=min_discard_cycles,
    )
    metadata = metadata | {
        "continuous_loop_output": True,
        "continuous_export_cycle_count": 1.0,
        "loop_endpoint_policy": "period_exclusive",
    }
    stability_metrics = window.attrs.get("cycle_stability_metrics")
    if not isinstance(stability_metrics, pd.DataFrame):
        stability_metrics = pd.DataFrame()
    return {"steady_state_one_cycle_frame": window, "metadata": metadata, "stability_metrics": stability_metrics}


def build_continuous_phase_aligned_command_profile(
    steady_state_one_cycle_frame: pd.DataFrame,
    *,
    freq_hz: float,
    waveform_type: str | None = None,
    correction_gain: float = 0.25,
    correction_gain_mode: str = "auto",
    voltage_limit_v: float = 5.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = steady_state_one_cycle_frame.copy()
    if frame.empty:
        return frame, {
            "continuous_first_modeling_available": False,
            "continuous_first_modeling_status": "empty_steady_state_window",
        }
    time_s = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
    measured = _first_numeric_column(frame, ("measured_field_normalized_mT", "normalized_measured_field_mT")).to_numpy(dtype=float)
    target = _first_numeric_column(frame, ("normalized_physical_target_output_mT", "physical_target_output_mT")).to_numpy(dtype=float)
    first_voltage = _continuous_first_voltage(frame).to_numpy(dtype=float)
    active_mask = np.isfinite(time_s) & np.isfinite(measured) & np.isfinite(target)
    measured_smoothed, smoothing_meta = smooth_measured_field_for_second_modeling(
        time_s,
        measured,
        time_s,
        active_mask,
        freq_hz=float(freq_hz),
        cycle_count=1.0,
    )
    measured_aligned, alignment_meta = align_measured_field_for_residual(
        time_s,
        target,
        measured_smoothed,
        active_mask,
        freq_hz=float(freq_hz),
        residual_alignment_mode="first_peak_aligned",
    )
    measured_for_modeling = measured_aligned if alignment_meta.get("phase_alignment_status") == "ok" else measured_smoothed
    residual = target - measured_for_modeling
    unit_delta = residual / 50.0 * float(voltage_limit_v)
    gain, gain_meta = compute_second_modeling_gain(
        unit_delta,
        first_voltage,
        active_mask,
        manual_gain=float(correction_gain),
        gain_mode=correction_gain_mode,
        voltage_limit_v=float(voltage_limit_v),
        tail_mask=np.zeros_like(active_mask, dtype=bool),
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
    command = pd.DataFrame(
        {
            "time_s": time_s,
            "first_modeled_voltage_v": first_modeled,
            "limited_voltage_v": limited,
            "correction_delta_v": correction_delta,
            "raw_correction_delta_v": raw_delta,
            "smoothed_correction_delta_v": arrays.get("smoothed_correction_delta_v", correction_delta),
            "measured_field_smoothed_mT": measured_smoothed,
            "measured_field_aligned_mT": measured_aligned,
            "residual_for_modeling_mT": residual,
            "continuous_loop_output": True,
            "loop_endpoint_policy": "period_exclusive",
            "continuous_export_cycle_count": 1.0,
            "freq_hz": float(freq_hz),
            "waveform_type": waveform_type,
        }
    )
    metadata = {
        **_base_metadata(freq_hz=freq_hz),
        **smoothing_meta,
        **alignment_meta,
        **gain_meta,
        **stabilization_meta,
        "continuous_first_modeling_available": True,
        "continuous_first_modeling_uses_phase_aligned_kernel": True,
        "continuous_first_modeling_tail_disabled": True,
        "continuous_first_modeling_cycle_count": 1.0,
        "continuous_loop_output": True,
        "continuous_modeling_kernel_source": "finite_second_modeling_shared_kernel",
        "continuous_export_cycle_count": 1.0,
        "loop_endpoint_policy": "period_exclusive",
        "fourier_resynthesis_involved": False,
        "harmonic_export_involved": False,
    }
    return command.reset_index(drop=True), metadata


def evaluate_cycle_stability(frame: pd.DataFrame, *, freq_hz: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = _coerce_continuous_source_frame(frame)
    period_s = 1.0 / max(float(freq_hz), 1e-12)
    boundary_meta = _detect_cycle_boundaries(source, freq_hz=freq_hz)
    cycle_starts = [float(item) for item in boundary_meta["cycle_start_times_s"]]
    rows: list[dict[str, Any]] = []
    previous_values: np.ndarray | None = None
    previous_peak_pp = np.nan
    for idx, (start_s, end_s) in enumerate(zip(cycle_starts[:-1], cycle_starts[1:])):
        mask = (source["time_s_abs"] >= start_s) & (source["time_s_abs"] < end_s)
        values = pd.to_numeric(source.loc[mask, "measured_field_effective_mT"], errors="coerce").to_numpy(dtype=float)
        voltage = pd.to_numeric(source.loc[mask, "raw_voltage_v"], errors="coerce").to_numpy(dtype=float)
        if values.size < 3:
            continue
        baseline = float(np.nanmedian(values))
        centered = values - baseline
        positive_peak = float(np.nanmax(centered))
        negative_peak = float(np.nanmin(centered))
        peak_pp = positive_peak - negative_peak
        drift = (
            abs(peak_pp - previous_peak_pp) / max(abs(previous_peak_pp), 1e-9) * 100.0
            if np.isfinite(previous_peak_pp)
            else np.nan
        )
        nrmse = np.nan
        corr = np.nan
        if previous_values is not None and previous_values.size >= 3:
            n = min(previous_values.size, centered.size)
            a = centered[:n]
            b = previous_values[:n]
            denom = max(float(np.nanmax(b) - np.nanmin(b)), 1e-9)
            nrmse = float(np.sqrt(np.nanmean((a - b) ** 2)) / denom * 100.0)
            if np.nanstd(a) > 0 and np.nanstd(b) > 0:
                corr = float(np.corrcoef(a, b)[0, 1])
        rows.append(
            {
                "cycle_index": idx,
                "cycle_start_s": float(start_s),
                "cycle_end_s": float(end_s),
                "positive_peak_mT": positive_peak,
                "negative_peak_mT": negative_peak,
                "peak_to_peak_mT": peak_pp,
                "baseline_mT": baseline,
                "phase_shift_s": 0.0,
                "shape_corr_to_previous_cycle": corr,
                "nrmse_to_previous_cycle": nrmse,
                "voltage_peak_v": float(np.nanmax(np.abs(voltage))) if voltage.size else np.nan,
                "field_peak_drift_pct": drift,
                "peak_drift_pct": drift,
                "baseline_drift_mT": 0.0 if len(rows) == 0 else abs(baseline - float(rows[-1]["baseline_mT"])),
            }
        )
        previous_values = centered
        previous_peak_pp = peak_pp
    metrics = pd.DataFrame(rows)
    metadata = {
        **boundary_meta,
        "startup_transient_detected": True,
        "startup_transient_excluded_cycles": DEFAULT_MIN_DISCARD_CYCLES,
        "steady_state_first_cycle_index": DEFAULT_MIN_DISCARD_CYCLES,
        "steady_state_detection_method": "cycle_metrics",
        "steady_state_detection_status": "ok" if not metrics.empty else "unavailable",
        "cycle_stability_score": float(100.0 - np.nanmedian(metrics["peak_drift_pct"].tail(3))) if "peak_drift_pct" in metrics else np.nan,
        "cycle_shape_nrmse_pct": float(np.nanmedian(metrics["nrmse_to_previous_cycle"].tail(3))) if "nrmse_to_previous_cycle" in metrics else np.nan,
        "peak_drift_pct": float(np.nanmedian(metrics["peak_drift_pct"].tail(3))) if "peak_drift_pct" in metrics else np.nan,
        "baseline_drift_mT": float(np.nanmedian(metrics["baseline_drift_mT"].tail(3))) if "baseline_drift_mT" in metrics else np.nan,
        "phase_drift_s": 0.0,
    }
    return metrics, metadata


def select_representative_steady_cycle(
    cycle_metrics: pd.DataFrame,
    *,
    min_discard_cycles: int = DEFAULT_MIN_DISCARD_CYCLES,
    mode: str = "last_stable_cycle",
) -> int:
    if cycle_metrics.empty or "cycle_index" not in cycle_metrics.columns:
        return int(min_discard_cycles)
    eligible = cycle_metrics[pd.to_numeric(cycle_metrics["cycle_index"], errors="coerce") >= int(min_discard_cycles)]
    if eligible.empty:
        eligible = cycle_metrics.tail(1)
    if mode == "best_single_cycle" and "peak_drift_pct" in eligible.columns:
        ranked = eligible.assign(_score=pd.to_numeric(eligible["peak_drift_pct"], errors="coerce").fillna(1e9))
        return int(ranked.sort_values(["_score", "cycle_index"]).iloc[0]["cycle_index"])
    return int(eligible.iloc[-1]["cycle_index"])


def build_continuous_actual_drive_review_case(
    frame: pd.DataFrame,
    *,
    waveform_type: str,
    freq_hz: float,
    purpose: str,
) -> dict[str, Any]:
    case = build_continuous_steady_state_modeling_case(frame, waveform_type=waveform_type, freq_hz=freq_hz)
    metadata = dict(case["metadata"])
    if purpose == "validation":
        metadata.update(
            {
                "validation_input_mode": "continuous_steady_state",
                "validation_startup_transient_excluded": True,
                "validation_steady_cycle_index": metadata.get("selected_cycle_index"),
                "validation_window_cycle_count": 1.0,
            }
        )
    else:
        metadata.update(
            {
                "second_modeling_input_mode": "continuous_steady_state",
                "second_drive_startup_transient_excluded": True,
                "second_drive_steady_cycle_index": metadata.get("selected_cycle_index"),
                "second_drive_steady_state_extraction_status": metadata.get("steady_state_extraction_status"),
                "second_drive_window_cycle_count": 1.0,
                "second_drive_actual_data_used": "steady_state_one_cycle_only",
            }
        )
    return {"steady_state_one_cycle_frame": case["steady_state_one_cycle_frame"], "metadata": metadata}


def evaluate_continuous_steady_state_validation(
    frame: pd.DataFrame,
    *,
    waveform_type: str,
    freq_hz: float,
) -> dict[str, Any]:
    result = build_continuous_actual_drive_review_case(frame, waveform_type=waveform_type, freq_hz=freq_hz, purpose="validation")
    window = result["steady_state_one_cycle_frame"]
    target = pd.to_numeric(window["normalized_physical_target_output_mT"], errors="coerce").to_numpy(dtype=float)
    measured = pd.to_numeric(window["measured_field_normalized_mT"], errors="coerce").to_numpy(dtype=float)
    residual = target - measured
    denom = max(float(np.nanmax(target) - np.nanmin(target)), 1e-9)
    metrics = {
        "positive_peak_error_pct": _safe_pct(np.nanmax(measured) - np.nanmax(target), max(abs(np.nanmax(target)), 1e-9)),
        "negative_trough_error_pct": _safe_pct(np.nanmin(measured) - np.nanmin(target), max(abs(np.nanmin(target)), 1e-9)),
        "frequency_error_pct": 0.0,
        "waveform_nrmse_pct": float(np.sqrt(np.nanmean(residual**2)) / denom * 100.0),
        "shape_correlation": float(np.corrcoef(target, measured)[0, 1]) if np.nanstd(target) > 0 and np.nanstd(measured) > 0 else np.nan,
        "peak_balance": float(abs(np.nanmax(measured)) - abs(np.nanmin(measured))),
    }
    return {"metrics": metrics, "metadata": result["metadata"], "steady_state_one_cycle_frame": window}


def _coerce_continuous_source_frame(frame: pd.DataFrame) -> pd.DataFrame:
    source, _metadata = adapt_continuous_source_frame(frame)
    return source


def _detect_cycle_boundaries(source: pd.DataFrame, *, freq_hz: float) -> dict[str, Any]:
    return detect_cycle_boundaries(source, freq_hz=freq_hz)


def _first_numeric_column(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    raise ValueError(f"missing one of required columns: {columns}")


def _continuous_first_voltage(frame: pd.DataFrame) -> pd.Series:
    for column in ("limited_voltage_v", "voltage_normalized_v", "raw_voltage_v", "Voltage1_V"):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.zeros(len(frame), dtype=float))


def _base_metadata(*, freq_hz: float) -> dict[str, Any]:
    return {
        "modeling_input_mode": "continuous_steady_state",
        "continuous_production_cycle_count": 1.0,
        "continuous_repeating_lut": True,
        "startup_transient_excluded": True,
        "steady_state_cycle_extraction_used": True,
        "zero_return_tail_enabled": False,
        "continuous_zero_return_tail_enabled": False,
        "period_s": 1.0 / max(float(freq_hz), 1e-12),
    }


def _safe_pct(delta: float, denom: float) -> float:
    return float(delta / max(abs(denom), 1e-9) * 100.0)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
