from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .finite_cycle_metrics import FiniteCycleMetrics, evaluate_finite_cycle_metrics

CandidatePredictor = Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame | None]

EXTENSION_FRACTIONS = (0.0, 0.05, 0.10, 0.15)
HOLD_FRACTIONS = (0.0, 0.05, 0.10)
TAPER_FRACTIONS = (0.0, 0.05, 0.10)
EPSILON = 1e-9


def evaluate_finite_timing_compensation(
    command_profile: pd.DataFrame,
    *,
    freq_hz: float,
    candidate_predictor: CandidatePredictor | None,
    max_daq_voltage_pp: float | None = None,
    timing_compensation_evaluate_only: bool = True,
    empirical_lag_s: float | None = None,
    finite_support_peak_delay_s: float | None = None,
    lcr_phase_delay_s: float | None = None,
    timebase_quality_suspect: bool = False,
    source_quality_suspect: bool = False,
) -> dict[str, Any]:
    """Evaluate finite timing compensation candidates without changing the baseline command.

    The prototype is intentionally evaluate-only by default. Callers provide a
    candidate_predictor so candidate scoring is based on a real forward
    prediction instead of copying baseline predicted-field values.
    """

    baseline = command_profile.copy(deep=True)
    target_snapshot = _target_snapshot(baseline)
    active_window = _active_window(baseline)
    baseline_metrics = evaluate_finite_cycle_metrics(baseline)
    baseline_score = _score_metrics(
        baseline_metrics,
        command_extension_s=0.0,
        voltage_limit_violation=False,
        command_smoothness_worsened=False,
    )
    phase_delay = _resolve_phase_delay(
        baseline_metrics=baseline_metrics,
        empirical_lag_s=empirical_lag_s,
        finite_support_peak_delay_s=finite_support_peak_delay_s,
        lcr_phase_delay_s=lcr_phase_delay_s,
    )
    base_report = _base_report(
        baseline=baseline,
        target_snapshot=target_snapshot,
        active_window=active_window,
        baseline_metrics=baseline_metrics,
        baseline_score=baseline_score,
        phase_delay=phase_delay,
        timing_compensation_evaluate_only=timing_compensation_evaluate_only,
        empirical_lag_s=empirical_lag_s,
        finite_support_peak_delay_s=finite_support_peak_delay_s,
        lcr_phase_delay_s=lcr_phase_delay_s,
    )

    if candidate_predictor is None:
        return {
            **base_report,
            "timing_candidate_prediction_available": False,
            "timing_route_selected": False,
            "timing_route_reject_reason": "candidate_forward_prediction_unavailable",
            "rejected_timing_candidates": [],
        }
    if timebase_quality_suspect or source_quality_suspect:
        reason = "timebase_quality_suspect" if timebase_quality_suspect else "source_quality_suspect"
        return {
            **base_report,
            "timing_candidate_prediction_available": False,
            "timing_route_selected": False,
            "timing_route_reject_reason": reason,
            "rejected_timing_candidates": [],
        }

    best_candidate: dict[str, Any] | None = None
    rejected: list[dict[str, Any]] = []
    for candidate in _candidate_grid(freq_hz):
        candidate_profile = _build_candidate_command_profile(
            baseline,
            candidate=candidate,
            active_window=active_window,
        )
        predicted_profile = candidate_predictor(candidate_profile.copy(deep=True), dict(candidate))
        if predicted_profile is None or predicted_profile.empty:
            rejected.append({**candidate, "reject_reason": "candidate_forward_prediction_unavailable"})
            continue
        predicted_profile = predicted_profile.copy(deep=True)
        target_unchanged = _target_snapshot(predicted_profile) == target_snapshot
        metrics_after = evaluate_finite_cycle_metrics(predicted_profile)
        voltage_limit_violation = _voltage_limit_violation(
            predicted_profile,
            max_daq_voltage_pp=max_daq_voltage_pp,
        )
        smoothness_worsened = _command_smoothness(predicted_profile) > _command_smoothness(baseline) + EPSILON
        reject_reasons = _reject_reasons(
            before=baseline_metrics,
            after=metrics_after,
            command_extension_s=float(candidate["command_extension_s"]),
            voltage_limit_violation=voltage_limit_violation,
            command_smoothness_worsened=smoothness_worsened,
            target_unchanged=target_unchanged,
        )
        score_after = _score_metrics(
            metrics_after,
            command_extension_s=float(candidate["command_extension_s"]),
            voltage_limit_violation=voltage_limit_violation,
            command_smoothness_worsened=smoothness_worsened,
        )
        candidate_summary = {
            **candidate,
            "timing_score_after": score_after,
            "active_nrmse_after": metrics_after.active_window_nrmse,
            "active_shape_corr_after": metrics_after.active_window_shape_corr,
            "terminal_peak_error_mT_after": metrics_after.terminal_peak_error_mT,
            "tail_residual_ratio_after": metrics_after.tail_residual_ratio,
            "voltage_limit_respected": not voltage_limit_violation,
            "target_unchanged": target_unchanged,
        }
        if reject_reasons:
            rejected.append({**candidate_summary, "reject_reason": "|".join(reject_reasons)})
            continue
        if best_candidate is None or score_after < float(best_candidate["timing_score_after"]):
            best_candidate = candidate_summary

    if best_candidate is None:
        return {
            **base_report,
            "timing_candidate_prediction_available": True,
            "timing_route_selected": False,
            "timing_route_reject_reason": _first_reject_reason(rejected),
            "rejected_timing_candidates": rejected,
        }

    return {
        **base_report,
        "timing_candidate_prediction_available": True,
        "timing_route_selected": True,
        "timing_route_reject_reason": "",
        "selected_timing_candidate": best_candidate,
        "rejected_timing_candidates": rejected,
        "timing_score_after": best_candidate["timing_score_after"],
        "command_extension_s": best_candidate["command_extension_s"],
        "command_hold_end_s": best_candidate["command_hold_end_s"],
        "command_nonzero_end_s": best_candidate["command_nonzero_end_s"],
        "predicted_peak_time_s": best_candidate["predicted_peak_time_s"],
        "predicted_settle_end_s": best_candidate["predicted_settle_end_s"],
        "voltage_limit_respected": best_candidate["voltage_limit_respected"],
    }


def _candidate_grid(freq_hz: float) -> list[dict[str, Any]]:
    period_s = 1.0 / float(freq_hz) if np.isfinite(freq_hz) and float(freq_hz) > 0 else 1.0
    candidates: list[dict[str, Any]] = []
    for extension_fraction in EXTENSION_FRACTIONS:
        for hold_fraction in HOLD_FRACTIONS:
            for taper_fraction in TAPER_FRACTIONS:
                extension_s = float(extension_fraction * period_s)
                candidates.append(
                    {
                        "timing_compensation_mode": "post_target_extension_hold_taper",
                        "command_extension_s": extension_s,
                        "hold_fraction": float(hold_fraction),
                        "taper_fraction": float(taper_fraction),
                    }
                )
    return candidates


def _build_candidate_command_profile(
    baseline: pd.DataFrame,
    *,
    candidate: dict[str, Any],
    active_window: dict[str, float],
) -> pd.DataFrame:
    frame = baseline.copy(deep=True)
    extension_s = float(candidate["command_extension_s"])
    hold_fraction = float(candidate["hold_fraction"])
    taper_fraction = float(candidate["taper_fraction"])
    target_end_s = float(active_window["target_active_end_s"])
    command_hold_end_s = target_end_s + extension_s * max(hold_fraction, 0.0)
    command_nonzero_end_s = target_end_s + extension_s
    time_values = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
    for column in ("recommended_voltage_v", "limited_voltage_v"):
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        values = _extend_command_values(
            values=values,
            time_values=time_values,
            target_end_s=target_end_s,
            command_hold_end_s=command_hold_end_s,
            command_nonzero_end_s=command_nonzero_end_s,
            taper_fraction=taper_fraction,
        )
        frame[column] = values
    frame["timing_candidate_command_extension_s"] = extension_s
    frame["timing_candidate_hold_fraction"] = hold_fraction
    frame["timing_candidate_taper_fraction"] = taper_fraction
    frame["timing_candidate_command_hold_end_s"] = command_hold_end_s
    frame["timing_candidate_command_nonzero_end_s"] = command_nonzero_end_s
    candidate["command_hold_end_s"] = command_hold_end_s
    candidate["command_nonzero_end_s"] = command_nonzero_end_s
    candidate["predicted_peak_time_s"] = float("nan")
    candidate["predicted_settle_end_s"] = command_nonzero_end_s
    return frame


def _extend_command_values(
    *,
    values: np.ndarray,
    time_values: np.ndarray,
    target_end_s: float,
    command_hold_end_s: float,
    command_nonzero_end_s: float,
    taper_fraction: float,
) -> np.ndarray:
    result = values.copy()
    finite = np.isfinite(time_values) & np.isfinite(result)
    if not finite.any() or command_nonzero_end_s <= target_end_s + EPSILON:
        return result
    active_before_end = finite & (time_values <= target_end_s + EPSILON)
    if not active_before_end.any():
        return result
    hold_value = float(result[np.flatnonzero(active_before_end)[-1]])
    extension_mask = finite & (time_values > target_end_s) & (time_values <= command_nonzero_end_s + EPSILON)
    if not extension_mask.any():
        return result
    result[extension_mask] = hold_value
    taper_start_s = command_hold_end_s
    if taper_fraction > 0.0:
        taper_mask = extension_mask & (time_values > taper_start_s)
        taper_duration_s = max(command_nonzero_end_s - taper_start_s, EPSILON)
        taper_progress = np.clip((time_values[taper_mask] - taper_start_s) / taper_duration_s, 0.0, 1.0)
        result[taper_mask] = hold_value * (1.0 - taper_progress)
    return result


def _reject_reasons(
    *,
    before: FiniteCycleMetrics,
    after: FiniteCycleMetrics,
    command_extension_s: float,
    voltage_limit_violation: bool,
    command_smoothness_worsened: bool,
    target_unchanged: bool,
) -> list[str]:
    reasons: list[str] = []
    if after.evaluation_status != "ok":
        reasons.append("candidate_metrics_unavailable")
    if not target_unchanged:
        reasons.append("target_changed")
    if np.isfinite(before.active_window_nrmse) and np.isfinite(after.active_window_nrmse):
        if after.active_window_nrmse > before.active_window_nrmse + EPSILON:
            reasons.append("active_nrmse_worsened")
    if np.isfinite(before.active_window_shape_corr) and np.isfinite(after.active_window_shape_corr):
        if after.active_window_shape_corr < before.active_window_shape_corr - EPSILON:
            reasons.append("active_shape_corr_worsened")
    if np.isfinite(before.tail_residual_ratio) and np.isfinite(after.tail_residual_ratio):
        if after.tail_residual_ratio > before.tail_residual_ratio + EPSILON:
            reasons.append("tail_residual_increased")
    if command_extension_s > 0.15 + EPSILON:
        reasons.append("command_extension_exceeds_clamp")
    if voltage_limit_violation:
        reasons.append("voltage_limit_violated")
    if command_smoothness_worsened:
        reasons.append("command_smoothness_worsened")
    return reasons


def _score_metrics(
    metrics: FiniteCycleMetrics,
    *,
    command_extension_s: float,
    voltage_limit_violation: bool,
    command_smoothness_worsened: bool,
) -> float:
    target_peak = max(abs(float(metrics.target_peak_mT)), EPSILON) if np.isfinite(metrics.target_peak_mT) else 1.0
    active = float(metrics.active_window_nrmse) if np.isfinite(metrics.active_window_nrmse) else 10.0
    peak = abs(float(metrics.terminal_peak_error_mT)) / target_peak if np.isfinite(metrics.terminal_peak_error_mT) else 10.0
    lag = abs(float(metrics.estimated_lag_seconds)) if np.isfinite(metrics.estimated_lag_seconds) else 1.0
    tail = float(metrics.tail_residual_ratio) if np.isfinite(metrics.tail_residual_ratio) else 10.0
    extension_penalty = float(command_extension_s)
    voltage_penalty = 100.0 if voltage_limit_violation else 0.0
    smoothness_penalty = 10.0 if command_smoothness_worsened else 0.0
    return float(active + 0.9 * peak + 0.5 * lag + 0.8 * tail + 0.2 * extension_penalty + voltage_penalty + smoothness_penalty)


def _resolve_phase_delay(
    *,
    baseline_metrics: FiniteCycleMetrics,
    empirical_lag_s: float | None,
    finite_support_peak_delay_s: float | None,
    lcr_phase_delay_s: float | None,
) -> dict[str, Any]:
    for source, value in (
        ("empirical_support_lag", empirical_lag_s),
        ("finite_support_peak_delay", finite_support_peak_delay_s),
        ("empirical_support_lag", baseline_metrics.estimated_lag_seconds),
        ("lcr_phase_prior", lcr_phase_delay_s),
    ):
        numeric = _optional_float(value)
        if numeric is not None and abs(numeric) > EPSILON:
            return {"phase_delay_source": source, "phase_delay_prior_s": numeric}
    return {"phase_delay_source": "unavailable", "phase_delay_prior_s": None}


def _base_report(
    *,
    baseline: pd.DataFrame,
    target_snapshot: tuple[tuple[str, tuple[float, ...]], ...],
    active_window: dict[str, float],
    baseline_metrics: FiniteCycleMetrics,
    baseline_score: float,
    phase_delay: dict[str, Any],
    timing_compensation_evaluate_only: bool,
    empirical_lag_s: float | None,
    finite_support_peak_delay_s: float | None,
    lcr_phase_delay_s: float | None,
) -> dict[str, Any]:
    command_nonzero_end_s = _command_nonzero_end_s(baseline)
    return {
        "compensation_route": "finite_timing_compensation",
        "timing_compensation_evaluate_only": bool(timing_compensation_evaluate_only),
        "timing_compensation_applied": False,
        "timing_compensation_mode": "post_target_extension_hold_taper",
        "timing_compensation_reason": "evaluate_only_candidate_scoring",
        "timing_route_selected": False,
        "timing_route_reject_reason": "",
        "selected_timing_candidate": None,
        "rejected_timing_candidates": [],
        "timing_candidate_count": len(_candidate_grid(float(baseline.get("freq_hz", pd.Series([1.0])).iloc[0]))),
        "timing_score_before": baseline_score,
        "timing_score_after": baseline_score,
        "command_extension_s": 0.0,
        "command_hold_end_s": command_nonzero_end_s,
        "predicted_peak_time_s": _predicted_peak_time_s(baseline),
        "predicted_settle_end_s": _predicted_settle_end_s(baseline),
        "phase_delay_source": phase_delay["phase_delay_source"],
        "phase_delay_prior_s": phase_delay["phase_delay_prior_s"],
        "empirical_lag_s": _optional_float(empirical_lag_s if empirical_lag_s is not None else baseline_metrics.estimated_lag_seconds),
        "finite_support_peak_delay_s": _optional_float(finite_support_peak_delay_s),
        "lcr_phase_delay_s": _optional_float(lcr_phase_delay_s),
        "voltage_limit_respected": True,
        "physical_target_unchanged": _target_snapshot(baseline) == target_snapshot,
        "target_duration_changed": False,
        "target_start_s": active_window["target_start_s"],
        "target_active_end_s": active_window["target_active_end_s"],
        "command_start_s": _command_start_s(baseline),
        "command_nonzero_end_s": command_nonzero_end_s,
        "timing_candidate_prediction_available": True,
        "timing_route_improvement_summary": {},
    }


def _active_window(frame: pd.DataFrame) -> dict[str, float]:
    time_values = pd.to_numeric(frame.get("time_s"), errors="coerce").to_numpy(dtype=float)
    active = frame.get("is_active_target", pd.Series([True] * len(frame))).astype(bool).to_numpy(dtype=bool)
    active_time = time_values[active & np.isfinite(time_values)]
    if active_time.size == 0:
        return {"target_start_s": float("nan"), "target_active_end_s": float("nan")}
    return {"target_start_s": float(np.nanmin(active_time)), "target_active_end_s": float(np.nanmax(active_time))}


def _target_snapshot(frame: pd.DataFrame) -> tuple[tuple[str, tuple[float, ...]], ...]:
    snapshot: list[tuple[str, tuple[float, ...]]] = []
    for column in ("physical_target_output_mT", "target_field_mT", "target_output"):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            snapshot.append((column, tuple(np.round(values, 12))))
    return tuple(snapshot)


def _command_nonzero_end_s(frame: pd.DataFrame) -> float:
    time_values = pd.to_numeric(frame.get("time_s"), errors="coerce").to_numpy(dtype=float)
    command_column = "limited_voltage_v" if "limited_voltage_v" in frame.columns else "recommended_voltage_v"
    command = pd.to_numeric(frame.get(command_column), errors="coerce").to_numpy(dtype=float)
    if command.size == 0 or time_values.size == 0:
        return float("nan")
    pp = float(np.nanmax(command) - np.nanmin(command)) if np.isfinite(command).any() else 0.0
    threshold = max(pp * 0.01, 1e-6)
    mask = np.isfinite(time_values) & np.isfinite(command) & (np.abs(command) > threshold)
    return float(np.nanmax(time_values[mask])) if mask.any() else float("nan")


def _command_start_s(frame: pd.DataFrame) -> float:
    time_values = pd.to_numeric(frame.get("time_s"), errors="coerce").to_numpy(dtype=float)
    command = pd.to_numeric(frame.get("recommended_voltage_v"), errors="coerce").to_numpy(dtype=float)
    if command.size == 0 or time_values.size == 0:
        return float("nan")
    pp = float(np.nanmax(command) - np.nanmin(command)) if np.isfinite(command).any() else 0.0
    threshold = max(pp * 0.01, 1e-6)
    mask = np.isfinite(time_values) & np.isfinite(command) & (np.abs(command) > threshold)
    return float(np.nanmin(time_values[mask])) if mask.any() else float("nan")


def _predicted_peak_time_s(frame: pd.DataFrame) -> float:
    if "predicted_field_mT" not in frame.columns or "time_s" not in frame.columns:
        return float("nan")
    predicted = pd.to_numeric(frame["predicted_field_mT"], errors="coerce").to_numpy(dtype=float)
    time_values = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(predicted) & np.isfinite(time_values)
    if not valid.any():
        return float("nan")
    valid_indices = np.flatnonzero(valid)
    peak_index = valid_indices[int(np.nanargmax(np.abs(predicted[valid])))]
    return float(time_values[peak_index])


def _predicted_settle_end_s(frame: pd.DataFrame) -> float:
    if "predicted_field_mT" not in frame.columns or "time_s" not in frame.columns:
        return float("nan")
    predicted = pd.to_numeric(frame["predicted_field_mT"], errors="coerce").to_numpy(dtype=float)
    time_values = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
    pp = float(np.nanmax(predicted) - np.nanmin(predicted)) if np.isfinite(predicted).any() else 0.0
    threshold = max(pp * 0.02, 1e-6)
    mask = np.isfinite(time_values) & np.isfinite(predicted) & (np.abs(predicted) > threshold)
    return float(np.nanmax(time_values[mask])) if mask.any() else float("nan")


def _voltage_limit_violation(frame: pd.DataFrame, *, max_daq_voltage_pp: float | None) -> bool:
    if max_daq_voltage_pp is None or not np.isfinite(float(max_daq_voltage_pp)):
        return False
    column = "limited_voltage_v" if "limited_voltage_v" in frame.columns else "recommended_voltage_v"
    if column not in frame.columns:
        return False
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).any():
        return False
    pp = float(np.nanmax(values) - np.nanmin(values))
    return bool(pp > float(max_daq_voltage_pp) + EPSILON)


def _command_smoothness(frame: pd.DataFrame) -> float:
    if "recommended_voltage_v" not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame["recommended_voltage_v"], errors="coerce").to_numpy(dtype=float)
    valid = values[np.isfinite(values)]
    if valid.size < 3:
        return 0.0
    return float(np.nanmax(np.abs(np.diff(valid, n=2))))


def _first_reject_reason(rejected: list[dict[str, Any]]) -> str:
    if not rejected:
        return "no_acceptable_timing_candidate"
    reason_counts: dict[str, int] = {}
    for item in rejected:
        reason = str(item.get("reject_reason") or "candidate_rejected")
        first = reason.split("|")[0]
        reason_counts[first] = reason_counts.get(first, 0) + 1
    return max(reason_counts.items(), key=lambda item: item[1])[0]


def _optional_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None
