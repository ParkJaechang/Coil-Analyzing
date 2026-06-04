from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .finite_actual_drive_normalization import peak_abs


def resolve_finite_tail_policy(freq_hz: float, mode: str = "auto", threshold_hz: float = 2.0) -> dict[str, Any]:
    normalized_mode = str(mode or "auto").strip().lower()
    if normalized_mode not in {"auto", "on", "off"}:
        normalized_mode = "auto"
    freq = float(freq_hz)
    threshold = float(threshold_hz)
    if normalized_mode == "on":
        enabled = True
        reason = "user_forced_on"
    elif normalized_mode == "off":
        enabled = False
        reason = "user_forced_off"
    elif np.isfinite(freq) and np.isfinite(threshold) and freq >= threshold:
        enabled = False
        reason = "auto_disabled_high_frequency"
    else:
        enabled = True
        reason = "auto_enabled_low_frequency"
    warning = (
        f"현재 주파수는 {threshold:g}Hz 이상입니다. 자동 정책에 따라 finite tail을 사용하지 않습니다."
        if reason == "auto_disabled_high_frequency"
        else None
    )
    status_message = (
        f"현재 주파수 {freq:g} Hz는 자동 정책에 따라 finite tail {'사용' if enabled else '사용 안 함'}입니다. "
        f"현재 자동 OFF 기준: {threshold:g} Hz"
    )
    if reason == "auto_disabled_high_frequency":
        warning = f"현재 주파수 {freq:g} Hz가 자동 OFF 기준 {threshold:g} Hz 이상입니다. 자동 정책에 따라 finite tail을 사용하지 않습니다."
    return {
        "finite_tail_mode": normalized_mode,
        "finite_tail_auto_threshold_hz": threshold,
        "finite_tail_auto_policy": "off_at_or_above_threshold",
        "finite_tail_effective_enabled": bool(enabled),
        "finite_tail_disabled_reason": None if enabled else reason,
        "finite_tail_user_override": normalized_mode in {"on", "off"},
        "finite_tail_policy_freq_hz": freq,
        "high_frequency_tail_auto_disabled": reason == "auto_disabled_high_frequency",
        "finite_tail_status_message": status_message,
        "finite_tail_warning_message": warning,
        "finite_tail_warning_message_dynamic": True,
    }


def trim_profile_to_active_duration(profile: pd.DataFrame, *, freq_hz: float, cycle_count: float) -> pd.DataFrame:
    if "time_s" not in profile.columns:
        return profile
    duration = float(cycle_count) / max(float(freq_hz), 1e-12)
    time_s = pd.to_numeric(profile["time_s"], errors="coerce")
    trimmed = profile.loc[time_s <= duration + 1e-12].copy()
    return trimmed.reset_index(drop=True) if not trimmed.empty else profile


def compute_second_modeling_gain(
    unit_delta_v: np.ndarray,
    first_voltage_v: np.ndarray,
    active_mask: np.ndarray,
    *,
    manual_gain: float,
    gain_mode: str,
    voltage_limit_v: float,
    tail_mask: np.ndarray | None = None,
    error_ratio_for_gain: np.ndarray | None = None,
) -> tuple[float, dict[str, Any]]:
    mode = "manual" if str(gain_mode).lower() == "manual" else "auto"
    manual = float(np.clip(manual_gain, 0.0, 1.0))
    active = np.asarray(active_mask, dtype=bool)
    tail = np.asarray(tail_mask, dtype=bool) if tail_mask is not None else np.zeros_like(active, dtype=bool)
    unit = np.abs(np.asarray(unit_delta_v, dtype=float)[active])
    active_unit = np.abs(np.asarray(unit_delta_v, dtype=float)[active & ~tail])
    tail_unit = np.abs(np.asarray(unit_delta_v, dtype=float)[tail])
    first = np.abs(np.asarray(first_voltage_v, dtype=float)[active])
    unit = unit[np.isfinite(unit)]
    active_unit = active_unit[np.isfinite(active_unit)]
    tail_unit = tail_unit[np.isfinite(tail_unit)]
    first = first[np.isfinite(first)]
    unit_peak = float(np.nanpercentile(unit, 95)) if unit.size else 0.0
    active_unit_peak = float(np.nanpercentile(active_unit, 95)) if active_unit.size else 0.0
    tail_unit_peak = float(np.nanpercentile(tail_unit, 95)) if tail_unit.size else 0.0
    first_peak = float(np.nanmax(first)) if first.size else 0.0
    headroom = np.maximum(float(voltage_limit_v) - first, 0.0) if first.size else np.array([], dtype=float)
    headroom_safe = float(np.nanpercentile(headroom, 20)) if headroom.size else 0.0
    if error_ratio_for_gain is not None:
        ratio_source = np.abs(np.asarray(error_ratio_for_gain, dtype=float)[active])
        error_ratio = ratio_source[np.isfinite(ratio_source)]
        error_ratio_basis = "field_residual_over_target_peak"
    else:
        voltage_limit_abs = max(abs(float(voltage_limit_v)), 1e-12)
        error_ratio = unit / voltage_limit_abs if unit.size else np.array([], dtype=float)
        error_ratio_basis = "unit_delta_over_voltage_limit"
    mean_error_ratio = float(np.nanmean(error_ratio)) if error_ratio.size else 0.0
    rms_error_ratio = float(np.sqrt(np.nanmean(np.square(error_ratio)))) if error_ratio.size else 0.0
    max_error_ratio = float(np.nanmax(error_ratio)) if error_ratio.size else 0.0
    p95_error_ratio = float(np.nanpercentile(error_ratio, 95)) if error_ratio.size else 0.0
    target_error_ratio_goal = 0.01
    goal_error_ratio = rms_error_ratio
    adaptive_drive_error_ratio = max(rms_error_ratio, 0.65 * p95_error_ratio, 0.45 * max_error_ratio)
    if goal_error_ratio <= target_error_ratio_goal:
        nominal = 0.0
        adaptive_gain_reason = "target_rms_error_goal_reached"
    else:
        # Apply the fraction of the residual needed to reduce the current
        # error estimate toward the 1% target.  Example: 10% -> gain 0.90,
        # 2% -> gain 0.50.  Headroom may still limit the final applied gain.
        nominal = 1.0 - target_error_ratio_goal / max(adaptive_drive_error_ratio, 1e-12)
        adaptive_gain_reason = "error_proportional_to_one_percent_goal"
    nominal = float(np.clip(nominal, 0.0, 1.0))
    if unit_peak <= 1e-9:
        auto = nominal
        headroom_limited_auto = nominal
        target_delta_peak = 0.0
    else:
        requested_delta_peak = unit_peak * nominal
        headroom_delta_peak = 0.70 * max(headroom_safe, 0.0)
        target_delta_peak = min(requested_delta_peak, headroom_delta_peak)
        headroom_limited_auto = target_delta_peak / max(unit_peak, 1e-9)
        auto = min(nominal, headroom_limited_auto)
    auto_clamped = 0.0 if nominal <= 0.0 else float(np.clip(auto, 0.0, 1.0))
    used = manual if mode == "manual" else auto_clamped
    return used, {
        "correction_gain_mode": mode,
        "correction_gain_auto": auto_clamped,
        "correction_gain_manual": manual,
        "correction_gain_used": float(used),
        "correction_gain": float(used),
        "auto_gain_unit_delta_peak_v": unit_peak,
        "auto_gain_active_unit_delta_peak_v": active_unit_peak,
        "auto_gain_tail_unit_delta_peak_v": tail_unit_peak,
        "auto_gain_first_voltage_peak_v": first_peak,
        "auto_gain_headroom_safe_v": headroom_safe,
        "auto_gain_target_delta_peak_v": float(target_delta_peak),
        "auto_gain_nominal_residual_fraction": nominal,
        "auto_gain_adaptive_drive_error_ratio": adaptive_drive_error_ratio,
        "auto_gain_formula": "max(0, 1 - target_error_ratio_goal / adaptive_drive_error_ratio)",
        "auto_gain_headroom_limited": bool(unit_peak > 1e-9 and headroom_limited_auto < nominal - 1e-12),
        "auto_gain_policy": "target_rms_error_goal_adaptive_headroom_limited",
        "auto_gain_error_ratio_basis": error_ratio_basis,
        "target_error_ratio_goal": target_error_ratio_goal,
        "goal_error_metric": "rms_error_ratio",
        "goal_error_ratio": goal_error_ratio,
        "mean_error_ratio": mean_error_ratio,
        "rms_error_ratio": rms_error_ratio,
        "max_error_ratio": max_error_ratio,
        "p95_error_ratio": p95_error_ratio,
        "goal_reached": bool(goal_error_ratio <= target_error_ratio_goal),
        "samples_above_1pct": int(np.sum(error_ratio > target_error_ratio_goal)) if error_ratio.size else 0,
        "adaptive_correction_gain": float(used),
        "adaptive_gain_reason": adaptive_gain_reason,
        "next_correction_recommended": bool(mode == "auto" and goal_error_ratio > target_error_ratio_goal),
        "auto_gain_clamped": bool(not np.isclose(auto, auto_clamped)),
        "tail_gain_used": float(used),
    }


def extend_profile_for_zero_tail(
    profile: pd.DataFrame,
    *,
    freq_hz: float,
    cycle_count: float,
    enabled: bool,
    tail_cycle_count: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    active_duration = float(cycle_count) / max(float(freq_hz), 1e-12)
    tail_cycles = float(np.clip(tail_cycle_count, 0.0, 1.0))
    tail_duration = tail_cycles / max(float(freq_hz), 1e-12) if enabled else 0.0
    base = {
        "post_cycle_zero_tail_enabled": bool(enabled and tail_duration > 0.0),
        "post_cycle_zero_tail_cycle_count": tail_cycles if enabled else 0.0,
        "post_cycle_zero_tail_duration_s": float(tail_duration),
        "post_cycle_zero_tail_target_field_mT": 0.0,
        "post_cycle_zero_tail_start_s": float(active_duration),
        "post_cycle_zero_tail_end_s": float(active_duration + tail_duration),
        "active_duration_s": float(active_duration),
        "total_command_duration_s": float(active_duration + tail_duration),
        "tail_voltage_taper_to_zero": bool(enabled and tail_duration > 0.0),
    }
    if not enabled or tail_duration <= 0.0:
        return _sanitize_time(profile, base)
    frame = profile.copy()
    time = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
    finite_time = time[np.isfinite(time)]
    diffs = np.diff(np.sort(finite_time)) if finite_time.size > 1 else np.array([], dtype=float)
    positive = diffs[np.isfinite(diffs) & (diffs > 0)]
    dt = float(np.nanmedian(positive)) if positive.size else tail_duration
    dt = dt if np.isfinite(dt) and dt > 0.0 else tail_duration
    last_time = float(np.nanmax(finite_time)) if finite_time.size else 0.0
    tail_end = active_duration + tail_duration
    if last_time < tail_end - 0.5 * dt:
        tail_times = np.arange(last_time + dt, tail_end + 0.5 * dt, dt, dtype=float)
        if tail_times.size:
            tail_rows = pd.DataFrame({column: np.nan for column in frame.columns}, index=np.arange(tail_times.size))
            tail_rows["time_s"] = tail_times
            voltage_columns = ("limited_voltage_v", "recommended_voltage_v", "first_modeled_voltage_v")
            for column in ("physical_target_output_mT", *voltage_columns):
                if column in tail_rows.columns:
                    tail_rows[column] = 0.0
            frame = pd.concat([frame, tail_rows], ignore_index=True)
    return _sanitize_time(frame, base)


def tail_mask(time_s: np.ndarray, *, freq_hz: float, cycle_count: float, tail_cycle_count: float, enabled: bool) -> np.ndarray:
    if not enabled or tail_cycle_count <= 0.0:
        return np.zeros(len(time_s), dtype=bool)
    active_duration = float(cycle_count) / max(float(freq_hz), 1e-12)
    tail_duration = float(np.clip(tail_cycle_count, 0.0, 1.0)) / max(float(freq_hz), 1e-12)
    values = np.asarray(time_s, dtype=float)
    return np.isfinite(values) & (values > active_duration + 1e-12) & (values <= active_duration + tail_duration + 1e-12)


def _sanitize_time(frame: pd.DataFrame, metadata: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    sanitized = frame.copy()
    before = len(sanitized)
    sanitized = sanitized.sort_values("time_s", kind="mergesort").drop_duplicates("time_s", keep="first").reset_index(drop=True)
    removed = before - len(sanitized)
    time = pd.to_numeric(sanitized["time_s"], errors="coerce").to_numpy(dtype=float)
    diffs = np.diff(time[np.isfinite(time)])
    metadata.update(
        {
            "second_command_time_monotonic": bool(diffs.size == 0 or np.all(diffs > 0)),
            "second_command_duplicate_time_count": int(removed),
            "active_tail_duplicate_time_removed": int(removed),
            "active_tail_duplicate_time_warning": "duplicates_removed" if removed else None,
        }
    )
    return sanitized, metadata
