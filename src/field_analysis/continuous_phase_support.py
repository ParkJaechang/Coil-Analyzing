from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .finite_phase_sync_math import midpoint_between_peak_pair, nearest_zero_crossing_time


def choose_supported_cycle(
    source: pd.DataFrame,
    *,
    cycle_starts: list[float],
    selected_cycle_index: int,
    min_cycle_index: int,
    period_s: float,
    exclude_terminal_cycles: bool = True,
    terminal_guard_cycle_count: int = 1,
) -> tuple[int, dict[str, Any]]:
    stop_meta = detect_command_stop(source)
    last_complete_cycle = len(cycle_starts) - 2
    rejected_count = 0
    for cycle_index in range(int(selected_cycle_index), int(min_cycle_index) - 1, -1):
        if cycle_index < 0 or cycle_index >= len(cycle_starts) - 1:
            continue
        start_s = float(cycle_starts[cycle_index])
        end_s = float(cycle_starts[cycle_index + 1])
        meta = compute_phase_support_metadata(source, start_s=start_s, end_s=end_s, period_s=period_s, command_stop_s=stop_meta["command_stop_s"])
        meta = {**stop_meta, **meta, **_terminal_meta(cycle_index, last_complete_cycle, exclude_terminal_cycles, terminal_guard_cycle_count)}
        if meta["selected_cycle_is_terminal"] or meta["field_support_uses_post_stop_data"]:
            rejected_count += 1
            continue
        if meta["phase_support_status"] == "ok":
            return cycle_index, {**meta, "stop_influenced_cycle_rejected_count": rejected_count}
    fallback_index = int(max(min_cycle_index, min(selected_cycle_index, len(cycle_starts) - 2)))
    if fallback_index >= 0 and fallback_index < len(cycle_starts) - 1:
        start_s = float(cycle_starts[fallback_index])
        end_s = float(cycle_starts[fallback_index + 1])
        meta = compute_phase_support_metadata(source, start_s=start_s, end_s=end_s, period_s=period_s, command_stop_s=stop_meta["command_stop_s"])
        terminal = _terminal_meta(fallback_index, last_complete_cycle, exclude_terminal_cycles, terminal_guard_cycle_count)
        return fallback_index, {**stop_meta, **meta, **terminal, "stop_influenced_cycle_rejected_count": rejected_count}
    return int(selected_cycle_index), {**stop_meta, "phase_support_status": "unavailable_no_cycle_boundary"}


def compute_phase_support_metadata(
    source: pd.DataFrame,
    *,
    start_s: float,
    end_s: float,
    period_s: float,
    command_stop_s: float | None = None,
) -> dict[str, Any]:
    time_s = pd.to_numeric(source["time_s_abs"], errors="coerce").to_numpy(dtype=float)
    finite_time = time_s[np.isfinite(time_s)]
    dt_s = float(np.nanmedian(np.diff(finite_time))) if finite_time.size > 1 else period_s / 100.0
    voltage = pd.to_numeric(source["raw_voltage_v"], errors="coerce").to_numpy(dtype=float)
    field = _field_signal(source)
    voltage_peak = _first_positive_peak_time(time_s, voltage, start_s, end_s)
    field_peak = _first_positive_peak_time(time_s, field, voltage_peak, min(end_s + 0.5 * period_s, np.nanmax(finite_time)))
    delay_s = max(float(field_peak - voltage_peak), 0.0) if np.isfinite(voltage_peak) and np.isfinite(field_peak) else 0.0
    support_margin_s = max(0.05 * float(period_s), 2.0 * max(dt_s, 1e-12))
    field_support_end_s = float(end_s + delay_s + support_margin_s)
    source_end_s = float(np.nanmax(finite_time)) if finite_time.size else float("nan")
    stop_s = float(command_stop_s) if command_stop_s is not None and np.isfinite(command_stop_s) else source_end_s
    uses_post_stop = bool(np.isfinite(stop_s) and field_support_end_s > stop_s + max(2.0 * dt_s, 1e-12))
    source_support_ok = bool(np.isfinite(source_end_s) and source_end_s >= field_support_end_s - max(2.0 * dt_s, 1e-12))
    support_ok = bool(source_support_ok and not uses_post_stop)
    status = "rejected_stop_influenced_phase_support" if uses_post_stop else ("ok" if support_ok else "insufficient_field_support")
    return {
        "voltage_model_start_s": float(start_s),
        "voltage_model_end_s": float(end_s),
        "field_support_start_s": float(start_s),
        "field_support_end_s": field_support_end_s,
        "estimated_phase_delay_s": float(delay_s),
        "support_margin_s": float(support_margin_s),
        "phase_support_status": status,
        "continuous_modeling_support_status": "ok" if support_ok else "unavailable",
        "voltage_first_peak_time_s": float(voltage_peak) if np.isfinite(voltage_peak) else None,
        "measured_first_peak_time_s": float(field_peak) if np.isfinite(field_peak) else None,
        "continuous_phase_delay_s": float(delay_s),
        "continuous_phase_delay_cycles": float(delay_s / max(float(period_s), 1e-12)),
        "field_support_uses_post_stop_data": uses_post_stop,
        "selected_cycle_phase_support_clear_of_stop": not uses_post_stop,
    }


def detect_command_stop(source: pd.DataFrame) -> dict[str, Any]:
    time_s = pd.to_numeric(source["time_s_abs"], errors="coerce").to_numpy(dtype=float)
    voltage = pd.to_numeric(source["raw_voltage_v"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(time_s) & np.isfinite(voltage)
    if finite.sum() < 3:
        return {
            "command_start_s": None,
            "command_stop_s": None,
            "command_stop_detection_method": "unavailable",
            "command_stop_detection_confidence": "low",
            "post_command_decay_region_detected": False,
        }
    t = time_s[finite]
    v = voltage[finite]
    dt = float(np.nanmedian(np.diff(np.sort(t)))) if t.size > 1 else 0.0
    peak = float(np.nanmax(np.abs(v))) if v.size else 0.0
    threshold = max(0.05, 0.03 * peak)
    nonzero = np.abs(v) >= threshold
    if not nonzero.any():
        return {
            "command_start_s": float(np.nanmin(t)),
            "command_stop_s": float(np.nanmax(t)),
            "command_stop_detection_method": "source_end_no_voltage_threshold",
            "command_stop_detection_confidence": "low",
            "post_command_decay_region_detected": False,
        }
    last = int(np.flatnonzero(nonzero)[-1])
    stop_s = float(min(t[last] + max(dt, 0.0), np.nanmax(t)))
    post_decay = bool(last < len(t) - 3 and np.nanmax(np.abs(v[last + 1 :])) < threshold)
    return {
        "command_start_s": float(t[int(np.flatnonzero(nonzero)[0])]),
        "command_stop_s": stop_s,
        "command_stop_detection_method": "voltage_nonzero_window",
        "command_stop_detection_confidence": "high" if post_decay else "medium",
        "post_command_decay_region_detected": post_decay,
    }


def _terminal_meta(cycle_index: int, last_complete_cycle: int, exclude: bool, guard: int) -> dict[str, Any]:
    terminal = bool(exclude and cycle_index >= last_complete_cycle - max(int(guard), 0) + 1)
    return {
        "exclude_terminal_cycles": bool(exclude),
        "terminal_guard_cycle_count": int(guard),
        "selected_cycle_is_terminal": terminal,
        "selected_cycle_stop_influence_risk": "high" if terminal else "low",
        "selected_cycle_stop_influence_status": "rejected_terminal_cycle" if terminal else "ok",
        "selected_cycle_rejected_reason": "rejected_terminal_cycle" if terminal else None,
    }


def build_support_frame(source: pd.DataFrame, *, start_s: float, support_end_s: float) -> pd.DataFrame:
    mask = (source["time_s_abs"] >= float(start_s)) & (source["time_s_abs"] <= float(support_end_s))
    support = source.loc[mask].copy()
    if not support.empty:
        support["time_s"] = pd.to_numeric(support["time_s_abs"], errors="coerce") - float(start_s)
    return support.reset_index(drop=True)


def interpolate_signal(target_time: np.ndarray, source_time: np.ndarray, source_values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(source_time) & np.isfinite(source_values)
    if not finite.any():
        return np.full(len(target_time), np.nan)
    order = np.argsort(source_time[finite])
    return np.interp(target_time, source_time[finite][order], source_values[finite][order], left=np.nan, right=np.nan)


def continuous_peak_alignment_metadata(
    time_s: np.ndarray,
    voltage: np.ndarray,
    measured: np.ndarray,
    *,
    period_s: float,
) -> dict[str, Any]:
    voltage_peak = _first_positive_peak_time(time_s, voltage, 0.0, period_s)
    measured_peak = _first_positive_peak_time(time_s, measured, voltage_peak, period_s + 0.5 * period_s)
    active_mask = np.isfinite(time_s) & (time_s >= 0.0) & (time_s <= period_s + 0.5 * period_s)
    midpoint_time, midpoint_polarity, midpoint_left, midpoint_right = midpoint_between_peak_pair(
        time_s,
        measured,
        active_mask,
        pair_start_number=2,
    )
    target_zero = nearest_zero_crossing_time(
        time_s,
        voltage,
        np.isfinite(time_s) & (time_s >= 0.0) & (time_s <= period_s),
        reference_time_s=midpoint_time,
    )
    if midpoint_time is not None and target_zero is not None:
        delay_s = max(float(midpoint_time - target_zero), 0.0)
        method = "peak_pair_midpoint_to_voltage_zero_crossing"
    else:
        delay_s = max(float(measured_peak - voltage_peak), 0.0) if np.isfinite(voltage_peak) and np.isfinite(measured_peak) else 0.0
        method = "field_peak_to_voltage_peak"
    return {
        "voltage_first_peak_time_s": float(voltage_peak) if np.isfinite(voltage_peak) else None,
        "measured_first_peak_time_s": float(measured_peak) if np.isfinite(measured_peak) else None,
        "continuous_phase_delay_s": float(delay_s),
        "continuous_phase_delay_cycles": float(delay_s / max(period_s, 1e-12)),
        "measured_aligned_first_peak_time_s": float(measured_peak - delay_s) if np.isfinite(measured_peak) else None,
        "continuous_phase_alignment_method": method,
        "continuous_phase_sync_midpoint_time_s": midpoint_time,
        "continuous_phase_sync_midpoint_left_peak_time_s": midpoint_left,
        "continuous_phase_sync_midpoint_right_peak_time_s": midpoint_right,
        "continuous_phase_sync_midpoint_polarity": midpoint_polarity,
        "continuous_phase_sync_target_zero_crossing_time_s": target_zero,
    }


def _field_signal(source: pd.DataFrame) -> np.ndarray:
    for column in ("measured_field_normalized_mT", "measured_field_effective_mT"):
        if column in source.columns:
            return pd.to_numeric(source[column], errors="coerce").to_numpy(dtype=float)
    return np.full(len(source), np.nan)


def _first_positive_peak_time(time_s: np.ndarray, values: np.ndarray, start_s: float, end_s: float) -> float:
    mask = np.isfinite(time_s) & np.isfinite(values) & (time_s >= float(start_s)) & (time_s < float(end_s))
    if not mask.any():
        return float("nan")
    local_time = time_s[mask]
    local_values = values[mask]
    if local_values.size >= 3:
        peak_level = 0.25 * max(float(np.nanmax(local_values)), 1e-12)
        for idx in range(1, local_values.size - 1):
            if local_values[idx] > peak_level and local_values[idx] >= local_values[idx - 1] and local_values[idx] >= local_values[idx + 1]:
                return float(local_time[idx])
    idx = int(np.nanargmax(local_values))
    return float(local_time[idx])
