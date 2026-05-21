from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def choose_supported_cycle(
    source: pd.DataFrame,
    *,
    cycle_starts: list[float],
    selected_cycle_index: int,
    min_cycle_index: int,
    period_s: float,
) -> tuple[int, dict[str, Any]]:
    for cycle_index in range(int(selected_cycle_index), int(min_cycle_index) - 1, -1):
        if cycle_index < 0 or cycle_index >= len(cycle_starts) - 1:
            continue
        start_s = float(cycle_starts[cycle_index])
        end_s = float(cycle_starts[cycle_index + 1])
        meta = compute_phase_support_metadata(source, start_s=start_s, end_s=end_s, period_s=period_s)
        if meta["phase_support_status"] == "ok":
            return cycle_index, meta
    fallback_index = int(max(min_cycle_index, min(selected_cycle_index, len(cycle_starts) - 2)))
    if fallback_index >= 0 and fallback_index < len(cycle_starts) - 1:
        start_s = float(cycle_starts[fallback_index])
        end_s = float(cycle_starts[fallback_index + 1])
        return fallback_index, compute_phase_support_metadata(source, start_s=start_s, end_s=end_s, period_s=period_s)
    return int(selected_cycle_index), {"phase_support_status": "unavailable_no_cycle_boundary"}


def compute_phase_support_metadata(
    source: pd.DataFrame,
    *,
    start_s: float,
    end_s: float,
    period_s: float,
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
    support_ok = bool(np.isfinite(source_end_s) and source_end_s >= field_support_end_s - max(2.0 * dt_s, 1e-12))
    return {
        "voltage_model_start_s": float(start_s),
        "voltage_model_end_s": float(end_s),
        "field_support_start_s": float(start_s),
        "field_support_end_s": field_support_end_s,
        "estimated_phase_delay_s": float(delay_s),
        "support_margin_s": float(support_margin_s),
        "phase_support_status": "ok" if support_ok else "insufficient_field_support",
        "continuous_modeling_support_status": "ok" if support_ok else "unavailable",
        "voltage_first_peak_time_s": float(voltage_peak) if np.isfinite(voltage_peak) else None,
        "measured_first_peak_time_s": float(field_peak) if np.isfinite(field_peak) else None,
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
    delay_s = max(float(measured_peak - voltage_peak), 0.0) if np.isfinite(voltage_peak) and np.isfinite(measured_peak) else 0.0
    return {
        "voltage_first_peak_time_s": float(voltage_peak) if np.isfinite(voltage_peak) else None,
        "measured_first_peak_time_s": float(measured_peak) if np.isfinite(measured_peak) else None,
        "continuous_phase_delay_s": float(delay_s),
        "continuous_phase_delay_cycles": float(delay_s / max(period_s, 1e-12)),
        "measured_aligned_first_peak_time_s": float(measured_peak - delay_s) if np.isfinite(measured_peak) else None,
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
    idx = int(np.nanargmax(local_values))
    return float(local_time[idx])
