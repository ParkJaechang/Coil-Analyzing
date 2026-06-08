from __future__ import annotations

from typing import Any

import numpy as np


def detect_actual_drive_timebase(
    raw_time: np.ndarray,
    voltage: np.ndarray,
    *,
    expected_active_duration_s: float,
) -> dict[str, Any]:
    candidates = {
        "seconds": 1.0,
        "milliseconds": 1.0 / 1000.0,
        "microseconds": 1.0 / 1_000_000.0,
    }
    best_name = "milliseconds"
    best_scale = candidates[best_name]
    best_duration = float("nan")
    best_score = float("inf")
    for name, scale in candidates.items():
        candidate_time = np.asarray(raw_time, dtype=float) * scale
        start_s, end_s = _nonzero_window(candidate_time, np.asarray(voltage, dtype=float))
        duration_s = float(end_s - start_s) if np.isfinite(start_s) and np.isfinite(end_s) else float("nan")
        score = abs(duration_s - expected_active_duration_s) if np.isfinite(duration_s) else float("inf")
        if score < best_score:
            best_name = name
            best_scale = scale
            best_duration = duration_s
            best_score = score
    time_s = np.asarray(raw_time, dtype=float) * best_scale
    finite_time = time_s[np.isfinite(time_s)]
    diffs = np.diff(finite_time) if finite_time.size > 1 else np.array([], dtype=float)
    duplicate_count = int(np.sum(np.isfinite(diffs) & np.isclose(diffs, 0.0, atol=1e-15)))
    monotonic = bool(finite_time.size <= 1 or np.all(diffs > 0.0))
    dt_median = float(np.nanmedian(diffs)) if diffs.size else float("nan")
    rel_error = (
        abs(best_duration - expected_active_duration_s) / max(abs(expected_active_duration_s), 1e-12)
        if np.isfinite(best_duration)
        else float("inf")
    )
    status = "ok" if monotonic and duplicate_count == 0 and rel_error <= 0.35 else "suspect_timebase"
    legacy = {"seconds": "s", "milliseconds": "ms", "microseconds": "us"}[best_name]
    return {
        "actual_drive_time_unit_detected": best_name,
        "legacy_time_unit": legacy,
        "time_scale_to_seconds": best_scale,
        "selected_time_unit_reason": "voltage_nonzero_duration_closest_to_expected_active_duration",
        "dt_median_s": dt_median,
        "voltage_nonzero_duration_s": best_duration,
        "actual_voltage_active_duration_s": best_duration,
        "expected_active_duration_s": float(expected_active_duration_s),
        "active_duration_ratio": (
            best_duration / max(abs(float(expected_active_duration_s)), 1e-12)
            if np.isfinite(best_duration)
            else float("nan")
        ),
        "timebase_status": status,
        "timebase_duration_relative_error": rel_error,
        "source_time_monotonic": monotonic,
        "duplicate_time_count": duplicate_count,
    }


def _nonzero_window(time_s: np.ndarray, values: np.ndarray, threshold_fraction: float = 0.02) -> tuple[float, float]:
    finite = np.isfinite(time_s) & np.isfinite(values)
    if finite.sum() < 2:
        return float("nan"), float("nan")
    signal = values[finite]
    pp = float(np.nanmax(signal) - np.nanmin(signal))
    threshold = max(abs(pp) * threshold_fraction, 1e-4)
    active = finite & (np.abs(values) > threshold)
    if not active.any():
        return float("nan"), float("nan")
    return float(np.nanmin(time_s[active])), float(np.nanmax(time_s[active]))
