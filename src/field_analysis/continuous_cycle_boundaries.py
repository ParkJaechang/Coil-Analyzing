from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def detect_cycle_boundaries(source: pd.DataFrame, *, freq_hz: float) -> dict[str, Any]:
    period_s = 1.0 / max(float(freq_hz), 1e-12)
    time_s = pd.to_numeric(source["time_s_abs"], errors="coerce").to_numpy(dtype=float)
    voltage = pd.to_numeric(source["raw_voltage_v"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(time_s) & np.isfinite(voltage)
    time_s = time_s[finite]
    voltage = voltage[finite]
    if time_s.size < 4:
        starts = fixed_period_cycle_starts(source, period_s)
        return boundary_metadata(
            method="fixed_period",
            status="insufficient_samples",
            starts=starts,
            period_s=period_s,
            confidence=0.0,
            fallback=True,
        )

    smoothed = smooth_for_zero_crossing(voltage, period_s=period_s, time_s=time_s)
    crossings: list[float] = []
    negative_crossings: list[float] = []
    for idx in range(len(smoothed) - 1):
        left = smoothed[idx]
        right = smoothed[idx + 1]
        if not (np.isfinite(left) and np.isfinite(right)):
            continue
        dt = time_s[idx + 1] - time_s[idx]
        denom = right - left
        if left <= 0.0 < right:
            crossing = time_s[idx] if abs(denom) < 1e-12 else time_s[idx] - left * dt / denom
            if not crossings or crossing - crossings[-1] >= 0.45 * period_s:
                crossings.append(float(crossing))
        elif left >= 0.0 > right:
            crossing = time_s[idx] if abs(denom) < 1e-12 else time_s[idx] - left * dt / denom
            if not negative_crossings or crossing - negative_crossings[-1] >= 0.20 * period_s:
                negative_crossings.append(float(crossing))

    if len(crossings) >= 2:
        intervals = np.diff(np.asarray(crossings, dtype=float))
        median_interval = float(np.nanmedian(intervals)) if intervals.size else period_s
        period_error = abs(median_interval - period_s) / max(period_s, 1e-12)
        confidence = float(max(0.0, min(1.0, 1.0 - period_error)))
        if confidence >= 0.65:
            return boundary_metadata(
                method="voltage_positive_zero_crossing",
                status="ok",
                starts=crossings,
                period_s=period_s,
                confidence=confidence,
                fallback=False,
                positive_count=len(crossings),
                negative_count=len(negative_crossings),
            )

    starts = fixed_period_cycle_starts(source, period_s)
    return boundary_metadata(
        method="fixed_period",
        status="insufficient_crossings",
        starts=starts,
        period_s=period_s,
        confidence=0.0,
        fallback=True,
        positive_count=len(crossings),
        negative_count=len(negative_crossings),
    )


def smooth_for_zero_crossing(voltage: np.ndarray, *, period_s: float, time_s: np.ndarray) -> np.ndarray:
    if voltage.size < 5:
        return voltage
    dt = float(np.nanmedian(np.diff(time_s))) if time_s.size > 1 else period_s / 80.0
    samples_per_period = max(int(round(period_s / max(dt, 1e-12))), 5)
    window = max(5, int(round(samples_per_period * 0.03)))
    if window % 2 == 0:
        window += 1
    window = min(window, max(5, voltage.size // 3 * 2 + 1))
    if window >= voltage.size:
        window = voltage.size - 1 if voltage.size % 2 == 0 else voltage.size
    if window < 5:
        return voltage
    series = pd.Series(voltage, dtype=float)
    return (
        series.rolling(window, center=True, min_periods=1)
        .median()
        .rolling(window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def fixed_period_cycle_starts(source: pd.DataFrame, period_s: float) -> list[float]:
    time_s = pd.to_numeric(source["time_s_abs"], errors="coerce").to_numpy(dtype=float)
    finite = time_s[np.isfinite(time_s)]
    if finite.size == 0:
        return []
    start = float(np.nanmin(finite))
    end = float(np.nanmax(finite))
    count = int(np.floor((end - start) / max(period_s, 1e-12))) + 1
    return [start + idx * period_s for idx in range(max(count + 1, 0))]


def boundary_metadata(
    *,
    method: str,
    status: str,
    starts: list[float],
    period_s: float,
    confidence: float,
    fallback: bool,
    positive_count: int = 0,
    negative_count: int = 0,
) -> dict[str, Any]:
    return {
        "cycle_boundary_method": method,
        "cycle_boundary_confidence": float(confidence),
        "detected_cycle_count": max(len(starts) - 1, 0),
        "period_s": float(period_s),
        "cycle_start_times_s": [float(item) for item in starts],
        "zero_cross_detection_status": status,
        "fixed_period_fallback_used": bool(fallback),
        "zero_crossing_count": int(positive_count + negative_count),
        "positive_going_zero_crossing_count": int(positive_count),
        "negative_going_zero_crossing_count": int(negative_count),
        "half_cycle_boundary_detected": bool(positive_count and negative_count),
        "half_cycle_boundary_rejected": bool(positive_count and negative_count),
    }
