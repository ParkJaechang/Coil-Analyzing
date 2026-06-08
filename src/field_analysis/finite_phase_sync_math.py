from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def phase_peak_detection_signal(values: np.ndarray, active_mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    active_count = int(np.asarray(active_mask, dtype=bool).sum())
    window = odd_window(min(max(7, active_count // 8), 101), max(active_count, 1))
    signal = pd.Series(np.asarray(values, dtype=float)).rolling(window=window, center=True, min_periods=1).median()
    signal = signal.rolling(window=window, center=True, min_periods=1).mean().to_numpy(dtype=float)
    return signal, {
        "phase_peak_detection_signal": "smoothed_normalized_measured_field",
        "phase_peak_detection_window_samples": int(window),
    }


def odd_window(value: int, max_size: int) -> int:
    size = max(1, min(int(value), int(max_size)))
    if size % 2 == 0:
        size -= 1
    return max(size, 1)


def dominant_peak_time(
    time_s: np.ndarray,
    values: np.ndarray,
    active_mask: np.ndarray,
    *,
    preferred_polarity: str | None = None,
) -> tuple[float | None, str | None, float | None]:
    peaks = peak_candidates(time_s, values, active_mask)
    if not peaks:
        return None, None, None
    if preferred_polarity in {"positive", "negative"}:
        preferred_sign = 1.0 if preferred_polarity == "positive" else -1.0
        same_polarity = [peak for peak in peaks if np.sign(peak[1]) == preferred_sign]
        if same_polarity:
            peaks = same_polarity
    peak_time, selected_value = max(peaks, key=lambda item: abs(float(item[1])))
    polarity = "positive" if float(selected_value) >= 0.0 else "negative"
    return float(peak_time), polarity, float(selected_value)


def reference_voltage_peak_for_measured_peak(
    time_s: np.ndarray,
    values: np.ndarray,
    active_mask: np.ndarray,
    *,
    preferred_polarity: str | None,
    measured_peak_rel_s: float | None,
    output_start_s: float,
) -> tuple[float | None, str | None, float | None]:
    peaks = peak_candidates(time_s, values, active_mask)
    if not peaks:
        return None, None, None
    if preferred_polarity in {"positive", "negative"}:
        preferred_sign = 1.0 if preferred_polarity == "positive" else -1.0
        same_polarity = [peak for peak in peaks if np.sign(peak[1]) == preferred_sign]
        if same_polarity:
            peaks = same_polarity
    if measured_peak_rel_s is not None and np.isfinite(measured_peak_rel_s):
        measured_peak_plot_time = float(output_start_s) + float(measured_peak_rel_s)
        selected_time, selected_value = min(peaks, key=lambda item: abs(float(item[0]) - measured_peak_plot_time))
    else:
        selected_time, selected_value = min(peaks, key=lambda item: float(item[0]))
    polarity = "positive" if float(selected_value) >= 0.0 else "negative"
    return float(selected_time), polarity, float(selected_value)


def midpoint_between_peak_pair(
    time_s: np.ndarray,
    values: np.ndarray,
    active_mask: np.ndarray,
    *,
    pair_start_number: int = 2,
) -> tuple[float | None, str | None, float | None, float | None]:
    peaks = sorted(peak_candidates(time_s, values, active_mask), key=lambda item: float(item[0]))
    if len(peaks) < int(pair_start_number) + 1:
        return None, None, None, None
    left_time, left_value = peaks[int(pair_start_number) - 1]
    right_time, right_value = peaks[int(pair_start_number)]
    midpoint = 0.5 * (float(left_time) + float(right_time))
    polarity = "positive_to_negative" if float(left_value) >= 0.0 and float(right_value) < 0.0 else (
        "negative_to_positive" if float(left_value) < 0.0 and float(right_value) >= 0.0 else "same_polarity"
    )
    return float(midpoint), polarity, float(left_time), float(right_time)


def nearest_zero_crossing_time(
    time_s: np.ndarray,
    values: np.ndarray,
    active_mask: np.ndarray,
    *,
    reference_time_s: float | None = None,
) -> float | None:
    time_arr = np.asarray(time_s, dtype=float)
    value_arr = np.asarray(values, dtype=float)
    active = np.asarray(active_mask, dtype=bool) & np.isfinite(time_arr) & np.isfinite(value_arr)
    if active.sum() < 2:
        return None
    indices = np.flatnonzero(active)
    zero_times: list[float] = []
    for left, right in zip(indices[:-1], indices[1:]):
        y0 = float(value_arr[left])
        y1 = float(value_arr[right])
        t0 = float(time_arr[left])
        t1 = float(time_arr[right])
        if y0 == 0.0:
            zero_times.append(t0)
        elif y0 * y1 < 0.0:
            frac = abs(y0) / max(abs(y0) + abs(y1), 1e-12)
            zero_times.append(t0 + (t1 - t0) * frac)
    if not zero_times:
        return None
    if reference_time_s is not None and np.isfinite(float(reference_time_s)):
        return float(min(zero_times, key=lambda value: abs(float(value) - float(reference_time_s))))
    return float(zero_times[0])


def peak_candidates(time_s: np.ndarray, values: np.ndarray, active_mask: np.ndarray) -> list[tuple[float, float]]:
    time_arr = np.asarray(time_s, dtype=float)
    value_arr = np.asarray(values, dtype=float)
    active = np.asarray(active_mask, dtype=bool) & np.isfinite(time_arr) & np.isfinite(value_arr)
    if active.sum() < 3:
        return []
    indices = np.flatnonzero(active)
    local = value_arr[indices]
    peaks: list[tuple[float, float]] = []
    for local_index in range(1, len(local) - 1):
        value = float(local[local_index])
        is_max = value >= float(local[local_index - 1]) and value >= float(local[local_index + 1])
        is_min = value <= float(local[local_index - 1]) and value <= float(local[local_index + 1])
        if is_max or is_min:
            source_index = int(indices[int(local_index)])
            peaks.append((float(time_arr[source_index]), value))
    if not peaks:
        selected = int(np.nanargmax(np.abs(local)))
        source_index = int(indices[selected])
        peaks = [(float(time_arr[source_index]), float(local[selected]))]
    return peaks


def phase_support_margin_s(time_s: np.ndarray) -> float:
    finite = np.asarray(time_s, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 3:
        return 0.0
    diffs = np.diff(np.sort(finite))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 0.0
    return float(2.0 * np.nanmedian(diffs))


def interp_no_extrapolation(source_time: np.ndarray, source_values: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    source_t = np.asarray(source_time, dtype=float)
    source_y = np.asarray(source_values, dtype=float)
    target_t = np.asarray(target_time, dtype=float)
    finite = np.isfinite(source_t) & np.isfinite(source_y)
    if finite.sum() < 2:
        return np.full_like(target_t, np.nan, dtype=float)
    order = np.argsort(source_t[finite])
    sorted_t = source_t[finite][order]
    sorted_y = source_y[finite][order]
    out = np.interp(target_t, sorted_t, sorted_y)
    out[(target_t < sorted_t[0]) | (target_t > sorted_t[-1])] = np.nan
    return out


def peak_abs(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.nanmax(np.abs(finite))) if finite.size else 0.0
