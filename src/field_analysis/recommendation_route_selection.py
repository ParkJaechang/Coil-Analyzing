from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _classify_support_state(
    support_freqs_hz: list[float] | np.ndarray,
    requested_freq_hz: float,
    *,
    exact_frequency_match: bool,
) -> str:
    """Classify support geometry for rollout/evaluation reporting."""

    if exact_frequency_match:
        return "exact"
    if requested_freq_hz is None or not np.isfinite(requested_freq_hz):
        return "unsupported"

    support_array = pd.to_numeric(pd.Series(support_freqs_hz), errors="coerce").dropna().to_numpy(dtype=float)
    if support_array.size == 0:
        return "unsupported"

    support_array = np.unique(np.sort(support_array))
    tolerance = max(abs(float(requested_freq_hz)) * 1e-6, 1e-9)
    min_freq = float(support_array[0])
    max_freq = float(support_array[-1])
    lower_exists = bool(np.any(support_array < float(requested_freq_hz) - tolerance))
    upper_exists = bool(np.any(support_array > float(requested_freq_hz) + tolerance))

    if float(requested_freq_hz) < min_freq - tolerance or float(requested_freq_hz) > max_freq + tolerance:
        return "out_of_hull"
    if lower_exists and upper_exists:
        return "interpolated_in_hull"
    return "interpolated_edge"


def _summarize_frequency_geometry(
    support_freqs_hz: list[float] | np.ndarray,
    requested_freq_hz: float,
    *,
    exact_frequency_match: bool,
) -> dict[str, Any]:
    support_array = pd.to_numeric(pd.Series(support_freqs_hz), errors="coerce").dropna().to_numpy(dtype=float)
    support_array = np.unique(np.sort(support_array))
    if support_array.size == 0 or requested_freq_hz is None or not np.isfinite(requested_freq_hz):
        return {
            "nearest_support_distance_hz": np.nan,
            "nearest_support_distance_log": np.nan,
            "lower_support_hz": np.nan,
            "upper_support_hz": np.nan,
            "bracket_span_hz": np.nan,
            "bracket_span_log": np.nan,
            "bracket_position_log": np.nan,
            "support_density_hz": np.nan,
        }

    requested = float(requested_freq_hz)
    tolerance = max(abs(requested) * 1e-6, 1e-9)
    nearest_distance_hz = float(np.min(np.abs(support_array - requested)))
    nearest_support = float(support_array[np.argmin(np.abs(support_array - requested))])
    nearest_distance_log = (
        float(abs(np.log(requested / nearest_support)))
        if requested > 0 and nearest_support > 0
        else np.nan
    )
    lower_values = support_array[support_array < requested - tolerance]
    upper_values = support_array[support_array > requested + tolerance]
    lower_support = float(lower_values[-1]) if lower_values.size else np.nan
    upper_support = float(upper_values[0]) if upper_values.size else np.nan

    bracket_span_hz = float(upper_support - lower_support) if np.isfinite(lower_support) and np.isfinite(upper_support) else np.nan
    bracket_span_log = (
        float(np.log(upper_support / lower_support))
        if np.isfinite(lower_support) and np.isfinite(upper_support) and lower_support > 0 and upper_support > 0
        else np.nan
    )
    bracket_position_log = (
        float(np.log(requested / lower_support) / np.log(upper_support / lower_support))
        if np.isfinite(lower_support)
        and np.isfinite(upper_support)
        and requested > 0
        and lower_support > 0
        and upper_support > 0
        and upper_support > lower_support
        else np.nan
    )
    support_density_hz = (
        float(len(support_array) / max(float(support_array[-1] - support_array[0]), 1e-9))
        if support_array.size >= 2
        else np.nan
    )
    if exact_frequency_match:
        bracket_position_log = 0.5
        bracket_span_log = 0.0
        nearest_distance_log = 0.0
        nearest_distance_hz = 0.0

    return {
        "nearest_support_distance_hz": nearest_distance_hz,
        "nearest_support_distance_log": nearest_distance_log,
        "lower_support_hz": lower_support,
        "upper_support_hz": upper_support,
        "bracket_span_hz": bracket_span_hz,
        "bracket_span_log": bracket_span_log,
        "bracket_position_log": bracket_position_log,
        "support_density_hz": support_density_hz,
    }
