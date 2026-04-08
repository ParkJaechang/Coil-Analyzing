"""Signal alignment utilities."""

from __future__ import annotations

import numpy as np
from scipy import signal


def estimate_delay_cross_correlation(
    reference: np.ndarray,
    target: np.ndarray,
    sample_rate_hz: float,
) -> float:
    if len(reference) != len(target):
        raise ValueError("Signals must have equal length for cross-correlation delay estimation.")
    ref = reference - np.nanmean(reference)
    tgt = target - np.nanmean(target)
    correlation = signal.correlate(tgt, ref, mode="full")
    lags = signal.correlation_lags(len(tgt), len(ref), mode="full")
    lag_samples = lags[int(np.argmax(correlation))]
    return float(lag_samples / sample_rate_hz)


def shift_signal(time_s: np.ndarray, values: np.ndarray, delay_s: float) -> np.ndarray:
    shifted_time = time_s - delay_s
    return np.interp(time_s, shifted_time, values, left=np.nan, right=np.nan)
