"""Core fundamental and waveform analysis routines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass
class FundamentalResult:
    frequency_hz: float
    amplitude_pk: float
    amplitude_rms: float
    phase_deg: float
    phase_delay_s: float
    raw_rms: float
    raw_pp: float
    crest_factor: float
    thd: float | None
    fitted: np.ndarray
    detrended: np.ndarray
    warnings: list[str]


def estimate_frequency_from_signal(time_s: np.ndarray, values: np.ndarray) -> float:
    cleaned = np.nan_to_num(values - np.nanmean(values))
    if len(cleaned) < 8:
        raise ValueError("Signal too short to estimate frequency.")
    sample_rate_hz = estimate_sample_rate(time_s)
    freqs = np.fft.rfftfreq(len(cleaned), d=1.0 / sample_rate_hz)
    spectrum = np.abs(np.fft.rfft(cleaned))
    if len(freqs) < 2:
        raise ValueError("Spectrum too short to estimate frequency.")
    spectrum[0] = 0.0
    peak_index = int(np.argmax(spectrum))
    return float(freqs[peak_index])


def estimate_sample_rate(time_s: np.ndarray) -> float:
    delta = np.diff(time_s)
    delta = delta[np.isfinite(delta) & (delta > 0)]
    if len(delta) == 0:
        raise ValueError("Invalid time axis.")
    return float(1.0 / np.median(delta))


def build_window_mask(
    time_s: np.ndarray,
    frequency_hz: float,
    cycle_start: int,
    cycle_count: int,
) -> np.ndarray:
    period_s = 1.0 / frequency_hz
    start_time = float(time_s[0] + cycle_start * period_s)
    end_time = float(start_time + cycle_count * period_s)
    return (time_s >= start_time) & (time_s <= end_time)


def apply_zero_phase_smoothing(
    values: np.ndarray,
    sample_rate_hz: float,
    frequency_hz: float,
    order: int,
    cutoff_ratio: float,
) -> np.ndarray:
    nyquist = sample_rate_hz / 2.0
    cutoff_hz = min(max(frequency_hz * (1.0 + cutoff_ratio), frequency_hz * 1.2), nyquist * 0.95)
    if cutoff_hz <= 0.0 or cutoff_hz >= nyquist:
        return values
    b, a = signal.butter(order, cutoff_hz / nyquist, btype="low")
    return signal.filtfilt(b, a, values)


def compute_zero_crossing_frequency(time_s: np.ndarray, values: np.ndarray) -> float | None:
    centered = values - np.nanmean(values)
    signs = np.signbit(centered)
    crossing_indices = np.where(np.diff(signs.astype(int)) > 0)[0]
    if len(crossing_indices) < 2:
        return None
    crossing_times = time_s[crossing_indices]
    periods = np.diff(crossing_times)
    periods = periods[periods > 0]
    if len(periods) == 0:
        return None
    return float(1.0 / np.mean(periods))


def compute_fundamental(
    time_s: np.ndarray,
    values: np.ndarray,
    frequency_hz: float,
    remove_offset: bool = True,
    detrend: bool = True,
    smoothing: bool = False,
    smoothing_order: int = 2,
    smoothing_cutoff_ratio: float = 0.2,
) -> FundamentalResult:
    warnings: list[str] = []
    sample_rate_hz = estimate_sample_rate(time_s)
    processed = np.asarray(values, dtype=float)
    if detrend:
        processed = signal.detrend(processed, type="linear")
    if remove_offset:
        processed = processed - np.nanmean(processed)
    if smoothing:
        processed = apply_zero_phase_smoothing(
            processed,
            sample_rate_hz=sample_rate_hz,
            frequency_hz=frequency_hz,
            order=smoothing_order,
            cutoff_ratio=smoothing_cutoff_ratio,
        )

    omega = 2.0 * np.pi * frequency_hz
    design = np.column_stack([np.sin(omega * time_s), np.cos(omega * time_s), np.ones_like(time_s)])
    coeffs, _, _, _ = np.linalg.lstsq(design, processed, rcond=None)
    sin_coeff, cos_coeff, dc_coeff = coeffs
    amplitude_pk = float(np.hypot(sin_coeff, cos_coeff))
    phase_rad = float(np.arctan2(cos_coeff, sin_coeff))
    fitted = design @ coeffs
    raw_rms = float(np.sqrt(np.mean(np.square(processed + dc_coeff))))
    raw_pp = float(np.nanmax(processed) - np.nanmin(processed))
    crest_factor = float(np.nanmax(np.abs(processed)) / raw_rms) if raw_rms else float("nan")

    coverage_cycles = (time_s[-1] - time_s[0]) * frequency_hz
    if coverage_cycles < 1.5:
        warnings.append("Analysis window covers less than 1.5 cycles.")
    if sample_rate_hz < 20.0 * frequency_hz:
        warnings.append("Sampling rate may be too low for stable phase estimation.")

    return FundamentalResult(
        frequency_hz=float(frequency_hz),
        amplitude_pk=amplitude_pk,
        amplitude_rms=float(amplitude_pk / np.sqrt(2.0)),
        phase_deg=float(np.degrees(phase_rad)),
        phase_delay_s=float(np.degrees(phase_rad) / (360.0 * frequency_hz)) if frequency_hz else float("nan"),
        raw_rms=raw_rms,
        raw_pp=raw_pp,
        crest_factor=crest_factor,
        thd=compute_thd(time_s, processed, frequency_hz),
        fitted=fitted,
        detrended=processed,
        warnings=warnings,
    )


def compute_thd(time_s: np.ndarray, values: np.ndarray, frequency_hz: float, max_harmonic: int = 5) -> float | None:
    sample_rate_hz = estimate_sample_rate(time_s)
    centered = values - np.nanmean(values)
    spectrum = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate_hz)
    if len(freqs) < 3:
        return None
    fundamental_idx = int(np.argmin(np.abs(freqs - frequency_hz)))
    fundamental_mag = np.abs(spectrum[fundamental_idx])
    if fundamental_mag == 0:
        return None
    harmonic_power = 0.0
    for harmonic in range(2, max_harmonic + 1):
        idx = int(np.argmin(np.abs(freqs - harmonic * frequency_hz)))
        harmonic_power += float(np.abs(spectrum[idx]) ** 2)
    return float(np.sqrt(harmonic_power) / fundamental_mag)
