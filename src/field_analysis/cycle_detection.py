from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal

from .models import CycleBoundary, CycleDetectionConfig, CycleDetectionResult


def detect_cycles(
    frame: pd.DataFrame,
    config: CycleDetectionConfig,
) -> CycleDetectionResult:
    """Detect repeated cycles using a selected reference channel."""

    working = frame.sort_values("time_s").reset_index(drop=True).copy()
    warnings: list[str] = []
    logs: list[str] = [f"cycle reference={config.reference_channel}"]

    if config.reference_channel not in working.columns:
        working["cycle_index"] = np.nan
        working["cycle_time_s"] = np.nan
        working["cycle_progress"] = np.nan
        return CycleDetectionResult(
            annotated_frame=working,
            boundaries=[],
            estimated_period_s=None,
            estimated_frequency_hz=None,
            reference_channel=config.reference_channel,
            warnings=[f"기준 채널 `{config.reference_channel}` 이 없습니다."],
            logs=logs,
        )

    time = pd.to_numeric(working["time_s"], errors="coerce").to_numpy(dtype=float)
    reference = pd.to_numeric(working[config.reference_channel], errors="coerce").to_numpy(dtype=float)
    estimated_period_s = (
        float(config.manual_period_s)
        if config.manual_period_s and config.manual_period_s > 0
        else estimate_period_seconds(time=time, values=reference)
    )
    estimated_frequency_hz = 1.0 / estimated_period_s if estimated_period_s and estimated_period_s > 0 else None
    logs.append(f"estimated_period_s={estimated_period_s}")

    if config.manual_start_s is not None and estimated_period_s:
        boundary_times = np.asarray(
            [
                float(config.manual_start_s) + cycle_index * estimated_period_s
                for cycle_index in range(config.expected_cycles + 1)
            ],
            dtype=float,
        )
        logs.append("수동 시작 시각과 주기로 boundary 생성")
    else:
        boundary_times = select_cycle_boundaries(
            time=time,
            values=reference,
            expected_cycles=config.expected_cycles,
            estimated_period_s=estimated_period_s,
            preferred_start_s=config.manual_start_s,
        )
        if len(boundary_times) < 2 and estimated_period_s:
            boundary_times = _fallback_uniform_boundaries(
                time=time,
                estimated_period_s=estimated_period_s,
                expected_cycles=config.expected_cycles,
            )
            warnings.append("zero crossing 기반 cycle 경계가 부족하여 균등 주기 추정으로 대체했습니다.")

    boundaries = build_boundaries_from_times(time, boundary_times)
    if len(boundaries) != config.expected_cycles:
        warnings.append(
            f"검출 cycle 수 {len(boundaries)}개가 기대값 {config.expected_cycles}개와 다릅니다."
        )

    working = annotate_cycles(working, boundaries, config.reference_channel)

    source_cycle_count = (
        int(working["source_cycle_no"].dropna().nunique())
        if "source_cycle_no" in working.columns
        else None
    )
    if source_cycle_count and abs(source_cycle_count - len(boundaries)) >= 2:
        warnings.append(
            f"원본 CycleNo 개수 {source_cycle_count}와 자동 검출 cycle 수 {len(boundaries)}가 크게 다릅니다."
        )

    if "freq_hz" in working.columns and estimated_frequency_hz is not None:
        declared_freq = pd.to_numeric(working["freq_hz"], errors="coerce").dropna()
        if not declared_freq.empty:
            freq_value = float(declared_freq.iloc[0])
            if abs(freq_value - estimated_frequency_hz) / max(freq_value, 1e-6) > 0.15:
                warnings.append(
                    f"명시 주파수 {freq_value:g}Hz 와 검출 주파수 {estimated_frequency_hz:g}Hz 가 15% 이상 다릅니다."
                )

    return CycleDetectionResult(
        annotated_frame=working,
        boundaries=boundaries,
        estimated_period_s=estimated_period_s,
        estimated_frequency_hz=estimated_frequency_hz,
        reference_channel=config.reference_channel,
        warnings=warnings,
        logs=logs,
    )


def estimate_period_seconds(time: np.ndarray, values: np.ndarray) -> float | None:
    """Estimate a dominant period using FFT, autocorrelation, and zero crossings."""

    valid = np.isfinite(time) & np.isfinite(values)
    if valid.sum() < 16:
        return None

    time = time[valid]
    values = values[valid]
    dt = np.median(np.diff(time))
    if not np.isfinite(dt) or dt <= 0:
        return None

    centered = values - np.nanmean(values)
    estimates: list[float] = []

    freqs = np.fft.rfftfreq(len(centered), d=dt)
    amplitudes = np.abs(np.fft.rfft(centered))
    valid_freq = freqs > 0.05
    if valid_freq.any():
        peak_freq = freqs[valid_freq][np.argmax(amplitudes[valid_freq])]
        if peak_freq > 0:
            estimates.append(float(1.0 / peak_freq))

    autocorr = signal.correlate(centered, centered, mode="full", method="fft")
    lags = signal.correlation_lags(len(centered), len(centered), mode="full")
    positive = lags > 0
    if positive.any():
        autocorr = autocorr[positive]
        lags = lags[positive]
        peak_indices, _ = signal.find_peaks(autocorr)
        if len(peak_indices) > 0:
            best_lag = lags[peak_indices[np.argmax(autocorr[peak_indices])]]
            estimates.append(float(best_lag * dt))

    crossings = _ascending_zero_crossings(time=time, values=centered)
    if len(crossings) >= 2:
        zero_period = float(np.median(np.diff(crossings)))
        if zero_period > 0:
            estimates.append(zero_period)

    finite_estimates = [estimate for estimate in estimates if np.isfinite(estimate) and estimate > 0]
    if not finite_estimates:
        return None
    return float(np.median(finite_estimates))


def select_cycle_boundaries(
    time: np.ndarray,
    values: np.ndarray,
    expected_cycles: int,
    estimated_period_s: float | None,
    preferred_start_s: float | None,
) -> np.ndarray:
    """Select the most coherent sequence of cycle boundary times."""

    centered = values - np.nanmean(values)
    crossings = _ascending_zero_crossings(time=time, values=centered)
    if len(crossings) < 2:
        return np.asarray([], dtype=float)

    if expected_cycles <= 0:
        return crossings

    required_points = expected_cycles + 1
    if len(crossings) < required_points:
        return crossings

    best_score = float("inf")
    best_window: np.ndarray | None = None

    for start_index in range(0, len(crossings) - required_points + 1):
        window = crossings[start_index : start_index + required_points]
        diffs = np.diff(window)
        if np.any(diffs <= 0):
            continue
        score = float(np.std(diffs))
        if estimated_period_s is not None:
            score += float(np.mean(np.abs(diffs - estimated_period_s)))
        if preferred_start_s is not None:
            score += abs(window[0] - preferred_start_s)
        if score < best_score:
            best_score = score
            best_window = window

    if best_window is not None:
        return best_window
    return crossings[:required_points]


def build_boundaries_from_times(
    time: np.ndarray,
    boundary_times: np.ndarray,
) -> list[CycleBoundary]:
    """Convert boundary times into row-index boundaries."""

    if len(boundary_times) < 2:
        return []

    boundaries: list[CycleBoundary] = []
    for cycle_index in range(len(boundary_times) - 1):
        start_s = float(boundary_times[cycle_index])
        end_s = float(boundary_times[cycle_index + 1])
        start_index = int(np.searchsorted(time, start_s, side="left"))
        end_index = int(np.searchsorted(time, end_s, side="right")) - 1
        if end_index <= start_index:
            continue
        boundaries.append(
            CycleBoundary(
                cycle_index=cycle_index + 1,
                start_index=start_index,
                end_index=end_index,
                start_s=start_s,
                end_s=end_s,
            )
        )
    return boundaries


def annotate_cycles(
    frame: pd.DataFrame,
    boundaries: list[CycleBoundary],
    reference_channel: str,
) -> pd.DataFrame:
    """Attach cycle and branch annotations to the measurement frame."""

    annotated = frame.copy()
    annotated["cycle_index"] = np.nan
    annotated["cycle_time_s"] = np.nan
    annotated["cycle_progress"] = np.nan
    annotated["branch_direction"] = ""

    for boundary in boundaries:
        mask = annotated.index.to_series().between(boundary.start_index, boundary.end_index, inclusive="both")
        duration = max(boundary.end_s - boundary.start_s, 1e-12)
        annotated.loc[mask, "cycle_index"] = boundary.cycle_index
        annotated.loc[mask, "cycle_time_s"] = annotated.loc[mask, "time_s"] - boundary.start_s
        annotated.loc[mask, "cycle_progress"] = annotated.loc[mask, "cycle_time_s"] / duration

    reference = pd.to_numeric(annotated[reference_channel], errors="coerce")
    derivative = np.gradient(
        reference.ffill().bfill().to_numpy(dtype=float),
        pd.to_numeric(annotated["time_s"], errors="coerce").to_numpy(dtype=float),
    )
    annotated["branch_direction"] = np.where(derivative >= 0, "rising", "falling")
    return annotated


def _ascending_zero_crossings(time: np.ndarray, values: np.ndarray) -> np.ndarray:
    crossings: list[float] = []
    for index in range(len(values) - 1):
        left = values[index]
        right = values[index + 1]
        if not np.isfinite(left) or not np.isfinite(right):
            continue
        if left <= 0 < right:
            delta = right - left
            ratio = 0.0 if delta == 0 else abs(left) / abs(delta)
            crossing = time[index] + ratio * (time[index + 1] - time[index])
            crossings.append(float(crossing))
    return np.asarray(crossings, dtype=float)


def _fallback_uniform_boundaries(
    time: np.ndarray,
    estimated_period_s: float,
    expected_cycles: int,
) -> np.ndarray:
    if expected_cycles <= 0 or estimated_period_s <= 0:
        return np.asarray([], dtype=float)

    start = float(time[0])
    end = start + expected_cycles * estimated_period_s
    if end > float(time[-1]):
        end = float(time[-1])
        start = end - expected_cycles * estimated_period_s
    return np.asarray(
        [start + index * estimated_period_s for index in range(expected_cycles + 1)],
        dtype=float,
    )
