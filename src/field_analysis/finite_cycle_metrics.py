from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_TARGET_CANDIDATES = (
    "aligned_target_field_mT",
    "target_field_mT",
    "aligned_target_output",
    "target_output",
)
DEFAULT_PREDICTED_CANDIDATES = (
    "predicted_field_mT",
    "expected_field_mT",
    "expected_output",
)
FINITE_METRIC_COLUMN_MAP = {
    "active_window_nrmse": "finite_active_nrmse",
    "active_window_rmse_mT": "finite_active_rmse_mT",
    "active_window_shape_corr": "finite_active_shape_corr",
    "terminal_peak_error_mT": "finite_terminal_peak_error_mT",
    "terminal_peak_error_ratio": "finite_terminal_peak_error_ratio",
    "terminal_value_error_mT": "finite_terminal_value_error_mT",
    "terminal_target_slope_sign": "finite_terminal_target_slope_sign",
    "terminal_predicted_slope_sign": "finite_terminal_predicted_slope_sign",
    "terminal_direction_match": "finite_terminal_direction_match",
    "terminal_slope_error_mT_per_s": "finite_terminal_slope_error_mT_per_s",
    "tail_residual_peak_mT": "finite_tail_residual_peak_mT",
    "tail_residual_ratio": "finite_tail_residual_ratio",
    "estimated_lag_seconds": "finite_estimated_lag_seconds",
    "active_sample_count": "finite_active_sample_count",
    "tail_sample_count": "finite_tail_sample_count",
    "target_peak_mT": "finite_target_peak_mT",
    "predicted_peak_mT": "finite_predicted_peak_mT",
    "active_end_s": "finite_active_end_s",
    "tail_end_s": "finite_tail_end_s",
    "evaluation_status": "finite_evaluation_status",
    "unavailable_reason": "finite_unavailable_reason",
}


@dataclass(slots=True)
class FiniteCycleMetrics:
    active_window_nrmse: float = float("nan")
    active_window_rmse_mT: float = float("nan")
    active_window_shape_corr: float = float("nan")
    terminal_peak_error_mT: float = float("nan")
    terminal_peak_error_ratio: float = float("nan")
    terminal_value_error_mT: float = float("nan")
    terminal_target_slope_sign: float = float("nan")
    terminal_predicted_slope_sign: float = float("nan")
    terminal_direction_match: bool | None = None
    terminal_slope_error_mT_per_s: float = float("nan")
    tail_residual_peak_mT: float = float("nan")
    tail_residual_ratio: float = float("nan")
    estimated_lag_seconds: float = float("nan")
    active_sample_count: int = 0
    tail_sample_count: int = 0
    target_peak_mT: float = float("nan")
    predicted_peak_mT: float = float("nan")
    active_end_s: float = float("nan")
    tail_end_s: float = float("nan")
    evaluation_status: str = "unavailable"
    unavailable_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_finite_cycle_metrics(
    command_profile: pd.DataFrame,
    *,
    target_candidates: tuple[str, ...] = DEFAULT_TARGET_CANDIDATES,
    predicted_candidates: tuple[str, ...] = DEFAULT_PREDICTED_CANDIDATES,
    time_column: str = "time_s",
    active_mask_column: str = "is_active_target",
) -> FiniteCycleMetrics:
    if command_profile.empty:
        return FiniteCycleMetrics(unavailable_reason="empty_command_profile")
    if time_column not in command_profile.columns:
        return FiniteCycleMetrics(unavailable_reason="missing_time_column")
    if active_mask_column not in command_profile.columns:
        return FiniteCycleMetrics(unavailable_reason="missing_active_mask")

    target_column = _resolve_first_existing_column(command_profile, target_candidates)
    predicted_column = _resolve_first_existing_column(command_profile, predicted_candidates)
    if target_column is None:
        return FiniteCycleMetrics(unavailable_reason="missing_target_column")
    if predicted_column is None:
        return FiniteCycleMetrics(unavailable_reason="missing_predicted_column")

    time_values = pd.to_numeric(command_profile[time_column], errors="coerce").to_numpy(dtype=float)
    target_values = pd.to_numeric(command_profile[target_column], errors="coerce").to_numpy(dtype=float)
    predicted_values = pd.to_numeric(command_profile[predicted_column], errors="coerce").to_numpy(dtype=float)
    active_mask = command_profile[active_mask_column].astype(bool).to_numpy(dtype=bool)

    valid_mask = np.isfinite(time_values) & np.isfinite(target_values) & np.isfinite(predicted_values)
    active_mask = active_mask & valid_mask
    if active_mask.sum() < 3:
        return FiniteCycleMetrics(unavailable_reason="insufficient_active_samples")

    active_indices = np.flatnonzero(active_mask)
    active_time = time_values[active_mask]
    active_target = target_values[active_mask]
    active_predicted = predicted_values[active_mask]
    active_end_index = int(active_indices[-1])
    active_end_s = float(time_values[active_end_index])
    tail_mask = valid_mask & (time_values > active_end_s + 1e-12)

    target_peak_to_peak = float(np.nanmax(active_target) - np.nanmin(active_target))
    normalization_scale = max(target_peak_to_peak / 2.0, 1e-12)
    rmse = float(np.sqrt(np.nanmean(np.square(active_predicted - active_target))))
    nrmse = rmse / normalization_scale
    shape_corr = _safe_corr(active_target - float(np.nanmean(active_target)), active_predicted - float(np.nanmean(active_predicted)))
    target_peak = float(np.nanmax(np.abs(active_target)))
    predicted_peak = float(np.nanmax(np.abs(active_predicted)))

    terminal_mask = _build_terminal_mask(
        time_values=time_values,
        active_mask=active_mask,
        command_profile=command_profile,
    )
    if terminal_mask.sum() < 2:
        terminal_mask = active_mask.copy()
    terminal_target = target_values[terminal_mask]
    terminal_predicted = predicted_values[terminal_mask]
    terminal_time = time_values[terminal_mask]

    terminal_target_peak = float(np.nanmax(np.abs(terminal_target)))
    terminal_predicted_peak = float(np.nanmax(np.abs(terminal_predicted)))
    terminal_peak_error = float(terminal_predicted_peak - terminal_target_peak)
    terminal_peak_error_ratio = terminal_peak_error / max(terminal_target_peak, 1e-12)
    terminal_value_error = float(predicted_values[active_end_index] - target_values[active_end_index])
    target_terminal_slope = _terminal_slope_rate(terminal_target, terminal_time)
    predicted_terminal_slope = _terminal_slope_rate(terminal_predicted, terminal_time)
    target_slope_sign = _slope_sign(target_terminal_slope)
    predicted_slope_sign = _slope_sign(predicted_terminal_slope)
    direction_match = (
        bool(target_slope_sign == predicted_slope_sign)
        if np.isfinite(target_slope_sign) and np.isfinite(predicted_slope_sign)
        else None
    )
    slope_error = (
        float(predicted_terminal_slope - target_terminal_slope)
        if np.isfinite(target_terminal_slope) and np.isfinite(predicted_terminal_slope)
        else float("nan")
    )

    tail_sample_count = int(tail_mask.sum())
    if tail_sample_count > 0:
        tail_predicted = predicted_values[tail_mask]
        tail_residual_peak = float(np.nanmax(np.abs(tail_predicted)))
        tail_end_s = float(np.nanmax(time_values[tail_mask]))
    else:
        tail_residual_peak = 0.0
        tail_end_s = float("nan")
    tail_residual_ratio = tail_residual_peak / max(target_peak, 1e-12)
    estimated_lag_seconds = _estimate_lag_seconds(active_time, active_target, active_predicted)

    return FiniteCycleMetrics(
        active_window_nrmse=nrmse,
        active_window_rmse_mT=rmse,
        active_window_shape_corr=shape_corr,
        terminal_peak_error_mT=terminal_peak_error,
        terminal_peak_error_ratio=terminal_peak_error_ratio,
        terminal_value_error_mT=terminal_value_error,
        terminal_target_slope_sign=target_slope_sign,
        terminal_predicted_slope_sign=predicted_slope_sign,
        terminal_direction_match=direction_match,
        terminal_slope_error_mT_per_s=slope_error,
        tail_residual_peak_mT=tail_residual_peak,
        tail_residual_ratio=tail_residual_ratio,
        estimated_lag_seconds=estimated_lag_seconds,
        active_sample_count=int(active_mask.sum()),
        tail_sample_count=tail_sample_count,
        target_peak_mT=target_peak,
        predicted_peak_mT=predicted_peak,
        active_end_s=active_end_s,
        tail_end_s=tail_end_s,
        evaluation_status="ok",
        unavailable_reason=None,
    )


def attach_finite_cycle_metrics(
    command_profile: pd.DataFrame,
    metrics: FiniteCycleMetrics,
) -> pd.DataFrame:
    if command_profile.empty:
        return command_profile
    payload = metrics.to_dict()
    for metric_name, column_name in FINITE_METRIC_COLUMN_MAP.items():
        command_profile[column_name] = payload.get(metric_name)
    return command_profile


def _resolve_first_existing_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _build_terminal_mask(
    *,
    time_values: np.ndarray,
    active_mask: np.ndarray,
    command_profile: pd.DataFrame,
) -> np.ndarray:
    active_time = time_values[active_mask]
    if active_time.size < 2:
        return active_mask.copy()
    active_end_s = float(np.nanmax(active_time))
    active_start_s = float(np.nanmin(active_time))
    active_duration_s = max(active_end_s - active_start_s, 0.0)
    freq_hz = _first_numeric(command_profile.get("freq_hz"))
    half_cycle_s = 0.5 / freq_hz if np.isfinite(freq_hz) and freq_hz > 0 else 0.0
    terminal_window_s = max(active_duration_s * 0.20, half_cycle_s)
    if not np.isfinite(terminal_window_s) or terminal_window_s <= 0:
        terminal_window_s = max(_median_time_step(active_time) * 4.0, 0.0)
    window_start_s = active_end_s - terminal_window_s
    terminal_mask = active_mask & (time_values >= window_start_s - 1e-12)
    if terminal_mask.sum() < 2:
        active_indices = np.flatnonzero(active_mask)
        fallback_indices = active_indices[-min(4, active_indices.size):]
        terminal_mask = np.zeros_like(active_mask, dtype=bool)
        terminal_mask[fallback_indices] = True
    return terminal_mask


def _estimate_lag_seconds(time_values: np.ndarray, target_values: np.ndarray, predicted_values: np.ndarray) -> float:
    if target_values.size < 3 or predicted_values.size < 3:
        return float("nan")
    dt = _median_time_step(time_values)
    if not np.isfinite(dt) or dt <= 0:
        return float("nan")
    target_centered = target_values - float(np.nanmean(target_values))
    predicted_centered = predicted_values - float(np.nanmean(predicted_values))
    max_shift = min(max(target_centered.size // 4, 1), 64)
    best_shift = 0
    best_score = -np.inf
    for shift in range(-max_shift, max_shift + 1):
        left, right = _aligned_segments(target_centered, predicted_centered, shift)
        score = _safe_corr(left, right)
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_shift = shift
    return float(best_shift * dt)


def _aligned_segments(left: np.ndarray, right: np.ndarray, shift: int) -> tuple[np.ndarray, np.ndarray]:
    if shift == 0:
        return left, right
    if shift > 0:
        return left[:-shift], right[shift:]
    return left[-shift:], right[:shift]


def _terminal_slope_rate(values: np.ndarray, time_values: np.ndarray) -> float:
    if values.size < 2 or time_values.size < 2:
        return float("nan")
    dt = float(time_values[-1] - time_values[-2])
    if not np.isfinite(dt) or abs(dt) <= 1e-12:
        return float("nan")
    return float((values[-1] - values[-2]) / dt)


def _slope_sign(slope_value: float) -> float:
    if not np.isfinite(slope_value):
        return float("nan")
    if abs(slope_value) <= 1e-6:
        return 0.0
    return float(np.sign(slope_value))


def _median_time_step(time_values: np.ndarray) -> float:
    if time_values.size < 2:
        return float("nan")
    deltas = np.diff(time_values)
    finite_deltas = deltas[np.isfinite(deltas) & (np.abs(deltas) > 1e-12)]
    if finite_deltas.size == 0:
        return float("nan")
    return float(np.median(finite_deltas))


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return float("nan")
    left_valid = left[valid]
    right_valid = right[valid]
    if np.allclose(np.nanstd(left_valid), 0.0) or np.allclose(np.nanstd(right_valid), 0.0):
        return float("nan")
    return float(np.corrcoef(left_valid, right_valid)[0, 1])


def _first_numeric(series: pd.Series | Any) -> float:
    if series is None:
        return float("nan")
    if isinstance(series, pd.Series):
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        return float(numeric.iloc[0]) if not numeric.empty else float("nan")
    try:
        return float(series)
    except (TypeError, ValueError):
        return float("nan")
