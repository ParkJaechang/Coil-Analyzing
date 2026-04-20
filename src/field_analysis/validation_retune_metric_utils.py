from __future__ import annotations

from .validation_retune_shared import *

def _resolve_target_level(frame: pd.DataFrame, target_output_type: str) -> tuple[float | None, str | None]:
    if "target_output_pp" in frame.columns:
        target_pp = _first_frame_numeric(frame, "target_output_pp")
        if target_pp is not None and np.isfinite(target_pp):
            return float(target_pp), "pp"
    if str(target_output_type) == "field":
        field_signal = _frame_signal_peak(frame, "target_field_mT")
        return field_signal, "peak"
    current_signal = _frame_signal_peak(frame, "target_current_a")
    return current_signal, "peak"


def _resolve_target_output_column(frame: pd.DataFrame) -> tuple[str, str]:
    candidates = (
        ("aligned_used_target_output", "aligned_used_target_output"),
        ("used_target_output", "used_target_output"),
        ("aligned_target_output", "aligned_target_output"),
        ("target_output", "target_output"),
        ("target_current_a", "target_current_a"),
        ("target_field_mT", "target_field_mT"),
    )
    for column, basis in candidates:
        if column in frame.columns:
            return column, basis
    if "target_current_a" in frame.columns:
        return "target_current_a", "target_current_a"
    if "target_field_mT" in frame.columns:
        return "target_field_mT", "target_field_mT"
    raise KeyError("target output column unavailable")


def _resolve_expected_output_column(frame: pd.DataFrame) -> str:
    for column in ("expected_output", "aligned_expected_output", "modeled_output", "expected_current_a", "expected_field_mT"):
        if column in frame.columns:
            return column
    target_column, _ = _resolve_target_output_column(frame)
    return target_column


def _resolve_bz_target_column(frame: pd.DataFrame) -> tuple[str, str]:
    candidates = (
        ("aligned_target_field_mT", "aligned_target_field_mT"),
        ("target_field_mT", "target_field_mT"),
        ("mapped_target_bz_effective_mT", "mapped_target_bz_validation_transfer"),
        ("expected_field_mT", "expected_field_surrogate"),
        ("modeled_field_mT", "modeled_field_surrogate"),
        ("support_scaled_field_mT", "support_scaled_field_surrogate"),
        ("bz_effective_mT", "measured_bz_effective"),
        ("bz_mT", "measured_bz_effective"),
    )
    for column, basis in candidates:
        if column in frame.columns:
            return column, basis
    raise KeyError("bz target column unavailable")


def _resolve_bz_expected_column(frame: pd.DataFrame) -> str:
    for column in ("expected_field_mT", "modeled_field_mT", "support_scaled_field_mT", "target_field_mT", "bz_effective_mT", "bz_mT"):
        if column in frame.columns:
            return column
    target_column, _ = _resolve_bz_target_column(frame)
    return target_column


def _interpolate_column(frame: pd.DataFrame, column: str, time_grid: np.ndarray) -> np.ndarray:
    if frame.empty or column not in frame.columns or "time_s" not in frame.columns:
        return np.full_like(time_grid, np.nan, dtype=float)
    time_values = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
    signal_values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(time_values) & np.isfinite(signal_values)
    if valid.sum() < 2:
        return np.full_like(time_grid, np.nan, dtype=float)
    return np.interp(time_grid, time_values[valid], signal_values[valid])


def _infer_fit_end_s(frame: pd.DataFrame) -> float:
    if frame.empty or "time_s" not in frame.columns:
        return float("nan")
    if "is_active_target" in frame.columns:
        active_mask = frame["is_active_target"].fillna(False).astype(bool)
        if active_mask.any():
            active_time = pd.to_numeric(frame.loc[active_mask, "time_s"], errors="coerce").dropna()
            if not active_time.empty:
                return float(active_time.max())
    time_series = pd.to_numeric(frame["time_s"], errors="coerce").dropna()
    return float(time_series.max()) if not time_series.empty else float("nan")


def _peak_to_peak(values: np.ndarray) -> float:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return float("nan")
    return float(np.nanmax(finite_values) - np.nanmin(finite_values))


def _correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    valid = np.isfinite(reference) & np.isfinite(candidate)
    if valid.sum() < 3:
        return float("nan")
    ref = reference[valid] - float(np.nanmean(reference[valid]))
    comp = candidate[valid] - float(np.nanmean(candidate[valid]))
    ref_std = float(np.nanstd(ref))
    comp_std = float(np.nanstd(comp))
    if ref_std <= 1e-12 or comp_std <= 1e-12:
        return float("nan")
    return float(np.clip(np.corrcoef(ref, comp)[0, 1], -1.0, 1.0))


def _estimate_phase_lag_seconds(reference: np.ndarray, candidate: np.ndarray, time_grid: np.ndarray) -> float:
    valid = np.isfinite(reference) & np.isfinite(candidate)
    if valid.sum() < 4:
        return float("nan")
    ref = reference[valid] - float(np.nanmean(reference[valid]))
    comp = candidate[valid] - float(np.nanmean(candidate[valid]))
    if np.nanstd(ref) <= 1e-12 or np.nanstd(comp) <= 1e-12:
        return float("nan")
    correlation = np.correlate(comp, ref, mode="full")
    lag_index = int(np.argmax(correlation) - (len(ref) - 1))
    if len(time_grid) < 2:
        return float("nan")
    dt = float(np.nanmedian(np.diff(time_grid)))
    if not np.isfinite(dt) or dt <= 0:
        return float("nan")
    return float(lag_index * dt)


def _detect_clipping(frame: pd.DataFrame, column: str, repeat_ratio_threshold: float = 0.05) -> bool:
    if frame.empty or column not in frame.columns:
        return False
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size < 8:
        return False
    rounded = np.round(finite_values, 6)
    max_repeat_ratio = float(np.mean(np.isclose(rounded, np.nanmax(rounded), atol=1e-6)))
    min_repeat_ratio = float(np.mean(np.isclose(rounded, np.nanmin(rounded), atol=1e-6)))
    return bool(max(max_repeat_ratio, min_repeat_ratio) >= float(repeat_ratio_threshold))


def _first_frame_numeric(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.iloc[0])


def _first_frame_bool(frame: pd.DataFrame, column: str) -> bool | None:
    if frame.empty or column not in frame.columns:
        return None
    series = frame[column].dropna()
    if series.empty:
        return None
    value = series.iloc[0]
    if isinstance(value, (bool, np.bool_,)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _detect_hardware_gate_violation(
    frame: pd.DataFrame,
    *,
    gain_shortfall_tolerance_pct: float = SOFT_HARDWARE_LIMIT_TOLERANCE_PCT,
) -> bool:
    if frame.empty:
        return False
    within_hardware = _first_frame_bool(frame, "within_hardware_limits")
    if within_hardware is True:
        return False

    within_daq = _first_frame_bool(frame, "within_daq_limit")
    if within_daq is False:
        return True

    required_gain_pct = _first_frame_numeric(frame, "required_amp_gain_pct")
    available_gain_pct = _first_frame_numeric(frame, "available_amp_gain_pct")
    peak_input_limit_margin = _first_frame_numeric(frame, "peak_input_limit_margin")
    p95_input_limit_margin = _first_frame_numeric(frame, "p95_input_limit_margin")

    nonnegative_input_margin = True
    for margin in (peak_input_limit_margin, p95_input_limit_margin):
        if margin is not None and np.isfinite(margin) and margin < 0.0:
            nonnegative_input_margin = False
            break
    if not nonnegative_input_margin:
        return True

    if (
        required_gain_pct is not None
        and available_gain_pct is not None
        and np.isfinite(required_gain_pct)
        and np.isfinite(available_gain_pct)
    ):
        gain_shortfall_pct = float(required_gain_pct - available_gain_pct)
        if gain_shortfall_pct <= float(gain_shortfall_tolerance_pct) and nonnegative_input_margin:
            return False
        return gain_shortfall_pct > float(gain_shortfall_tolerance_pct)

    return bool(within_hardware is False)


def _first_frame_text(frame: pd.DataFrame, column: str) -> str | None:
    if frame.empty or column not in frame.columns:
        return None
    series = frame[column].dropna()
    if series.empty:
        return None
    return canonicalize_waveform_type(str(series.iloc[0])) or str(series.iloc[0])


def _frame_signal_peak(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return None
    return float(np.nanmax(np.abs(finite_values)))


def _safe_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric


__all__ = [name for name in globals() if not name.startswith('__')]

