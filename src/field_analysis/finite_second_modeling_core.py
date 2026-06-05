from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


SUPPORTED_SECOND_MODELING_CYCLES = (1.0, 1.5)


def review_diagnostic_metadata(review_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline_source": review_meta.get("baseline_source"),
        "baseline_mT": review_meta.get("baseline_mT"),
        "actual_drive_time_unit": review_meta.get("actual_drive_time_unit"),
        "actual_drive_time_unit_detected": review_meta.get("actual_drive_time_unit_detected"),
        "selected_time_unit_reason": review_meta.get("selected_time_unit_reason"),
        "dt_median_s": review_meta.get("dt_median_s"),
        "voltage_nonzero_duration_s": review_meta.get("voltage_nonzero_duration_s"),
        "actual_voltage_active_duration_s": review_meta.get("actual_voltage_active_duration_s"),
        "expected_active_duration_s": review_meta.get("expected_active_duration_s"),
        "active_duration_ratio": review_meta.get("active_duration_ratio"),
        "timebase_status": review_meta.get("timebase_status"),
        "source_time_monotonic": review_meta.get("source_time_monotonic"),
        "duplicate_time_count": review_meta.get("duplicate_time_count"),
        "field_convention": review_meta.get("field_convention"),
        "effective_field_convention": review_meta.get("effective_field_convention"),
        "hallbz_effective_sign": review_meta.get("hallbz_effective_sign"),
        "hallbz_sign_applied": review_meta.get("hallbz_sign_applied"),
        "hallbz_sign_inverted": review_meta.get("hallbz_sign_inverted"),
        "hallbz_sign_auto_corrected": review_meta.get("hallbz_sign_auto_corrected"),
        "hallbz_sign_selection_status": review_meta.get("hallbz_sign_selection_status"),
        "hallbz_target_dominant_peak_sign": review_meta.get("hallbz_target_dominant_peak_sign"),
        "hallbz_negative_convention_dominant_peak_sign": review_meta.get("hallbz_negative_convention_dominant_peak_sign"),
        "hallbz_positive_convention_dominant_peak_sign": review_meta.get("hallbz_positive_convention_dominant_peak_sign"),
        "hallbz_negative_convention_corr": review_meta.get("hallbz_negative_convention_corr"),
        "hallbz_positive_convention_corr": review_meta.get("hallbz_positive_convention_corr"),
    }


def is_supported_cycle(cycle_count: float) -> bool:
    return any(abs(float(cycle_count) - cycle) <= 1e-9 for cycle in SUPPORTED_SECOND_MODELING_CYCLES)


def normalize_residual_alignment_mode(value: str) -> str:
    if value == "pointwise":
        return "pointwise"
    if value == "first_peak_aligned":
        return "first_peak_aligned"
    return "first_peak_aligned_stabilized"


def first_voltage(profile: pd.DataFrame) -> np.ndarray:
    for column in (
        "first_modeled_voltage_v",
        "baseline_limited_voltage_v",
        "feedback_corrected_limited_voltage_v",
        "limited_voltage_v",
        "recommended_voltage_v",
    ):
        if column in profile.columns:
            return pd.to_numeric(profile[column], errors="coerce").to_numpy(dtype=float)
    return np.zeros(len(profile), dtype=float)


def first_limited_voltage(profile: pd.DataFrame, fallback: np.ndarray) -> np.ndarray:
    if "limited_voltage_v" in profile.columns:
        return pd.to_numeric(profile["limited_voltage_v"], errors="coerce").to_numpy(dtype=float)
    return np.asarray(fallback, dtype=float).copy()


def optional_voltage(profile: pd.DataFrame, column: str) -> np.ndarray | None:
    if column not in profile.columns:
        return None
    return pd.to_numeric(profile[column], errors="coerce").to_numpy(dtype=float)


def target_field(profile: pd.DataFrame, review: pd.DataFrame, time_s: np.ndarray) -> np.ndarray:
    if "physical_target_output_mT" in profile.columns:
        return pd.to_numeric(profile["physical_target_output_mT"], errors="coerce").to_numpy(dtype=float)
    return interp(review["time_s"], review["normalized_physical_target_output_mT"], time_s)


def active_mask(time_s: np.ndarray, *, freq_hz: float, cycle_count: float) -> np.ndarray:
    duration = float(cycle_count) / max(float(freq_hz), 1e-12)
    values = np.asarray(time_s, dtype=float)
    return np.isfinite(values) & (values >= -1e-12) & (values <= duration + 1e-12)


def interp(source_time: Any, source_values: Any, target_time: Any) -> np.ndarray:
    x = pd.to_numeric(pd.Series(source_time), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(source_values), errors="coerce").to_numpy(dtype=float)
    t = pd.to_numeric(pd.Series(target_time), errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() == 0:
        return np.full(len(t), np.nan)
    order = np.argsort(x[finite])
    return np.interp(t, x[finite][order], y[finite][order], left=np.nan, right=np.nan)


def source_covers_target_active_window(source_time: Any, target_active_time: np.ndarray) -> bool:
    source = pd.to_numeric(pd.Series(source_time), errors="coerce").to_numpy(dtype=float)
    target = np.asarray(target_active_time, dtype=float)
    source = source[np.isfinite(source)]
    target = target[np.isfinite(target)]
    if source.size < 2 or target.size < 1:
        return False
    return bool(np.nanmin(source) <= np.nanmin(target) + 1e-12 and np.nanmax(source) >= np.nanmax(target) - 1e-12)


__all__ = [
    "active_mask",
    "first_limited_voltage",
    "first_voltage",
    "interp",
    "is_supported_cycle",
    "normalize_residual_alignment_mode",
    "optional_voltage",
    "review_diagnostic_metadata",
    "source_covers_target_active_window",
    "target_field",
]
