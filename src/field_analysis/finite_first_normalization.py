from __future__ import annotations

from typing import Any

import numpy as np


def coerce_measured_field_centered(
    values: np.ndarray,
    column: str,
    active_mask: np.ndarray,
) -> tuple[np.ndarray, str, bool, dict[str, Any]]:
    raw = np.asarray(values, dtype=float)
    active = np.asarray(active_mask, dtype=bool) & np.isfinite(raw)
    if column in {"HallBz", "HallZ", "raw_hallbz_mT"}:
        effective = -raw
        active_effective = np.asarray(active_mask, dtype=bool) & np.isfinite(effective)
        baseline = float(np.nanmedian(effective[active_effective])) if np.any(active_effective) else 0.0
        centered = effective - baseline
        return centered, "raw_hallbz_effective_centered", True, {
            "measured_abs_peak_raw_mT": _peak_abs(raw[active]) if np.any(active) else 0.0,
            "measured_field_pre_normalization_baseline_mT": baseline,
        }
    baseline = float(np.nanmedian(raw[active])) if np.any(active) else 0.0
    centered = raw - baseline
    return centered, "actual_measured_field_centered", False, {
        "measured_abs_peak_raw_mT": _peak_abs(raw[active]) if np.any(active) else 0.0,
        "measured_field_pre_normalization_baseline_mT": baseline,
    }


def normalize_smoothed_field_to_pm50(
    raw_values: np.ndarray,
    smoothed_centered: np.ndarray,
    active_mask: np.ndarray,
    center_meta: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    smoothed = np.asarray(smoothed_centered, dtype=float)
    active = np.asarray(active_mask, dtype=bool) & np.isfinite(smoothed)
    if np.any(active):
        active_values = smoothed[active]
        active_min = float(np.nanmin(active_values))
        active_max = float(np.nanmax(active_values))
        normalization_center = 0.5 * (active_max + active_min)
        half_range = 0.5 * (active_max - active_min)
    else:
        active_min = active_max = normalization_center = 0.0
        half_range = 0.0
    scale = 50.0 / half_range if half_range > 1e-12 else 1.0
    normalized = (smoothed - normalization_center) * scale
    raw_peak = float(
        center_meta.get(
            "measured_abs_peak_raw_mT",
            _peak_abs(np.asarray(raw_values, dtype=float)[np.asarray(active_mask, dtype=bool)]),
        )
    )
    baseline = float(center_meta.get("measured_field_pre_normalization_baseline_mT", 0.0))
    return normalized, measured_normalization_metadata(
        raw_peak=raw_peak,
        effective_peak=half_range,
        scale=scale,
        status="ok" if half_range > 1e-12 else "zero_range",
        center=normalization_center,
        baseline=baseline,
        active_min=active_min,
        active_max=active_max,
        half_range=half_range,
    )


def measured_normalization_metadata(
    *,
    raw_peak: float,
    effective_peak: float,
    scale: float,
    status: str,
    center: float,
    baseline: float,
    active_min: float,
    active_max: float,
    half_range: float,
) -> dict[str, Any]:
    return {
        "measured_abs_peak_raw_mT": float(raw_peak),
        "measured_abs_peak_effective_mT": float(effective_peak),
        "measured_field_scale_to_50mT": float(scale),
        "measured_field_normalization_status": status,
        "measured_field_normalization_mode": "smoothed_midrange_to_pm50mT",
        "measured_field_normalization_center_mT": float(center),
        "measured_field_pre_normalization_baseline_mT": float(baseline),
        "measured_field_total_offset_removed_mT": float(baseline + center),
        "measured_field_smoothed_active_min_mT": float(active_min),
        "measured_field_smoothed_active_max_mT": float(active_max),
        "measured_field_smoothed_half_range_mT": float(half_range),
    }


def _peak_abs(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.nanmax(np.abs(finite))) if finite.size else 0.0
