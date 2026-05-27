from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def coerce_measured_field_centered(
    values: np.ndarray,
    column: str,
    active_mask: np.ndarray,
) -> tuple[np.ndarray, str, bool, dict[str, Any]]:
    raw = np.asarray(values, dtype=float)
    active = np.asarray(active_mask, dtype=bool) & np.isfinite(raw)
    if column in {"HallBz", "HallZ", "raw_hallbz_mT"}:
        effective = -raw
        return effective, "raw_hallbz_effective_scale_only", True, {
            "measured_abs_peak_raw_mT": _peak_abs(raw[active]) if np.any(active) else 0.0,
            "measured_field_pre_normalization_baseline_mT": 0.0,
        }
    return raw, "actual_measured_field_scale_only", False, {
        "measured_abs_peak_raw_mT": _peak_abs(raw[active]) if np.any(active) else 0.0,
        "measured_field_pre_normalization_baseline_mT": 0.0,
    }


def normalize_smoothed_field_to_pm50(
    raw_values: np.ndarray,
    smoothed_centered: np.ndarray,
    active_mask: np.ndarray,
    center_meta: dict[str, Any],
    *,
    target_peak_mT: float = 50.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    smoothed = np.asarray(smoothed_centered, dtype=float)
    active = np.asarray(active_mask, dtype=bool) & np.isfinite(smoothed)
    if np.any(active):
        active_values = smoothed[active]
        active_min = float(np.nanmin(active_values))
        active_max = float(np.nanmax(active_values))
        peak = float(np.nanmax(np.abs(active_values)))
    else:
        active_min = active_max = peak = 0.0
    target_peak = abs(float(target_peak_mT)) if np.isfinite(float(target_peak_mT)) else 50.0
    if target_peak <= 1e-12:
        target_peak = 50.0
    scale = target_peak / peak if peak > 1e-12 else 1.0
    normalized = smoothed * scale
    raw_peak = float(
        center_meta.get(
            "measured_abs_peak_raw_mT",
            _peak_abs(np.asarray(raw_values, dtype=float)[np.asarray(active_mask, dtype=bool)]),
        )
    )
    baseline = float(center_meta.get("measured_field_pre_normalization_baseline_mT", 0.0))
    return normalized, measured_normalization_metadata(
        raw_peak=raw_peak,
        effective_peak=peak,
        scale=scale,
        status="ok" if peak > 1e-12 else "zero_peak",
        center=0.0,
        baseline=baseline,
        active_min=active_min,
        active_max=active_max,
        peak=peak,
        target_peak=target_peak,
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
    peak: float,
    target_peak: float = 50.0,
) -> dict[str, Any]:
    return {
        "measured_abs_peak_raw_mT": float(raw_peak),
        "measured_abs_peak_effective_mT": float(effective_peak),
        "measured_field_scale_to_50mT": float(scale),
        "measured_field_scale_to_target_peak_mT": float(scale),
        "field_modeling_normalization_reference_mT": float(target_peak),
        "measured_field_normalization_status": status,
        "measured_field_normalization_mode": "scale_only_abs_peak_to_target_peak",
        "measured_field_normalization_center_mT": float(center),
        "measured_field_pre_normalization_baseline_mT": float(baseline),
        "measured_field_total_offset_removed_mT": float(baseline + center),
        "measured_field_smoothed_active_min_mT": float(active_min),
        "measured_field_smoothed_active_max_mT": float(active_max),
        "measured_field_smoothed_abs_peak_mT": float(peak),
    }


def scale_target_field_columns_to_peak(
    frame: pd.DataFrame,
    active_mask: np.ndarray,
    *,
    target_peak_mT: float,
    target_peak_source: str,
) -> dict[str, Any]:
    active = np.asarray(active_mask, dtype=bool)
    columns = (
        "physical_target_output_mT",
        "target_field_mT",
        "target_output",
        "aligned_target_output",
    )
    scaled_columns: list[str] = []
    original_peak = float("nan")
    scaled_peak = float("nan")
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        peak = _peak_abs(values[active])
        if not np.isfinite(peak) or peak <= 1e-12:
            continue
        scale = float(target_peak_mT) / peak if np.isfinite(target_peak_mT) and target_peak_mT > 1e-12 else 1.0
        frame[column] = values * scale
        scaled_columns.append(column)
        if not np.isfinite(original_peak):
            original_peak = peak
            scaled_peak = _peak_abs(pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)[active])
    return {
        "target_field_original_peak_mT": original_peak,
        "target_field_scale_applied": (float(target_peak_mT) / original_peak)
        if np.isfinite(original_peak) and original_peak > 1e-12 and np.isfinite(target_peak_mT)
        else 1.0,
        "target_field_scaled_peak_mT": scaled_peak,
        "target_field_scaled_columns": tuple(scaled_columns),
        "target_field_scale_source": target_peak_source,
    }


def _peak_abs(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.nanmax(np.abs(finite))) if finite.size else 0.0
