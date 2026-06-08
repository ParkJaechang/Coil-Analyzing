from __future__ import annotations

from typing import Any

import numpy as np


def field_per_volt_response_metadata(
    input_voltage_v: np.ndarray,
    active_mask: np.ndarray,
    *,
    target_peak_mT: float,
    field_scale_to_target: float,
    prefix: str,
) -> dict[str, Any]:
    """Describe the measured field response behind first-model voltage scaling.

    field_scale_to_target is target_peak / measured_peak.  Therefore the
    original measured peak is target_peak / field_scale_to_target, and the
    response ratio is measured_peak / original input voltage peak.
    """
    voltage_peak = _peak_abs(np.asarray(input_voltage_v, dtype=float)[np.asarray(active_mask, dtype=bool)])
    scale = float(field_scale_to_target)
    target_peak = float(target_peak_mT)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    measured_peak = target_peak / scale if target_peak > 1e-12 else float("nan")
    field_per_volt = measured_peak / voltage_peak if voltage_peak > 1e-12 and np.isfinite(measured_peak) else float("nan")
    voltage_per_field = 1.0 / field_per_volt if np.isfinite(field_per_volt) and abs(field_per_volt) > 1e-12 else float("nan")
    return {
        "field_per_volt_mT_per_v": field_per_volt,
        "voltage_per_field_v_per_mT": voltage_per_field,
        "field_per_volt_source": "measured_peak_div_input_voltage_peak",
        "input_voltage_peak_for_field_per_volt_v": voltage_peak,
        "measured_peak_for_field_per_volt_mT": measured_peak,
        "target_peak_for_field_per_volt_mT": target_peak,
        "base_voltage_scale_from_field_per_volt": scale,
        "correction_delta_mode": "residual_div_field_per_volt",
        "correction_delta_v_uses_field_per_volt": True,
        f"{prefix}_field_per_volt_mT_per_v": field_per_volt,
        f"{prefix}_voltage_per_field_v_per_mT": voltage_per_field,
        f"{prefix}_input_voltage_peak_for_field_per_volt_v": voltage_peak,
        f"{prefix}_measured_peak_for_field_per_volt_mT": measured_peak,
    }


def residual_to_voltage_delta_from_field_per_volt(
    residual_mT: np.ndarray,
    field_per_volt_mT_per_v: float,
) -> np.ndarray:
    residual = np.asarray(residual_mT, dtype=float)
    if not np.isfinite(field_per_volt_mT_per_v) or abs(float(field_per_volt_mT_per_v)) <= 1e-12:
        return np.full_like(residual, np.nan, dtype=float)
    return residual / float(field_per_volt_mT_per_v)


def _peak_abs(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.nanmax(np.abs(finite))) if finite.size else 0.0


__all__ = [
    "field_per_volt_response_metadata",
    "residual_to_voltage_delta_from_field_per_volt",
]
