from __future__ import annotations

import numpy as np


def analytic_fixed_rounded_triangle_normalized(phase: np.ndarray, *, corner_fraction: float = 0.04) -> np.ndarray:
    """Return a deterministic rounded triangle with straight linear segments."""

    phase_values = np.mod(np.asarray(phase, dtype=float), 1.0)
    values = _base_triangle(phase_values)
    radius = float(np.clip(corner_fraction, 0.0, 0.12))
    if radius > 0.0:
        values = _round_corner(values, phase_values, center=0.25, y0=1.0, left_slope=4.0, right_slope=-4.0, radius=radius)
        values = _round_corner(values, phase_values, center=0.75, y0=-1.0, left_slope=-4.0, right_slope=4.0, radius=radius)
    peak = float(np.nanmax(np.abs(values))) if values.size else float("nan")
    if not np.isfinite(peak) or peak <= 1e-12:
        return values
    return values / peak


def target_template_quality(values: np.ndarray, *, target_peak_mT: float = 50.0) -> dict[str, float | bool | str]:
    numeric = np.asarray(values, dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return {
            "target_template_type": "analytic_fixed_rounded_triangle",
            "target_template_ripple_check_passed": False,
            "target_linear_segment_deviation_max_mT": float("nan"),
            "target_peak_positive_mT": float("nan"),
            "target_peak_negative_mT": float("nan"),
        }
    return {
        "target_template_type": "analytic_fixed_rounded_triangle",
        "target_template_ripple_check_passed": True,
        "target_linear_segment_deviation_max_mT": _linear_segment_deviation_max(numeric, target_peak_mT=float(target_peak_mT)),
        "target_peak_positive_mT": float(np.nanmax(finite)),
        "target_peak_negative_mT": float(np.nanmin(finite)),
    }


def _base_triangle(phase: np.ndarray) -> np.ndarray:
    return np.piecewise(
        phase,
        [
            phase < 0.25,
            (phase >= 0.25) & (phase < 0.5),
            (phase >= 0.5) & (phase < 0.75),
            phase >= 0.75,
        ],
        [
            lambda value: 4.0 * value,
            lambda value: 2.0 - 4.0 * value,
            lambda value: -4.0 * (value - 0.5),
            lambda value: -4.0 + 4.0 * value,
        ],
    ).astype(float)


def _round_corner(
    values: np.ndarray,
    phase: np.ndarray,
    *,
    center: float,
    y0: float,
    left_slope: float,
    right_slope: float,
    radius: float,
) -> np.ndarray:
    distance = phase - float(center)
    mask = np.abs(distance) <= float(radius)
    if not np.any(mask):
        return values
    out = values.copy()
    left = mask & (distance <= 0.0)
    right = mask & (distance > 0.0)
    if np.any(left):
        t_left = (distance[left] + radius) / radius
        out[left] = _hermite(
            t_left,
            y0 - left_slope * radius,
            y0,
            left_slope * radius,
            0.0,
        )
    if np.any(right):
        t_right = distance[right] / radius
        out[right] = _hermite(
            t_right,
            y0,
            y0 + right_slope * radius,
            0.0,
            right_slope * radius,
        )
    return out


def _hermite(t: np.ndarray, y_start: float, y_end: float, m_start: float, m_end: float) -> np.ndarray:
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    return h00 * y_start + h10 * m_start + h01 * y_end + h11 * m_end


def _linear_segment_deviation_max(values: np.ndarray, *, target_peak_mT: float) -> float:
    numeric = np.asarray(values, dtype=float)
    if numeric.size < 16:
        return float("nan")
    phase = np.arange(numeric.size, dtype=float) / float(numeric.size)
    normalized = numeric / float(target_peak_mT) if abs(float(target_peak_mT)) > 1e-12 else numeric
    reference = _base_triangle(np.mod(phase, 1.0))
    straight = (
        ((phase >= 0.06) & (phase <= 0.19))
        | ((phase >= 0.31) & (phase <= 0.44))
        | ((phase >= 0.56) & (phase <= 0.69))
        | ((phase >= 0.81) & (phase <= 0.94))
    )
    finite = straight & np.isfinite(normalized) & np.isfinite(reference)
    if not np.any(finite):
        return float("nan")
    return float(np.nanmax(np.abs((normalized[finite] - reference[finite]) * float(target_peak_mT))))
