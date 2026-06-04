from __future__ import annotations

from typing import Any

import numpy as np


def peak_error_metrics(
    target_mT: np.ndarray,
    measured_mT: np.ndarray,
    mask: np.ndarray,
    *,
    reference_peak_mT: float,
) -> dict[str, Any]:
    """Return peak-to-peak and signed peak error ratios on the evaluation window."""

    target = np.asarray(target_mT, dtype=float)
    measured = np.asarray(measured_mT, dtype=float)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(target) & np.isfinite(measured)
    if target.size != measured.size or target.size != valid.size or not np.any(valid):
        return _empty_peak_error_metrics()

    target_eval = target[valid]
    measured_eval = measured[valid]
    target_pos = _safe_nanmax(target_eval)
    target_neg = _safe_nanmin(target_eval)
    measured_pos = _safe_nanmax(measured_eval)
    measured_neg = _safe_nanmin(measured_eval)
    target_pp = target_pos - target_neg if np.isfinite(target_pos) and np.isfinite(target_neg) else float("nan")
    measured_pp = (
        measured_pos - measured_neg if np.isfinite(measured_pos) and np.isfinite(measured_neg) else float("nan")
    )
    peak_ref = abs(float(reference_peak_mT)) if np.isfinite(reference_peak_mT) and abs(float(reference_peak_mT)) > 1e-12 else _safe_nanmax(np.abs(target_eval))
    pp_ref = abs(target_pp) if np.isfinite(target_pp) and abs(target_pp) > 1e-12 else 2.0 * peak_ref

    positive_error_mT = measured_pos - target_pos if np.isfinite(measured_pos) and np.isfinite(target_pos) else float("nan")
    negative_error_mT = measured_neg - target_neg if np.isfinite(measured_neg) and np.isfinite(target_neg) else float("nan")
    pp_error_mT = measured_pp - target_pp if np.isfinite(measured_pp) and np.isfinite(target_pp) else float("nan")

    return {
        "target_positive_peak_mT": target_pos,
        "target_negative_peak_mT": target_neg,
        "measured_positive_peak_mT": measured_pos,
        "measured_negative_peak_mT": measured_neg,
        "target_peak_to_peak_mT": target_pp,
        "measured_peak_to_peak_mT": measured_pp,
        "positive_peak_error_mT": positive_error_mT,
        "negative_peak_error_mT": negative_error_mT,
        "peak_to_peak_error_mT": pp_error_mT,
        "positive_peak_error_ratio": _safe_abs_ratio(positive_error_mT, peak_ref),
        "negative_peak_error_ratio": _safe_abs_ratio(negative_error_mT, peak_ref),
        "peak_to_peak_error_ratio": _safe_abs_ratio(pp_error_mT, pp_ref),
        "peak_error_reference_peak_mT": peak_ref,
        "peak_error_reference_pp_mT": pp_ref,
    }


def _empty_peak_error_metrics() -> dict[str, Any]:
    return {
        "target_positive_peak_mT": float("nan"),
        "target_negative_peak_mT": float("nan"),
        "measured_positive_peak_mT": float("nan"),
        "measured_negative_peak_mT": float("nan"),
        "target_peak_to_peak_mT": float("nan"),
        "measured_peak_to_peak_mT": float("nan"),
        "positive_peak_error_mT": float("nan"),
        "negative_peak_error_mT": float("nan"),
        "peak_to_peak_error_mT": float("nan"),
        "positive_peak_error_ratio": float("nan"),
        "negative_peak_error_ratio": float("nan"),
        "peak_to_peak_error_ratio": float("nan"),
        "peak_error_reference_peak_mT": float("nan"),
        "peak_error_reference_pp_mT": float("nan"),
    }


def _safe_nanmax(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.nanmax(finite)) if finite.size else float("nan")


def _safe_nanmin(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.nanmin(finite)) if finite.size else float("nan")


def _safe_abs_ratio(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference) or abs(reference) <= 1e-12:
        return float("nan")
    return float(abs(value) / abs(reference))
