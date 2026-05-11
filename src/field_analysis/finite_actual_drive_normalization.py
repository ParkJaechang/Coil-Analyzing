from __future__ import annotations

from typing import Any

import numpy as np


def normalize_peak_to_limit(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    limit: float,
    unavailable_status: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.asarray(values, dtype=float)
    finite_mask = np.asarray(mask, dtype=bool) & np.isfinite(source)
    source_peak = peak_abs(source[finite_mask])
    if not np.isfinite(source_peak) or source_peak <= 1e-12:
        return np.zeros_like(source, dtype=float), {
            "status": unavailable_status,
            "source_peak": source_peak,
            "scale_factor": float("nan"),
        }
    scale = float(limit) / source_peak
    return source * scale, {
        "status": "ok",
        "source_peak": source_peak,
        "scale_factor": scale,
    }


def peak_abs(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.nanmax(np.abs(finite)))
