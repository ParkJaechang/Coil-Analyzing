from __future__ import annotations

from typing import Any

import numpy as np


def _clip_lcr_weight(weight: float | None) -> float:
    if weight is None:
        return 0.0
    try:
        value = float(weight)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def resolve_lcr_runtime_policy(
    *,
    requested_lcr_weight: float | None,
    lcr_prior_available: bool,
    exact_field_support_present: bool,
    support_point_count: int,
    waveform_type: str | None = None,
    official_band_applied: bool = True,
) -> dict[str, Any]:
    requested_weight = _clip_lcr_weight(requested_lcr_weight)
    normalized_waveform = str(waveform_type or "").strip().lower()

    if (
        not lcr_prior_available
        or requested_weight <= 0.0
        or normalized_waveform not in {"", "sine"}
        or not official_band_applied
    ):
        return {
            "lcr_usage_mode": "disabled",
            "lcr_weight": 0.0,
            "requested_lcr_weight": requested_weight,
            "exact_field_support_present": bool(exact_field_support_present),
            "lcr_phase_anchor_used": False,
            "lcr_gain_prior_used": False,
        }

    if exact_field_support_present:
        return {
            "lcr_usage_mode": "audit_only",
            "lcr_weight": 0.0,
            "requested_lcr_weight": requested_weight,
            "exact_field_support_present": True,
            "lcr_phase_anchor_used": False,
            "lcr_gain_prior_used": False,
        }

    support_count = max(int(support_point_count), 0)
    capped_weight = min(requested_weight, 0.25 if support_count <= 1 else 0.15)
    prior_used = capped_weight > 0.0
    return {
        "lcr_usage_mode": "weak_prior",
        "lcr_weight": float(capped_weight),
        "requested_lcr_weight": requested_weight,
        "exact_field_support_present": False,
        "lcr_phase_anchor_used": bool(prior_used),
        "lcr_gain_prior_used": bool(prior_used),
    }
