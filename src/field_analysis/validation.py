from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ValidationReport:
    """Validation gate output shared by recommendation and export layers."""

    in_support: bool
    exact_freq_match: bool
    exact_cycle_match: bool
    shape_quality: float
    expected_error_band: float
    allow_auto_recommendation: bool
    reasons: list[str] = field(default_factory=list)
