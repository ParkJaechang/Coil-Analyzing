from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from .canonical_runs import CanonicalRun, Regime, TargetLevelKind, TargetType
from .validation import ValidationReport


@dataclass(slots=True)
class TargetRequest:
    """Requested target specification kept separate from measured run definitions."""

    regime: Regime
    target_waveform: str
    freq_hz: float | None
    command_waveform: str | None = None
    commanded_cycles: float | None = None
    target_type: TargetType = "unknown"
    target_level_value: float | None = None
    target_level_kind: TargetLevelKind | None = None
    role: Literal["interactive", "batch"] = "interactive"
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RecommendationOptions:
    """Execution options for the recommendation service."""

    current_channel: str = "i_sum_signed"
    field_channel: str = "bz_mT"
    max_daq_voltage_pp: float = 20.0
    amp_gain_at_100_pct: float = 20.0
    amp_gain_limit_pct: float = 100.0
    amp_max_output_pk_v: float = 180.0
    default_support_amp_gain_pct: float = 100.0
    allow_target_extrapolation: bool = True
    allow_output_extrapolation: bool = True
    frequency_mode: str = "exact"
    preview_tail_cycles: float = 0.25
    lcr_measurements: pd.DataFrame | None = None
    lcr_blend_weight: float = 0.0
    apply_startup_correction: bool = False
    startup_transition_cycles: float = 0.25
    startup_correction_strength: float = 0.65
    startup_preview_cycle_count: int = 3


@dataclass(slots=True)
class LegacyRecommendationContext:
    """Legacy objects still required behind the PR2 adapter boundary."""

    per_test_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    analysis_lookup: dict[str, Any] = field(default_factory=dict)
    transient_measurements: list = field(default_factory=list)
    transient_preprocess_results: list = field(default_factory=list)
    transient_canonical_runs: list[CanonicalRun] = field(default_factory=list)
    validation_measurements: list = field(default_factory=list)
    validation_preprocess_results: list = field(default_factory=list)


@dataclass(slots=True)
class RecommendationResult:
    """Standard recommendation output consumed by UI and export layers."""

    selected_regime: Literal["continuous", "transient"]
    preview_only: bool
    allow_auto_download: bool
    recommended_time_s: np.ndarray
    recommended_input_v: np.ndarray
    predicted_current_a: np.ndarray | None
    predicted_bx_mT: np.ndarray | None
    predicted_by_mT: np.ndarray | None
    predicted_bz_mT: np.ndarray | None
    validation_report: ValidationReport | None
    warnings: list[str] = field(default_factory=list)
    debug_info: dict[str, Any] = field(default_factory=dict)
    engine_summary: dict[str, Any] = field(default_factory=dict)
    support_summary: dict[str, Any] = field(default_factory=dict)
    confidence_summary: dict[str, Any] = field(default_factory=dict)
    command_profile: pd.DataFrame | None = None
    lookup_table: pd.DataFrame | None = None
    support_table: pd.DataFrame | None = None
    legacy_payload: dict[str, Any] | None = None


@dataclass(slots=True)
class RecommendationPolicy:
    """Policy gate for interpolated steady-state auto recommendation."""

    min_surface_confidence: float = 0.85
    min_harmonic_fill_ratio: float = 0.6
    max_predicted_error_band: float = 0.15
    min_input_limit_margin: float = 0.10
    min_support_runs: int = 2
    allow_interpolated_auto: bool = False
    margin_source: Literal["gain", "peak", "p95"] = "gain"
    version: str = "manual"

    @classmethod
    def from_config(cls, config: "RecommendationPolicyConfig") -> "RecommendationPolicy":
        return cls(
            min_surface_confidence=config.thresholds.min_surface_confidence,
            min_harmonic_fill_ratio=config.thresholds.min_harmonic_fill_ratio,
            max_predicted_error_band=config.thresholds.max_predicted_error_band,
            min_input_limit_margin=config.thresholds.min_input_limit_margin,
            min_support_runs=config.thresholds.min_support_runs,
            allow_interpolated_auto=config.allow_interpolated_auto,
            margin_source=config.margin_source,
            version=config.version,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "allow_interpolated_auto": bool(self.allow_interpolated_auto),
            "margin_source": self.margin_source,
            "thresholds": {
                "min_surface_confidence": float(self.min_surface_confidence),
                "min_harmonic_fill_ratio": float(self.min_harmonic_fill_ratio),
                "max_predicted_error_band": float(self.max_predicted_error_band),
                "min_input_limit_margin": float(self.min_input_limit_margin),
                "min_support_runs": int(self.min_support_runs),
            },
        }


@dataclass(slots=True, frozen=True)
class RecommendationPolicyThresholds:
    """Threshold bundle externalized from hard-coded policy decisions."""

    min_surface_confidence: float = 0.85
    min_harmonic_fill_ratio: float = 0.6
    max_predicted_error_band: float = 0.15
    min_input_limit_margin: float = 0.10
    min_support_runs: int = 2


@dataclass(slots=True, frozen=True)
class RecommendationPolicyConfig:
    """Versioned policy configuration used to build rollout policy objects."""

    version: str = "v2"
    thresholds: RecommendationPolicyThresholds = field(default_factory=RecommendationPolicyThresholds)
    allow_interpolated_auto: bool = True
    margin_source: Literal["gain", "peak", "p95"] = "gain"


@dataclass(slots=True)
class PolicyDecision:
    """Final recommendation policy decision after engine/model evaluation."""

    allow_auto_recommendation: bool
    preview_only: bool
    reasons: list[str] = field(default_factory=list)
    policy_flags: set[str] = field(default_factory=set)


DEFAULT_RECOMMENDATION_POLICY_CONFIG = RecommendationPolicyConfig()
DEFAULT_RECOMMENDATION_POLICY = RecommendationPolicy.from_config(DEFAULT_RECOMMENDATION_POLICY_CONFIG)
