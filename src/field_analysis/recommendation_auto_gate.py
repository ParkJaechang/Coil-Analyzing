from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from .recommendation_constants import OFFICIAL_OPERATION_MAX_FREQ_HZ
from .recommendation_models import PolicyDecision, RecommendationPolicy, RecommendationResult, TargetRequest


def _resolve_policy_input_limit_margin(
    confidence_summary: dict[str, Any],
    margin_source: Literal["gain", "peak", "p95"],
) -> float:
    if margin_source == "peak":
        value = confidence_summary.get("peak_input_limit_margin")
    elif margin_source == "p95":
        value = confidence_summary.get("p95_input_limit_margin")
    else:
        value = confidence_summary.get("gain_input_limit_margin", confidence_summary.get("input_limit_margin"))
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])


def _prediction_shape_gate(
    confidence_summary: dict[str, Any],
    *,
    freq_hz: float | None,
    mode: Literal["strict", "exact_release"] = "strict",
) -> tuple[list[str], set[str]]:
    reasons: list[str] = []
    flags: set[str] = set()

    if mode == "exact_release":
        min_shape_corr = 0.85
        max_nrmse = 0.35
    else:
        min_shape_corr = 0.92
        max_nrmse = 0.20

    predicted_shape_corr = float(pd.to_numeric(pd.Series([confidence_summary.get("predicted_shape_corr")]), errors="coerce").iloc[0])
    predicted_nrmse = float(pd.to_numeric(pd.Series([confidence_summary.get("predicted_nrmse")]), errors="coerce").iloc[0])
    predicted_phase_lag_cycles = float(
        pd.to_numeric(pd.Series([confidence_summary.get("predicted_phase_lag_cycles")]), errors="coerce").iloc[0]
    )
    predicted_phase_lag_seconds = float(
        pd.to_numeric(pd.Series([confidence_summary.get("predicted_phase_lag")]), errors="coerce").iloc[0]
    )
    predicted_clipping = bool(confidence_summary.get("predicted_clipping", False))

    if np.isfinite(predicted_shape_corr) and predicted_shape_corr < min_shape_corr:
        reasons.append("predicted_shape_corr_below_threshold")
        flags.add("predicted_shape_low")

    if np.isfinite(predicted_nrmse) and predicted_nrmse > max_nrmse:
        reasons.append("predicted_nrmse_above_threshold")
        flags.add("predicted_nrmse_high")

    if np.isfinite(predicted_phase_lag_cycles):
        if abs(predicted_phase_lag_cycles) > 0.10:
            reasons.append("predicted_phase_lag_above_threshold")
            flags.add("predicted_phase_lag_high")
    elif np.isfinite(predicted_phase_lag_seconds) and np.isfinite(freq_hz) and float(freq_hz) > 0:
        if abs(predicted_phase_lag_seconds * float(freq_hz)) > 0.10:
            reasons.append("predicted_phase_lag_above_threshold")
            flags.add("predicted_phase_lag_high")

    if predicted_clipping:
        reasons.append("predicted_clipping_detected")
        flags.add("predicted_clipping")

    return reasons, flags


def evaluate_recommendation_policy(
    result: RecommendationResult,
    target: TargetRequest,
    policy: RecommendationPolicy,
) -> PolicyDecision:
    """Apply rollout policy after engine/model evaluation."""

    exact_frequency_match = bool(result.engine_summary.get("exact_frequency_match"))
    selected_engine = str(result.engine_summary.get("selected_engine", "legacy"))
    confidence = result.confidence_summary or {}
    reasons: list[str] = []
    flags: set[str] = set()
    requested_freq_hz = float(target.freq_hz) if target.freq_hz is not None and np.isfinite(target.freq_hz) else np.nan

    if exact_frequency_match:
        shape_reasons, shape_flags = _prediction_shape_gate(
            confidence,
            freq_hz=requested_freq_hz,
            mode="exact_release",
        )
        if np.isfinite(requested_freq_hz) and requested_freq_hz > float(OFFICIAL_OPERATION_MAX_FREQ_HZ):
            flags.add("official_support_band_exceeded")
            reasons.append(f"official_support_band_exceeded(>{OFFICIAL_OPERATION_MAX_FREQ_HZ:g}Hz)")
            return PolicyDecision(
                allow_auto_recommendation=False,
                preview_only=True,
                reasons=reasons,
                policy_flags=flags,
            )
        allow = bool(result.validation_report.allow_auto_recommendation) if result.validation_report is not None else False
        if not allow:
            reasons.extend(list(result.validation_report.reasons) if result.validation_report is not None else ["validation_blocked"])
            flags.add("exact_support_validation_block")
        field_prediction_source = str(confidence.get("field_prediction_source") or "")
        zero_field_reason = str(confidence.get("zero_field_reason") or "")
        field_prediction_available = bool(confidence.get("field_prediction_available", True))
        field_prediction_status = str(confidence.get("field_prediction_status") or "")
        request_kind = str(result.engine_summary.get("request_kind") or "")
        waveform_prediction_gate = request_kind == "waveform_compensation"
        if waveform_prediction_gate and field_prediction_source in {"support_blended_preview", "zero_fill_fallback"}:
            reasons.append(
                "exact_route_support_blended_preview_bug"
                if field_prediction_source == "support_blended_preview"
                else "zero_fill_fallback"
            )
            flags.add("exact_route_support_preview_bug")
            allow = False
        if waveform_prediction_gate and target.target_type == "field" and (
            not field_prediction_available
            or field_prediction_status == "unavailable"
            or field_prediction_source in {"target_leak_suspect"}
        ):
            reasons.append(
                zero_field_reason
                or str(confidence.get("field_prediction_unavailable_reason") or "")
                or field_prediction_source
                or "field_prediction_unavailable"
            )
            flags.add("exact_field_prediction_block")
            allow = False
        flags.update(shape_flags)
        if shape_flags:
            flags.add("exact_support_shape_quality_advisory")
        if not reasons and allow:
            flags.add("exact_support_auto")
        return PolicyDecision(
            allow_auto_recommendation=bool(allow),
            preview_only=not bool(allow),
            reasons=reasons,
            policy_flags=flags,
        )

    shape_reasons, shape_flags = _prediction_shape_gate(confidence, freq_hz=requested_freq_hz)

    if selected_engine != "harmonic_surface":
        flags.add("legacy_engine_preview_only")
        return PolicyDecision(
            allow_auto_recommendation=False,
            preview_only=True,
            reasons=[str(result.engine_summary.get("fallback_reason") or "legacy_engine_used")],
            policy_flags=flags,
        )

    if not bool(policy.allow_interpolated_auto):
        reasons.append("interpolated_auto_disabled")
        flags.add("interpolated_auto_disabled")

    if target.target_type != "current":
        reasons.append("interpolated_auto_current_only")
        flags.add("target_type_not_allowed")

    support_run_count = int(confidence.get("support_run_count", 0))
    if support_run_count < int(policy.min_support_runs):
        reasons.append("support_runs_below_threshold")
        flags.add("insufficient_support_runs")

    harmonic_fill_ratio = float(confidence.get("harmonic_fill_ratio", 0.0) or 0.0)
    if harmonic_fill_ratio < float(policy.min_harmonic_fill_ratio):
        reasons.append("harmonic_fill_ratio_below_threshold")
        flags.add("insufficient_harmonics")

    predicted_error_band = float(confidence.get("predicted_error_band", np.inf) or np.inf)
    if not np.isfinite(predicted_error_band) or predicted_error_band > float(policy.max_predicted_error_band):
        reasons.append("predicted_error_band_above_threshold")
        flags.add("predicted_error_high")

    input_limit_margin = _resolve_policy_input_limit_margin(confidence, policy.margin_source)
    if not np.isfinite(input_limit_margin) or input_limit_margin < float(policy.min_input_limit_margin):
        reasons.append("input_limit_margin_below_threshold")
        flags.add("low_input_headroom")

    surface_confidence = float(confidence.get("surface_confidence", 0.0) or 0.0)
    if surface_confidence < float(policy.min_surface_confidence):
        reasons.append("surface_confidence_below_threshold")
        flags.add("surface_confidence_low")

    fallback_reason = result.engine_summary.get("fallback_reason")
    if fallback_reason:
        reasons.append(str(fallback_reason))
        flags.add("fallback_reason_present")

    reasons.extend(shape_reasons)
    flags.update(shape_flags)

    allow = not reasons
    if allow:
        flags.add("interpolated_auto_allowed")
    return PolicyDecision(
        allow_auto_recommendation=allow,
        preview_only=not allow,
        reasons=reasons,
        policy_flags=flags,
    )
