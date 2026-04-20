from __future__ import annotations

from typing import Any

import numpy as np

from .canonical_runs import CanonicalRun
from .field_prediction_debug import build_prediction_debug_from_recommendation
from .recommendation_auto_gate import evaluate_recommendation_policy
from .recommendation_exact_runtime import (
    _filter_finite_support_entries_for_scope_lock,
    _find_provisional_finite_recipe,
    build_support_report,
    _resolve_artifact_scope_lock,
)
from .recommendation_legacy_bridge import (
    build_finite_support_entries,
    recommend_voltage_waveform,
    synthesize_current_waveform_compensation,
    synthesize_finite_empirical_compensation,
)
from .recommendation_models import (
    DEFAULT_RECOMMENDATION_POLICY_CONFIG,
    LegacyRecommendationContext,
    RecommendationOptions,
    RecommendationPolicy,
    RecommendationPolicyConfig,
    RecommendationResult,
    TargetRequest,
)
from .recommendation_service_finalize import build_no_payload_result, finalize_recommendation_result
from .recommendation_surface_runtime import _recommend_continuous_harmonic_surface
from .validation import ValidationReport

_recommend_voltage_waveform = recommend_voltage_waveform
_synthesize_current_waveform_compensation = synthesize_current_waveform_compensation
_synthesize_finite_empirical_compensation = synthesize_finite_empirical_compensation


def _build_prediction_debug_payload(
    *,
    command_profile,
    target: TargetRequest,
    engine_summary: dict[str, Any],
    confidence_summary: dict[str, Any],
    request_kind: str,
    legacy_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    return build_prediction_debug_from_recommendation(
        command_profile=command_profile,
        target_output_type=str(target.target_type),
        engine_summary=engine_summary,
        confidence_summary=confidence_summary,
        request_kind=request_kind,
        legacy_payload=legacy_payload,
    )


def recommend(
    continuous_runs: list[CanonicalRun],
    transient_runs: list[CanonicalRun],
    validation_runs: list[CanonicalRun],
    target: TargetRequest,
    options: RecommendationOptions,
    legacy_context: LegacyRecommendationContext,
    policy: RecommendationPolicy | None = None,
    policy_config: RecommendationPolicyConfig | None = None,
) -> RecommendationResult:
    validation_report = build_support_report(continuous_runs, transient_runs, target)
    artifact_scope_lock = _resolve_artifact_scope_lock(target)
    request_kind = str(target.context.get("request_kind", "waveform_compensation"))
    warnings = list(validation_report.reasons)
    steady_state_engine = "legacy"
    surface_debug: dict[str, Any] = {}
    provisional_recipe: dict[str, Any] | None = None
    active_policy = policy or RecommendationPolicy.from_config(policy_config or DEFAULT_RECOMMENDATION_POLICY_CONFIG)
    policy_snapshot = active_policy.snapshot()

    legacy_payload: dict[str, Any] | None
    if request_kind == "size_lut":
        legacy_payload = _recommend_voltage_waveform(
            per_test_summary=legacy_context.per_test_summary,
            analyses_by_test_id=legacy_context.analysis_lookup,
            waveform_type=target.target_waveform,
            freq_hz=float(target.freq_hz) if target.freq_hz is not None else np.nan,
            target_metric=str(target.context["target_metric"]),
            target_value=float(target.context["target_value"]),
            frequency_mode=str(target.context.get("frequency_mode", options.frequency_mode)),
            finite_cycle_mode=bool(target.context.get("finite_cycle_mode", False)),
            target_cycle_count=target.commanded_cycles,
            preview_tail_cycles=float(target.context.get("preview_tail_cycles", options.preview_tail_cycles)),
            max_daq_voltage_pp=options.max_daq_voltage_pp,
            amp_gain_at_100_pct=options.amp_gain_at_100_pct,
            amp_gain_limit_pct=options.amp_gain_limit_pct,
            amp_max_output_pk_v=options.amp_max_output_pk_v,
            default_support_amp_gain_pct=options.default_support_amp_gain_pct,
            allow_target_extrapolation=options.allow_target_extrapolation,
        )
    elif target.regime == "transient":
        support_entries = build_finite_support_entries(
            transient_measurements=legacy_context.transient_measurements,
            transient_preprocess_results=legacy_context.transient_preprocess_results,
            transient_canonical_runs=legacy_context.transient_canonical_runs,
            current_channel=options.current_channel,
            field_channel=options.field_channel,
        )
        support_entries = _filter_finite_support_entries_for_scope_lock(
            support_entries,
            target=target,
            artifact_scope_lock=artifact_scope_lock,
        )
        if target.target_type == "current" and target.target_level_value is not None and np.isfinite(target.target_level_value):
            level_tolerance = max(abs(float(target.target_level_value)) * 0.15, 1.0)
            exact_recipe_match = any(
                (entry.get("waveform_type") == target.command_waveform or entry.get("waveform_type") == target.target_waveform)
                and np.isfinite(entry.get("freq_hz", np.nan))
                and np.isclose(float(entry.get("freq_hz", np.nan)), float(target.freq_hz), atol=1e-9)
                and target.commanded_cycles is not None
                and np.isfinite(entry.get("requested_cycle_count", np.nan))
                and np.isclose(float(entry.get("requested_cycle_count", np.nan)), float(target.commanded_cycles), atol=1e-9)
                and np.isfinite(entry.get("requested_current_pp", entry.get("current_pp", np.nan)))
                and abs(float(entry.get("requested_current_pp", entry.get("current_pp", np.nan))) - float(target.target_level_value)) <= level_tolerance
                for entry in support_entries
            )
            if artifact_scope_lock is not None and artifact_scope_lock.get("bucket") in {"provisional_preview", "missing_exact", "reference_only"}:
                exact_recipe_match = False
            if not exact_recipe_match:
                provisional_recipe = _find_provisional_finite_recipe(transient_runs, target)
                if artifact_scope_lock is not None and artifact_scope_lock.get("bucket") == "missing_exact":
                    provisional_recipe = None
                validation_report = ValidationReport(
                    in_support=False,
                    exact_freq_match=validation_report.exact_freq_match,
                    exact_cycle_match=validation_report.exact_cycle_match,
                    shape_quality=validation_report.shape_quality,
                    expected_error_band=max(float(validation_report.expected_error_band), 0.25),
                    allow_auto_recommendation=False,
                    reasons=list(
                        dict.fromkeys(
                            [
                                *list(validation_report.reasons),
                                "exact level support ??곸벉",
                                *(
                                    [
                                        (
                                            "provisional preview only: "
                                            f"{provisional_recipe['freq_hz']:g} Hz / {provisional_recipe['cycles']:g} cycle / "
                                            f"{provisional_recipe['source_level_pp']:g} pp exact recipe??"
                                            f"{provisional_recipe['target_level_pp']:g} pp嚥??????노립 ?袁⑸뻻 ??筌?鈺곌퀬鍮"
                                        )
                                    ]
                                    if provisional_recipe is not None
                                    else []
                                ),
                            ]
                        )
                    ),
                )
        legacy_payload = _synthesize_finite_empirical_compensation(
            finite_support_entries=support_entries,
            waveform_type=target.command_waveform or target.target_waveform,
            freq_hz=float(target.freq_hz) if target.freq_hz is not None else np.nan,
            target_cycle_count=target.commanded_cycles,
            target_output_type=target.target_type,
            target_output_pp=float(target.target_level_value) if target.target_level_value is not None else np.nan,
            current_channel=options.current_channel,
            field_channel=options.field_channel,
            max_daq_voltage_pp=options.max_daq_voltage_pp,
            amp_gain_at_100_pct=options.amp_gain_at_100_pct,
            amp_gain_limit_pct=options.amp_gain_limit_pct,
            amp_max_output_pk_v=options.amp_max_output_pk_v,
            default_support_amp_gain_pct=options.default_support_amp_gain_pct,
            preview_tail_cycles=options.preview_tail_cycles,
        )
    else:
        harmonic_surface_payload, surface_debug = _recommend_continuous_harmonic_surface(
            continuous_runs=continuous_runs,
            target=target,
            options=options,
            legacy_context=legacy_context,
            validation_report=validation_report,
        )
        if harmonic_surface_payload is not None:
            legacy_payload = harmonic_surface_payload
            steady_state_engine = "harmonic_surface"
        else:
            legacy_payload = _synthesize_current_waveform_compensation(
                per_test_summary=legacy_context.per_test_summary,
                analyses_by_test_id=legacy_context.analysis_lookup,
                waveform_type=target.command_waveform or target.target_waveform,
                freq_hz=float(target.freq_hz) if target.freq_hz is not None else np.nan,
                target_current_pp_a=float(target.target_level_value) if target.target_level_value is not None else np.nan,
                current_channel=options.current_channel,
                field_channel=options.field_channel,
                target_output_type=target.target_type,
                target_output_pp=float(target.target_level_value) if target.target_level_value is not None else np.nan,
                frequency_mode=str(target.context.get("frequency_mode", options.frequency_mode)),
                finite_cycle_mode=bool(target.context.get("finite_cycle_mode", False)),
                target_cycle_count=target.commanded_cycles,
                preview_tail_cycles=float(target.context.get("preview_tail_cycles", options.preview_tail_cycles)),
                max_daq_voltage_pp=options.max_daq_voltage_pp,
                amp_gain_at_100_pct=options.amp_gain_at_100_pct,
                amp_gain_limit_pct=options.amp_gain_limit_pct,
                amp_max_output_pk_v=options.amp_max_output_pk_v,
                default_support_amp_gain_pct=options.default_support_amp_gain_pct,
                allow_output_extrapolation=options.allow_output_extrapolation,
                lcr_measurements=options.lcr_measurements,
                lcr_blend_weight=options.lcr_blend_weight,
                apply_startup_correction=options.apply_startup_correction,
                startup_transition_cycles=options.startup_transition_cycles,
                startup_correction_strength=options.startup_correction_strength,
                startup_preview_cycle_count=options.startup_preview_cycle_count,
            )

    if legacy_payload is None:
        warnings.append("legacy recommendation 野껉퀗?든몴?筌띾슢諭억쭪? 筌륁궢六??щ빍??")
        return build_no_payload_result(
            target=target,
            validation_report=validation_report,
            warnings=warnings,
            request_kind=request_kind,
            validation_run_count=len(validation_runs),
            steady_state_engine=steady_state_engine,
            surface_debug=surface_debug,
            artifact_scope_lock=artifact_scope_lock,
            policy_snapshot=policy_snapshot,
        )

    return finalize_recommendation_result(
        legacy_payload=legacy_payload,
        target=target,
        options=options,
        validation_report=validation_report,
        warnings=warnings,
        request_kind=request_kind,
        validation_runs=validation_runs,
        steady_state_engine=steady_state_engine,
        surface_debug=surface_debug,
        artifact_scope_lock=artifact_scope_lock,
        policy_snapshot=policy_snapshot,
        active_policy=active_policy,
        provisional_recipe=provisional_recipe,
        build_prediction_debug_payload=_build_prediction_debug_payload,
    )
