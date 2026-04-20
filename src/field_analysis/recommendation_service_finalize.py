from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .field_prediction_debug import sanitize_unavailable_exact_field_prediction
from .recommendation_auto_gate import evaluate_recommendation_policy
from .recommendation_legacy_bridge import (
    _extract_command_profile,
    _extract_lookup_table,
    _extract_series,
    _extract_support_table,
)
from .recommendation_models import RecommendationPolicy, RecommendationResult, TargetRequest
from .recommendation_shape_metrics import _compute_prediction_shape_metrics
from .validation import ValidationReport


def build_no_payload_result(
    *,
    target: TargetRequest,
    validation_report: ValidationReport,
    warnings: list[str],
    request_kind: str,
    validation_run_count: int,
    steady_state_engine: str,
    surface_debug: dict[str, Any],
    artifact_scope_lock: dict[str, Any] | None,
    policy_snapshot: dict[str, Any],
) -> RecommendationResult:
    return RecommendationResult(
        selected_regime=target.regime,
        preview_only=True,
        allow_auto_download=False,
        recommended_time_s=np.array([], dtype=float),
        recommended_input_v=np.array([], dtype=float),
        predicted_current_a=None,
        predicted_bx_mT=None,
        predicted_by_mT=None,
        predicted_bz_mT=None,
        validation_report=validation_report,
        warnings=warnings,
        debug_info={
            "request_kind": request_kind,
            "validation_run_count": validation_run_count,
            "steady_state_engine": steady_state_engine,
            "harmonic_surface_debug": surface_debug,
            "artifact_scope_lock": artifact_scope_lock,
            "policy_version": policy_snapshot["version"],
            "policy_snapshot": policy_snapshot,
            "policy_decision": "blocked",
            "policy_reasons": ["no_recommendation_payload"],
            "auto_gate_reasons": ["no_recommendation_payload"],
            "allow_auto_download": False,
        },
        engine_summary={
            "selected_engine": steady_state_engine,
            "request_kind": request_kind,
            "exact_frequency_match": validation_report.exact_freq_match,
            "support_state": str(
                (artifact_scope_lock or {}).get("support_state")
                or surface_debug.get("support_state")
                or ("exact" if validation_report.exact_freq_match else "unsupported")
            ),
            "preview_only": True,
            "allow_auto_download": False,
            "request_route": str((artifact_scope_lock or {}).get("request_route") or "unsupported"),
            "solver_route": str(surface_debug.get("solver_route") or "no_payload"),
            "artifact_scope_bucket": (artifact_scope_lock or {}).get("bucket"),
            "artifact_scope_status": (artifact_scope_lock or {}).get("status"),
            "artifact_scope_locked": artifact_scope_lock is not None,
        },
        support_summary=dict(surface_debug.get("surface_support", {}))
        if isinstance(surface_debug.get("surface_support"), dict)
        else {},
        confidence_summary=dict(surface_debug.get("confidence_summary", {}))
        if isinstance(surface_debug.get("confidence_summary"), dict)
        else {},
        legacy_payload=None,
    )


def finalize_recommendation_result(
    *,
    legacy_payload: dict[str, Any],
    target: TargetRequest,
    options: Any,
    validation_report: ValidationReport,
    warnings: list[str],
    request_kind: str,
    validation_runs: list[Any],
    steady_state_engine: str,
    surface_debug: dict[str, Any],
    artifact_scope_lock: dict[str, Any] | None,
    policy_snapshot: dict[str, Any],
    active_policy: RecommendationPolicy,
    provisional_recipe: dict[str, Any] | None,
    build_prediction_debug_payload: Any,
) -> RecommendationResult:
    command_profile = _extract_command_profile(legacy_payload)
    lookup_table = _extract_lookup_table(legacy_payload)
    support_table = _extract_support_table(legacy_payload)

    preview_only = not validation_report.allow_auto_recommendation
    input_limit_margin = float("nan")
    if not command_profile.empty and {"available_amp_gain_pct", "required_amp_gain_pct"}.issubset(command_profile.columns):
        input_limit_margin = float(
            pd.to_numeric(command_profile["available_amp_gain_pct"], errors="coerce").iloc[0]
            - pd.to_numeric(command_profile["required_amp_gain_pct"], errors="coerce").iloc[0]
        )
    support_state = str(surface_debug.get("support_state") or ("exact" if validation_report.exact_freq_match else "unsupported"))
    if provisional_recipe is not None:
        support_state = "provisional_preview"
    if artifact_scope_lock is not None:
        support_state = str(artifact_scope_lock.get("support_state") or support_state)

    engine_summary = {
        "selected_engine": steady_state_engine,
        "request_kind": request_kind,
        "exact_frequency_match": validation_report.exact_freq_match,
        "support_state": support_state,
        "preview_only": preview_only,
        "allow_auto_download": not preview_only,
        "fallback_reason": surface_debug.get("reason") if steady_state_engine != "harmonic_surface" else None,
    }
    support_summary = dict(surface_debug.get("surface_support", {})) if isinstance(surface_debug.get("surface_support"), dict) else {}
    confidence_summary = dict(surface_debug.get("confidence_summary", {})) if isinstance(surface_debug.get("confidence_summary"), dict) else {}
    confidence_summary.setdefault("predicted_error_band", float(validation_report.expected_error_band))
    confidence_summary.setdefault("input_limit_margin", input_limit_margin)
    confidence_summary.update(_compute_prediction_shape_metrics(command_profile))
    if provisional_recipe is not None:
        confidence_summary["provisional_scale_ratio"] = float(provisional_recipe["scale_ratio"])

    payload_route_fields: dict[str, Any] = {}
    if isinstance(legacy_payload, dict):
        for key in (
            "request_route",
            "plot_source",
            "selected_support_waveform",
            "selected_support_id",
            "selected_support_family",
            "support_selection_reason",
            "support_family_metric",
            "support_family_value",
            "estimated_family_level",
            "support_family_lock_applied",
            "support_bz_to_current_ratio",
            "active_window_start_s",
            "active_window_end_s",
            "active_duration_s",
            "zero_padded_fraction",
            "harmonic_weights_used",
            "field_prediction_source_hint",
            "expected_current_source_hint",
            "field_prediction_status",
            "field_prediction_unavailable_reason",
            "field_prediction_fallback_reason",
            "field_prediction_hierarchy",
            "exact_field_direct_available",
            "exact_field_direct_reason",
            "same_recipe_surrogate_candidate_available",
            "same_recipe_surrogate_applied",
            "same_recipe_surrogate_ratio",
            "surrogate_scope",
            "used_lcr_prior",
            "lcr_blend_weight",
            "requested_lcr_weight",
            "lcr_usage_mode",
            "exact_field_support_present",
            "lcr_phase_anchor_used",
            "lcr_gain_prior_used",
        ):
            if legacy_payload.get(key) is not None:
                payload_route_fields[key] = legacy_payload.get(key)

    derived_request_route = payload_route_fields.get("request_route")
    if support_state == "provisional_preview":
        derived_request_route = "provisional"
    elif support_state == "unsupported":
        derived_request_route = "unsupported"
    if artifact_scope_lock is not None:
        derived_request_route = str(artifact_scope_lock.get("request_route") or derived_request_route or "unsupported")

    if payload_route_fields:
        engine_summary.update(
            {
                "request_route": derived_request_route,
                "solver_route": legacy_payload.get("mode"),
                "plot_source": payload_route_fields.get("plot_source"),
                "selected_support_waveform": payload_route_fields.get("selected_support_waveform"),
                "selected_support_id": payload_route_fields.get("selected_support_id"),
                "selected_support_family": payload_route_fields.get("selected_support_family"),
                "support_selection_reason": payload_route_fields.get("support_selection_reason"),
                "support_family_metric": payload_route_fields.get("support_family_metric"),
                "support_family_value": payload_route_fields.get("support_family_value"),
                "estimated_family_level": payload_route_fields.get("estimated_family_level"),
                "support_family_lock_applied": payload_route_fields.get("support_family_lock_applied"),
                "support_bz_to_current_ratio": payload_route_fields.get("support_bz_to_current_ratio"),
                "field_prediction_source_hint": payload_route_fields.get("field_prediction_source_hint"),
                "expected_current_source_hint": payload_route_fields.get("expected_current_source_hint"),
                "field_prediction_status": payload_route_fields.get("field_prediction_status"),
                "field_prediction_unavailable_reason": payload_route_fields.get("field_prediction_unavailable_reason"),
                "field_prediction_fallback_reason": payload_route_fields.get("field_prediction_fallback_reason"),
                "field_prediction_hierarchy": payload_route_fields.get("field_prediction_hierarchy"),
                "exact_field_direct_available": payload_route_fields.get("exact_field_direct_available"),
                "exact_field_direct_reason": payload_route_fields.get("exact_field_direct_reason"),
                "same_recipe_surrogate_candidate_available": payload_route_fields.get("same_recipe_surrogate_candidate_available"),
                "same_recipe_surrogate_applied": payload_route_fields.get("same_recipe_surrogate_applied"),
                "same_recipe_surrogate_ratio": payload_route_fields.get("same_recipe_surrogate_ratio"),
                "surrogate_scope": payload_route_fields.get("surrogate_scope"),
                "used_lcr_prior": payload_route_fields.get("used_lcr_prior"),
                "lcr_blend_weight": payload_route_fields.get("lcr_blend_weight"),
                "requested_lcr_weight": payload_route_fields.get("requested_lcr_weight"),
                "lcr_usage_mode": payload_route_fields.get("lcr_usage_mode"),
                "exact_field_support_present": payload_route_fields.get("exact_field_support_present"),
                "lcr_phase_anchor_used": payload_route_fields.get("lcr_phase_anchor_used"),
                "lcr_gain_prior_used": payload_route_fields.get("lcr_gain_prior_used"),
            }
        )
        support_summary.update(
            {
                "active_window_start_s": payload_route_fields.get("active_window_start_s"),
                "active_window_end_s": payload_route_fields.get("active_window_end_s"),
                "active_duration_s": payload_route_fields.get("active_duration_s"),
                "zero_padded_fraction": payload_route_fields.get("zero_padded_fraction"),
                "selected_support_id": payload_route_fields.get("selected_support_id"),
                "selected_support_family": payload_route_fields.get("selected_support_family"),
                "support_selection_reason": payload_route_fields.get("support_selection_reason"),
                "support_family_metric": payload_route_fields.get("support_family_metric"),
                "support_family_value": payload_route_fields.get("support_family_value"),
                "estimated_family_level": payload_route_fields.get("estimated_family_level"),
                "support_family_lock_applied": payload_route_fields.get("support_family_lock_applied"),
            }
        )
        confidence_summary.setdefault("harmonic_weights_used", payload_route_fields.get("harmonic_weights_used"))
        for key in (
            "selected_support_id",
            "selected_support_family",
            "support_selection_reason",
            "support_family_metric",
            "support_family_value",
            "estimated_family_level",
            "support_family_lock_applied",
            "support_bz_to_current_ratio",
            "field_prediction_source_hint",
            "expected_current_source_hint",
            "field_prediction_status",
            "field_prediction_unavailable_reason",
            "field_prediction_fallback_reason",
            "field_prediction_hierarchy",
            "exact_field_direct_available",
            "exact_field_direct_reason",
            "same_recipe_surrogate_candidate_available",
            "same_recipe_surrogate_applied",
            "same_recipe_surrogate_ratio",
            "surrogate_scope",
            "used_lcr_prior",
            "lcr_blend_weight",
            "requested_lcr_weight",
            "lcr_usage_mode",
            "exact_field_support_present",
            "lcr_phase_anchor_used",
            "lcr_gain_prior_used",
        ):
            if payload_route_fields.get(key) is not None:
                confidence_summary.setdefault(key, payload_route_fields.get(key))
    else:
        engine_summary["solver_route"] = legacy_payload.get("mode")
        engine_summary["request_route"] = derived_request_route

    engine_summary["artifact_scope_bucket"] = (artifact_scope_lock or {}).get("bucket")
    engine_summary["artifact_scope_status"] = (artifact_scope_lock or {}).get("status")
    engine_summary["artifact_scope_locked"] = artifact_scope_lock is not None
    prediction_debug = build_prediction_debug_payload(
        command_profile=command_profile,
        target=target,
        engine_summary=engine_summary,
        confidence_summary=confidence_summary,
        request_kind=request_kind,
        legacy_payload=legacy_payload,
    )
    command_profile = sanitize_unavailable_exact_field_prediction(
        command_profile,
        prediction_debug,
        target_output_type=str(target.target_type),
    )
    confidence_summary.update(prediction_debug)
    engine_summary["field_prediction_source"] = prediction_debug.get("field_prediction_source")
    engine_summary["expected_current_source"] = prediction_debug.get("expected_current_source")
    engine_summary["field_prediction_available"] = bool(prediction_debug.get("field_prediction_available", False))
    engine_summary["zero_field_reason"] = prediction_debug.get("zero_field_reason")
    predicted_field = _extract_series(command_profile, ["modeled_field_mT", "expected_field_mT", "support_scaled_field_mT"])

    if artifact_scope_lock is not None and artifact_scope_lock.get("bucket") in {"provisional_preview", "missing_exact", "reference_only"}:
        preview_only = True
    result = RecommendationResult(
        selected_regime=target.regime,
        preview_only=preview_only,
        allow_auto_download=not preview_only,
        recommended_time_s=_extract_series(command_profile, ["time_s"]),
        recommended_input_v=_extract_series(command_profile, ["limited_voltage_v", "recommended_voltage_v"]),
        predicted_current_a=_extract_series(command_profile, ["modeled_current_a", "expected_current_a", "support_scaled_current_a"]),
        predicted_bx_mT=None,
        predicted_by_mT=None,
        predicted_bz_mT=predicted_field if options.field_channel == "bz_mT" else None,
        validation_report=validation_report,
        warnings=warnings,
        debug_info={
            "request_kind": request_kind,
            "validation_run_count": len(validation_runs),
            "support_reasons": list(validation_report.reasons),
            "steady_state_engine": steady_state_engine,
            "harmonic_surface_debug": surface_debug,
            "artifact_scope_lock": artifact_scope_lock,
            "policy_snapshot": policy_snapshot,
            "engine_summary": engine_summary,
            "support_summary": support_summary,
            "confidence_summary": confidence_summary,
            "provisional_recipe": provisional_recipe,
            "support_state": support_state,
            "allow_auto_download": not preview_only,
            "solver_route": engine_summary.get("solver_route"),
            **prediction_debug,
            **payload_route_fields,
        },
        engine_summary=engine_summary,
        support_summary=support_summary,
        confidence_summary=confidence_summary,
        command_profile=command_profile if not command_profile.empty else None,
        lookup_table=lookup_table if not lookup_table.empty else None,
        support_table=support_table if not support_table.empty else None,
        legacy_payload=legacy_payload,
    )
    if result.command_profile is not None and not result.command_profile.empty:
        result.command_profile["support_state"] = support_state
        result.command_profile["request_route"] = result.engine_summary.get("request_route")
        result.command_profile.attrs["engine_summary"] = dict(result.engine_summary)
        result.command_profile.attrs["confidence_summary"] = dict(result.confidence_summary)
        result.command_profile.attrs["prediction_debug"] = dict(prediction_debug)

    policy_decision = evaluate_recommendation_policy(result=result, target=target, policy=active_policy)
    result.preview_only = bool(policy_decision.preview_only)
    result.allow_auto_download = bool(policy_decision.allow_auto_recommendation)
    result.engine_summary["preview_only"] = bool(policy_decision.preview_only)
    result.engine_summary["allow_auto_download"] = bool(policy_decision.allow_auto_recommendation)
    result.engine_summary["auto_recommendation"] = bool(policy_decision.allow_auto_recommendation)
    result.engine_summary["policy_flags"] = sorted(policy_decision.policy_flags)
    result.engine_summary["policy_version"] = policy_snapshot["version"]
    result.engine_summary["policy_margin_source"] = policy_snapshot.get("margin_source")
    result.debug_info["policy_version"] = policy_snapshot["version"]
    result.debug_info["policy_snapshot"] = policy_snapshot
    result.debug_info["policy_decision"] = "auto" if policy_decision.allow_auto_recommendation else "preview_only"
    result.debug_info["policy_reasons"] = list(policy_decision.reasons)
    result.debug_info["auto_gate_reasons"] = list(policy_decision.reasons)
    result.debug_info["policy_flags"] = sorted(policy_decision.policy_flags)
    result.debug_info["allow_auto_download"] = bool(policy_decision.allow_auto_recommendation)
    result.debug_info["support_state"] = result.engine_summary.get("support_state")
    result.debug_info["request_route"] = result.engine_summary.get("request_route")
    result.debug_info["solver_route"] = result.engine_summary.get("solver_route")
    for key, value in prediction_debug.items():
        result.debug_info[key] = value
    for key in (
        "predicted_shape_corr",
        "predicted_nrmse",
        "predicted_phase_lag",
        "predicted_phase_lag_cycles",
        "predicted_clipping",
        "harmonic_weights_used",
        "zero_padded_fraction",
        "active_duration_s",
    ):
        if key in result.confidence_summary and key not in result.debug_info:
            result.debug_info[key] = result.confidence_summary.get(key)
    if provisional_recipe is not None:
        result.engine_summary["provisional_recipe"] = provisional_recipe
        result.support_summary["provisional_recipe"] = provisional_recipe
    if artifact_scope_lock is not None and artifact_scope_lock.get("bucket") in {"provisional_preview", "missing_exact", "reference_only"}:
        result.preview_only = True
        result.allow_auto_download = False
        result.engine_summary["preview_only"] = True
        result.engine_summary["allow_auto_download"] = False
        result.engine_summary["auto_recommendation"] = False
        result.engine_summary["policy_flags"] = sorted(set(result.engine_summary.get("policy_flags", [])) | {"artifact_scope_locked"})
        result.debug_info["policy_flags"] = sorted(set(result.debug_info.get("policy_flags", [])) | {"artifact_scope_locked"})
        result.debug_info["allow_auto_download"] = False
        if result.command_profile is not None and not result.command_profile.empty:
            result.command_profile["support_state"] = result.engine_summary.get("support_state")
            result.command_profile["request_route"] = result.engine_summary.get("request_route")
    if result.command_profile is not None and not result.command_profile.empty:
        result.command_profile.attrs["engine_summary"] = dict(result.engine_summary)
        result.command_profile.attrs["confidence_summary"] = dict(result.confidence_summary)
        result.command_profile.attrs["prediction_debug"] = {
            **prediction_debug,
            "request_route": result.engine_summary.get("request_route"),
            "solver_route": result.engine_summary.get("solver_route"),
            "plot_source": result.engine_summary.get("plot_source"),
            "allow_auto_download": result.allow_auto_download,
        }
    return result
