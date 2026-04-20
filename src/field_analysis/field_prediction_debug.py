from __future__ import annotations

import ast
from typing import Any

import numpy as np
import pandas as pd

from .shape_amplitude_debug import infer_shape_amplitude_debug_fields


DEFAULT_FIELD_PREDICTION_HIERARCHY = [
    "exact_field_direct",
    "current_to_bz_surrogate",
    "unavailable",
]


def coerce_debug_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return {}
        if isinstance(parsed, dict):
            return dict(parsed)
    return {}


def _extract_signal(frame: pd.DataFrame | None, columns: tuple[str, ...]) -> np.ndarray:
    if frame is None or frame.empty:
        return np.array([], dtype=float)
    for column in columns:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return np.array([], dtype=float)


def _frame_value(frame: pd.DataFrame | None, column: str) -> Any:
    if frame is None or frame.empty or column not in frame.columns:
        return None
    return frame[column].iloc[0]


def _attr_mapping(frame: pd.DataFrame | None, key: str) -> dict[str, Any]:
    if frame is None:
        return {}
    value = frame.attrs.get(key)
    return dict(value) if isinstance(value, dict) else {}


def signal_peak_to_peak(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.nanmax(finite) - np.nanmin(finite))


def nonzero_fraction(values: np.ndarray, *, atol: float = 1e-9) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(np.abs(finite) > float(atol)))


def signal_correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    valid = np.isfinite(reference) & np.isfinite(candidate)
    if valid.sum() < 8:
        return float("nan")
    ref = reference[valid]
    comp = candidate[valid]
    if np.nanstd(ref) <= 1e-12 or np.nanstd(comp) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(ref, comp)[0, 1])


def prediction_objective_audit(
    *,
    target_output_type: str,
    harmonic_weights_used: dict[str, Any],
) -> dict[str, Any]:
    if str(target_output_type) == "field":
        return {
            "loss_target_type": "field",
            "w_bz_nrmse": 1.0,
            "w_bz_shape": 1.0,
            "w_bz_phase": 1.0,
            "w_bz_pp": 1.0,
            "w_current_limit": 0.15,
            "w_voltage_limit": 0.15,
            "harmonic_weights_used": harmonic_weights_used,
            "objective_weight_source": "route_inferred",
        }
    return {
        "loss_target_type": "current",
        "w_bz_nrmse": 0.0,
        "w_bz_shape": 0.0,
        "w_bz_phase": 0.0,
        "w_bz_pp": 0.0,
        "w_current_limit": 1.0,
        "w_voltage_limit": 0.25,
        "harmonic_weights_used": harmonic_weights_used,
        "objective_weight_source": "route_inferred",
    }


def _detect_target_leak_suspect(
    *,
    command_profile: pd.DataFrame,
    target_output_type: str,
    request_route: str,
    solver_route: str,
) -> tuple[bool, str | None]:
    if str(request_route) != "exact" or str(solver_route) != "finite_exact_direct" or str(target_output_type) != "current":
        return False, None
    target_output = _extract_signal(command_profile, ("target_output", "used_target_output"))
    expected_output = _extract_signal(command_profile, ("expected_output", "expected_current_a", "modeled_output"))
    if target_output.size == 0 or expected_output.size == 0:
        return False, None
    leak_corr = signal_correlation(target_output, expected_output)
    target_pp = signal_peak_to_peak(target_output)
    expected_pp = signal_peak_to_peak(expected_output)
    pp_ratio = (
        float(expected_pp / target_pp)
        if np.isfinite(expected_pp) and np.isfinite(target_pp) and float(target_pp) > 1e-9
        else float("nan")
    )
    if np.isfinite(leak_corr) and leak_corr >= 0.999 and np.isfinite(pp_ratio) and 0.95 <= pp_ratio <= 1.05:
        return True, "expected_output_matches_target_template"
    return False, None


def _resolve_field_prediction_source(
    *,
    source_hint: str,
    request_route: str,
    solver_route: str,
    plot_source: str,
    target_output_type: str,
    command_profile: pd.DataFrame,
    target_leak_suspect: bool,
) -> str:
    source_map = {
        "exact_field_direct": "exact_field_direct",
        "current_to_bz_surrogate": "current_to_bz_surrogate",
        "support_scaled_field_mT": "support_blended_preview",
        "support_blended_preview": "support_blended_preview",
        "validation_transfer": "validation_transfer",
        "validation_voltage_to_bz_transfer": "validation_transfer",
        "reference_support_transfer_fallback": "reference_support_transfer_fallback",
        "reference_voltage_to_bz_transfer": "reference_support_transfer_fallback",
        "target_leak_suspect": "target_leak_suspect",
        "zero_fill_fallback": "zero_fill_fallback",
    }
    if source_hint in source_map:
        return source_map[source_hint]
    if target_leak_suspect:
        return "target_leak_suspect"
    if str(request_route) == "exact" and str(plot_source) == "support_blended_preview":
        return "support_blended_preview"
    if str(plot_source) == "support_blended_preview" or "support_scaled_field_mT" in command_profile.columns:
        return "support_blended_preview"
    if str(target_output_type) == "current" and str(solver_route) != "finite_exact_direct":
        return "current_to_bz_surrogate"
    if str(request_route) == "exact":
        return "exact_field_direct"
    return "reference_support_transfer_fallback"


def _resolve_expected_current_source(
    *,
    explicit_hint: str,
    request_route: str,
    solver_route: str,
    plot_source: str,
    target_output_type: str,
    expected_current: np.ndarray,
    target_leak_suspect: bool,
) -> str:
    if explicit_hint:
        return explicit_hint
    if target_leak_suspect and str(target_output_type) == "current":
        return "target_template"
    if str(solver_route) == "validation_residual_second_stage" and expected_current.size:
        return "validation_transfer"
    if str(plot_source) == "support_blended_preview":
        return "support_blended_preview"
    if str(request_route) == "exact" and expected_current.size:
        return "exact_current_direct"
    if expected_current.size and str(target_output_type) == "field":
        return "field_penalty_surrogate"
    return "missing"


def build_prediction_debug(
    *,
    command_profile: pd.DataFrame,
    validation_frame: pd.DataFrame | None,
    target_output_type: str,
    current_channel: str,
    field_channel: str,
    request_route: str,
    solver_route: str,
    plot_source: str,
    source_hint: str,
    status_hint: str,
    unavailable_reason_hint: str,
    fallback_reason_hint: str,
    hierarchy_hint: Any,
    harmonic_weights_hint: Any,
    explicit_expected_current_source: str = "",
    exact_field_direct_available: Any = None,
    exact_field_direct_reason: Any = None,
    same_recipe_surrogate_candidate_available: Any = None,
    same_recipe_surrogate_applied: Any = None,
    same_recipe_surrogate_ratio: Any = None,
    surrogate_scope: Any = None,
    selected_support_id: Any = None,
    selected_support_family: Any = None,
    support_selection_reason: Any = None,
    support_family_metric: Any = None,
    support_family_value: Any = None,
    support_family_lock_applied: Any = None,
    support_bz_to_current_ratio: Any = None,
) -> dict[str, Any]:
    harmonic_weights_used = coerce_debug_mapping(harmonic_weights_hint)
    expected_field = _extract_signal(command_profile, ("expected_field_mT", "modeled_field_mT", "support_scaled_field_mT"))
    direct_field = _extract_signal(command_profile, ("exact_field_direct_mT",))
    expected_current = _extract_signal(command_profile, ("expected_current_a", "modeled_current_a", "support_scaled_current_a"))
    target_output = _extract_signal(command_profile, ("target_output", "aligned_target_output", "used_target_output", "aligned_used_target_output"))
    target_field = _extract_signal(command_profile, ("target_field_mT", "aligned_target_field_mT", "used_target_field_mT", "aligned_used_target_field_mT"))
    validation_bz = _extract_signal(validation_frame, (field_channel, "bz_effective_mT", "bz_raw_mT"))

    bz_channel_present = bool(validation_bz.size and np.isfinite(validation_bz).any())
    if not bz_channel_present:
        bz_channel_present = bool(expected_field.size and np.isfinite(expected_field).any())
    bz_nonzero_fraction = nonzero_fraction(validation_bz if validation_bz.size else expected_field)

    field_pp = signal_peak_to_peak(expected_field)
    target_field_reference = target_field if target_field.size else target_output
    target_field_pp = signal_peak_to_peak(target_field_reference)
    amplitude_ratio = (
        float(field_pp / target_field_pp)
        if np.isfinite(field_pp) and np.isfinite(target_field_pp) and float(target_field_pp) > 1e-9
        else float("nan")
    )
    zero_field_reason: str | None = None
    field_prediction_required = str(target_output_type) == "field" or str(solver_route) == "finite_exact_direct"
    if str(request_route) == "exact" and field_prediction_required:
        if not bz_channel_present:
            zero_field_reason = "missing_bz_channel"
        elif not np.isfinite(field_pp) or field_pp <= 1e-6:
            zero_field_reason = "expected_field_near_zero"
        elif np.isfinite(amplitude_ratio) and amplitude_ratio < 0.02:
            zero_field_reason = "expected_field_collapse"

    target_leak_suspect, target_leak_reason = _detect_target_leak_suspect(
        command_profile=command_profile,
        target_output_type=target_output_type,
        request_route=request_route,
        solver_route=solver_route,
    )
    field_prediction_source = _resolve_field_prediction_source(
        source_hint=str(source_hint or "").strip(),
        request_route=request_route,
        solver_route=solver_route,
        plot_source=plot_source,
        target_output_type=target_output_type,
        command_profile=command_profile,
        target_leak_suspect=target_leak_suspect,
    )
    expected_current_source = _resolve_expected_current_source(
        explicit_hint=str(explicit_expected_current_source or "").strip(),
        request_route=request_route,
        solver_route=solver_route,
        plot_source=plot_source,
        target_output_type=target_output_type,
        expected_current=expected_current,
        target_leak_suspect=target_leak_suspect,
    )

    force_unavailable_reason: str | None = None
    if str(request_route) == "exact" and str(target_output_type) == "field":
        if field_prediction_source == "support_blended_preview":
            force_unavailable_reason = "exact_route_support_blended_preview_bug"
        elif field_prediction_source == "target_leak_suspect":
            force_unavailable_reason = "target_leak_suspect"
        elif field_prediction_source == "zero_fill_fallback":
            force_unavailable_reason = "zero_fill_fallback"
        elif zero_field_reason is not None:
            force_unavailable_reason = zero_field_reason

    field_prediction_status = str(status_hint or "").strip() or ("unavailable" if force_unavailable_reason else "available")
    if force_unavailable_reason:
        field_prediction_status = "unavailable"
    field_prediction_unavailable_reason = str(unavailable_reason_hint or "").strip() or force_unavailable_reason
    if not field_prediction_unavailable_reason:
        field_prediction_unavailable_reason = None

    field_prediction_hierarchy = hierarchy_hint if isinstance(hierarchy_hint, list) else list(DEFAULT_FIELD_PREDICTION_HIERARCHY)
    objective_audit = prediction_objective_audit(
        target_output_type=target_output_type,
        harmonic_weights_used=harmonic_weights_used,
    )
    return {
        "field_prediction_source": field_prediction_source,
        "expected_current_source": expected_current_source,
        "request_route": request_route or None,
        "solver_route": solver_route or None,
        "plot_source": plot_source or None,
        "bz_channel_present": bz_channel_present,
        "bz_nonzero_fraction": bz_nonzero_fraction,
        "zero_field_reason": zero_field_reason,
        "field_prediction_status": field_prediction_status,
        "field_prediction_available": field_prediction_status != "unavailable",
        "field_prediction_unavailable_reason": field_prediction_unavailable_reason,
        "field_prediction_fallback_reason": str(fallback_reason_hint or "").strip() or None,
        "field_prediction_hierarchy": field_prediction_hierarchy,
        "exact_field_direct_available": exact_field_direct_available,
        "exact_field_direct_reason": exact_field_direct_reason,
        "same_recipe_surrogate_candidate_available": same_recipe_surrogate_candidate_available,
        "same_recipe_surrogate_applied": same_recipe_surrogate_applied,
        "same_recipe_surrogate_ratio": same_recipe_surrogate_ratio,
        "surrogate_scope": surrogate_scope,
        "target_leak_suspect": target_leak_suspect,
        "target_leak_reason": target_leak_reason,
        "exact_field_direct_pp": signal_peak_to_peak(direct_field),
        "predicted_field_pp": field_pp,
        "target_field_pp": target_field_pp,
        "predicted_field_to_target_pp_ratio": amplitude_ratio,
        "selected_support_id": selected_support_id,
        "selected_support_family": selected_support_family,
        "support_selection_reason": support_selection_reason,
        "support_family_metric": support_family_metric,
        "support_family_value": support_family_value,
        "support_family_lock_applied": support_family_lock_applied,
        "support_bz_to_current_ratio": support_bz_to_current_ratio,
        **objective_audit,
    }


def build_prediction_debug_from_recommendation(
    *,
    command_profile: pd.DataFrame,
    target_output_type: str,
    engine_summary: dict[str, Any],
    confidence_summary: dict[str, Any],
    request_kind: str = "waveform_compensation",
    legacy_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = build_prediction_debug(
        command_profile=command_profile,
        validation_frame=None,
        target_output_type=target_output_type,
        current_channel="expected_current_a",
        field_channel="expected_field_mT",
        request_route=str(engine_summary.get("request_route") or ""),
        solver_route=str(engine_summary.get("solver_route") or ""),
        plot_source=str(engine_summary.get("plot_source") or ""),
        source_hint=str(engine_summary.get("field_prediction_source_hint") or confidence_summary.get("field_prediction_source_hint") or ""),
        status_hint=str(engine_summary.get("field_prediction_status") or confidence_summary.get("field_prediction_status") or ""),
        unavailable_reason_hint=str(engine_summary.get("field_prediction_unavailable_reason") or confidence_summary.get("field_prediction_unavailable_reason") or ""),
        fallback_reason_hint=str(engine_summary.get("field_prediction_fallback_reason") or confidence_summary.get("field_prediction_fallback_reason") or ""),
        hierarchy_hint=engine_summary.get("field_prediction_hierarchy") or confidence_summary.get("field_prediction_hierarchy"),
        harmonic_weights_hint=confidence_summary.get("harmonic_weights_used"),
        explicit_expected_current_source=str(engine_summary.get("expected_current_source_hint") or confidence_summary.get("expected_current_source_hint") or ""),
        exact_field_direct_available=engine_summary.get("exact_field_direct_available", confidence_summary.get("exact_field_direct_available")),
        exact_field_direct_reason=engine_summary.get("exact_field_direct_reason", confidence_summary.get("exact_field_direct_reason")),
        same_recipe_surrogate_candidate_available=engine_summary.get("same_recipe_surrogate_candidate_available", confidence_summary.get("same_recipe_surrogate_candidate_available")),
        same_recipe_surrogate_applied=engine_summary.get("same_recipe_surrogate_applied", confidence_summary.get("same_recipe_surrogate_applied")),
        same_recipe_surrogate_ratio=engine_summary.get("same_recipe_surrogate_ratio", confidence_summary.get("same_recipe_surrogate_ratio")),
        surrogate_scope=engine_summary.get("surrogate_scope", confidence_summary.get("surrogate_scope")),
        selected_support_id=engine_summary.get("selected_support_id", confidence_summary.get("selected_support_id")),
        selected_support_family=engine_summary.get("selected_support_family", confidence_summary.get("selected_support_family")),
        support_selection_reason=engine_summary.get("support_selection_reason", confidence_summary.get("support_selection_reason")),
        support_family_metric=engine_summary.get("support_family_metric", confidence_summary.get("support_family_metric")),
        support_family_value=engine_summary.get("support_family_value", confidence_summary.get("support_family_value")),
        support_family_lock_applied=engine_summary.get("support_family_lock_applied", confidence_summary.get("support_family_lock_applied")),
        support_bz_to_current_ratio=engine_summary.get("support_bz_to_current_ratio", confidence_summary.get("support_bz_to_current_ratio")),
    )
    payload.update(
        infer_shape_amplitude_debug_fields(
            request_kind=str(request_kind),
            target_type=str(target_output_type),
            engine_summary=engine_summary,
            confidence_summary=confidence_summary,
            legacy_payload=legacy_payload,
        )
    )
    return payload


def build_prediction_debug_from_profile(
    *,
    command_profile: pd.DataFrame,
    validation_frame: pd.DataFrame | None,
    target_output_type: str,
    current_channel: str,
    field_channel: str,
) -> dict[str, Any]:
    prediction_debug = _attr_mapping(command_profile, "prediction_debug")
    engine_summary = _attr_mapping(command_profile, "engine_summary")
    confidence_summary = _attr_mapping(command_profile, "confidence_summary")
    bz_projection = _attr_mapping(command_profile, "bz_projection")
    return build_prediction_debug(
        command_profile=command_profile,
        validation_frame=validation_frame,
        target_output_type=target_output_type,
        current_channel=current_channel,
        field_channel=field_channel,
        request_route=str(prediction_debug.get("request_route") or _frame_value(command_profile, "request_route") or engine_summary.get("request_route") or ""),
        solver_route=str(prediction_debug.get("solver_route") or engine_summary.get("solver_route") or ""),
        plot_source=str(prediction_debug.get("plot_source") or _frame_value(command_profile, "plot_source") or engine_summary.get("plot_source") or ""),
        source_hint=str(prediction_debug.get("field_prediction_source_hint") or prediction_debug.get("field_prediction_source") or bz_projection.get("source") or ""),
        status_hint=str(prediction_debug.get("field_prediction_status") or ""),
        unavailable_reason_hint=str(prediction_debug.get("field_prediction_unavailable_reason") or ""),
        fallback_reason_hint=str(prediction_debug.get("field_prediction_fallback_reason") or ""),
        hierarchy_hint=prediction_debug.get("field_prediction_hierarchy"),
        harmonic_weights_hint=prediction_debug.get("harmonic_weights_used") or confidence_summary.get("harmonic_weights_used") or _frame_value(command_profile, "harmonic_weights_used"),
        explicit_expected_current_source=str(prediction_debug.get("expected_current_source") or ""),
        exact_field_direct_available=prediction_debug.get("exact_field_direct_available"),
        exact_field_direct_reason=prediction_debug.get("exact_field_direct_reason"),
        same_recipe_surrogate_candidate_available=prediction_debug.get("same_recipe_surrogate_candidate_available"),
        same_recipe_surrogate_applied=prediction_debug.get("same_recipe_surrogate_applied"),
        same_recipe_surrogate_ratio=prediction_debug.get("same_recipe_surrogate_ratio"),
        surrogate_scope=prediction_debug.get("surrogate_scope"),
        selected_support_id=prediction_debug.get("selected_support_id") or engine_summary.get("selected_support_id"),
        selected_support_family=prediction_debug.get("selected_support_family") or engine_summary.get("selected_support_family"),
        support_selection_reason=prediction_debug.get("support_selection_reason") or engine_summary.get("support_selection_reason"),
        support_family_metric=prediction_debug.get("support_family_metric") or engine_summary.get("support_family_metric"),
        support_family_value=prediction_debug.get("support_family_value") or engine_summary.get("support_family_value"),
        support_family_lock_applied=prediction_debug.get("support_family_lock_applied", engine_summary.get("support_family_lock_applied")),
        support_bz_to_current_ratio=prediction_debug.get("support_bz_to_current_ratio") or engine_summary.get("support_bz_to_current_ratio"),
    )


def sanitize_unavailable_exact_field_prediction(
    command_profile: pd.DataFrame,
    prediction_debug: dict[str, Any],
    *,
    target_output_type: str,
) -> pd.DataFrame:
    if command_profile.empty or str(target_output_type) != "field":
        return command_profile
    if str(prediction_debug.get("request_route") or "") != "exact":
        return command_profile
    if str(prediction_debug.get("field_prediction_status") or "") != "unavailable":
        return command_profile
    sanitized = command_profile.copy()
    nan_values = np.full(len(sanitized), np.nan, dtype=float)
    for column in ("expected_field_mT", "modeled_field_mT", "support_scaled_field_mT", "expected_output", "modeled_output"):
        if column in sanitized.columns:
            sanitized[column] = nan_values
    return sanitized
