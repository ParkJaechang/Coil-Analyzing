from __future__ import annotations

from .validation_retune_shared import *
from .validation_retune_provenance import *
from .validation_retune_alignment import *
from .validation_retune_comparison import *
from .validation_retune_corrected_export import *
from .validation_retune_acceptance import *
from .validation_retune_metric_utils import *

def execute_validation_retune(
    *,
    base_profile: pd.DataFrame,
    validation_candidate: dict[str, Any],
    validation_frame: pd.DataFrame,
    export_file_prefix: str,
    target_output_type: str,
    current_channel: str,
    field_channel: str,
    max_daq_voltage_pp: float,
    amp_gain_at_100_pct: float,
    amp_gain_limit_pct: float,
    amp_max_output_pk_v: float,
    support_amp_gain_pct: float,
    correction_gain: float,
    max_iterations: int,
    improvement_threshold: float,
    original_recommendation_id: str | None = None,
    source_selection: dict[str, Any] | None = None,
    iteration_index: int | None = None,
) -> RetuneResult | None:
    exact_path_preview = infer_exact_path(base_profile=base_profile, target_output_type=target_output_type)
    effective_correction_gain = float(correction_gain)
    if exact_path_preview == EXACT_PATH_FINITE:
        effective_correction_gain = min(
            effective_correction_gain,
            float(RETUNE_ACCEPTANCE_POLICY["finite_exact_max_correction_gain"]),
        )
    correction_rule = build_correction_rule(
        correction_gain=effective_correction_gain,
        max_iterations=max_iterations,
        improvement_threshold=improvement_threshold,
    )
    validation_run = build_validation_run(
        base_profile=base_profile,
        validation_candidate=validation_candidate,
        export_file_prefix=export_file_prefix,
        target_output_type=target_output_type,
        original_recommendation_id=original_recommendation_id,
        source_selection=source_selection,
        iteration_index=iteration_index,
        correction_rule=correction_rule,
    )
    canonical_validation_frame = _canonicalize_validation_frame(
        base_profile=base_profile,
        validation_frame=validation_frame,
        target_output_type=target_output_type,
        current_channel=current_channel,
        field_channel=field_channel,
    )
    base_profile_with_bz = _ensure_bz_target_mapping(
        reference_profile=base_profile,
        validation_frame=canonical_validation_frame,
        target_output_type=target_output_type,
        current_channel=current_channel,
        field_channel=field_channel,
    )
    window_info = canonical_validation_frame.attrs.get("validation_window", {})
    validation_run.metadata["validation_window"] = {
        "applied": bool(window_info.get("applied", False)),
        "start_s": _safe_float(window_info.get("start_s")),
        "end_s": _safe_float(window_info.get("end_s")),
        "score": _safe_float(window_info.get("score")),
        "output_column": str(window_info.get("output_column") or ""),
    }
    validation_run.metadata["requested_correction_gain"] = float(correction_gain)
    validation_run.metadata["effective_correction_gain"] = float(effective_correction_gain)
    validation_run.metadata["bz_target_mapping"] = dict(base_profile_with_bz.attrs.get("bz_target_mapping", {}))
    baseline_prediction_debug = build_prediction_debug_snapshot(
        command_profile=base_profile_with_bz,
        validation_frame=canonical_validation_frame,
        target_output_type=target_output_type,
        current_channel=current_channel,
        field_channel=field_channel,
    )
    validation_run.metadata["baseline_prediction_debug"] = dict(baseline_prediction_debug)
    baseline_comparison = build_validation_comparison(
        command_profile=base_profile_with_bz,
        validation_frame=canonical_validation_frame,
        label="before_retune",
        comparison_source="actual",
        target_output_type=target_output_type,
        current_channel=current_channel,
        field_channel=field_channel,
        metric_domain=TARGET_OUTPUT_DOMAIN,
    )
    baseline_bz_comparison = build_validation_comparison(
        command_profile=base_profile_with_bz,
        validation_frame=canonical_validation_frame,
        label="before_retune_bz",
        comparison_source="actual",
        target_output_type=target_output_type,
        current_channel=current_channel,
        field_channel=field_channel,
        metric_domain=BZ_EFFECTIVE_DOMAIN,
    )
    loop_result = run_validation_recommendation_loop(
        command_profile=base_profile,
        validation_frame=canonical_validation_frame,
        target_output_type=target_output_type,
        current_channel=current_channel,
        field_channel=field_channel,
        correction_gain=effective_correction_gain,
        max_iterations=max_iterations,
        improvement_threshold=improvement_threshold,
        max_daq_voltage_pp=max_daq_voltage_pp,
        amp_gain_at_100_pct=amp_gain_at_100_pct,
        support_amp_gain_pct=support_amp_gain_pct,
        amp_gain_limit_pct=amp_gain_limit_pct,
        amp_max_output_pk_v=amp_max_output_pk_v,
    )
    if loop_result is None:
        return None

    corrected_profile = loop_result["command_profile"].copy()
    corrected_profile.attrs = {
        **dict(getattr(base_profile, "attrs", {})),
        **dict(getattr(corrected_profile, "attrs", {})),
    }
    corrected_profile = _ensure_predicted_output_from_reference_transfer(
        reference_profile=base_profile,
        corrected_profile=corrected_profile,
        target_output_type=target_output_type,
    )
    corrected_profile = _project_bz_from_validation_transfer(
        reference_profile=base_profile_with_bz,
        corrected_profile=corrected_profile,
        validation_frame=canonical_validation_frame,
        field_channel=field_channel,
    )
    corrected_profile = _ensure_bz_target_mapping(
        reference_profile=corrected_profile,
        validation_frame=canonical_validation_frame,
        target_output_type=target_output_type,
        current_channel=current_channel,
        field_channel=field_channel,
    )
    corrected_profile = _ensure_bz_surrogate_columns(
        reference_profile=base_profile_with_bz,
        corrected_profile=corrected_profile,
    )
    validation_run.metadata["corrected_bz_projection"] = dict(corrected_profile.attrs.get("bz_projection", {}))
    corrected_prediction_debug = build_prediction_debug_snapshot(
        command_profile=corrected_profile,
        validation_frame=canonical_validation_frame,
        target_output_type=target_output_type,
        current_channel=current_channel,
        field_channel=field_channel,
    )
    validation_run.metadata["corrected_prediction_debug"] = dict(corrected_prediction_debug)
    corrected_comparison = build_validation_comparison(
        command_profile=corrected_profile,
        validation_frame=canonical_validation_frame,
        label="after_retune",
        comparison_source="predicted",
        target_output_type=target_output_type,
        current_channel=current_channel,
        field_channel=field_channel,
        metric_domain=TARGET_OUTPUT_DOMAIN,
    )
    corrected_bz_comparison = build_validation_comparison(
        command_profile=corrected_profile,
        validation_frame=canonical_validation_frame,
        label="after_retune_bz",
        comparison_source="predicted",
        target_output_type=target_output_type,
        current_channel=current_channel,
        field_channel=field_channel,
        metric_domain=BZ_EFFECTIVE_DOMAIN,
    )
    overlay_frame = build_validation_overlay_frame(
        base_profile=base_profile_with_bz,
        validation_frame=canonical_validation_frame,
        corrected_profile=corrected_profile,
        target_output_type=target_output_type,
        current_channel=current_channel,
        field_channel=field_channel,



    )
    loop_summary = {
        "mode": str(loop_result.get("mode") or "validation_retune"),
        "iteration_count": int(loop_result.get("iteration_count", 0)),
        "stop_reason": str(loop_result.get("stop_reason") or "unknown"),
        "validation_rmse_reference": _safe_float(loop_result.get("validation_rmse_reference")),
        "validation_nrmse_reference": _safe_float(loop_result.get("validation_nrmse_reference")),
        "predicted_rmse_final": _safe_float(loop_result.get("predicted_rmse_final")),
        "predicted_nrmse_final": _safe_float(loop_result.get("predicted_nrmse_final")),
        "within_hardware_limits": bool(loop_result.get("within_hardware_limits", False)),
        "correction_gain": float(effective_correction_gain),
        "requested_correction_gain": float(correction_gain),
        "effective_correction_gain": float(effective_correction_gain),
        "improvement_threshold": float(improvement_threshold),
        "correction_rule": validation_run.correction_rule,
        "exact_path": validation_run.exact_path,
    }
    quality_badge = build_retune_quality_badge_payload(corrected_bz_comparison)
    acceptance_decision = build_retune_acceptance_decision(
        validation_run=validation_run,
        baseline_comparison=baseline_comparison,
        corrected_comparison=corrected_comparison,
        baseline_bz_comparison=baseline_bz_comparison,
        corrected_bz_comparison=corrected_bz_comparison,
    )
    quality_badge["candidate_status"] = acceptance_decision["decision"]
    quality_badge["candidate_status_label"] = acceptance_decision["label"]
    quality_badge["candidate_status_tone"] = acceptance_decision["tone"]
    quality_badge["preferred_output_id"] = acceptance_decision["preferred_output_id"]
    quality_badge["rejection_reason"] = acceptance_decision["rejection_reason"]
    artifact_payload = build_retune_artifact_payload(
        validation_run=validation_run,
        baseline_comparison=baseline_comparison,
        corrected_comparison=corrected_comparison,
        baseline_bz_comparison=baseline_bz_comparison,
        corrected_bz_comparison=corrected_bz_comparison,
        loop_summary=loop_summary,
        quality_badge=quality_badge,
        acceptance_decision=acceptance_decision,
        baseline_prediction_debug=baseline_prediction_debug,
        corrected_prediction_debug=corrected_prediction_debug,
    )
    return RetuneResult(
        validation_run=validation_run,
        baseline_comparison=baseline_comparison,
        corrected_comparison=corrected_comparison,
        baseline_bz_comparison=baseline_bz_comparison,
        corrected_bz_comparison=corrected_bz_comparison,
        overlay_frame=overlay_frame,
        corrected_command_profile=corrected_profile,
        iteration_table=loop_result.get("iteration_table", pd.DataFrame()).copy(),
        loop_summary=loop_summary,
        artifact_payload=artifact_payload,
        quality_label=str(quality_badge["label"]),
        quality_tone=str(quality_badge["tone"]),
        quality_reasons=list(quality_badge["reasons"]),
        quality_badge=quality_badge,
        acceptance_decision=acceptance_decision,
        preferred_output_id=str(acceptance_decision.get("preferred_output_id") or ""),
    )


def _ensure_bz_surrogate_columns(
    *,
    reference_profile: pd.DataFrame,
    corrected_profile: pd.DataFrame,
) -> pd.DataFrame:
    for column in ("expected_field_mT", "modeled_field_mT", "support_scaled_field_mT"):
        if column in corrected_profile.columns:
            candidate = pd.to_numeric(corrected_profile[column], errors="coerce").to_numpy(dtype=float)
            if _is_signal_stable(candidate, min_pp=1e-3):
                return corrected_profile
    if not any(column in reference_profile.columns for column in ("expected_field_mT", "modeled_field_mT", "support_scaled_field_mT")):
        return corrected_profile
    try:
        reference_bz_column = _resolve_bz_expected_column(reference_profile)
        corrected_output_column = _resolve_expected_output_column(corrected_profile)
        reference_output_column = _resolve_expected_output_column(reference_profile)
    except KeyError:
        return corrected_profile

    reference_bz = pd.to_numeric(reference_profile[reference_bz_column], errors="coerce").to_numpy(dtype=float)
    reference_output = pd.to_numeric(reference_profile[reference_output_column], errors="coerce").to_numpy(dtype=float)
    corrected_output = pd.to_numeric(corrected_profile[corrected_output_column], errors="coerce").to_numpy(dtype=float)
    reference_output_pp = _peak_to_peak(reference_output)
    reference_bz_pp = _peak_to_peak(reference_bz)
    scale = 1.0
    if np.isfinite(reference_output_pp) and reference_output_pp > 0 and np.isfinite(reference_bz_pp):
        scale = float(reference_bz_pp / reference_output_pp)
    corrected_profile["expected_field_mT"] = corrected_output * scale
    corrected_profile["modeled_field_mT"] = corrected_profile["expected_field_mT"]
    corrected_profile.attrs["bz_projection"] = {
        "available": True,
        "reason_code": None,
        "source": "current_to_bz_surrogate",
    }
    return corrected_profile


def _resolve_metric_status(
    *,
    metric_domain: str,
    target_basis: str,
    comparison_source: str,
    output_column: str,
    validation_frame: pd.DataFrame,
    target_output: np.ndarray,
    comparison_output: np.ndarray,
    nrmse: float,
    shape_corr: float,
    phase_lag_s: float,
    clipping_detected: bool,
    saturation_detected: bool,
) -> tuple[bool, str | None, list[str], int]:
    valid_mask = np.isfinite(target_output) & np.isfinite(comparison_output)
    valid_sample_count = int(valid_mask.sum())
    if metric_domain != BZ_EFFECTIVE_DOMAIN:
        metrics_available = bool(
            valid_sample_count >= 8
            and np.isfinite(nrmse)
            and np.isfinite(shape_corr)
            and np.isfinite(phase_lag_s)
        )
        return metrics_available, (None if metrics_available else "other"), [], valid_sample_count

    reason_codes: list[str] = []
    critical_reason_codes: list[str] = []
    if comparison_source == "actual" and output_column not in validation_frame.columns:
        critical_reason_codes.append("missing_bz_channel")

    target_pp = _peak_to_peak(target_output)
    target_std = float(np.nanstd(target_output[valid_mask])) if valid_sample_count else float("nan")
    comparison_std = float(np.nanstd(comparison_output[valid_mask])) if valid_sample_count else float("nan")
    surrogate_target = "surrogate" in str(target_basis) or "mapped_target" in str(target_basis)

    if comparison_source == "actual" and (clipping_detected or saturation_detected):
        reason_codes.append("clipped_actual")
    if valid_sample_count < 16:
        critical_reason_codes.append("insufficient_active_window")
    if not np.isfinite(target_pp) or target_pp <= 1e-6:
        critical_reason_codes.append("surrogate_unstable" if surrogate_target else "invalid_target_mapping")
    if valid_sample_count >= 16 and (
        not np.isfinite(target_std)
        or target_std <= 1e-9
        or not np.isfinite(comparison_std)
        or comparison_std <= 1e-9
    ):
        critical_reason_codes.append("surrogate_unstable" if surrogate_target else "unstable_alignment")
    if valid_sample_count >= 16 and (not np.isfinite(shape_corr) or not np.isfinite(phase_lag_s)):
        critical_reason_codes.append("unstable_alignment")

    metrics_available = bool(
        not critical_reason_codes
        and np.isfinite(nrmse)
        and np.isfinite(shape_corr)
        and np.isfinite(phase_lag_s)
    )
    reason_codes.extend(code for code in critical_reason_codes if code not in reason_codes)
    unavailable_reason = None
    if not metrics_available:
        ordered = [
            "missing_bz_channel",
            "invalid_target_mapping",
            "surrogate_unstable",
            "insufficient_active_window",
            "unstable_alignment",
            "clipped_actual",
            "other",
        ]
        for code in ordered:
            if code in reason_codes or code in critical_reason_codes:
                unavailable_reason = code
                break
        if unavailable_reason is None:
            unavailable_reason = "other"
            reason_codes.append(unavailable_reason)
    return metrics_available, unavailable_reason, reason_codes, valid_sample_count



__all__ = [name for name in globals() if not name.startswith('__')]

