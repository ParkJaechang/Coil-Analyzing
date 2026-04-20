from __future__ import annotations

from .validation_retune_shared import *
from .validation_retune_metric_utils import *

def build_retune_quality_badge_payload(comparison: ValidationComparison) -> dict[str, Any]:
    thresholds = QUALITY_BADGE_POLICY["thresholds"]
    labels = QUALITY_BADGE_POLICY["labels"]
    reasons: list[str] = []
    clipping = bool(comparison.clipping_detected)
    saturation = bool(comparison.saturation_detected)
    nrmse = comparison.nrmse
    shape_corr = comparison.shape_corr
    phase_lag_s = abs(comparison.phase_lag_s)
    metrics_available = bool(comparison.metrics_available)
    missing_metric = not metrics_available

    if clipping or saturation:
        reasons.append("clipping/saturation 媛먯?")
    if missing_metric:
        reasons.append(f"Bz metric unavailable ({comparison.unavailable_reason or 'other'})")
    if np.isfinite(nrmse) and nrmse > float(thresholds["repro_good_max_nrmse"]):
        reasons.append(f"Bz NRMSE {nrmse:.2%}")
    if np.isfinite(shape_corr) and shape_corr < float(thresholds["repro_good_min_shape_corr"]):
        reasons.append(f"Bz shape corr {shape_corr:.3f}")
    if np.isfinite(phase_lag_s) and phase_lag_s > float(thresholds["repro_good_max_phase_lag_s"]):
        reasons.append(f"Bz phase lag {phase_lag_s:.4f}s")

    if clipping or saturation or (
        np.isfinite(nrmse) and nrmse > float(thresholds["caution_max_nrmse"])
    ) or (
        np.isfinite(shape_corr) and shape_corr < float(thresholds["caution_min_shape_corr"])
    ) or (
        np.isfinite(phase_lag_s) and phase_lag_s > float(thresholds["caution_max_phase_lag_s"])
    ):
        label = labels["retune"]
        tone = "red"
    elif missing_metric or reasons:
        label = labels["caution"]
        tone = "orange"
    else:
        label = labels["good"]
        tone = "green"
        reasons = ["Bz NRMSE / shape corr / phase lag媛 exact retune 湲곗? ?댁뿉 ?덉뒿?덈떎."]

    return {
        "label": label,
        "tone": tone,
        "reasons": reasons,
        "reason_codes": list(comparison.reason_codes),
        "metrics_available": metrics_available,
        "unavailable_reason": comparison.unavailable_reason,
        "evaluation_status": "evaluated" if metrics_available else "unevaluable",
        "evaluation_label": QUALITY_EVALUATION_STATUS_LABELS["evaluated" if metrics_available else "unevaluable"],
        "metric_domain": QUALITY_BADGE_POLICY["metric_domain"],
        "basis": {
            "comparison_label": comparison.label,
            "comparison_source": comparison.comparison_source,
            "target_basis": comparison.target_basis,
        },
        "criteria": QUALITY_BADGE_POLICY,
    }


def build_retune_quality_badge(comparison: ValidationComparison) -> tuple[str, str, list[str]]:
    badge = build_retune_quality_badge_payload(comparison)
    return str(badge["label"]), str(badge["tone"]), list(badge["reasons"])


def _build_retune_quality_badge(comparison: ValidationComparison) -> tuple[str, str, list[str]]:
    return build_retune_quality_badge(comparison)


def _extend_unique_reason_codes(target: list[str], *groups: list[str]) -> list[str]:
    for group in groups:
        for code in group:
            text = str(code or "").strip()
            if text and text not in target:
                target.append(text)
    return target


def _comparison_metric_snapshot(comparison: ValidationComparison) -> dict[str, Any]:
    return {
        "label": comparison.label,
        "nrmse": comparison.nrmse,
        "shape_corr": comparison.shape_corr,
        "phase_lag_s": comparison.phase_lag_s,
        "abs_phase_lag_s": abs(comparison.phase_lag_s) if np.isfinite(comparison.phase_lag_s) else float("nan"),
        "clipping_detected": comparison.clipping_detected,
        "saturation_detected": comparison.saturation_detected,
        "metrics_available": comparison.metrics_available,
        "unavailable_reason": comparison.unavailable_reason,
        "reason_codes": list(comparison.reason_codes),
        "valid_sample_count": comparison.valid_sample_count,
        "target_basis": comparison.target_basis,
        "comparison_source": comparison.comparison_source,
    }


def build_retune_acceptance_decision(
    *,
    validation_run: ValidationRun,
    baseline_comparison: ValidationComparison,
    corrected_comparison: ValidationComparison,
    baseline_bz_comparison: ValidationComparison,
    corrected_bz_comparison: ValidationComparison,
) -> dict[str, Any]:
    decision = "evaluation_failed"
    reason_codes: list[str] = []
    preferred_output_kind = "baseline"
    preferred_output_id = validation_run.lut_id
    rejection_reason: str | None = None
    accepted = False

    try:
        min_improvement = RETUNE_ACCEPTANCE_POLICY["min_improvement"]
        max_tolerated_degradation = RETUNE_ACCEPTANCE_POLICY["max_tolerated_degradation"]
        baseline_clipped = bool(
            baseline_bz_comparison.clipping_detected or baseline_bz_comparison.saturation_detected
        )
        corrected_clipped = bool(
            corrected_bz_comparison.clipping_detected or corrected_bz_comparison.saturation_detected
        )
        candidate_clipping = corrected_clipped and not baseline_clipped
        bz_nrmse_improvement = _safe_float(baseline_bz_comparison.nrmse - corrected_bz_comparison.nrmse)
        bz_shape_corr_improvement = _safe_float(corrected_bz_comparison.shape_corr - baseline_bz_comparison.shape_corr)
        bz_phase_lag_improvement_s = _safe_float(
            abs(baseline_bz_comparison.phase_lag_s) - abs(corrected_bz_comparison.phase_lag_s)
        )
        target_nrmse_improvement = _safe_float(baseline_comparison.nrmse - corrected_comparison.nrmse)
        target_shape_corr_improvement = _safe_float(corrected_comparison.shape_corr - baseline_comparison.shape_corr)
        target_phase_lag_improvement_s = _safe_float(
            abs(baseline_comparison.phase_lag_s) - abs(corrected_comparison.phase_lag_s)
        )
        valid_sample_floor = min(
            int(baseline_bz_comparison.valid_sample_count),
            int(corrected_bz_comparison.valid_sample_count),
        )
        mapping_meta = validation_run.metadata.get("bz_target_mapping", {}) or {}
        projection_meta = validation_run.metadata.get("corrected_bz_projection", {}) or {}
        weak_bz_mapping = (
            not bool(mapping_meta.get("available", False))
            or str(mapping_meta.get("reason_code") or "") in {"invalid_target_mapping", "surrogate_unstable", "insufficient_active_window", "missing_bz_channel"}
            or (
                validation_run.exact_path == EXACT_PATH_FINITE
                and str(mapping_meta.get("basis") or "").startswith("mapped_target")
            )
        )
        unstable_transfer_estimate = (
            not bool(projection_meta.get("available", False))
            or str(projection_meta.get("source") or "") == "reference_voltage_to_bz_transfer"
            or str(projection_meta.get("reason_code") or "") in {"insufficient_active_window", "surrogate_unstable", "other"}
        )

        if baseline_clipped:
            _extend_unique_reason_codes(reason_codes, ["clipped_actual"])
        if valid_sample_floor < int(RETUNE_ACCEPTANCE_POLICY["min_valid_samples"]):
            _extend_unique_reason_codes(reason_codes, ["insufficient_valid_samples"])

        metrics_available = bool(
            baseline_bz_comparison.metrics_available and corrected_bz_comparison.metrics_available
        )
        if not metrics_available:
            decision = "metrics_unavailable"
            rejection_reason = (
                corrected_bz_comparison.unavailable_reason
                or baseline_bz_comparison.unavailable_reason
                or "metrics_unavailable"
            )
            _extend_unique_reason_codes(
                reason_codes,
                list(baseline_bz_comparison.reason_codes),
                list(corrected_bz_comparison.reason_codes),
            )
            if weak_bz_mapping:
                _extend_unique_reason_codes(reason_codes, ["weak_bz_mapping"])
            if unstable_transfer_estimate:
                _extend_unique_reason_codes(reason_codes, ["unstable_transfer_estimate"])
            if validation_run.exact_path == EXACT_PATH_FINITE:
                _extend_unique_reason_codes(reason_codes, ["finite_alignment_sensitive"])
        else:
            material_improvement = any(
                (
                    bz_nrmse_improvement is not None
                    and bz_nrmse_improvement >= float(min_improvement["bz_nrmse"]),
                    bz_shape_corr_improvement is not None
                    and bz_shape_corr_improvement >= float(min_improvement["bz_shape_corr"]),
                    bz_phase_lag_improvement_s is not None
                    and bz_phase_lag_improvement_s >= float(min_improvement["bz_phase_lag_s"]),
                    baseline_clipped and not corrected_clipped,
                )
            )
            material_degradation = any(
                (
                    bz_nrmse_improvement is not None
                    and bz_nrmse_improvement <= -float(max_tolerated_degradation["bz_nrmse"]),
                    bz_shape_corr_improvement is not None
                    and bz_shape_corr_improvement <= -float(max_tolerated_degradation["bz_shape_corr"]),
                    bz_phase_lag_improvement_s is not None
                    and bz_phase_lag_improvement_s <= -float(max_tolerated_degradation["bz_phase_lag_s"]),
                    candidate_clipping,
                )
            )
            if material_degradation:
                decision = "degraded_and_rejected"
                rejection_reason = "degraded_candidate"
                if candidate_clipping:
                    _extend_unique_reason_codes(reason_codes, ["candidate_clipping"])
                if weak_bz_mapping:
                    _extend_unique_reason_codes(reason_codes, ["weak_bz_mapping"])
                if unstable_transfer_estimate:
                    _extend_unique_reason_codes(reason_codes, ["unstable_transfer_estimate"])
                if validation_run.exact_path == EXACT_PATH_FINITE:
                    _extend_unique_reason_codes(reason_codes, ["finite_alignment_sensitive"])
                target_material_improvement = any(
                    (
                        target_nrmse_improvement is not None and target_nrmse_improvement > 0.0,
                        target_shape_corr_improvement is not None and target_shape_corr_improvement > 0.0,
                        target_phase_lag_improvement_s is not None and target_phase_lag_improvement_s > 0.0,
                    )
                )
                if target_material_improvement:
                    _extend_unique_reason_codes(reason_codes, ["correction_overfit"])
                rejection_reason = reason_codes[0] if reason_codes else rejection_reason
            elif material_improvement:
                decision = "improved_and_accepted"
                preferred_output_kind = "corrected_candidate"
                preferred_output_id = validation_run.corrected_lut_id
                accepted = True
            else:
                decision = "no_material_change"
                rejection_reason = "no_material_improvement"
                _extend_unique_reason_codes(reason_codes, ["no_material_improvement"])
    except Exception:
        decision = "evaluation_failed"
        rejection_reason = "evaluation_failed"
        _extend_unique_reason_codes(reason_codes, ["other"])

    label = RETUNE_ACCEPTANCE_DECISION_LABELS[decision]
    tone = RETUNE_ACCEPTANCE_DECISION_TONES[decision]
    preferred_source_kind = SOURCE_KIND_CORRECTED if preferred_output_kind == "corrected_candidate" else validation_run.source_kind
    return {
        "decision": decision,
        "label": label,
        "tone": tone,
        "accepted": accepted,
        "preferred_output_kind": preferred_output_kind,
        "preferred_output_id": preferred_output_id,
        "preferred_output_source_kind": preferred_source_kind,
        "baseline_output_id": validation_run.lut_id,
        "baseline_output_source_kind": validation_run.source_kind,
        "corrected_candidate_id": validation_run.corrected_lut_id,
        "corrected_candidate_source_kind": SOURCE_KIND_CORRECTED,
        "rejection_reason": rejection_reason,
        "reason_codes": reason_codes,
        "policy": RETUNE_ACCEPTANCE_POLICY,
        "metric_snapshot": {
            "baseline": _comparison_metric_snapshot(baseline_bz_comparison),
            "corrected": _comparison_metric_snapshot(corrected_bz_comparison),
            "target_output_baseline": _comparison_metric_snapshot(baseline_comparison),
            "target_output_corrected": _comparison_metric_snapshot(corrected_comparison),
            "improvements": {
                "bz_nrmse": bz_nrmse_improvement,
                "bz_shape_corr": bz_shape_corr_improvement,
                "bz_phase_lag_s": bz_phase_lag_improvement_s,
                "target_output_nrmse": target_nrmse_improvement,
                "target_output_shape_corr": target_shape_corr_improvement,
                "target_output_phase_lag_s": target_phase_lag_improvement_s,
            },
        },
    }





__all__ = [name for name in globals() if not name.startswith('__')]

