from __future__ import annotations

from .validation_retune_shared import *
from .validation_retune_metric_utils import *

def build_retune_artifact_payload(
    *,
    validation_run: ValidationRun,
    baseline_comparison: ValidationComparison,
    corrected_comparison: ValidationComparison,
    baseline_bz_comparison: ValidationComparison,
    corrected_bz_comparison: ValidationComparison,
    loop_summary: dict[str, Any],
    quality_badge: dict[str, Any],
    acceptance_decision: dict[str, Any],
    baseline_prediction_debug: dict[str, Any],
    corrected_prediction_debug: dict[str, Any],
) -> dict[str, Any]:
    bz_metric_comparison = {
        "baseline": {
            "nrmse": baseline_bz_comparison.nrmse,
            "shape_corr": baseline_bz_comparison.shape_corr,
            "phase_lag_s": baseline_bz_comparison.phase_lag_s,
            "clipping_detected": baseline_bz_comparison.clipping_detected,
            "saturation_detected": baseline_bz_comparison.saturation_detected,
            "metrics_available": baseline_bz_comparison.metrics_available,
            "unavailable_reason": baseline_bz_comparison.unavailable_reason,
            "reason_codes": list(baseline_bz_comparison.reason_codes),
        },
        "corrected": {
            "nrmse": corrected_bz_comparison.nrmse,
            "shape_corr": corrected_bz_comparison.shape_corr,
            "phase_lag_s": corrected_bz_comparison.phase_lag_s,
            "clipping_detected": corrected_bz_comparison.clipping_detected,
            "saturation_detected": corrected_bz_comparison.saturation_detected,
            "metrics_available": corrected_bz_comparison.metrics_available,
            "unavailable_reason": corrected_bz_comparison.unavailable_reason,
            "reason_codes": list(corrected_bz_comparison.reason_codes),
        },
    }
    return {
        "schema_version": "validation_retune_v2",
        "provenance": {
            "original_recommendation_id": validation_run.original_recommendation_id,
            "lut_id": validation_run.lut_id,
            "source_kind": validation_run.source_kind,
            "source_selection_id": validation_run.source_selection_id,
            "source_lut_filename": validation_run.source_lut_filename,
            "source_profile_path": validation_run.source_profile_path,
            "validation_run_id": validation_run.validation_run_id,
            "corrected_lut_id": validation_run.corrected_lut_id,
            "iteration_index": validation_run.iteration_index,
            "created_at": validation_run.created_at,
            "exact_path": validation_run.exact_path,
            "correction_rule": validation_run.correction_rule,
        },
        "validation_run": asdict(validation_run),
        "baseline_comparison": asdict(baseline_comparison),
        "corrected_comparison": asdict(corrected_comparison),
        "baseline_bz_comparison": asdict(baseline_bz_comparison),
        "corrected_bz_comparison": asdict(corrected_bz_comparison),
        "baseline_metrics": {
            TARGET_OUTPUT_DOMAIN: asdict(baseline_comparison),
            BZ_EFFECTIVE_DOMAIN: asdict(baseline_bz_comparison),
        },
        "corrected_metrics": {
            TARGET_OUTPUT_DOMAIN: asdict(corrected_comparison),
            BZ_EFFECTIVE_DOMAIN: asdict(corrected_bz_comparison),
        },
        "before_after_metrics": {
            TARGET_OUTPUT_DOMAIN: {
                "before": asdict(baseline_comparison),
                "after": asdict(corrected_comparison),
            },
            BZ_EFFECTIVE_DOMAIN: {
                "before": asdict(baseline_bz_comparison),
                "after": asdict(corrected_bz_comparison),
            },
        },
        "prediction_debug": {
            "baseline": baseline_prediction_debug,
            "corrected": corrected_prediction_debug,
        },
        "bz_metric_comparison": bz_metric_comparison,
        "acceptance_decision": acceptance_decision,
        "preferred_output": {
            "output_id": acceptance_decision.get("preferred_output_id"),
            "output_kind": acceptance_decision.get("preferred_output_kind"),
            "source_kind": acceptance_decision.get("preferred_output_source_kind"),
        },
        "preferred_output_id": acceptance_decision.get("preferred_output_id"),
        "preferred_output_kind": acceptance_decision.get("preferred_output_kind"),
        "preferred_output_source_kind": acceptance_decision.get("preferred_output_source_kind"),
        "rejection_reason": acceptance_decision.get("rejection_reason"),
        "candidate_status": acceptance_decision.get("decision"),
        "candidate_status_label": acceptance_decision.get("label"),
        "candidate_status_tone": acceptance_decision.get("tone"),
        "metrics_availability": {
            "baseline_bz": {
                "available": baseline_bz_comparison.metrics_available,
                "reason": baseline_bz_comparison.unavailable_reason,
                "reason_codes": list(baseline_bz_comparison.reason_codes),
            },
            "corrected_bz": {
                "available": corrected_bz_comparison.metrics_available,
                "reason": corrected_bz_comparison.unavailable_reason,
                "reason_codes": list(corrected_bz_comparison.reason_codes),
            },
        },
        "loop_summary": loop_summary,
        "quality_badge": quality_badge,
        "retune_delta": {
            "target_output_nrmse_improvement": _safe_float(baseline_comparison.nrmse - corrected_comparison.nrmse),
            "target_output_shape_corr_improvement": _safe_float(corrected_comparison.shape_corr - baseline_comparison.shape_corr),
            "target_output_phase_lag_improvement_s": _safe_float(abs(baseline_comparison.phase_lag_s) - abs(corrected_comparison.phase_lag_s)),
            "bz_nrmse_improvement": _safe_float(baseline_bz_comparison.nrmse - corrected_bz_comparison.nrmse),
            "bz_shape_corr_improvement": _safe_float(corrected_bz_comparison.shape_corr - baseline_bz_comparison.shape_corr),
            "bz_phase_lag_improvement_s": _safe_float(abs(baseline_bz_comparison.phase_lag_s) - abs(corrected_bz_comparison.phase_lag_s)),
        },
    }


def save_retune_artifacts(
    *,
    retune_result: RetuneResult,
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = retune_result.validation_run.corrected_lut_id
    corrected_waveform_path = output_dir / f"{prefix}_waveform.csv"
    corrected_lut_path = output_dir / f"{prefix}_control_lut.csv"
    corrected_formula_path = output_dir / f"{prefix}_formula.txt"
    report_json_path = output_dir / f"{prefix}_validation_report.json"
    report_md_path = output_dir / f"{prefix}_validation_report.md"
    retune_result_json_path = output_dir / f"{prefix}_retune_result.json"
    retune_result_md_path = output_dir / f"{prefix}_retune_result.md"

    retune_result.corrected_command_profile.to_csv(corrected_waveform_path, index=False, encoding="utf-8-sig")
    control_lut = build_control_lut(retune_result.corrected_command_profile, value_column="limited_voltage_v", sample_count=128)
    if control_lut is not None:
        control_lut.to_csv(corrected_lut_path, index=False, encoding="utf-8-sig")
    control_formula = build_control_formula(retune_result.corrected_command_profile, value_column="limited_voltage_v")
    if control_formula is not None:
        corrected_formula_path.write_text(str(control_formula["formula_text"]), encoding="utf-8")

    artifact_paths = {
        "corrected_waveform_csv": corrected_waveform_path.as_posix(),
        "validation_report_json": report_json_path.as_posix(),
        "validation_report_md": report_md_path.as_posix(),
        "retune_result_json": retune_result_json_path.as_posix(),
        "retune_result_md": retune_result_md_path.as_posix(),
    }
    if corrected_lut_path.exists():
        artifact_paths["corrected_control_lut_csv"] = corrected_lut_path.as_posix()
    if corrected_formula_path.exists():
        artifact_paths["corrected_formula_txt"] = corrected_formula_path.as_posix()
    artifact_manifest = {
        "artifact_prefix": prefix,
        "output_dir": output_dir.as_posix(),
        "artifact_paths": artifact_paths,
        "required_artifacts": list(REQUIRED_CORRECTED_ARTIFACT_KEYS),
        "complete": all(key in artifact_paths for key in REQUIRED_CORRECTED_ARTIFACT_KEYS),
    }
    retune_result.artifact_paths = artifact_paths
    retune_result.artifact_payload["artifact_paths"] = artifact_paths
    retune_result.artifact_payload["artifact_manifest"] = artifact_manifest
    validation_payload = to_jsonable(retune_result.artifact_payload)
    retune_payload = to_jsonable(
        {
            **retune_result.artifact_payload,
            "iteration_table": retune_result.iteration_table.to_dict(orient="records"),
            "overlay_columns": list(retune_result.overlay_frame.columns),
            "overlay_row_count": int(len(retune_result.overlay_frame)),
        }
    )
    report_json_path.write_text(json.dumps(validation_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md_path.write_text(_build_validation_report_markdown(retune_result), encoding="utf-8")
    retune_result_json_path.write_text(json.dumps(retune_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    retune_result_md_path.write_text(_build_retune_result_markdown(retune_result), encoding="utf-8")
    return artifact_paths


def _build_validation_report_markdown(retune_result: RetuneResult) -> str:
    run = retune_result.validation_run
    before = retune_result.baseline_comparison
    after = retune_result.corrected_comparison
    before_bz = retune_result.baseline_bz_comparison
    after_bz = retune_result.corrected_bz_comparison
    loop = retune_result.loop_summary
    artifact_paths = retune_result.artifact_payload.get("artifact_paths", {})
    lines = [
        "# Validation Report",
        "",
        "## Provenance",
        f"- exact_path: `{run.exact_path}`",
        f"- source_kind: `{run.source_kind}`",
        f"- lut_id: `{run.lut_id}`",
        f"- original_recommendation_id: `{run.original_recommendation_id}`",
        f"- corrected_lut_id: `{run.corrected_lut_id}`",
        f"- validation_run_id: `{run.validation_run_id}`",
        f"- source_lut_filename: `{run.source_lut_filename or 'n/a'}`",
        f"- iteration_index: `{run.iteration_index}`",
        f"- created_at: `{run.created_at}`",
        f"- correction_rule: `{run.correction_rule}`",
        "",
        "## Quality Badge",
        f"- label: `{retune_result.quality_label}`",
        f"- tone: `{retune_result.quality_tone}`",
        f"- reasons: `{'; '.join(retune_result.quality_reasons) if retune_result.quality_reasons else 'n/a'}`",
        "",
        build_quality_badge_markdown(),
        "",
        "## Before / After Metrics (Target Output Domain)",
        f"- before NRMSE: `{before.nrmse:.4f}`",
        f"- after NRMSE: `{after.nrmse:.4f}`",
        f"- before shape corr: `{before.shape_corr:.4f}`",
        f"- after shape corr: `{after.shape_corr:.4f}`",
        f"- before phase lag (s): `{before.phase_lag_s:.6f}`",
        f"- after phase lag (s): `{after.phase_lag_s:.6f}`",
        "",
        "## Before / After Metrics (Bz Effective Domain)",
        f"- before Bz NRMSE: `{before_bz.nrmse:.4f}`",
        f"- after Bz NRMSE: `{after_bz.nrmse:.4f}`",
        f"- before Bz shape corr: `{before_bz.shape_corr:.4f}`",
        f"- after Bz shape corr: `{after_bz.shape_corr:.4f}`",
        f"- before Bz phase lag (s): `{before_bz.phase_lag_s:.6f}`",
        f"- after Bz phase lag (s): `{after_bz.phase_lag_s:.6f}`",
        "",
        "## Retune Loop",
        f"- iteration_count: `{loop.get('iteration_count')}`",
        f"- stop_reason: `{loop.get('stop_reason')}`",
        f"- within_hardware_limits: `{loop.get('within_hardware_limits')}`",
        "",
        "## Artifacts",
        *(f"- {name}: `{path}`" for name, path in artifact_paths.items()),
    ]
    return "\n".join(lines) + "\n"


def _build_retune_result_markdown(retune_result: RetuneResult) -> str:
    iteration_rows = retune_result.iteration_table.head(8).to_dict(orient="records")
    lines = [
        "# Retune Result",
        "",
        f"- corrected_lut_id: `{retune_result.validation_run.corrected_lut_id}`",
        f"- validation_run_id: `{retune_result.validation_run.validation_run_id}`",
        f"- exact_path: `{retune_result.validation_run.exact_path}`",
        f"- quality_label: `{retune_result.quality_label}`",
        "",
        "## Iteration Preview",
    ]
    if iteration_rows:
        for row in iteration_rows:
            row_text = ", ".join(f"{key}={value}" for key, value in row.items())
            lines.append(f"- {row_text}")
    else:
        lines.append("- iteration_table unavailable")
    lines.extend(
        [
            "",
            "## Artifact Manifest",
            *(f"- {key}: `{value}`" for key, value in retune_result.artifact_paths.items()),
        ]
    )
    return "\n".join(lines) + "\n"





__all__ = [name for name in globals() if not name.startswith('__')]

