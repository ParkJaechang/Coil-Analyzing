
import ast
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from .compensation import run_validation_recommendation_loop
from .control_formula import build_control_formula, build_control_lut
from .field_prediction_debug import build_prediction_debug_from_profile
from .utils import canonicalize_waveform_type


TARGET_OUTPUT_DOMAIN = "target_output"
BZ_EFFECTIVE_DOMAIN = "bz_effective"
SOURCE_KIND_RECOMMENDATION = "recommendation"
SOURCE_KIND_EXPORT = "export"
SOURCE_KIND_CORRECTED = "corrected"
SOURCE_KIND_UNKNOWN = "unknown"
EXACT_PATH_CURRENT = "exact_current"
EXACT_PATH_FIELD = "exact_field"
EXACT_PATH_FINITE = "finite_exact"
REQUIRED_CORRECTED_ARTIFACT_KEYS = (
    "corrected_waveform_csv",
    "corrected_control_lut_csv",
    "corrected_formula_txt",
    "validation_report_json",
    "validation_report_md",
    "retune_result_json",
    "retune_result_md",
)

QUALITY_BADGE_POLICY: dict[str, Any] = {
    "version": "bz_effective_v1",
    "metric_domain": BZ_EFFECTIVE_DOMAIN,
    "thresholds": {
        "repro_good_max_nrmse": 0.15,
        "caution_max_nrmse": 0.30,
        "repro_good_min_shape_corr": 0.97,
        "caution_min_shape_corr": 0.90,
        "repro_good_max_phase_lag_s": 0.02,
        "caution_max_phase_lag_s": 0.05,
        "clipping_forces_retune": True,
    },
    "labels": {
        "good": "재현 양호",
        "caution": "주의",
        "retune": "재보정 권장",
    },
}

METRIC_UNAVAILABLE_REASON_CODES = {
    "missing_bz_channel",
    "clipped_actual",
    "unstable_alignment",
    "insufficient_active_window",
    "invalid_target_mapping",
    "surrogate_unstable",
    "other",
}
QUALITY_EVALUATION_STATUS_LABELS = {
    "evaluated": "평가 가능",
    "unevaluable": "평가 불가",
}
RETUNE_ACCEPTANCE_DECISION_LABELS = {
    "improved_and_accepted": "보정 채택",
    "no_material_change": "변화 미미",
    "degraded_and_rejected": "재보정 후보 기각",
    "metrics_unavailable": "평가 불가",
    "evaluation_failed": "평가 실패",
}
RETUNE_ACCEPTANCE_POLICY: dict[str, Any] = {
    "version": "retune_acceptance_v1",
    "min_improvement": {
        "bz_nrmse": 0.01,
        "bz_shape_corr": 0.005,
        "bz_phase_lag_s": 0.005,
    },
    "max_tolerated_degradation": {
        "bz_nrmse": 0.01,
        "bz_shape_corr": 0.01,
        "bz_phase_lag_s": 0.01,
    },
    "min_valid_samples": 64,
    "finite_exact_max_correction_gain": 0.35,
}
RETUNE_ACCEPTANCE_DECISION_TONES = {
    "improved_and_accepted": "green",
    "no_material_change": "orange",
    "degraded_and_rejected": "red",
    "metrics_unavailable": "orange",
    "evaluation_failed": "red",
}
SOFT_HARDWARE_LIMIT_TOLERANCE_PCT = 0.25


@dataclass(slots=True)
class ValidationRun:
    """Resolved linkage between one selected exact source and one validation run."""

    export_file_prefix: str
    lut_id: str
    source_kind: str
    source_selection_id: str
    source_lut_filename: str | None
    source_profile_path: str | None
    original_recommendation_id: str | None
    validation_run_id: str
    corrected_lut_id: str
    iteration_index: int
    exact_path: str
    target_output_type: str
    waveform_type: str | None
    freq_hz: float | None
    commanded_cycles: float | None
    finite_cycle_mode: bool
    target_level_value: float | None
    target_level_kind: str | None
    selected_validation_test_id: str
    selected_validation_source_file: str | None
    measured_file_name: str | None
    created_at: str
    correction_rule: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationComparison:
    """Measured-vs-target comparison snapshot on a common time grid."""

    label: str
    output_column: str
    rmse: float
    nrmse: float
    shape_corr: float
    phase_lag_s: float
    pp_error: float
    peak_error: float
    clipping_detected: bool
    saturation_detected: bool
    metric_domain: str = TARGET_OUTPUT_DOMAIN
    target_basis: str = "target_output"
    comparison_source: str = "actual"
    sample_count: int = 0
    fit_end_s: float = float("nan")
    metrics_available: bool = True
    unavailable_reason: str | None = None
    reason_codes: list[str] = field(default_factory=list)
    valid_sample_count: int = 0
    active_window_start_s: float = float("nan")
    active_window_end_s: float = float("nan")


@dataclass(slots=True)
class RetuneResult:
    """Validation-driven one-step retune result and linked artifacts."""

    validation_run: ValidationRun
    baseline_comparison: ValidationComparison
    corrected_comparison: ValidationComparison
    baseline_bz_comparison: ValidationComparison
    corrected_bz_comparison: ValidationComparison
    overlay_frame: pd.DataFrame
    corrected_command_profile: pd.DataFrame
    iteration_table: pd.DataFrame
    loop_summary: dict[str, Any]
    artifact_payload: dict[str, Any]
    quality_label: str
    quality_tone: str
    quality_reasons: list[str] = field(default_factory=list)
    quality_badge: dict[str, Any] = field(default_factory=dict)
    acceptance_decision: dict[str, Any] = field(default_factory=dict)
    preferred_output_id: str | None = None
    artifact_paths: dict[str, str] = field(default_factory=dict)


def iso_now() -> str:
    return datetime.now().isoformat(timespec="microseconds")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, pd.DataFrame):
        return to_jsonable(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return to_jsonable(value.to_list())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _coerce_debug_mapping(value: Any) -> dict[str, Any]:
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


def _first_attr_mapping(frame: pd.DataFrame, key: str) -> dict[str, Any]:
    if not isinstance(frame, pd.DataFrame):
        return {}
    return _coerce_debug_mapping(getattr(frame, "attrs", {}).get(key))


def _first_frame_value(frame: pd.DataFrame, column: str) -> Any:
    if frame.empty or column not in frame.columns:
        return None
    series = frame[column].dropna()
    if series.empty:
        return None
    return series.iloc[0]


def _extract_profile_signal(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    if frame.empty:
        return np.array([], dtype=float)
    for column in columns:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return np.array([], dtype=float)


def _nonzero_fraction(values: np.ndarray, *, atol: float = 1e-9) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(np.abs(finite) > float(atol)))


def _sanitize_identifier(value: str) -> str:
    sanitized = re.sub(r"[^\w.-]+", "_", str(value).strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized


def normalize_corrected_lineage_root(identifier: str | None) -> str:
    if not identifier:
        return ""
    cleaned = str(identifier).strip()
    cleaned = re.sub(
        r"(?:__|_)(?:corrected|retuned)(?:[_-]?(?:control[_-]?lut|waveform|formula))?(?:[_-]?iter(?:ation)?\d+)?$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"(?:__|_)(?:corrected|retuned)_?iter(?:ation)?\d+$", "", cleaned, flags=re.IGNORECASE)
    return _sanitize_identifier(cleaned)


def parse_corrected_iteration_index(identifier: str | None) -> int | None:
    if not identifier:
        return None
    match = re.search(r"(?:corrected|retuned)[_-]?iter(?:ation)?0*(\d+)", str(identifier), flags=re.IGNORECASE)
    if match:
        return max(int(match.group(1)), 1)
    if re.search(r"(?:corrected|retuned)", str(identifier), flags=re.IGNORECASE):
        return 1
    return None


def build_corrected_lut_id(
    *,
    lut_id: str | None,
    original_recommendation_id: str | None,
    iteration_index: int,
) -> str:
    lineage_root = normalize_corrected_lineage_root(original_recommendation_id or lut_id or "validation_retune")
    if not lineage_root:
        lineage_root = "validation_retune"
    return f"{lineage_root}__corrected_iter{max(int(iteration_index), 1):02d}"


def build_quality_badge_markdown() -> str:
    thresholds = QUALITY_BADGE_POLICY["thresholds"]
    labels = QUALITY_BADGE_POLICY["labels"]
    return "\n".join(
        [
            "## Quality Badge Rule",
            f"- metric domain: `{QUALITY_BADGE_POLICY['metric_domain']}` (`bz_mT`, global rule `bz_effective = -bz_raw`)",
            f"- `{labels['good']}`: Bz NRMSE <= `{thresholds['repro_good_max_nrmse']:.2f}`, shape corr >= `{thresholds['repro_good_min_shape_corr']:.2f}`, |phase lag| <= `{thresholds['repro_good_max_phase_lag_s']:.2f}s`, clipping/saturation 없음",
            f"- `{labels['caution']}`: green 기준은 벗어나지만 Bz NRMSE <= `{thresholds['caution_max_nrmse']:.2f}`, shape corr >= `{thresholds['caution_min_shape_corr']:.2f}`, |phase lag| <= `{thresholds['caution_max_phase_lag_s']:.2f}s`, clipping/saturation 없음",
            f"- `{labels['retune']}`: clipping/saturation 감지 또는 Bz NRMSE > `{thresholds['caution_max_nrmse']:.2f}` 또는 shape corr < `{thresholds['caution_min_shape_corr']:.2f}` 또는 |phase lag| > `{thresholds['caution_max_phase_lag_s']:.2f}s`",
        ]
    )


def _infer_source_kind(*, lut_id: str, profile_path: str | None) -> str:
    lowered_id = str(lut_id or "").strip().lower()
    lowered_path = str(profile_path or "").replace("\\", "/").lower()
    if parse_corrected_iteration_index(lowered_id) is not None or "/validation_retune/" in lowered_path:
        return SOURCE_KIND_CORRECTED
    if "/recommendation_library/" in lowered_path:
        return SOURCE_KIND_RECOMMENDATION
    if "/export_validation/" in lowered_path or lowered_id.startswith("control_formula_"):
        return SOURCE_KIND_EXPORT
    return SOURCE_KIND_UNKNOWN


def normalize_retune_source_selection(
    *,
    export_file_prefix: str,
    original_recommendation_id: str | None = None,
    source_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selection = dict(source_selection or {})
    profile_path = (
        selection.get("profile_csv_path")
        or selection.get("source_profile_path")
        or selection.get("waveform_csv_path")
    )
    control_lut_path = selection.get("control_lut_path")
    lut_id = str(
        selection.get("lut_id")
        or selection.get("source_lut_id")
        or selection.get("source_id")
        or selection.get("corrected_lut_id")
        or original_recommendation_id
        or export_file_prefix
    )
    source_kind = str(selection.get("source_kind") or _infer_source_kind(lut_id=lut_id, profile_path=profile_path)).strip().lower()
    source_lut_filename = selection.get("source_lut_filename")
    if not source_lut_filename:
        candidate_path = profile_path or control_lut_path
        if candidate_path:
            source_lut_filename = Path(str(candidate_path)).name
    original_id = selection.get("original_recommendation_id") or original_recommendation_id
    if not original_id:
        original_id = normalize_corrected_lineage_root(lut_id) if source_kind == SOURCE_KIND_CORRECTED else lut_id
    return {
        "source_kind": source_kind or SOURCE_KIND_UNKNOWN,
        "source_selection_id": str(selection.get("selection_id") or f"{source_kind or SOURCE_KIND_UNKNOWN}::{lut_id}"),
        "lut_id": lut_id,
        "source_profile_path": str(profile_path) if profile_path else None,
        "control_lut_path": str(control_lut_path) if control_lut_path else None,
        "source_lut_filename": str(source_lut_filename) if source_lut_filename else None,
        "original_recommendation_id": str(original_id) if original_id else None,
        "iteration_index": selection.get("iteration_index"),
    }


def _resolve_iteration_index(
    *,
    explicit_iteration_index: int | None,
    source_selection: dict[str, Any],
) -> int:
    if explicit_iteration_index is not None:
        return max(int(explicit_iteration_index), 1)
    existing = source_selection.get("iteration_index")
    if existing is not None:
        return max(int(existing), 1)
    source_kind = str(source_selection.get("source_kind") or SOURCE_KIND_UNKNOWN)
    source_identifier = (
        source_selection.get("lut_id")
        or source_selection.get("source_selection_id")
        or source_selection.get("original_recommendation_id")
    )
    parsed = parse_corrected_iteration_index(str(source_identifier) if source_identifier is not None else None)
    if source_kind == SOURCE_KIND_CORRECTED:
        return (parsed or 0) + 1
    return parsed or 1


def infer_exact_path(*, base_profile: pd.DataFrame, target_output_type: str) -> str:
    finite_cycle_mode = bool(
        base_profile["finite_cycle_mode"].iloc[0]
        if not base_profile.empty and "finite_cycle_mode" in base_profile.columns
        else False
    )
    if finite_cycle_mode:
        return EXACT_PATH_FINITE
    return EXACT_PATH_FIELD if str(target_output_type).strip().lower() == "field" else EXACT_PATH_CURRENT


def build_correction_rule(
    *,
    correction_gain: float,
    max_iterations: int,
    improvement_threshold: float,
    mode: str = "validation_residual_recommendation_loop",
) -> str:
    return (
        f"{mode}[correction_gain={float(correction_gain):.4g};"
        f"max_iterations={int(max_iterations)};"
        f"improvement_threshold={float(improvement_threshold):.4g}]"
    )


def build_validation_run(
    *,
    base_profile: pd.DataFrame,
    validation_candidate: dict[str, Any],
    export_file_prefix: str,
    target_output_type: str,
    original_recommendation_id: str | None = None,
    source_selection: dict[str, Any] | None = None,
    iteration_index: int | None = None,
    correction_rule: str | None = None,
) -> ValidationRun:
    source_info = normalize_retune_source_selection(
        export_file_prefix=export_file_prefix,
        original_recommendation_id=original_recommendation_id,
        source_selection=source_selection,
    )
    waveform_type = _first_frame_text(base_profile, "waveform_type")
    freq_hz = _first_frame_numeric(base_profile, "freq_hz")
    commanded_cycles = _first_frame_numeric(base_profile, "target_cycle_count")
    finite_cycle_mode = bool(
        base_profile["finite_cycle_mode"].iloc[0]
        if not base_profile.empty and "finite_cycle_mode" in base_profile.columns
        else False
    )
    target_level_value, target_level_kind = _resolve_target_level(base_profile, target_output_type)
    selected_validation_test_id = str(validation_candidate.get("test_id") or "unknown")
    source_file = validation_candidate.get("source_file")
    measured_file_name = None
    parsed = validation_candidate.get("parsed")
    if parsed is not None:
        measured_file_name = getattr(parsed, "source_file", None)
    created_at = iso_now()
    validation_run_id = f"{selected_validation_test_id}::{created_at}"
    resolved_iteration_index = _resolve_iteration_index(
        explicit_iteration_index=iteration_index,
        source_selection=source_info,
    )
    corrected_lut_id = build_corrected_lut_id(
        lut_id=source_info.get("lut_id"),
        original_recommendation_id=source_info.get("original_recommendation_id"),
        iteration_index=resolved_iteration_index,
    )
    exact_path = infer_exact_path(base_profile=base_profile, target_output_type=target_output_type)
    return ValidationRun(
        export_file_prefix=export_file_prefix,
        lut_id=str(source_info["lut_id"]),
        source_kind=str(source_info["source_kind"]),
        source_selection_id=str(source_info["source_selection_id"]),
        source_lut_filename=source_info.get("source_lut_filename"),
        source_profile_path=source_info.get("source_profile_path"),
        original_recommendation_id=source_info.get("original_recommendation_id"),
        validation_run_id=validation_run_id,
        corrected_lut_id=corrected_lut_id,
        iteration_index=resolved_iteration_index,
        exact_path=exact_path,
        target_output_type=str(target_output_type),
        waveform_type=waveform_type,
        freq_hz=freq_hz,
        commanded_cycles=commanded_cycles,
        finite_cycle_mode=finite_cycle_mode,
        target_level_value=target_level_value,
        target_level_kind=target_level_kind,
        selected_validation_test_id=selected_validation_test_id,
        selected_validation_source_file=str(source_file) if source_file is not None else None,
        measured_file_name=str(measured_file_name) if measured_file_name is not None else None,
        created_at=created_at,
        correction_rule=str(correction_rule or "validation_residual_recommendation_loop"),
        metadata={
            "candidate_score": float(validation_candidate.get("score", np.nan)),
            "candidate_eligible": bool(validation_candidate.get("eligible", False)),
            "candidate_freq_hz": _safe_float(validation_candidate.get("freq_hz")),
            "candidate_output_pp": _safe_float(validation_candidate.get("output_pp")),
            "candidate_label": str(validation_candidate.get("label") or ""),
            "source_control_lut_path": source_info.get("control_lut_path"),
        },
    )





__all__ = [name for name in globals() if not name.startswith('__')]

