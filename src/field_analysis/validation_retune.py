
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



def _copy_frame_with_attrs(frame: pd.DataFrame) -> pd.DataFrame:
    copied = frame.copy()
    copied.attrs = dict(getattr(frame, "attrs", {}))
    return copied


def _prediction_objective_audit(
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
) -> tuple[bool, str | None, float | None]:
    if str(request_route) != "exact" or str(target_output_type) != "current" or str(solver_route) != "finite_exact_direct":
        return False, None, None
    try:
        target_column, _ = _resolve_target_output_column(command_profile)
        expected_column = _resolve_expected_output_column(command_profile)
    except KeyError:
        return False, None, None
    target_output = pd.to_numeric(command_profile[target_column], errors="coerce").to_numpy(dtype=float)
    expected_output = pd.to_numeric(command_profile[expected_column], errors="coerce").to_numpy(dtype=float)
    if target_output.size == 0 or expected_output.size == 0:
        return False, None, None
    leak_corr = _correlation(target_output, expected_output)
    target_pp = _peak_to_peak(target_output)
    expected_pp = _peak_to_peak(expected_output)
    pp_ratio = (
        float(expected_pp / target_pp)
        if np.isfinite(expected_pp) and np.isfinite(target_pp) and float(target_pp) > 1e-9
        else float("nan")
    )
    if np.isfinite(leak_corr) and leak_corr >= 0.999 and np.isfinite(pp_ratio) and 0.95 <= pp_ratio <= 1.05:
        return True, "expected_output_matches_target_template", float(leak_corr)
    return False, None, float(leak_corr) if np.isfinite(leak_corr) else None


def build_prediction_debug_snapshot(
    *,
    command_profile: pd.DataFrame,
    validation_frame: pd.DataFrame | None,
    target_output_type: str,
    current_channel: str,
    field_channel: str,
) -> dict[str, Any]:
    return build_prediction_debug_from_profile(
        command_profile=command_profile,
        validation_frame=validation_frame,
        target_output_type=target_output_type,
        current_channel=current_channel,
        field_channel=field_channel,
    )



def _valid_time_signal(frame: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    if frame.empty or "time_s" not in frame.columns or column not in frame.columns:
        return np.array([], dtype=float), np.array([], dtype=float)
    time_values = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
    signal_values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(time_values) & np.isfinite(signal_values)
    if valid.sum() < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    ordered = np.argsort(time_values[valid])
    return time_values[valid][ordered], signal_values[valid][ordered]


def _normalized_peak_ratio(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_pp = _peak_to_peak(reference)
    candidate_pp = _peak_to_peak(candidate)
    if not np.isfinite(reference_pp) or not np.isfinite(candidate_pp) or reference_pp <= 1e-9 or candidate_pp <= 1e-9:
        return float("nan")
    return float(min(candidate_pp / reference_pp, reference_pp / candidate_pp))


def _max_aligned_shape_score(reference: np.ndarray, candidate: np.ndarray) -> float:
    valid = np.isfinite(reference) & np.isfinite(candidate)
    if valid.sum() < 8:
        return float("nan")
    ref = reference[valid] - float(np.nanmean(reference[valid]))
    comp = candidate[valid] - float(np.nanmean(candidate[valid]))
    ref_std = float(np.nanstd(ref))
    comp_std = float(np.nanstd(comp))
    if ref_std <= 1e-12 or comp_std <= 1e-12:
        return float("nan")
    correlation = np.correlate(comp, ref, mode="full")
    denom = float(len(ref) * ref_std * comp_std)
    if not np.isfinite(denom) or denom <= 1e-12:
        return float("nan")
    return float(np.clip(np.nanmax(correlation) / denom, -1.0, 1.0))


def _shift_signal_by_seconds(signal: np.ndarray, lag_s: float, time_grid: np.ndarray) -> np.ndarray:
    if len(signal) != len(time_grid):
        return signal
    if not np.isfinite(lag_s) or abs(float(lag_s)) <= 1e-12:
        return signal
    valid = np.isfinite(signal) & np.isfinite(time_grid)
    if valid.sum() < 2:
        return signal
    shifted = np.full_like(signal, np.nan, dtype=float)
    shifted_source_time = time_grid[valid] - float(lag_s)
    shifted[valid] = np.interp(
        shifted_source_time,
        time_grid[valid],
        signal[valid],
        left=0.0,
        right=0.0,
    )
    return shifted


def _is_signal_stable(values: np.ndarray, *, min_valid: int = 16, min_pp: float = 1e-6) -> bool:
    finite_values = values[np.isfinite(values)]
    if finite_values.size < min_valid:
        return False
    signal_pp = _peak_to_peak(finite_values)
    signal_std = float(np.nanstd(finite_values)) if finite_values.size else float("nan")
    return bool(np.isfinite(signal_pp) and signal_pp > min_pp and np.isfinite(signal_std) and signal_std > min_pp / 4.0)


def _canonicalize_validation_frame(
    *,
    base_profile: pd.DataFrame,
    validation_frame: pd.DataFrame,
    target_output_type: str,
    current_channel: str,
    field_channel: str,
) -> pd.DataFrame:
    canonical = _copy_frame_with_attrs(validation_frame)
    canonical.attrs["validation_window"] = {
        "applied": False,
        "start_s": 0.0,
        "end_s": _infer_fit_end_s(validation_frame),
        "score": float("nan"),
        "output_column": field_channel if str(target_output_type) == "field" else current_channel,
    }
    output_column = field_channel if str(target_output_type) == "field" else current_channel
    if canonical.empty or "time_s" not in canonical.columns or output_column not in canonical.columns:
        return canonical

    fit_end_s = _infer_fit_end_s(base_profile)
    if not np.isfinite(fit_end_s) or fit_end_s <= 0:
        return canonical
    target_column, _ = _resolve_target_output_column(base_profile)
    validation_time, validation_output = _valid_time_signal(canonical, output_column)
    if len(validation_time) < 16:
        return canonical
    if validation_time[-1] <= float(fit_end_s) + 1e-9:
        in_window = canonical["time_s"].between(0.0, float(fit_end_s), inclusive="both")
        cropped = _copy_frame_with_attrs(canonical.loc[in_window].copy())
        cropped.attrs["validation_window"] = {
            "applied": False,
            "start_s": 0.0,
            "end_s": float(fit_end_s),
            "score": float("nan"),
            "output_column": output_column,
        }
        return cropped if not cropped.empty else canonical

    sample_count = max(256, min(len(base_profile), len(validation_time)) * 2)
    time_grid = np.linspace(0.0, float(fit_end_s), sample_count)
    target_output = _interpolate_column(base_profile, target_column, time_grid)
    if not _is_signal_stable(target_output, min_pp=1e-3):
        return canonical

    candidate_starts = validation_time[validation_time <= validation_time[-1] - float(fit_end_s) + 1e-9]
    if candidate_starts.size == 0:
        return canonical
    stride = max(int(candidate_starts.size // 400), 1)
    best_start = float(candidate_starts[0])
    best_score = float("-inf")
    for start_s in candidate_starts[::stride]:
        measured = np.interp(time_grid + float(start_s), validation_time, validation_output)
        aligned_corr = _max_aligned_shape_score(target_output, measured)
        amplitude_ratio = _normalized_peak_ratio(target_output, measured)
        score = float((aligned_corr if np.isfinite(aligned_corr) else -1.0) + 0.35 * (amplitude_ratio if np.isfinite(amplitude_ratio) else 0.0))
        if score > best_score:
            best_score = score
            best_start = float(start_s)

    window_end_s = best_start + float(fit_end_s)
    mask = pd.to_numeric(canonical["time_s"], errors="coerce").between(best_start - 1e-9, window_end_s + 1e-9, inclusive="both")
    if not mask.any():
        return canonical
    cropped = _copy_frame_with_attrs(canonical.loc[mask].copy())
    cropped["time_s"] = pd.to_numeric(cropped["time_s"], errors="coerce") - float(best_start)
    cropped.attrs["validation_window"] = {
        "applied": bool(best_start > 1e-9),
        "start_s": float(best_start),
        "end_s": float(window_end_s),
        "score": float(best_score),
        "output_column": output_column,
    }
    return cropped


def _estimate_signal_scale(source_signal: np.ndarray, target_signal: np.ndarray) -> float:
    valid = np.isfinite(source_signal) & np.isfinite(target_signal)
    if valid.sum() < 8:
        return float("nan")
    centered_source = source_signal[valid] - float(np.nanmean(source_signal[valid]))
    centered_target = target_signal[valid] - float(np.nanmean(target_signal[valid]))
    denom = float(np.dot(centered_source, centered_source))
    if not np.isfinite(denom) or denom <= 1e-12:
        return float("nan")
    return float(np.dot(centered_source, centered_target) / denom)


def _project_signal_from_reference_transfer(
    *,
    reference_profile: pd.DataFrame,
    corrected_profile: pd.DataFrame,
    reference_signal_column: str,
) -> np.ndarray | None:
    reference_voltage_column = "limited_voltage_v" if "limited_voltage_v" in reference_profile.columns else "recommended_voltage_v"
    corrected_voltage_column = "limited_voltage_v" if "limited_voltage_v" in corrected_profile.columns else "recommended_voltage_v"
    if (
        reference_voltage_column not in reference_profile.columns
        or corrected_voltage_column not in corrected_profile.columns
        or "time_s" not in reference_profile.columns
        or "time_s" not in corrected_profile.columns
        or reference_signal_column not in reference_profile.columns
    ):
        return None

    profile_time = pd.to_numeric(corrected_profile["time_s"], errors="coerce").to_numpy(dtype=float)
    corrected_voltage = pd.to_numeric(corrected_profile[corrected_voltage_column], errors="coerce").to_numpy(dtype=float)
    reference_voltage = _interpolate_column(reference_profile, reference_voltage_column, profile_time)
    reference_signal = _interpolate_column(reference_profile, reference_signal_column, profile_time)
    if not _is_signal_stable(reference_voltage, min_pp=1e-3) or not _is_signal_stable(reference_signal, min_pp=1e-3):
        return None

    corrected_voltage_centered = corrected_voltage - float(np.nanmean(corrected_voltage))
    reference_voltage_centered = reference_voltage - float(np.nanmean(reference_voltage))
    reference_signal_centered = reference_signal - float(np.nanmean(reference_signal))
    reference_voltage_fft = np.fft.rfft(reference_voltage_centered)
    reference_signal_fft = np.fft.rfft(reference_signal_centered)
    corrected_voltage_fft = np.fft.rfft(corrected_voltage_centered)
    valid_transfer_mask = np.abs(reference_voltage_fft) > 1e-9
    if int(np.count_nonzero(valid_transfer_mask)) < 2:
        return None
    transfer = np.zeros_like(reference_signal_fft, dtype=np.complex128)
    transfer[valid_transfer_mask] = reference_signal_fft[valid_transfer_mask] / reference_voltage_fft[valid_transfer_mask]
    projected_signal = np.fft.irfft(corrected_voltage_fft * transfer, n=len(profile_time))
    return projected_signal if _is_signal_stable(projected_signal, min_pp=1e-3) else None





def _ensure_predicted_output_from_reference_transfer(
    *,
    reference_profile: pd.DataFrame,
    corrected_profile: pd.DataFrame,
    target_output_type: str,
) -> pd.DataFrame:
    enriched = _copy_frame_with_attrs(corrected_profile)
    if str(target_output_type) == "field":
        existing = pd.to_numeric(enriched.get("expected_field_mT", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
        if _is_signal_stable(existing, min_pp=1e-3):
            return enriched
        reference_signal_column = "expected_field_mT" if "expected_field_mT" in reference_profile.columns else "expected_output"
        projected_signal = _project_signal_from_reference_transfer(
            reference_profile=reference_profile,
            corrected_profile=enriched,
            reference_signal_column=reference_signal_column,
        )
        if projected_signal is None:
            return enriched
        enriched["expected_field_mT"] = projected_signal
        enriched["expected_output"] = projected_signal
        enriched["modeled_field_mT"] = projected_signal
        enriched["modeled_output"] = projected_signal
        return enriched

    existing = pd.to_numeric(enriched.get("expected_current_a", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    if _is_signal_stable(existing, min_pp=1e-3):
        return enriched
    reference_signal_column = "expected_current_a" if "expected_current_a" in reference_profile.columns else "expected_output"
    projected_signal = _project_signal_from_reference_transfer(
        reference_profile=reference_profile,
        corrected_profile=enriched,
        reference_signal_column=reference_signal_column,
    )
    if projected_signal is None:
        return enriched
    enriched["expected_current_a"] = projected_signal
    enriched["expected_output"] = projected_signal
    enriched["modeled_current_a"] = projected_signal
    enriched["modeled_output"] = projected_signal
    return enriched


def _ensure_bz_target_mapping(
    *,
    reference_profile: pd.DataFrame,
    validation_frame: pd.DataFrame,
    target_output_type: str,
    current_channel: str,
    field_channel: str,
) -> pd.DataFrame:
    enriched = _copy_frame_with_attrs(reference_profile)
    try:
        target_field = pd.to_numeric(enriched["target_field_mT"], errors="coerce").to_numpy(dtype=float)
        if _is_signal_stable(target_field, min_pp=1e-3):
            enriched.attrs["bz_target_mapping"] = {
                "available": True,
                "reason_code": None,
                "basis": "target_field_mT",
            }
            return enriched
    except KeyError:
        pass

    target_column, _ = _resolve_target_output_column(enriched)
    if field_channel not in validation_frame.columns:
        enriched.attrs["bz_target_mapping"] = {
            "available": False,
            "reason_code": "missing_bz_channel",
            "basis": "target_output",
        }
        return enriched
    drive_column = field_channel if str(target_output_type) == "field" else current_channel
    if drive_column not in validation_frame.columns:
        enriched.attrs["bz_target_mapping"] = {
            "available": False,
            "reason_code": "invalid_target_mapping",
            "basis": "target_output",
        }
        return enriched

    profile_time = pd.to_numeric(enriched["time_s"], errors="coerce").to_numpy(dtype=float)
    target_output = pd.to_numeric(enriched[target_column], errors="coerce").to_numpy(dtype=float)
    actual_drive = _interpolate_column(validation_frame, drive_column, profile_time)
    actual_bz = _interpolate_column(validation_frame, field_channel, profile_time)
    if not _is_signal_stable(actual_drive, min_pp=1e-3) or not _is_signal_stable(actual_bz, min_pp=1e-3):
        enriched.attrs["bz_target_mapping"] = {
            "available": False,
            "reason_code": "insufficient_active_window",
            "basis": "target_output",
        }
        return enriched

    scale = _estimate_signal_scale(actual_drive, actual_bz)
    phase_lag_s = _estimate_phase_lag_seconds(actual_drive, actual_bz, profile_time)
    mapped_target = _shift_signal_by_seconds(target_output * scale, phase_lag_s, profile_time)
    if not np.isfinite(scale) or not _is_signal_stable(mapped_target, min_pp=1e-3):
        enriched.attrs["bz_target_mapping"] = {
            "available": False,
            "reason_code": "surrogate_unstable",
            "basis": "target_output",
        }
        return enriched

    enriched["mapped_target_bz_effective_mT"] = mapped_target
    enriched["mapped_target_bz_scale"] = float(scale)
    enriched["mapped_target_bz_phase_lag_s"] = float(phase_lag_s) if np.isfinite(phase_lag_s) else float("nan")
    enriched.attrs["bz_target_mapping"] = {
        "available": True,
        "reason_code": None,
        "basis": "mapped_target_bz_effective_mT",
        "scale": float(scale),
        "phase_lag_s": float(phase_lag_s) if np.isfinite(phase_lag_s) else None,
    }
    return enriched


def _project_bz_from_validation_transfer(
    *,
    reference_profile: pd.DataFrame,
    corrected_profile: pd.DataFrame,
    validation_frame: pd.DataFrame,
    field_channel: str,
) -> pd.DataFrame:
    enriched = _copy_frame_with_attrs(corrected_profile)
    for column in ("expected_field_mT", "modeled_field_mT", "support_scaled_field_mT"):
        if column in enriched.columns:
            candidate = pd.to_numeric(enriched[column], errors="coerce").to_numpy(dtype=float)
            if _is_signal_stable(candidate, min_pp=1e-3):
                enriched.attrs["bz_projection"] = {
                    "available": True,
                    "reason_code": None,
                    "source": column,
                }
                return enriched

    if field_channel not in validation_frame.columns or "daq_input_v" not in validation_frame.columns:
        projected_bz = _project_signal_from_reference_transfer(
            reference_profile=reference_profile,
            corrected_profile=enriched,
            reference_signal_column=_resolve_bz_expected_column(reference_profile),
        )
        if projected_bz is not None:
            enriched["expected_field_mT"] = projected_bz
            enriched["modeled_field_mT"] = projected_bz
            enriched.attrs["bz_projection"] = {
                "available": True,
                "reason_code": None,
                "source": "reference_voltage_to_bz_transfer",
            }
            return enriched
        enriched.attrs["bz_projection"] = {
            "available": False,
            "reason_code": "missing_bz_channel",
            "source": "validation_transfer",
        }
        return enriched

    voltage_column = "limited_voltage_v" if "limited_voltage_v" in enriched.columns else "recommended_voltage_v"
    if voltage_column not in enriched.columns or "time_s" not in enriched.columns:
        enriched.attrs["bz_projection"] = {
            "available": False,
            "reason_code": "other",
            "source": "validation_transfer",
        }
        return enriched

    profile_time = pd.to_numeric(enriched["time_s"], errors="coerce").to_numpy(dtype=float)
    corrected_voltage = pd.to_numeric(enriched[voltage_column], errors="coerce").to_numpy(dtype=float)
    validation_voltage = _interpolate_column(validation_frame, "daq_input_v", profile_time)
    validation_bz = _interpolate_column(validation_frame, field_channel, profile_time)
    if not _is_signal_stable(validation_voltage, min_pp=1e-3) or not _is_signal_stable(validation_bz, min_pp=1e-3):
        projected_bz = _project_signal_from_reference_transfer(
            reference_profile=reference_profile,
            corrected_profile=enriched,
            reference_signal_column=_resolve_bz_expected_column(reference_profile),
        )
        if projected_bz is not None:
            enriched["expected_field_mT"] = projected_bz
            enriched["modeled_field_mT"] = projected_bz
            enriched.attrs["bz_projection"] = {
                "available": True,
                "reason_code": None,
                "source": "reference_voltage_to_bz_transfer",
            }
            return enriched
        enriched.attrs["bz_projection"] = {
            "available": False,
            "reason_code": "insufficient_active_window",
            "source": "validation_transfer",
        }
        return enriched

    corrected_voltage_centered = corrected_voltage - float(np.nanmean(corrected_voltage))
    validation_voltage_centered = validation_voltage - float(np.nanmean(validation_voltage))
    validation_bz_centered = validation_bz - float(np.nanmean(validation_bz))
    validation_voltage_fft = np.fft.rfft(validation_voltage_centered)
    validation_bz_fft = np.fft.rfft(validation_bz_centered)
    corrected_voltage_fft = np.fft.rfft(corrected_voltage_centered)
    transfer = np.zeros_like(validation_bz_fft, dtype=np.complex128)
    valid_transfer_mask = np.abs(validation_voltage_fft) > 1e-9
    if int(np.count_nonzero(valid_transfer_mask)) < 2:
        projected_bz = _project_signal_from_reference_transfer(
            reference_profile=reference_profile,
            corrected_profile=enriched,
            reference_signal_column=_resolve_bz_expected_column(reference_profile),
        )
        if projected_bz is not None:
            enriched["expected_field_mT"] = projected_bz
            enriched["modeled_field_mT"] = projected_bz
            enriched.attrs["bz_projection"] = {
                "available": True,
                "reason_code": None,
                "source": "reference_voltage_to_bz_transfer",
            }
            return enriched
        enriched.attrs["bz_projection"] = {
            "available": False,
            "reason_code": "surrogate_unstable",
            "source": "validation_transfer",
        }
        return enriched
    transfer[valid_transfer_mask] = validation_bz_fft[valid_transfer_mask] / validation_voltage_fft[valid_transfer_mask]
    projected_bz = np.fft.irfft(corrected_voltage_fft * transfer, n=len(profile_time))
    if not _is_signal_stable(projected_bz, min_pp=1e-3):
        projected_bz = _project_signal_from_reference_transfer(
            reference_profile=reference_profile,
            corrected_profile=enriched,
            reference_signal_column=_resolve_bz_expected_column(reference_profile),
        )
        if projected_bz is not None:
            enriched["expected_field_mT"] = projected_bz
            enriched["modeled_field_mT"] = projected_bz
            enriched.attrs["bz_projection"] = {
                "available": True,
                "reason_code": None,
                "source": "reference_voltage_to_bz_transfer",
            }
            return enriched
        enriched.attrs["bz_projection"] = {
            "available": False,
            "reason_code": "surrogate_unstable",
            "source": "validation_transfer",
        }
        return enriched

    enriched["expected_field_mT"] = projected_bz
    enriched["modeled_field_mT"] = projected_bz
    enriched.attrs["bz_projection"] = {
        "available": True,
        "reason_code": None,
        "source": "validation_voltage_to_bz_transfer",
    }
    return enriched



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



def build_validation_comparison(
    *,
    command_profile: pd.DataFrame,
    validation_frame: pd.DataFrame,
    label: str,
    comparison_source: str,
    target_output_type: str,
    current_channel: str,
    field_channel: str,
    metric_domain: str = TARGET_OUTPUT_DOMAIN,
) -> ValidationComparison:
    if metric_domain == BZ_EFFECTIVE_DOMAIN:
        output_column = field_channel
        target_column, target_basis = _resolve_bz_target_column(command_profile)
        expected_column = _resolve_bz_expected_column(command_profile)
    else:
        output_column = field_channel if str(target_output_type) == "field" else current_channel
        target_column, target_basis = _resolve_target_output_column(command_profile)
        expected_column = _resolve_expected_output_column(command_profile)

    fit_end_s = _infer_fit_end_s(command_profile)
    validation_end_s = _infer_fit_end_s(validation_frame)
    end_s = min(fit_end_s, validation_end_s) if np.isfinite(validation_end_s) and validation_end_s > 0 else fit_end_s
    if not np.isfinite(end_s) or end_s <= 0:
        end_s = max(_safe_float(command_profile.get("time_s", pd.Series(dtype=float)).max()), 1.0)
    sample_count = max(256, min(len(command_profile), len(validation_frame)) * 4 if len(validation_frame) else len(command_profile) * 4)
    time_grid = np.linspace(0.0, float(end_s), max(int(sample_count), 256))

    target_output = _interpolate_column(command_profile, target_column, time_grid)
    comparison_output = (
        _interpolate_column(validation_frame, output_column, time_grid)
        if comparison_source == "actual"
        else _interpolate_column(command_profile, expected_column, time_grid)
    )
    error = comparison_output - target_output
    finite_error = error[np.isfinite(error)]
    target_pp = _peak_to_peak(target_output)
    rmse = float(np.sqrt(np.nanmean(np.square(finite_error)))) if finite_error.size else float("nan")
    denom = max(target_pp / 2.0, 1e-12) if np.isfinite(target_pp) and target_pp > 0 else float("nan")
    nrmse = rmse / denom if np.isfinite(denom) else float("nan")
    pp_error = _peak_to_peak(comparison_output) - target_pp if np.isfinite(target_pp) else float("nan")
    peak_error = float(np.nanmax(np.abs(finite_error))) if finite_error.size else float("nan")
    shape_corr = _correlation(target_output, comparison_output)
    phase_lag_s = _estimate_phase_lag_seconds(target_output, comparison_output, time_grid)
    clipping_detected = _detect_clipping(validation_frame, output_column) or _detect_hardware_gate_violation(command_profile)
    saturation_detected = _detect_clipping(validation_frame, "daq_input_v")
    metrics_available, unavailable_reason, reason_codes, valid_sample_count = _resolve_metric_status(
        metric_domain=metric_domain,
        target_basis=target_basis,
        comparison_source=comparison_source,
        output_column=output_column,
        validation_frame=validation_frame,
        target_output=target_output,
        comparison_output=comparison_output,
        nrmse=nrmse,
        shape_corr=shape_corr,
        phase_lag_s=phase_lag_s,
        clipping_detected=bool(clipping_detected),
        saturation_detected=bool(saturation_detected),
    )



    window_info = validation_frame.attrs.get("validation_window", {})

    return ValidationComparison(
        label=label,
        output_column=output_column,
        rmse=rmse,
        nrmse=nrmse,
        shape_corr=shape_corr,
        phase_lag_s=phase_lag_s,
        pp_error=pp_error,
        peak_error=peak_error,
        clipping_detected=bool(clipping_detected),
        saturation_detected=bool(saturation_detected),
        metric_domain=metric_domain,
        target_basis=target_basis,
        comparison_source=comparison_source,
        sample_count=int(len(time_grid)),
        fit_end_s=float(end_s),
        metrics_available=bool(metrics_available),
        unavailable_reason=unavailable_reason,
        reason_codes=reason_codes,
        valid_sample_count=valid_sample_count,
        active_window_start_s=_safe_float(window_info.get("start_s")),
        active_window_end_s=_safe_float(window_info.get("end_s")),
    )


def build_validation_overlay_frame(
    *,
    base_profile: pd.DataFrame,
    validation_frame: pd.DataFrame,
    corrected_profile: pd.DataFrame,
    target_output_type: str,
    current_channel: str,
    field_channel: str,
) -> pd.DataFrame:
    output_column = field_channel if str(target_output_type) == "field" else current_channel
    target_column, _ = _resolve_target_output_column(base_profile)
    predicted_column = _resolve_expected_output_column(base_profile)
    corrected_column = _resolve_expected_output_column(corrected_profile)
    fit_end_s = _infer_fit_end_s(base_profile)
    validation_end_s = _infer_fit_end_s(validation_frame)
    corrected_end_s = _infer_fit_end_s(corrected_profile)
    finite_candidates = [value for value in (fit_end_s, validation_end_s, corrected_end_s) if np.isfinite(value) and value > 0]
    end_s = min(finite_candidates) if finite_candidates else 1.0
    sample_count = max(256, min(len(base_profile), len(validation_frame), len(corrected_profile)) * 4 if len(validation_frame) and len(corrected_profile) else 256)
    time_grid = np.linspace(0.0, float(end_s), max(int(sample_count), 256))
    target_bz_column, _ = _resolve_bz_target_column(base_profile)
    predicted_bz_column = _resolve_bz_expected_column(base_profile)
    corrected_bz_column = _resolve_bz_expected_column(corrected_profile)
    return pd.DataFrame(
        {
            "time_s": time_grid,
            "target_output": _interpolate_column(base_profile, target_column, time_grid),
            "predicted_output": _interpolate_column(base_profile, predicted_column, time_grid),
            "actual_measured": _interpolate_column(validation_frame, output_column, time_grid),
            "corrected_prediction": _interpolate_column(corrected_profile, corrected_column, time_grid),
            "target_bz_effective": _interpolate_column(base_profile, target_bz_column, time_grid),
            "predicted_bz_effective": _interpolate_column(base_profile, predicted_bz_column, time_grid),
            "actual_bz_effective": _interpolate_column(validation_frame, field_channel, time_grid),
            "corrected_bz_effective": _interpolate_column(corrected_profile, corrected_bz_column, time_grid),
        }
    )



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



def _resolve_target_level(frame: pd.DataFrame, target_output_type: str) -> tuple[float | None, str | None]:
    if "target_output_pp" in frame.columns:
        target_pp = _first_frame_numeric(frame, "target_output_pp")
        if target_pp is not None and np.isfinite(target_pp):
            return float(target_pp), "pp"
    if str(target_output_type) == "field":
        field_signal = _frame_signal_peak(frame, "target_field_mT")
        return field_signal, "peak"
    current_signal = _frame_signal_peak(frame, "target_current_a")
    return current_signal, "peak"


def _resolve_target_output_column(frame: pd.DataFrame) -> tuple[str, str]:
    candidates = (
        ("aligned_used_target_output", "aligned_used_target_output"),
        ("used_target_output", "used_target_output"),
        ("aligned_target_output", "aligned_target_output"),
        ("target_output", "target_output"),
        ("target_current_a", "target_current_a"),
        ("target_field_mT", "target_field_mT"),
    )
    for column, basis in candidates:
        if column in frame.columns:
            return column, basis
    if "target_current_a" in frame.columns:
        return "target_current_a", "target_current_a"
    if "target_field_mT" in frame.columns:
        return "target_field_mT", "target_field_mT"
    raise KeyError("target output column unavailable")


def _resolve_expected_output_column(frame: pd.DataFrame) -> str:
    for column in ("expected_output", "aligned_expected_output", "modeled_output", "expected_current_a", "expected_field_mT"):
        if column in frame.columns:
            return column
    target_column, _ = _resolve_target_output_column(frame)
    return target_column


def _resolve_bz_target_column(frame: pd.DataFrame) -> tuple[str, str]:
    candidates = (
        ("aligned_target_field_mT", "aligned_target_field_mT"),
        ("target_field_mT", "target_field_mT"),
        ("mapped_target_bz_effective_mT", "mapped_target_bz_validation_transfer"),
        ("expected_field_mT", "expected_field_surrogate"),
        ("modeled_field_mT", "modeled_field_surrogate"),
        ("support_scaled_field_mT", "support_scaled_field_surrogate"),
        ("bz_effective_mT", "measured_bz_effective"),
        ("bz_mT", "measured_bz_effective"),
    )
    for column, basis in candidates:
        if column in frame.columns:
            return column, basis
    raise KeyError("bz target column unavailable")


def _resolve_bz_expected_column(frame: pd.DataFrame) -> str:
    for column in ("expected_field_mT", "modeled_field_mT", "support_scaled_field_mT", "target_field_mT", "bz_effective_mT", "bz_mT"):
        if column in frame.columns:
            return column
    target_column, _ = _resolve_bz_target_column(frame)
    return target_column


def _interpolate_column(frame: pd.DataFrame, column: str, time_grid: np.ndarray) -> np.ndarray:
    if frame.empty or column not in frame.columns or "time_s" not in frame.columns:
        return np.full_like(time_grid, np.nan, dtype=float)
    time_values = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
    signal_values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(time_values) & np.isfinite(signal_values)
    if valid.sum() < 2:
        return np.full_like(time_grid, np.nan, dtype=float)
    return np.interp(time_grid, time_values[valid], signal_values[valid])


def _infer_fit_end_s(frame: pd.DataFrame) -> float:
    if frame.empty or "time_s" not in frame.columns:
        return float("nan")
    if "is_active_target" in frame.columns:
        active_mask = frame["is_active_target"].fillna(False).astype(bool)
        if active_mask.any():
            active_time = pd.to_numeric(frame.loc[active_mask, "time_s"], errors="coerce").dropna()
            if not active_time.empty:
                return float(active_time.max())
    time_series = pd.to_numeric(frame["time_s"], errors="coerce").dropna()
    return float(time_series.max()) if not time_series.empty else float("nan")


def _peak_to_peak(values: np.ndarray) -> float:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return float("nan")
    return float(np.nanmax(finite_values) - np.nanmin(finite_values))


def _correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    valid = np.isfinite(reference) & np.isfinite(candidate)
    if valid.sum() < 3:
        return float("nan")
    ref = reference[valid] - float(np.nanmean(reference[valid]))
    comp = candidate[valid] - float(np.nanmean(candidate[valid]))
    ref_std = float(np.nanstd(ref))
    comp_std = float(np.nanstd(comp))
    if ref_std <= 1e-12 or comp_std <= 1e-12:
        return float("nan")
    return float(np.clip(np.corrcoef(ref, comp)[0, 1], -1.0, 1.0))


def _estimate_phase_lag_seconds(reference: np.ndarray, candidate: np.ndarray, time_grid: np.ndarray) -> float:
    valid = np.isfinite(reference) & np.isfinite(candidate)
    if valid.sum() < 4:
        return float("nan")
    ref = reference[valid] - float(np.nanmean(reference[valid]))
    comp = candidate[valid] - float(np.nanmean(candidate[valid]))
    if np.nanstd(ref) <= 1e-12 or np.nanstd(comp) <= 1e-12:
        return float("nan")
    correlation = np.correlate(comp, ref, mode="full")
    lag_index = int(np.argmax(correlation) - (len(ref) - 1))
    if len(time_grid) < 2:
        return float("nan")
    dt = float(np.nanmedian(np.diff(time_grid)))
    if not np.isfinite(dt) or dt <= 0:
        return float("nan")
    return float(lag_index * dt)


def _detect_clipping(frame: pd.DataFrame, column: str, repeat_ratio_threshold: float = 0.05) -> bool:
    if frame.empty or column not in frame.columns:
        return False
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size < 8:
        return False
    rounded = np.round(finite_values, 6)
    max_repeat_ratio = float(np.mean(np.isclose(rounded, np.nanmax(rounded), atol=1e-6)))
    min_repeat_ratio = float(np.mean(np.isclose(rounded, np.nanmin(rounded), atol=1e-6)))
    return bool(max(max_repeat_ratio, min_repeat_ratio) >= float(repeat_ratio_threshold))


def _first_frame_numeric(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.iloc[0])


def _first_frame_bool(frame: pd.DataFrame, column: str) -> bool | None:
    if frame.empty or column not in frame.columns:
        return None
    series = frame[column].dropna()
    if series.empty:
        return None
    value = series.iloc[0]
    if isinstance(value, (bool, np.bool_,)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _detect_hardware_gate_violation(
    frame: pd.DataFrame,
    *,
    gain_shortfall_tolerance_pct: float = SOFT_HARDWARE_LIMIT_TOLERANCE_PCT,
) -> bool:
    if frame.empty:
        return False
    within_hardware = _first_frame_bool(frame, "within_hardware_limits")
    if within_hardware is True:
        return False

    within_daq = _first_frame_bool(frame, "within_daq_limit")
    if within_daq is False:
        return True

    required_gain_pct = _first_frame_numeric(frame, "required_amp_gain_pct")
    available_gain_pct = _first_frame_numeric(frame, "available_amp_gain_pct")
    peak_input_limit_margin = _first_frame_numeric(frame, "peak_input_limit_margin")
    p95_input_limit_margin = _first_frame_numeric(frame, "p95_input_limit_margin")

    nonnegative_input_margin = True
    for margin in (peak_input_limit_margin, p95_input_limit_margin):
        if margin is not None and np.isfinite(margin) and margin < 0.0:
            nonnegative_input_margin = False
            break
    if not nonnegative_input_margin:
        return True

    if (
        required_gain_pct is not None
        and available_gain_pct is not None
        and np.isfinite(required_gain_pct)
        and np.isfinite(available_gain_pct)
    ):
        gain_shortfall_pct = float(required_gain_pct - available_gain_pct)
        if gain_shortfall_pct <= float(gain_shortfall_tolerance_pct) and nonnegative_input_margin:
            return False
        return gain_shortfall_pct > float(gain_shortfall_tolerance_pct)

    return bool(within_hardware is False)


def _first_frame_text(frame: pd.DataFrame, column: str) -> str | None:
    if frame.empty or column not in frame.columns:
        return None
    series = frame[column].dropna()
    if series.empty:
        return None
    return canonicalize_waveform_type(str(series.iloc[0])) or str(series.iloc[0])


def _frame_signal_peak(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return None
    return float(np.nanmax(np.abs(finite_values)))


def _safe_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric

