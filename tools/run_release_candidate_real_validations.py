from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.analysis import analyze_measurements, combine_analysis_frames  # noqa: E402
from field_analysis.canonicalize import CANONICAL_SCHEMA_VERSION, CanonicalizeConfig, canonicalize_batch  # noqa: E402
from field_analysis.models import CycleDetectionConfig, PreprocessConfig  # noqa: E402
from field_analysis.parser import parse_measurement_file  # noqa: E402
from field_analysis.recommendation_service import (  # noqa: E402
    LegacyRecommendationContext,
    RecommendationOptions,
    RecommendationResult,
    TargetRequest,
    recommend,
)
from field_analysis.schema_config import load_schema_config  # noqa: E402
from field_analysis.validation_retune import (  # noqa: E402
    SOURCE_KIND_EXPORT,
    SOURCE_KIND_RECOMMENDATION,
    build_prediction_debug_snapshot,
    execute_validation_retune,
    save_retune_artifacts,
)
from run_real_validation_retune import (  # noqa: E402
    CURRENT_CHANNEL,
    FIELD_CHANNEL,
    OUTPUT_DIR,
    VALIDATION_FILE_PATH as FINITE_VALIDATION_FILE_PATH,
    _build_exact_base_profile,
    _bz_effective_signal,
    _current_signal,
    _find_support_file,
    _measurement_time_seconds,
    _peak_to_peak,
    _read_measurement_csv,
    _voltage_signal,
)

try:  # noqa: E402
    from policy_validation_runtime import run_case as _policy_run_case
except Exception:  # pragma: no cover - runtime fallback for broken UI imports
    _policy_run_case = None


REAL_UPLOAD_DIR = REPO_ROOT / ".coil_analyzer" / "uploads"
SUMMARY_JSON_PATH = REPO_ROOT / "artifacts" / "bz_first_exact_matrix" / "real_validation_suite_result.json"
SUMMARY_MD_PATH = REPO_ROOT / "artifacts" / "bz_first_exact_matrix" / "real_validation_suite_result.md"
FIELD_PREDICTION_DEBUG_JSON_PATH = REPO_ROOT / "artifacts" / "bz_first_exact_matrix" / "field_prediction_debug_report.json"
FIELD_PREDICTION_DEBUG_MD_PATH = REPO_ROOT / "artifacts" / "bz_first_exact_matrix" / "field_prediction_debug_report.md"
EXACT_FIELD_ROUTE_AUDIT_JSON_PATH = REPO_ROOT / "artifacts" / "bz_first_exact_matrix" / "exact_field_route_audit.json"
EXACT_FIELD_ROUTE_AUDIT_MD_PATH = REPO_ROOT / "artifacts" / "bz_first_exact_matrix" / "exact_field_route_audit.md"
LEVEL_SENSITIVITY_JSON_PATH = REPO_ROOT / "artifacts" / "bz_first_exact_matrix" / "level_sensitivity_diagnosis.json"
LEVEL_SENSITIVITY_MD_PATH = REPO_ROOT / "artifacts" / "bz_first_exact_matrix" / "level_sensitivity_diagnosis.md"
FINITE_TRIANGLE_AUDIT_JSON_PATH = REPO_ROOT / "artifacts" / "bz_first_exact_matrix" / "finite_triangle_expected_source_audit.json"
FINITE_TRIANGLE_AUDIT_MD_PATH = REPO_ROOT / "artifacts" / "bz_first_exact_matrix" / "finite_triangle_expected_source_audit.md"
FALLBACK_HIERARCHY_JSON_PATH = REPO_ROOT / "artifacts" / "bz_first_exact_matrix" / "fallback_hierarchy_audit.json"
FALLBACK_HIERARCHY_MD_PATH = REPO_ROOT / "artifacts" / "bz_first_exact_matrix" / "fallback_hierarchy_audit.md"
REASON_CODE_BREAKDOWN_JSON_PATH = REPO_ROOT / "artifacts" / "bz_first_exact_matrix" / "reason_code_breakdown.json"
REASON_CODE_BREAKDOWN_MD_PATH = REPO_ROOT / "artifacts" / "bz_first_exact_matrix" / "reason_code_breakdown.md"
RETUNE_HISTORY_PATH = REPO_ROOT.parent / "outputs" / "field_analysis_app_state" / "validation_retune_history.json"
ARTIFACT_GENERATOR = REPO_ROOT / "tools" / "generate_bz_first_artifacts.py"
UPLOAD_STATE_DIR = REPO_ROOT.parent / "outputs" / "field_analysis_app_state" / "uploads"
CONTINUOUS_UPLOAD_DIR = UPLOAD_STATE_DIR / "continuous"
TRANSIENT_UPLOAD_DIR = UPLOAD_STATE_DIR / "transient"


CASE_CONFIGS: list[dict[str, Any]] = [
    {
        "case_id": "continuous_current_exact_real_0p5hz",
        "label": "continuous current exact real validation",
        "mode": "continuous",
        "file_path": REAL_UPLOAD_DIR / "6c92e4f555df4fe7b6106580b286b88c_0.5Hz_9V_38gain.csv",
        "waveform": "sine",
        "freq_hz": 0.5,
        "target_type": "current",
        "export_prefix": "real_continuous_current_exact_validation_0p5hz",
        "source_kind": SOURCE_KIND_RECOMMENDATION,
        "source_lut_id": "control_formula_sine_0.5Hz_current_20",
        "support_state_expected": "exact",
        "software_status": "operational",
    },
    {
        "case_id": "continuous_field_exact_real_0p25hz",
        "label": "continuous field exact real validation",
        "mode": "continuous",
        "file_path": REAL_UPLOAD_DIR / "a45d8cb65a6b4e36b21481394aec8c0b_0.25Hz_9V_36gain.csv",
        "waveform": "sine",
        "freq_hz": 0.25,
        "target_type": "field",
        "export_prefix": "real_continuous_field_exact_validation_0p25hz",
        "source_kind": SOURCE_KIND_RECOMMENDATION,
        "source_lut_id": "control_formula_sine_0.25Hz_field_20",
        "support_state_expected": "exact",
        "software_status": "software_ready_bench_pending",
    },
    {
        "case_id": "finite_exact_real_1p25hz_1p25cycle_20pp",
        "label": "finite exact real validation",
        "mode": "finite",
        "file_path": FINITE_VALIDATION_FILE_PATH,
        "waveform": "sine",
        "freq_hz": 1.25,
        "target_type": "current",
        "export_prefix": "real_finite_exact_validation_1p25hz_1p25cycle_20pp",
        "source_kind": SOURCE_KIND_EXPORT,
        "source_lut_id": "ff132682eb37f728_1.25hz_1.25cycle_20pp",
        "support_state_expected": "exact",
        "software_status": "operational",
    },
]


@dataclass(slots=True)
class BackendProbeRuntime:
    continuous_runs: list[Any]
    transient_runs: list[Any]
    validation_runs: list[Any]
    recommendation_options: RecommendationOptions
    legacy_context: LegacyRecommendationContext


def _iter_upload_files(upload_dir: Path) -> list[Path]:
    if not upload_dir.exists():
        return []
    return sorted(path for path in upload_dir.rglob("*") if path.is_file())


def _parse_measurements_from_dir(upload_dir: Path, *, schema: Any, expected_cycles: int) -> list[Any]:
    parsed_measurements: list[Any] = []
    for path in _iter_upload_files(upload_dir):
        try:
            parsed_measurements.extend(
                parse_measurement_file(
                    file_name=path.name,
                    file_bytes=path.read_bytes(),
                    schema=schema,
                    expected_cycles=expected_cycles,
                    target_current_mode=schema.target_current_mode,
                )
            )
        except Exception:
            continue
    return parsed_measurements


@lru_cache(maxsize=1)
def _load_backend_probe_runtime() -> BackendProbeRuntime:
    schema = load_schema_config(None)
    canonicalize_config = CanonicalizeConfig(
        preferred_field_axis=FIELD_CHANNEL,
        uniform_resample=True,
        custom_current_alpha=1.0,
        custom_current_beta=1.0,
    )
    continuous_measurements = _parse_measurements_from_dir(
        CONTINUOUS_UPLOAD_DIR,
        schema=schema,
        expected_cycles=int(schema.default_expected_cycles),
    )
    transient_measurements = _parse_measurements_from_dir(
        TRANSIENT_UPLOAD_DIR,
        schema=schema,
        expected_cycles=max(1, int(schema.default_expected_cycles)),
    )
    continuous_runs = canonicalize_batch(
        continuous_measurements,
        regime="continuous",
        role="train",
        config=canonicalize_config,
    )
    transient_runs = canonicalize_batch(
        transient_measurements,
        regime="transient",
        role="train",
        config=canonicalize_config,
    )
    preprocess_config = PreprocessConfig(
        custom_current_alpha=1.0,
        custom_current_beta=1.0,
        projection_vector=(0.0, 0.0, 1.0),
    )
    cycle_config = CycleDetectionConfig(
        reference_channel="daq_input_v",
        expected_cycles=int(schema.default_expected_cycles),
    )
    continuous_analyses = analyze_measurements(
        parsed_measurements=continuous_measurements,
        preprocess_config=preprocess_config,
        cycle_config=cycle_config,
        current_channel=CURRENT_CHANNEL,
        main_field_axis=FIELD_CHANNEL,
        canonical_runs=continuous_runs,
    )
    transient_analyses = analyze_measurements(
        parsed_measurements=transient_measurements,
        preprocess_config=preprocess_config,
        cycle_config=cycle_config,
        current_channel=CURRENT_CHANNEL,
        main_field_axis=FIELD_CHANNEL,
        canonical_runs=transient_runs,
    )
    _per_cycle_summary, per_test_summary, _coverage = combine_analysis_frames(
        analyses=continuous_analyses,
        reference_test_id=None,
        field_axis=FIELD_CHANNEL,
    )
    analysis_lookup = {
        analysis.parsed.normalized_frame["test_id"].iloc[0]: analysis
        for analysis in continuous_analyses
        if not analysis.parsed.normalized_frame.empty
    }
    legacy_context = LegacyRecommendationContext(
        per_test_summary=per_test_summary,
        analysis_lookup=analysis_lookup,
        transient_measurements=transient_measurements,
        transient_preprocess_results=[analysis.preprocess for analysis in transient_analyses],
        transient_canonical_runs=transient_runs,
        validation_measurements=[],
        validation_preprocess_results=[],
    )
    return BackendProbeRuntime(
        continuous_runs=continuous_runs,
        transient_runs=transient_runs,
        validation_runs=[],
        recommendation_options=RecommendationOptions(
            current_channel=CURRENT_CHANNEL,
            field_channel=FIELD_CHANNEL,
            max_daq_voltage_pp=20.0,
            amp_gain_at_100_pct=20.0,
            amp_gain_limit_pct=100.0,
            amp_max_output_pk_v=180.0,
            default_support_amp_gain_pct=100.0,
            allow_target_extrapolation=True,
            allow_output_extrapolation=True,
            frequency_mode="exact",
            preview_tail_cycles=0.25,
        ),
        legacy_context=legacy_context,
    )


def run_case(
    *,
    waveform: str,
    target_type: str,
    freq_hz: float,
    target_level: float,
    finite_cycle_mode: bool = False,
    target_cycle_count: float | None = None,
) -> tuple[RecommendationResult, dict[str, Any], BackendProbeRuntime | Any]:
    if _policy_run_case is not None:
        return _policy_run_case(
            waveform=waveform,
            target_type=target_type,
            freq_hz=freq_hz,
            target_level=target_level,
            finite_cycle_mode=finite_cycle_mode,
            target_cycle_count=target_cycle_count,
        )

    runtime = _load_backend_probe_runtime()
    result = recommend(
        continuous_runs=runtime.continuous_runs,
        transient_runs=runtime.transient_runs,
        validation_runs=runtime.validation_runs,
        target=TargetRequest(
            regime="transient" if finite_cycle_mode else "continuous",
            target_waveform=waveform,
            command_waveform=waveform,
            freq_hz=float(freq_hz),
            commanded_cycles=None if target_cycle_count is None else float(target_cycle_count),
            target_type=target_type,
            target_level_value=float(target_level),
            target_level_kind="pp",
            context={
                "request_kind": "waveform_compensation",
                "frequency_mode": "exact",
                "finite_cycle_mode": bool(finite_cycle_mode),
                "preview_tail_cycles": 0.25,
            },
        ),
        options=runtime.recommendation_options,
        legacy_context=runtime.legacy_context,
    )
    summary = {
        "reason": (
            str(result.validation_report.reasons[0])
            if result.validation_report is not None and result.validation_report.reasons
            else str(result.engine_summary.get("support_state") or "unknown")
        ),
        "state": str(result.engine_summary.get("support_state") or "unknown"),
        "status_label": str(result.engine_summary.get("support_state") or "unknown"),
        "finite_cycle_mode": bool(finite_cycle_mode),
        "apply_exact_enabled": False,
        "apply_provisional_enabled": False,
        "measurement_recommendations": [],
    }
    return result, summary, runtime


def _safe_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if pd.notna(numeric) else None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _sample_series(frame: pd.DataFrame | None, column: str, count: int = 12) -> list[float | None]:
    if frame is None or frame.empty or column not in frame.columns:
        return []
    series = pd.to_numeric(frame[column], errors="coerce").head(count)
    return [_safe_float(value) for value in series.to_list()]


def _extract_signal(frame: pd.DataFrame | None, columns: list[str]) -> np.ndarray:
    if frame is None or frame.empty:
        return np.array([], dtype=float)
    for column in columns:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return np.array([], dtype=float)


def _signal_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 8:
        return None
    left = left[valid]
    right = right[valid]
    if np.allclose(np.nanstd(left), 0.0) or np.allclose(np.nanstd(right), 0.0):
        return None
    return _safe_float(np.corrcoef(left, right)[0, 1])


def _signal_nrmse(reference: np.ndarray, candidate: np.ndarray) -> float | None:
    ref = np.asarray(reference, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    valid = np.isfinite(ref) & np.isfinite(cand)
    if valid.sum() < 8:
        return None
    ref = ref[valid]
    cand = cand[valid]
    ref_pp = float(np.nanmax(ref) - np.nanmin(ref))
    if not np.isfinite(ref_pp) or ref_pp <= 1e-9:
        return None
    return _safe_float(np.sqrt(np.mean(np.square(cand - ref))) / ref_pp)


def _resample_profile_signal(profile: pd.DataFrame | None, columns: list[str], count: int = 128) -> tuple[list[float], str | None, float | None]:
    if profile is None or profile.empty:
        return [], None, None
    signal_column = next((column for column in columns if column in profile.columns), None)
    if signal_column is None:
        return [], None, None
    if "cycle_progress" in profile.columns:
        x = pd.to_numeric(profile["cycle_progress"], errors="coerce").to_numpy(dtype=float)
    elif "time_s" in profile.columns:
        time_values = pd.to_numeric(profile["time_s"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(time_values).any():
            return [], signal_column, None
        duration = float(np.nanmax(time_values) - np.nanmin(time_values))
        if not np.isfinite(duration) or duration <= 1e-9:
            return [], signal_column, None
        x = (time_values - float(np.nanmin(time_values))) / duration
    else:
        return [], signal_column, None
    y = pd.to_numeric(profile[signal_column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 8:
        return [], signal_column, None
    ordered = np.argsort(x[valid])
    x_valid = x[valid][ordered]
    y_valid = y[valid][ordered]
    x_unique, unique_indices = np.unique(x_valid, return_index=True)
    y_unique = y_valid[unique_indices]
    if x_unique.size < 8:
        return [], signal_column, None
    if x_unique[0] > 0.0:
        x_unique = np.insert(x_unique, 0, 0.0)
        y_unique = np.insert(y_unique, 0, y_unique[0])
    if x_unique[-1] < 1.0:
        x_unique = np.append(x_unique, 1.0)
        y_unique = np.append(y_unique, y_unique[-1])
    grid = np.linspace(0.0, 1.0, max(int(count), 32))
    samples = np.interp(grid, x_unique, y_unique)
    pp = float(np.nanmax(samples) - np.nanmin(samples)) if np.isfinite(samples).any() else None
    return [float(value) for value in samples], signal_column, _safe_float(pp)


def _selected_support_id(result: Any) -> str | None:
    if not isinstance(getattr(result, "legacy_payload", None), dict):
        return None
    payload = result.legacy_payload
    selected = payload.get("selected_support_id")
    if selected:
        return str(selected)
    for key in ("support_test_id", "nearest_test_id"):
        value = payload.get(key)
        if value:
            return str(value)
    support_tests = payload.get("support_tests_used")
    if isinstance(support_tests, list) and support_tests:
        return str(support_tests[0])
    return None


def _runtime_prediction_snapshot(result: Any) -> dict[str, Any]:
    command_profile = getattr(result, "command_profile", None)
    if command_profile is None or command_profile.empty:
        return {}
    legacy_payload = result.legacy_payload if isinstance(getattr(result, "legacy_payload", None), dict) else {}
    if "target_output_type" in command_profile.columns:
        target_type_series = command_profile["target_output_type"].dropna()
        target_output_type = str(target_type_series.iloc[0]) if not target_type_series.empty else ("field" if "target_field_mT" in command_profile.columns else "current")
    else:
        target_output_type = "field" if "target_field_mT" in command_profile.columns else "current"
    debug = build_prediction_debug_snapshot(
        command_profile=command_profile,
        validation_frame=None,
        target_output_type=target_output_type,
        current_channel=CURRENT_CHANNEL,
        field_channel=FIELD_CHANNEL,
    )
    support_profile = legacy_payload.get("nearest_profile")
    support_shape_columns = ["measured_field_mT", "bz_mT"] if target_output_type == "field" else ["measured_current_a", "i_sum_signed"]
    support_shape_samples, support_shape_channel, support_shape_pp = _resample_profile_signal(
        support_profile if isinstance(support_profile, pd.DataFrame) else None,
        support_shape_columns,
    )
    return {
        **debug,
        "selected_support_id": _selected_support_id(result),
        "selected_support_family": legacy_payload.get("selected_support_family"),
        "support_selection_reason": legacy_payload.get("support_selection_reason"),
        "support_family_metric": legacy_payload.get("support_family_metric"),
        "support_family_value": _safe_float(legacy_payload.get("support_family_value")),
        "support_family_lock_applied": bool(legacy_payload.get("support_family_lock_applied", False)),
        "support_bz_to_current_ratio": _safe_float(legacy_payload.get("support_bz_to_current_ratio")),
        "support_amp_gain_pct": _safe_float(legacy_payload.get("support_amp_gain_pct")),
        "required_amp_gain_pct": _safe_float(legacy_payload.get("required_amp_gain_pct")),
        "available_amp_gain_pct": _safe_float(legacy_payload.get("available_amp_gain_pct")),
        "within_hardware_limits": bool(legacy_payload.get("within_hardware_limits", False)),
        "selected_support_shape_channel": support_shape_channel,
        "selected_support_shape_samples": support_shape_samples,
        "selected_support_shape_pp": support_shape_pp,
        "support_tests_used": (
            list(legacy_payload.get("support_tests_used", []))
            if legacy_payload
            else []
        ),
        "policy_reasons": list(getattr(result, "debug_info", {}).get("policy_reasons", [])),
        "predicted_shape_corr": _safe_float(getattr(result, "debug_info", {}).get("predicted_shape_corr")),
        "predicted_nrmse": _safe_float(getattr(result, "debug_info", {}).get("predicted_nrmse")),
        "predicted_phase_lag": _safe_float(getattr(result, "debug_info", {}).get("predicted_phase_lag")),
        "predicted_phase_lag_cycles": _safe_float(getattr(result, "debug_info", {}).get("predicted_phase_lag_cycles")),
        "predicted_clipping": bool(getattr(result, "debug_info", {}).get("predicted_clipping", False)),
        "allow_auto_download": bool(getattr(result, "allow_auto_download", False)),
        "preview_only": bool(getattr(result, "preview_only", True)),
    }


def _field_route_probe(target_level: float) -> dict[str, Any]:
    recommendation_result, route_summary, _runtime = run_case(
        waveform="sine",
        target_type="field",
        freq_hz=0.25,
        target_level=float(target_level),
    )
    snapshot = _runtime_prediction_snapshot(recommendation_result)
    return {
        "target_level_pp": float(target_level),
        "route_reason": str(route_summary.get("reason") or ""),
        "support_state": str(recommendation_result.engine_summary.get("support_state") or ""),
        **snapshot,
    }


def _finite_triangle_probe(target_level: float) -> dict[str, Any]:
    recommendation_result, route_summary, _runtime = run_case(
        waveform="triangle",
        target_type="current",
        freq_hz=1.25,
        target_level=float(target_level),
        finite_cycle_mode=True,
        target_cycle_count=1.25,
    )
    command_profile = recommendation_result.command_profile
    snapshot = _runtime_prediction_snapshot(recommendation_result)
    return {
        "target_level_pp": float(target_level),
        "route_reason": str(route_summary.get("reason") or ""),
        "support_state": str(recommendation_result.engine_summary.get("support_state") or ""),
        **snapshot,
        "target_output_preview": _sample_series(command_profile, "target_output"),
        "expected_output_preview": _sample_series(command_profile, "expected_output"),
        "expected_field_preview": _sample_series(command_profile, "expected_field_mT"),
}


def _diagnose_level_sensitivity(probes: list[dict[str, Any]]) -> dict[str, Any]:
    if len(probes) < 2:
        return {"diagnosis": "insufficient_samples"}
    first = probes[0]
    last = probes[-1]
    support_id_changed = first.get("selected_support_id") != last.get("selected_support_id")
    support_family_changed = first.get("selected_support_family") != last.get("selected_support_family")
    prediction_source_changed = (
        first.get("field_prediction_source") != last.get("field_prediction_source")
        or first.get("field_prediction_status") != last.get("field_prediction_status")
    )
    clipping_changed = bool(first.get("predicted_clipping")) != bool(last.get("predicted_clipping"))
    hardware_limit_changed = bool(first.get("within_hardware_limits")) != bool(last.get("within_hardware_limits"))
    amp_gain_changed = (
        _safe_float(first.get("support_amp_gain_pct")) is not None
        and _safe_float(last.get("support_amp_gain_pct")) is not None
        and not np.isclose(float(first["support_amp_gain_pct"]), float(last["support_amp_gain_pct"]), atol=1e-9)
    )
    limit_condition_changed = clipping_changed or hardware_limit_changed or amp_gain_changed
    shape_delta = None
    if _safe_float(first.get("predicted_shape_corr")) is not None and _safe_float(last.get("predicted_shape_corr")) is not None:
        shape_delta = abs(float(first["predicted_shape_corr"]) - float(last["predicted_shape_corr"]))
    support_shape_corr = _signal_corr(
        np.asarray(first.get("selected_support_shape_samples", []), dtype=float),
        np.asarray(last.get("selected_support_shape_samples", []), dtype=float),
    )
    support_shape_nrmse = _signal_nrmse(
        np.asarray(first.get("selected_support_shape_samples", []), dtype=float),
        np.asarray(last.get("selected_support_shape_samples", []), dtype=float),
    )
    support_shape_pp_ratio = None
    if _safe_float(first.get("selected_support_shape_pp")) is not None and _safe_float(last.get("selected_support_shape_pp")) is not None:
        first_pp = float(first["selected_support_shape_pp"])
        last_pp = float(last["selected_support_shape_pp"])
        if np.isfinite(first_pp) and abs(first_pp) > 1e-9:
            support_shape_pp_ratio = _safe_float(last_pp / first_pp)

    reason_codes: list[str] = []
    if support_id_changed or support_family_changed:
        reason_codes.append("support_id_switch")
    if prediction_source_changed:
        reason_codes.append("prediction_source_switch")
    if limit_condition_changed:
        reason_codes.append("limit_induced_switch")
    if support_shape_corr is not None and support_shape_corr < 0.92:
        reason_codes.append("true_nonlinear_shape_change")
    if reason_codes:
        diagnosis = reason_codes[0]
    elif shape_delta is not None and shape_delta > 0.25:
        diagnosis = "solver_or_objective_instability"
    else:
        diagnosis = "minor_runtime_variation"
    return {
        "diagnosis": diagnosis,
        "reason_codes": reason_codes,
        "same_solver_route": first.get("solver_route") == last.get("solver_route"),
        "same_selected_support_id": not support_id_changed,
        "same_selected_support_family": not support_family_changed,
        "same_field_prediction_source": not prediction_source_changed,
        "support_id_changed": support_id_changed,
        "support_family_changed": support_family_changed,
        "prediction_source_changed": prediction_source_changed,
        "limit_condition_changed": limit_condition_changed,
        "clipping_changed": clipping_changed,
        "support_shape_channel": first.get("selected_support_shape_channel") or last.get("selected_support_shape_channel"),
        "measured_support_shape_corr": support_shape_corr,
        "measured_support_shape_nrmse": support_shape_nrmse,
        "measured_support_shape_pp_ratio": support_shape_pp_ratio,
        "predicted_shape_corr_delta": shape_delta,
    }

def _record_retune_history(retune_summary: dict[str, Any]) -> None:
    history = _load_json(RETUNE_HISTORY_PATH)
    entries = history.get("retunes", [])
    if not isinstance(entries, list):
        entries = []
    retune_id = str(retune_summary.get("corrected_lut_id") or retune_summary.get("retune_id") or "")
    updated = [item for item in entries if str(item.get("corrected_lut_id") or item.get("retune_id") or "") != retune_id]
    updated.append(retune_summary)
    updated.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    _write_json(RETUNE_HISTORY_PATH, {"retunes": updated})


def _build_validation_frame(path: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    _, frame = _read_measurement_csv(path)
    time_s = _measurement_time_seconds(frame)
    current = _current_signal(frame)
    bz_effective = _bz_effective_signal(frame)
    voltage = _voltage_signal(frame)
    validation_frame = pd.DataFrame(
        {
            "time_s": time_s,
            "daq_input_v": voltage,
            CURRENT_CHANNEL: current,
            FIELD_CHANNEL: bz_effective,
            "bz_effective_mT": bz_effective,
            "bz_raw_mT": -bz_effective,
        }
    )
    return validation_frame, {
        "current_pp": _peak_to_peak(current),
        "field_pp": _peak_to_peak(bz_effective),
        "voltage_pp": _peak_to_peak(voltage),
    }


def _build_history_entry(retune_result: Any, artifact_paths: dict[str, str]) -> dict[str, Any]:
    return {
        "retune_id": retune_result.validation_run.export_file_prefix,
        "created_at": retune_result.validation_run.created_at,
        "original_recommendation_id": retune_result.validation_run.original_recommendation_id,
        "validation_run_id": retune_result.validation_run.validation_run_id,
        "validation_test_id": retune_result.validation_run.selected_validation_test_id,
        "lut_id": retune_result.validation_run.lut_id,
        "source_kind": retune_result.validation_run.source_kind,
        "source_lut_filename": retune_result.validation_run.source_lut_filename,
        "corrected_lut_id": retune_result.validation_run.corrected_lut_id,
        "iteration_index": retune_result.validation_run.iteration_index,
        "exact_path": retune_result.validation_run.exact_path,
        "correction_rule": retune_result.validation_run.correction_rule,
        "target_output_type": retune_result.validation_run.target_output_type,
        "waveform_type": retune_result.validation_run.waveform_type,
        "freq_hz": retune_result.validation_run.freq_hz,
        "commanded_cycles": retune_result.validation_run.commanded_cycles,
        "target_level_value": retune_result.validation_run.target_level_value,
        "target_level_kind": retune_result.validation_run.target_level_kind,
        "quality_label": retune_result.quality_label,
        "quality_tone": retune_result.quality_tone,
        "quality_reasons": retune_result.quality_reasons,
        "acceptance_decision": retune_result.acceptance_decision,
        "preferred_output_id": retune_result.preferred_output_id,
        "candidate_status": retune_result.acceptance_decision.get("decision"),
        "candidate_status_label": retune_result.acceptance_decision.get("label"),
        "rejection_reason": retune_result.acceptance_decision.get("rejection_reason"),
        "artifact_paths": artifact_paths,
        "before_nrmse": retune_result.baseline_comparison.nrmse,
        "after_nrmse": retune_result.corrected_comparison.nrmse,
        "before_shape_corr": retune_result.baseline_comparison.shape_corr,
        "after_shape_corr": retune_result.corrected_comparison.shape_corr,
        "before_bz_nrmse": retune_result.baseline_bz_comparison.nrmse,
        "after_bz_nrmse": retune_result.corrected_bz_comparison.nrmse,
        "before_bz_shape_corr": retune_result.baseline_bz_comparison.shape_corr,
        "after_bz_shape_corr": retune_result.corrected_bz_comparison.shape_corr,
    }


def _run_single_case(case_config: dict[str, Any]) -> dict[str, Any]:
    file_path = Path(case_config["file_path"])
    if not file_path.exists():
        raise FileNotFoundError(f"validation source file not found: {file_path}")

    validation_frame, signal_stats = _build_validation_frame(file_path)
    target_type = str(case_config["target_type"])
    target_level = float(signal_stats["current_pp"]) if target_type == "current" else float(signal_stats["field_pp"])

    if str(case_config["mode"]) == "finite":
        support_path = _find_support_file()
        base_profile = _build_exact_base_profile(support_path)
        route_reason = "finite exact support profile reuse"
        support_state = "exact"
        support_amp_gain_pct = 100.0
        source_selection = {
            "source_kind": str(case_config["source_kind"]),
            "lut_id": str(case_config["source_lut_id"]),
            "profile_csv_path": support_path.as_posix(),
            "source_lut_filename": support_path.name,
        }
        original_recommendation_id = str(case_config["source_lut_id"])
    else:
        recommendation_result, route_summary, _runtime = run_case(
            waveform=str(case_config["waveform"]),
            target_type=target_type,
            freq_hz=float(case_config["freq_hz"]),
            target_level=target_level,
        )
        support_state = str(recommendation_result.engine_summary.get("support_state") or "")
        if support_state != str(case_config["support_state_expected"]):
            raise RuntimeError(f"{case_config['case_id']}: expected support_state=exact but got {support_state!r}")
        if recommendation_result.preview_only or not recommendation_result.allow_auto_download:
            raise RuntimeError(
                f"{case_config['case_id']}: recommendation stayed preview-only "
                f"(preview_only={recommendation_result.preview_only}, allow_auto_download={recommendation_result.allow_auto_download})"
            )
        if recommendation_result.command_profile is None or recommendation_result.command_profile.empty:
            raise RuntimeError(f"{case_config['case_id']}: missing exact command_profile")
        base_profile = recommendation_result.command_profile
        route_reason = str(route_summary.get("reason") or "")
        support_amp_gain_pct = (
            float(pd.to_numeric(base_profile.get("support_amp_gain_pct"), errors="coerce").iloc[0])
            if "support_amp_gain_pct" in base_profile.columns
            else 100.0
        )
        source_selection = {
            "source_kind": str(case_config["source_kind"]),
            "lut_id": str(case_config["source_lut_id"]),
            "source_lut_filename": f"{case_config['source_lut_id']}.csv",
        }
        original_recommendation_id = str(case_config["source_lut_id"])

    validation_candidate = {
        "label": f"{case_config['label']} | {file_path.name}",
        "test_id": file_path.stem,
        "source_file": file_path.as_posix(),
        "waveform_type": str(case_config["waveform"]),
        "freq_hz": float(case_config["freq_hz"]),
        "output_pp": target_level,
        "score": 0.0,
        "eligible": True,
        "eligibility_reason": "release candidate real validation input",
    }
    retune_result = execute_validation_retune(
        base_profile=base_profile,
        validation_candidate=validation_candidate,
        validation_frame=validation_frame,
        export_file_prefix=str(case_config["export_prefix"]),
        target_output_type=target_type,
        current_channel=CURRENT_CHANNEL,
        field_channel=FIELD_CHANNEL,
        max_daq_voltage_pp=20.0,
        amp_gain_at_100_pct=20.0,
        amp_gain_limit_pct=100.0,
        amp_max_output_pk_v=180.0,
        support_amp_gain_pct=support_amp_gain_pct,
        correction_gain=0.7,
        max_iterations=2,
        improvement_threshold=0.0,
        original_recommendation_id=original_recommendation_id,
        source_selection=source_selection,
    )
    if retune_result is None:
        raise RuntimeError(f"{case_config['case_id']}: validation retune returned None")

    artifact_paths = save_retune_artifacts(retune_result=retune_result, output_dir=OUTPUT_DIR)
    _record_retune_history(_build_history_entry(retune_result, artifact_paths))
    report_payload = dict(retune_result.artifact_payload)

    return {
        "case_id": str(case_config["case_id"]),
        "label": str(case_config["label"]),
        "validation_source_file": file_path.as_posix(),
        "support_state": support_state,
        "software_status": str(case_config["software_status"]),
        "target_type": target_type,
        "target_level_pp": target_level,
        "quality_label": retune_result.quality_label,
        "quality_tone": retune_result.quality_tone,
        "quality_reasons": retune_result.quality_reasons,
        "acceptance_decision": retune_result.acceptance_decision,
        "preferred_output_id": retune_result.preferred_output_id,
        "candidate_status": retune_result.acceptance_decision.get("decision"),
        "candidate_status_label": retune_result.acceptance_decision.get("label"),
        "rejection_reason": retune_result.acceptance_decision.get("rejection_reason"),
        "baseline_nrmse": _safe_float(retune_result.baseline_comparison.nrmse),
        "corrected_nrmse": _safe_float(retune_result.corrected_comparison.nrmse),
        "baseline_bz_nrmse": _safe_float(retune_result.baseline_bz_comparison.nrmse),
        "corrected_bz_nrmse": _safe_float(retune_result.corrected_bz_comparison.nrmse),
        "baseline_shape_corr": _safe_float(retune_result.baseline_comparison.shape_corr),
        "corrected_shape_corr": _safe_float(retune_result.corrected_comparison.shape_corr),
        "baseline_bz_shape_corr": _safe_float(retune_result.baseline_bz_comparison.shape_corr),
        "corrected_bz_shape_corr": _safe_float(retune_result.corrected_bz_comparison.shape_corr),
        "baseline_bz_phase_lag_s": _safe_float(retune_result.baseline_bz_comparison.phase_lag_s),
        "corrected_bz_phase_lag_s": _safe_float(retune_result.corrected_bz_comparison.phase_lag_s),
        "baseline_bz_clipping_detected": bool(retune_result.baseline_bz_comparison.clipping_detected),
        "corrected_bz_clipping_detected": bool(retune_result.corrected_bz_comparison.clipping_detected),
        "baseline_bz_metrics_available": bool(retune_result.baseline_bz_comparison.metrics_available),
        "corrected_bz_metrics_available": bool(retune_result.corrected_bz_comparison.metrics_available),
        "baseline_bz_unavailable_reason": retune_result.baseline_bz_comparison.unavailable_reason,
        "corrected_bz_unavailable_reason": retune_result.corrected_bz_comparison.unavailable_reason,
        "baseline_bz_reason_codes": list(retune_result.baseline_bz_comparison.reason_codes),
        "corrected_bz_reason_codes": list(retune_result.corrected_bz_comparison.reason_codes),
        "corrected_lut_id": retune_result.validation_run.corrected_lut_id,
        "validation_run_id": retune_result.validation_run.validation_run_id,
        "artifact_paths": artifact_paths,
        "route_reason": route_reason,
        "exact_path": retune_result.validation_run.exact_path,
        "validation_window": retune_result.validation_run.metadata.get("validation_window", {}),
        "baseline_metrics": report_payload.get("baseline_metrics", {}),
        "corrected_metrics": report_payload.get("corrected_metrics", {}),
        "preferred_output": report_payload.get("preferred_output", {}),
        "metrics_availability": report_payload.get("metrics_availability", {}),
        "prediction_debug": report_payload.get("prediction_debug", {}),
    }


def _write_field_prediction_debug_report(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "cases": [
            {
                "case_id": case["case_id"],
                "label": case["label"],
                "exact_path": case["exact_path"],
                "support_state": case["support_state"],
                "baseline_prediction_debug": case.get("prediction_debug", {}).get("baseline", {}),
                "corrected_prediction_debug": case.get("prediction_debug", {}).get("corrected", {}),
                "baseline_metrics": case.get("baseline_metrics", {}),
                "corrected_metrics": case.get("corrected_metrics", {}),
                "preferred_output": case.get("preferred_output", {}),
                "metrics_availability": case.get("metrics_availability", {}),
                "rejection_reason": case.get("rejection_reason"),
            }
            for case in case_results
        ],
    }
    _write_json(FIELD_PREDICTION_DEBUG_JSON_PATH, payload)
    lines = ["# Field Prediction Debug Report", ""]
    for case in payload["cases"]:
        lines.extend(
            [
                f"## {case['label']}",
                f"- exact_path: `{case['exact_path']}`",
                f"- baseline.field_prediction_source: `{case['baseline_prediction_debug'].get('field_prediction_source')}`",
                f"- baseline.field_prediction_status: `{case['baseline_prediction_debug'].get('field_prediction_status')}`",
                f"- baseline.zero_field_reason: `{case['baseline_prediction_debug'].get('zero_field_reason')}`",
                f"- baseline.field_prediction_unavailable_reason: `{case['baseline_prediction_debug'].get('field_prediction_unavailable_reason')}`",
                f"- baseline.field_prediction_fallback_reason: `{case['baseline_prediction_debug'].get('field_prediction_fallback_reason')}`",
                f"- corrected.field_prediction_source: `{case['corrected_prediction_debug'].get('field_prediction_source')}`",
                f"- corrected.field_prediction_status: `{case['corrected_prediction_debug'].get('field_prediction_status')}`",
                f"- corrected.zero_field_reason: `{case['corrected_prediction_debug'].get('zero_field_reason')}`",
                f"- preferred_output: `{case['preferred_output'].get('output_id')}` / `{case['preferred_output'].get('output_kind')}`",
                "",
            ]
        )
    _write_markdown(FIELD_PREDICTION_DEBUG_MD_PATH, lines)
    return payload


def _write_exact_field_route_audit() -> dict[str, Any]:
    probes = [_field_route_probe(level) for level in (20.0, 40.0)]
    payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "probes": probes,
        "acceptance": {
            "exact_route_support_blended_preview_detected": any(probe.get("field_prediction_source") == "support_blended_preview" for probe in probes),
            "exact_route_zero_without_reason_detected": any(
                probe.get("request_route") == "exact"
                and (probe.get("predicted_field_pp") in {0.0, None} or probe.get("field_prediction_status") == "unavailable")
                and not (probe.get("field_prediction_unavailable_reason") or probe.get("zero_field_reason"))
                for probe in probes
            ),
            "exact_route_zero_fill_as_main_source_detected": any(probe.get("field_prediction_source") == "zero_fill_fallback" for probe in probes),
        },
    }
    _write_json(EXACT_FIELD_ROUTE_AUDIT_JSON_PATH, payload)
    lines = [
        "# Exact Field Route Audit",
        "",
        f"- support_blended_preview_on_exact: `{payload['acceptance']['exact_route_support_blended_preview_detected']}`",
        f"- zero_without_reason_on_exact: `{payload['acceptance']['exact_route_zero_without_reason_detected']}`",
        f"- zero_fill_as_main_source_on_exact: `{payload['acceptance']['exact_route_zero_fill_as_main_source_detected']}`",
        "",
    ]
    for probe in probes:
        lines.extend(
            [
                f"## target_level={probe['target_level_pp']}",
                f"- request_route: `{probe.get('request_route')}`",
                f"- solver_route: `{probe.get('solver_route')}`",
                f"- plot_source: `{probe.get('plot_source')}`",
                f"- field_prediction_source: `{probe.get('field_prediction_source')}`",
                f"- field_prediction_status: `{probe.get('field_prediction_status')}`",
                f"- zero_field_reason: `{probe.get('zero_field_reason')}`",
                f"- field_prediction_unavailable_reason: `{probe.get('field_prediction_unavailable_reason')}`",
                f"- field_prediction_fallback_reason: `{probe.get('field_prediction_fallback_reason')}`",
                f"- support_selection_reason: `{probe.get('support_selection_reason')}`",
                f"- selected_support_id: `{probe.get('selected_support_id')}`",
                f"- selected_support_family: `{probe.get('selected_support_family')}`",
                f"- allow_auto_download: `{probe.get('allow_auto_download')}`",
                "",
            ]
        )
    _write_markdown(EXACT_FIELD_ROUTE_AUDIT_MD_PATH, lines)
    return payload


def _write_level_sensitivity_diagnosis() -> dict[str, Any]:
    field_probes = [_field_route_probe(level) for level in (20.0, 40.0)]
    finite_probes = [_finite_triangle_probe(level) for level in (10.0, 20.0)]
    payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "continuous_field_same_freq_diff_pp": {
            "requested_freq_hz": 0.25,
            "probes": field_probes,
            "diagnosis": _diagnose_level_sensitivity(field_probes),
        },
        "finite_triangle_same_freq_diff_pp": {
            "requested_freq_hz": 1.25,
            "target_cycle_count": 1.25,
            "probes": finite_probes,
            "diagnosis": _diagnose_level_sensitivity(finite_probes),
        },
    }
    _write_json(LEVEL_SENSITIVITY_JSON_PATH, payload)
    lines = [
        "# Level Sensitivity Diagnosis",
        "",
        f"- continuous_field diagnosis: `{payload['continuous_field_same_freq_diff_pp']['diagnosis']['diagnosis']}`",
        f"- finite_triangle diagnosis: `{payload['finite_triangle_same_freq_diff_pp']['diagnosis']['diagnosis']}`",
        f"- continuous_field reason_codes: `{payload['continuous_field_same_freq_diff_pp']['diagnosis'].get('reason_codes', [])}`",
        f"- finite_triangle reason_codes: `{payload['finite_triangle_same_freq_diff_pp']['diagnosis'].get('reason_codes', [])}`",
        "",
    ]
    _write_markdown(LEVEL_SENSITIVITY_MD_PATH, lines)
    return payload


def _write_finite_triangle_expected_source_audit() -> dict[str, Any]:
    probes = [_finite_triangle_probe(level) for level in (10.0, 20.0)]
    payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "probes": probes,
        "acceptance": {
            "support_blended_preview_on_exact_detected": any(probe.get("field_prediction_source") == "support_blended_preview" for probe in probes),
            "target_leak_suspect_detected": any(bool(probe.get("target_leak_suspect")) for probe in probes),
        },
    }
    _write_json(FINITE_TRIANGLE_AUDIT_JSON_PATH, payload)
    lines = [
        "# Finite Triangle Expected Source Audit",
        "",
        f"- support_blended_preview_on_exact: `{payload['acceptance']['support_blended_preview_on_exact_detected']}`",
        f"- target_leak_suspect_detected: `{payload['acceptance']['target_leak_suspect_detected']}`",
        "",
    ]
    for probe in probes:
        lines.extend(
            [
                f"## target_level={probe['target_level_pp']}",
                f"- field_prediction_source: `{probe.get('field_prediction_source')}`",
                f"- zero_field_reason: `{probe.get('zero_field_reason')}`",
                f"- target_leak_suspect: `{probe.get('target_leak_suspect')}`",
                f"- target_leak_reason: `{probe.get('target_leak_reason')}`",
                "",
            ]
        )
    _write_markdown(FINITE_TRIANGLE_AUDIT_MD_PATH, lines)
    return payload


def _write_fallback_hierarchy_audit(case_results: list[dict[str, Any]], exact_field_route_audit: dict[str, Any]) -> dict[str, Any]:
    field_case = next((case for case in case_results if case.get("exact_path") == "exact_field"), None)
    payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "exact_field_route_probes": [
            {
                "target_level_pp": probe.get("target_level_pp"),
                "field_prediction_source": probe.get("field_prediction_source"),
                "field_prediction_status": probe.get("field_prediction_status"),
                "field_prediction_available": probe.get("field_prediction_available"),
                "field_prediction_unavailable_reason": probe.get("field_prediction_unavailable_reason"),
                "field_prediction_fallback_reason": probe.get("field_prediction_fallback_reason"),
                "field_prediction_hierarchy": probe.get("field_prediction_hierarchy"),
                "exact_field_direct_available": probe.get("exact_field_direct_available"),
                "exact_field_direct_reason": probe.get("exact_field_direct_reason"),
                "same_recipe_surrogate_candidate_available": probe.get("same_recipe_surrogate_candidate_available"),
                "same_recipe_surrogate_applied": probe.get("same_recipe_surrogate_applied"),
                "same_recipe_surrogate_ratio": probe.get("same_recipe_surrogate_ratio"),
                "surrogate_scope": probe.get("surrogate_scope"),
                "allow_auto_download": probe.get("allow_auto_download"),
            }
            for probe in exact_field_route_audit.get("probes", [])
        ],
        "real_validation_exact_field": field_case.get("prediction_debug", {}) if isinstance(field_case, dict) else {},
    }
    payload["acceptance"] = {
        "zero_fill_as_main_prediction_detected": any(
            probe.get("field_prediction_source") == "zero_fill_fallback"
            for probe in payload["exact_field_route_probes"]
        ),
        "unavailable_without_reason_detected": any(
            probe.get("field_prediction_status") == "unavailable"
            and not probe.get("field_prediction_unavailable_reason")
            for probe in payload["exact_field_route_probes"]
        ),
    }
    _write_json(FALLBACK_HIERARCHY_JSON_PATH, payload)
    lines = [
        "# Fallback Hierarchy Audit",
        "",
        f"- zero_fill_as_main_prediction_detected: `{payload['acceptance']['zero_fill_as_main_prediction_detected']}`",
        f"- unavailable_without_reason_detected: `{payload['acceptance']['unavailable_without_reason_detected']}`",
        "",
    ]
    for probe in payload["exact_field_route_probes"]:
        lines.extend(
            [
                f"## target_level={probe.get('target_level_pp')}",
                f"- field_prediction_source: `{probe.get('field_prediction_source')}`",
                f"- field_prediction_status: `{probe.get('field_prediction_status')}`",
                f"- field_prediction_unavailable_reason: `{probe.get('field_prediction_unavailable_reason')}`",
                f"- field_prediction_fallback_reason: `{probe.get('field_prediction_fallback_reason')}`",
                f"- same_recipe_surrogate_applied: `{probe.get('same_recipe_surrogate_applied')}`",
                "",
            ]
        )
    _write_markdown(FALLBACK_HIERARCHY_MD_PATH, lines)
    return payload


def _write_reason_code_breakdown(
    case_results: list[dict[str, Any]],
    exact_field_route_audit: dict[str, Any],
    level_sensitivity: dict[str, Any],
) -> dict[str, Any]:
    def _bump(counter: dict[str, int], key: str | None) -> None:
        if key:
            counter[key] = counter.get(key, 0) + 1

    candidate_status_counts: dict[str, int] = {}
    rejection_reason_counts: dict[str, int] = {}
    field_unavailable_reason_counts: dict[str, int] = {}
    field_fallback_reason_counts: dict[str, int] = {}
    metric_reason_code_counts: dict[str, int] = {}
    level_diagnosis_counts: dict[str, int] = {}
    level_reason_code_counts: dict[str, int] = {}

    for case in case_results:
        _bump(candidate_status_counts, str(case.get("candidate_status") or ""))
        _bump(rejection_reason_counts, str(case.get("rejection_reason") or ""))
        prediction_debug = case.get("prediction_debug", {})
        for key in ("baseline", "corrected"):
            debug = prediction_debug.get(key, {})
            _bump(field_unavailable_reason_counts, str(debug.get("field_prediction_unavailable_reason") or ""))
            _bump(field_fallback_reason_counts, str(debug.get("field_prediction_fallback_reason") or ""))
        for metrics_key in ("baseline_metrics", "corrected_metrics"):
            bz_metrics = case.get(metrics_key, {}).get("bz_effective", {})
            for reason_code in bz_metrics.get("reason_codes", []) or []:
                _bump(metric_reason_code_counts, str(reason_code))

    for probe in exact_field_route_audit.get("probes", []):
        _bump(field_unavailable_reason_counts, str(probe.get("field_prediction_unavailable_reason") or probe.get("zero_field_reason") or ""))
        _bump(field_fallback_reason_counts, str(probe.get("field_prediction_fallback_reason") or ""))

    for key in ("continuous_field_same_freq_diff_pp", "finite_triangle_same_freq_diff_pp"):
        diagnosis = level_sensitivity.get(key, {}).get("diagnosis", {})
        _bump(level_diagnosis_counts, str(diagnosis.get("diagnosis") or ""))
        for reason_code in diagnosis.get("reason_codes", []) or []:
            _bump(level_reason_code_counts, str(reason_code))

    payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "candidate_status_counts": candidate_status_counts,
        "rejection_reason_counts": rejection_reason_counts,
        "field_prediction_unavailable_reason_counts": field_unavailable_reason_counts,
        "field_prediction_fallback_reason_counts": field_fallback_reason_counts,
        "bz_metric_reason_code_counts": metric_reason_code_counts,
        "level_diagnosis_counts": level_diagnosis_counts,
        "level_reason_code_counts": level_reason_code_counts,
    }
    _write_json(REASON_CODE_BREAKDOWN_JSON_PATH, payload)
    lines = [
        "# Reason Code Breakdown",
        "",
        f"- candidate_status_counts: `{candidate_status_counts}`",
        f"- rejection_reason_counts: `{rejection_reason_counts}`",
        f"- field_prediction_unavailable_reason_counts: `{field_unavailable_reason_counts}`",
        f"- field_prediction_fallback_reason_counts: `{field_fallback_reason_counts}`",
        f"- bz_metric_reason_code_counts: `{metric_reason_code_counts}`",
        f"- level_reason_code_counts: `{level_reason_code_counts}`",
        "",
    ]
    _write_markdown(REASON_CODE_BREAKDOWN_MD_PATH, lines)
    return payload


def _write_summary_markdown(payload: dict[str, Any]) -> None:
    lines = [
        "# Release Candidate Real Validation Suite",
        "",
        "- 실제 validation 파일 기준으로 exact current / exact field / finite exact 3건을 end-to-end 재실행한 결과입니다.",
        "- 각 케이스는 recommendation 또는 past export를 source로 선택하고 corrected LUT 산출물까지 생성합니다.",
        "",
    ]
    for case in payload.get("cases", []):
        lines.extend(
            [
                f"## {case['label']}",
                f"- validation_source_file: `{case['validation_source_file']}`",
                f"- exact_path: `{case['exact_path']}`",
                f"- support_state: `{case['support_state']}`",
                f"- quality_label: `{case['quality_label']}`",
                f"- baseline_nrmse: `{case['baseline_nrmse']}`",
                f"- corrected_nrmse: `{case['corrected_nrmse']}`",
                f"- baseline_bz_nrmse: `{case['baseline_bz_nrmse']}`",
                f"- corrected_bz_nrmse: `{case['corrected_bz_nrmse']}`",
                f"- corrected_lut_id: `{case['corrected_lut_id']}`",
                f"- report_path: `{case['artifact_paths'].get('validation_report_json')}`",
                "",
            ]
        )
    SUMMARY_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    case_results = [_run_single_case(case_config) for case_config in CASE_CONFIGS]
    field_prediction_debug = _write_field_prediction_debug_report(case_results)
    exact_field_route_audit = _write_exact_field_route_audit()
    level_sensitivity = _write_level_sensitivity_diagnosis()
    finite_triangle_audit = _write_finite_triangle_expected_source_audit()
    fallback_hierarchy_audit = _write_fallback_hierarchy_audit(case_results, exact_field_route_audit)
    reason_code_breakdown = _write_reason_code_breakdown(case_results, exact_field_route_audit, level_sensitivity)

    completed = subprocess.run(
        [sys.executable, str(ARTIFACT_GENERATOR)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"artifact generator failed after release candidate validations:\n{completed.stderr}")

    payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "cases": case_results,
        "field_prediction_debug_report": FIELD_PREDICTION_DEBUG_JSON_PATH.as_posix(),
        "exact_field_route_audit": EXACT_FIELD_ROUTE_AUDIT_JSON_PATH.as_posix(),
        "level_sensitivity_diagnosis": LEVEL_SENSITIVITY_JSON_PATH.as_posix(),
        "finite_triangle_expected_source_audit": FINITE_TRIANGLE_AUDIT_JSON_PATH.as_posix(),
        "fallback_hierarchy_audit": FALLBACK_HIERARCHY_JSON_PATH.as_posix(),
        "reason_code_breakdown": REASON_CODE_BREAKDOWN_JSON_PATH.as_posix(),
        "field_prediction_debug_summary": {
            "case_count": len(field_prediction_debug.get("cases", [])),
        },
        "exact_field_route_summary": exact_field_route_audit.get("acceptance", {}),
        "level_sensitivity_summary": {
            "continuous_field": level_sensitivity.get("continuous_field_same_freq_diff_pp", {}).get("diagnosis", {}),
            "finite_triangle": level_sensitivity.get("finite_triangle_same_freq_diff_pp", {}).get("diagnosis", {}),
        },
        "finite_triangle_audit_summary": finite_triangle_audit.get("acceptance", {}),
        "fallback_hierarchy_summary": fallback_hierarchy_audit.get("acceptance", {}),
        "reason_code_breakdown_summary": reason_code_breakdown,
        "generator_stdout": completed.stdout.strip(),
    }
    _write_json(SUMMARY_JSON_PATH, payload)
    _write_summary_markdown(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
