from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
FIELD_ANALYSIS_SRC = ROOT.parent / "src"
if str(FIELD_ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(FIELD_ANALYSIS_SRC))

import generate_bz_first_artifacts as artifact_script
import report_lcr_influence_audit as lcr_audit_script
import report_exact_and_finite_scope as scope_script
from field_analysis.canonicalize import CanonicalizeConfig, canonicalize_run
from field_analysis.lut import recommend_voltage_waveform
from field_analysis.models import (
    CycleDetectionResult,
    DatasetAnalysis,
    ParsedMeasurement,
    PreprocessResult,
    SheetPreview,
)
from field_analysis.compensation import _select_nearest_support_row
from field_analysis.field_prediction_debug import sanitize_unavailable_exact_field_prediction
from field_analysis.level_sensitivity_analysis import build_level_sensitivity_diagnosis
from field_analysis.recommendation_service import (
    LegacyRecommendationContext,
    RecommendationPolicy,
    RecommendationPolicyConfig,
    RecommendationPolicyThresholds,
    RecommendationOptions,
    RecommendationResult,
    TargetRequest,
    _find_provisional_finite_recipe,
    build_support_report,
    evaluate_recommendation_policy,
    recommend,
)
from field_analysis.utils import canonicalize_waveform_type
from field_analysis.validation import ValidationReport


def _normalized_waveform(phase: np.ndarray | float, waveform_type: str) -> np.ndarray:
    waveform_type = canonicalize_waveform_type(waveform_type) or "sine"
    phase_array = np.mod(np.asarray(phase, dtype=float), 1.0)
    if waveform_type == "triangle":
        waveform = np.empty_like(phase_array)
        rising = phase_array < 0.25
        falling = (phase_array >= 0.25) & (phase_array < 0.75)
        waveform[rising] = 4.0 * phase_array[rising]
        waveform[falling] = 2.0 - 4.0 * phase_array[falling]
        waveform[~(rising | falling)] = -4.0 + 4.0 * phase_array[~(rising | falling)]
        return waveform
    return np.sin(2.0 * np.pi * phase_array)


def _build_dummy_analysis(
    phase_shift_cycles: float = 0.0,
    waveform_type: str = "sine",
    freq_hz: float = 0.5,
    test_id: str = "t1",
) -> tuple[pd.DataFrame, DatasetAnalysis]:
    waveform_type = canonicalize_waveform_type(waveform_type) or "sine"
    per_test_summary = pd.DataFrame(
        [
            {
                "test_id": test_id,
                "waveform_type": waveform_type,
                "freq_hz": freq_hz,
                "current_pp_target_a": 10.0,
                "achieved_current_pp_a_mean": 10.0,
                "achieved_bz_mT_pp_mean": 20.0,
                "achieved_bmag_mT_pp_mean": 20.0,
                "daq_input_v_pp_mean": 4.0,
                "amp_gain_setting_mean": 50.0,
            }
        ]
    )

    phase = np.linspace(0.0, 1.0, 32)
    rows: list[dict[str, float]] = []
    for cycle_index in range(4):
        cycle_time = phase * (1.0 / freq_hz)
        for cycle_progress, cycle_time_s in zip(phase, cycle_time, strict=False):
            shifted_phase = cycle_progress + phase_shift_cycles
            rows.append(
                {
                    "cycle_index": float(cycle_index),
                    "cycle_progress": float(cycle_progress),
                    "cycle_time_s": float(cycle_time_s),
                    "time_s": float(cycle_time_s + cycle_index * (1.0 / freq_hz)),
                    "freq_hz": freq_hz,
                    "daq_input_v": 2.0 * _normalized_waveform(shifted_phase, waveform_type),
                    "i_sum_signed": 5.0 * _normalized_waveform(shifted_phase, waveform_type),
                    "bz_mT": 10.0 * _normalized_waveform(shifted_phase - 0.1, waveform_type),
                    "current_pp_target_a": 10.0,
                    "current_pk_target_a": 5.0,
                    "waveform_type": waveform_type,
                    "cycle_total_expected": 4.0,
                    "source_cycle_no": np.nan,
                }
            )
    annotated_frame = pd.DataFrame(rows)
    per_cycle_summary = pd.DataFrame(
        [
            {
                "cycle_index": cycle_index,
                "achieved_current_pp_a": 10.0,
                "achieved_bz_mT_pp": 20.0,
                "daq_input_v_pp": 4.0,
            }
            for cycle_index in range(4)
        ]
    )

    preview = SheetPreview("main", 0, 0, [], 0, {}, [], {}, [], [])
    parsed = ParsedMeasurement(
        source_file="file.csv",
        file_type="csv",
        sheet_name="main",
        structure_preview=preview,
        metadata={"waveform": waveform_type, "cycle": "4"},
        mapping={},
        raw_frame=annotated_frame.copy(),
        normalized_frame=annotated_frame.copy(),
        warnings=[],
        logs=[],
    )
    preprocess = PreprocessResult(
        corrected_frame=annotated_frame.copy(),
        offsets={},
        lags=[],
        warnings=[],
        logs=[],
    )
    cycle_detection = CycleDetectionResult(
        annotated_frame=annotated_frame.copy(),
        boundaries=[],
        estimated_period_s=1.0 / freq_hz,
        estimated_frequency_hz=freq_hz,
        reference_channel="daq_input_v",
        warnings=[],
        logs=[],
    )
    analysis = DatasetAnalysis(
        parsed=parsed,
        preprocess=preprocess,
        cycle_detection=cycle_detection,
        per_cycle_summary=per_cycle_summary,
        per_test_summary=per_test_summary.copy(),
        warnings=[],
    )
    return per_test_summary, analysis


def _build_dummy_transient_support(
    *,
    waveform_type: str = "sine",
    freq_hz: float = 1.0,
    cycle_count: float = 1.0,
    requested_level_pp: float = 20.0,
    test_id: str = "tx1",
) -> tuple[object, LegacyRecommendationContext]:
    waveform_type = canonicalize_waveform_type(waveform_type) or "sine"
    points = 257
    duration_s = float(cycle_count) / float(freq_hz)
    time_s = np.linspace(0.0, duration_s, points)
    phase = np.linspace(0.0, float(cycle_count), points)
    drive = _normalized_waveform(phase, waveform_type)
    current_signal = (float(requested_level_pp) / 2.0) * drive
    field_signal = float(requested_level_pp) * drive

    frame = pd.DataFrame(
        {
            "time_s": time_s,
            "daq_input_v": 2.0 * drive,
            "i_sum_signed": current_signal,
            "bz_mT": field_signal,
            "freq_hz": float(freq_hz),
            "waveform_type": waveform_type,
            "cycle_total_expected": float(cycle_count),
            "source_cycle_no": np.nan,
        }
    )
    preview = SheetPreview("main", 0, 0, [], 0, {}, [], {}, [], [])
    parsed = ParsedMeasurement(
        source_file=f"{test_id}.csv",
        file_type="csv",
        sheet_name="main",
        structure_preview=preview,
        metadata={"waveform": waveform_type, "cycle": f"{cycle_count}", "current": f"{requested_level_pp}pp"},
        mapping={},
        raw_frame=frame.copy(),
        normalized_frame=frame.copy(),
        warnings=[],
        logs=[],
    )
    preprocess = PreprocessResult(
        corrected_frame=frame.copy(),
        offsets={},
        lags=[],
        warnings=[],
        logs=[],
    )
    canonical = canonicalize_run(parsed, regime="transient", role="train", config=CanonicalizeConfig())
    canonical.target_level_value = float(requested_level_pp)
    canonical.target_level_kind = "pp"
    canonical.commanded_cycles = float(cycle_count)
    legacy_context = LegacyRecommendationContext(
        transient_measurements=[parsed],
        transient_preprocess_results=[preprocess],
        transient_canonical_runs=[canonical],
    )
    return canonical, legacy_context


def _build_dummy_lcr_measurements() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "freq_hz": [0.25, 0.5, 1.0, 2.0, 3.0],
            "rs_ohm": [2.4, 2.5, 2.7, 3.0, 3.3],
            "ls_h": [0.012, 0.012, 0.011, 0.010, 0.009],
            "cs_f": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )


def _merge_transient_legacy_contexts(*contexts: LegacyRecommendationContext) -> LegacyRecommendationContext:
    merged = LegacyRecommendationContext()
    for context in contexts:
        merged.transient_measurements.extend(context.transient_measurements)
        merged.transient_preprocess_results.extend(context.transient_preprocess_results)
        merged.transient_canonical_runs.extend(context.transient_canonical_runs)
    return merged


def _write_exact_matrix_artifact(
    path: Path,
    *,
    continuous_current_exact: list[dict[str, object]] | None = None,
    continuous_field_exact: list[dict[str, object]] | None = None,
    finite_exact: list[dict[str, object]] | None = None,
    provisional: list[dict[str, object]] | None = None,
    missing: list[dict[str, object]] | None = None,
    reference_only: list[dict[str, object]] | None = None,
    promotion_state: str = "provisional_only",
) -> Path:
    continuous_current_exact = continuous_current_exact or []
    continuous_field_exact = continuous_field_exact or []
    finite_exact = finite_exact or []
    provisional = provisional or []
    missing = missing or []
    reference_only = reference_only or []
    payload = {
        "generated_at": "2026-04-16T00:00:00+00:00",
        "schema_version": "exact_matrix_final_v3",
        "counts": {
            "continuous_current_exact_cells": len(continuous_current_exact),
            "continuous_field_exact_rows": len(continuous_field_exact),
            "finite_exact_cells": len(finite_exact),
            "provisional_cells": len(provisional),
            "missing_exact_cells": len(missing),
            "reference_only_cells": len(reference_only),
        },
        "continuous_current_exact_matrix": {"cells": continuous_current_exact, "summary": []},
        "continuous_field_exact_matrix": {"summary": continuous_field_exact},
        "finite_exact_matrix": {
            "cells": finite_exact,
            "summary": [],
            "promotion_status": {"state": promotion_state},
        },
        "provisional_cell": {"cells": provisional},
        "missing_exact_cell": {"cells": missing},
        "reference_only": {"cells": reference_only},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_catalog_profile(
    path: Path,
    *,
    waveform: str,
    freq_hz: float,
    target_type: str,
    level_pp: float,
    cycle_count: float | None = None,
    request_route: str = "exact",
    plot_source: str = "exact_prediction",
) -> None:
    sample_count = 64
    total_cycles = cycle_count if cycle_count is not None else 1.0
    phase = np.linspace(0.0, total_cycles, sample_count)
    time_s = np.linspace(0.0, total_cycles / freq_hz, sample_count)
    waveform_signal = np.sin(2.0 * np.pi * phase)
    frame = pd.DataFrame(
        {
            "time_s": time_s,
            "cycle_progress": phase,
            "waveform_type": waveform,
            "freq_hz": freq_hz,
            "target_output_type": target_type,
            "target_output_pp": level_pp,
            "target_cycle_count": cycle_count if cycle_count is not None else np.nan,
            "request_route": request_route,
            "plot_source": plot_source,
            "within_hardware_limits": True,
            "peak_input_limit_margin": 0.5,
            "expected_field_mT": 10.0 * waveform_signal,
            "modeled_field_mT": 10.0 * waveform_signal,
            "target_current_a": (level_pp / 2.0) * waveform_signal if target_type == "current" else np.nan,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def test_recommend_route_matches_legacy_scalar_lut() -> None:
    per_test_summary, analysis = _build_dummy_analysis()
    legacy = recommend_voltage_waveform(
        per_test_summary=per_test_summary,
        analyses_by_test_id={"t1": analysis},
        waveform_type="sine",
        freq_hz=0.5,
        target_metric="achieved_bz_mT_pp_mean",
        target_value=20.0,
        frequency_mode="exact",
    )
    canonical = canonicalize_run(analysis.parsed, regime="continuous", role="train", config=CanonicalizeConfig())

    result = recommend(
        continuous_runs=[canonical],
        transient_runs=[],
        validation_runs=[],
        target=TargetRequest(
            regime="continuous",
            target_waveform="sine",
            command_waveform="sine",
            freq_hz=0.5,
            target_type="field",
            target_level_value=20.0,
            target_level_kind="pp",
            context={
                "request_kind": "size_lut",
                "target_metric": "achieved_bz_mT_pp_mean",
                "target_value": 20.0,
                "frequency_mode": "exact",
                "finite_cycle_mode": False,
                "preview_tail_cycles": 0.25,
            },
        ),
        options=RecommendationOptions(current_channel="i_sum_signed", field_channel="bz_mT"),
        legacy_context=LegacyRecommendationContext(
            per_test_summary=per_test_summary,
            analysis_lookup={"t1": analysis},
        ),
    )

    assert legacy is not None
    assert result.legacy_payload is not None
    assert np.isclose(result.legacy_payload["limited_voltage_pp"], legacy["limited_voltage_pp"])
    assert np.isclose(result.legacy_payload["estimated_current_pp"], legacy["estimated_current_pp"])
    assert result.preview_only is False


def test_build_support_report_blocks_auto_when_transient_cycle_is_missing() -> None:
    per_test_summary, analysis = _build_dummy_analysis(freq_hz=1.0)
    transient_run = canonicalize_run(analysis.parsed, regime="transient", role="train", config=CanonicalizeConfig())

    report = build_support_report(
        continuous_runs=[],
        transient_runs=[transient_run],
        target=TargetRequest(
            regime="transient",
            target_waveform="sine",
            command_waveform="sine",
            freq_hz=1.0,
            commanded_cycles=1.25,
            target_type="field",
            target_level_value=20.0,
            target_level_kind="pp",
        ),
    )

    assert report.exact_freq_match is True
    assert report.exact_cycle_match is False
    assert report.allow_auto_recommendation is False


def test_build_support_report_accepts_finite_exact_triangle_recipe() -> None:
    canonical, _legacy_context = _build_dummy_transient_support(
        waveform_type="triangle",
        freq_hz=1.0,
        cycle_count=1.0,
        requested_level_pp=20.0,
        test_id="tri_tx1",
    )

    report = build_support_report(
        continuous_runs=[],
        transient_runs=[canonical],
        target=TargetRequest(
            regime="transient",
            target_waveform="triangle",
            command_waveform="triangle",
            freq_hz=1.0,
            commanded_cycles=1.0,
            target_type="current",
            target_level_value=20.0,
            target_level_kind="pp",
            context={"request_kind": "waveform_compensation", "frequency_mode": "exact", "finite_cycle_mode": True},
        )
    )

    assert report.in_support is True
    assert report.exact_freq_match is True
    assert report.exact_cycle_match is True
    assert report.allow_auto_recommendation is True


def test_find_provisional_recipe_for_missing_finite_sine_20pp() -> None:
    _canonical, legacy_context = _build_dummy_transient_support(
        waveform_type="sine",
        freq_hz=1.0,
        cycle_count=1.0,
        requested_level_pp=10.0,
        test_id="sine_tx10",
    )

    provisional = _find_provisional_finite_recipe(
        list(legacy_context.transient_canonical_runs),
        TargetRequest(
            regime="transient",
            target_waveform="sine",
            command_waveform="sine",
            freq_hz=1.0,
            commanded_cycles=1.0,
            target_type="current",
            target_level_value=20.0,
            target_level_kind="pp",
        ),
    )

    assert provisional is not None
    assert provisional["source_level_pp"] == 10.0
    assert provisional["target_level_pp"] == 20.0
    assert provisional["scale_ratio"] == 2.0


def test_artifact_scope_lock_keeps_provisional_runtime_preview_for_placeholder_cell(tmp_path: Path) -> None:
    exact_matrix_path = _write_exact_matrix_artifact(
        tmp_path / "exact_matrix_final.json",
        provisional=[
            {
                "waveform": "sine",
                "freq_hz": 1.0,
                "cycles": 1.0,
                "level_pp_a": 20.0,
                "source_exact_level_pp_a": 10.0,
                "scale_ratio": 2.0,
                "status": "provisional_preview",
            }
        ],
        missing=[
            {
                "waveform": "sine",
                "freq_hz": 1.0,
                "cycles": 1.0,
                "level_pp_a": 20.0,
                "status": "missing_exact",
            }
        ],
        promotion_state="provisional_only",
    )
    canonical_10, legacy_10 = _build_dummy_transient_support(
        waveform_type="sine",
        freq_hz=1.0,
        cycle_count=1.0,
        requested_level_pp=10.0,
        test_id="scope_lock_10pp",
    )
    canonical_20, legacy_20 = _build_dummy_transient_support(
        waveform_type="sine",
        freq_hz=1.0,
        cycle_count=1.0,
        requested_level_pp=20.0,
        test_id="scope_lock_20pp_placeholder",
    )
    legacy_context = _merge_transient_legacy_contexts(legacy_10, legacy_20)

    result = recommend(
        continuous_runs=[],
        transient_runs=[canonical_10, canonical_20],
        validation_runs=[],
        target=TargetRequest(
            regime="transient",
            target_waveform="sine",
            command_waveform="sine",
            freq_hz=1.0,
            commanded_cycles=1.0,
            target_type="current",
            target_level_value=20.0,
            target_level_kind="pp",
            context={
                "request_kind": "waveform_compensation",
                "frequency_mode": "exact",
                "finite_cycle_mode": True,
                "exact_matrix_artifact_path": str(exact_matrix_path),
            },
        ),
        options=RecommendationOptions(current_channel="i_sum_signed", field_channel="bz_mT"),
        legacy_context=legacy_context,
    )

    assert result.engine_summary["support_state"] == "provisional_preview"
    assert result.engine_summary["request_route"] == "provisional"
    assert result.preview_only is True
    assert result.allow_auto_download is False
    assert result.debug_info["artifact_scope_lock"]["bucket"] == "provisional_preview"
    assert "artifact_scope_locked" in result.debug_info["policy_flags"]


def test_artifact_scope_lock_promotes_runtime_to_exact_after_exact_cell_is_present(tmp_path: Path) -> None:
    exact_matrix_path = _write_exact_matrix_artifact(
        tmp_path / "exact_matrix_final.json",
        finite_exact=[
            {
                "waveform": "sine",
                "freq_hz": 1.0,
                "cycles": 1.0,
                "level_pp_a": 20.0,
                "status": "certified_exact",
            }
        ],
        promotion_state="promoted_to_exact",
    )
    canonical_20, legacy_20 = _build_dummy_transient_support(
        waveform_type="sine",
        freq_hz=1.0,
        cycle_count=1.0,
        requested_level_pp=20.0,
        test_id="scope_lock_promoted_20pp",
    )

    result = recommend(
        continuous_runs=[],
        transient_runs=[canonical_20],
        validation_runs=[],
        target=TargetRequest(
            regime="transient",
            target_waveform="sine",
            command_waveform="sine",
            freq_hz=1.0,
            commanded_cycles=1.0,
            target_type="current",
            target_level_value=20.0,
            target_level_kind="pp",
            context={
                "request_kind": "waveform_compensation",
                "frequency_mode": "exact",
                "finite_cycle_mode": True,
                "exact_matrix_artifact_path": str(exact_matrix_path),
            },
        ),
        options=RecommendationOptions(current_channel="i_sum_signed", field_channel="bz_mT"),
        legacy_context=legacy_20,
    )

    assert result.engine_summary["support_state"] == "exact"
    assert result.engine_summary["request_route"] == "exact"
    assert result.preview_only is False
    assert result.allow_auto_download is True
    assert result.debug_info["artifact_scope_lock"]["bucket"] == "exact"


def test_non_placeholder_promotion_aligns_matrix_roi_catalog_and_runtime_route(tmp_path: Path) -> None:
    continuous_dir = tmp_path / "uploads" / "continuous"
    transient_dir = tmp_path / "uploads" / "transient"
    sine_dir = transient_dir / "sinusidal"
    sine_dir.mkdir(parents=True, exist_ok=True)
    (sine_dir / "1hz_1cycle_10pp.csv").touch()
    (sine_dir / "1hz_1cycle_20pp.csv").touch()

    baseline_scope = scope_script.build_scope_payload(
        continuous_dir=continuous_dir,
        transient_dir=transient_dir,
    )
    baseline_exact_matrix = artifact_script.build_exact_matrix(baseline_scope)
    baseline_roi = artifact_script.build_measurement_roi_priority(scope=baseline_scope, validation_catalog=[])

    (sine_dir / "abcd1234_1hz_1cycle_20pp.csv").touch()

    promoted_scope = scope_script.build_scope_payload(
        continuous_dir=continuous_dir,
        transient_dir=transient_dir,
    )
    exact_matrix_payload = artifact_script.build_exact_matrix(promoted_scope)
    promoted_roi = artifact_script.build_measurement_roi_priority(scope=promoted_scope, validation_catalog=[])

    exact_matrix_path = tmp_path / "artifacts" / "exact_matrix_final.json"
    artifact_script.write_json(exact_matrix_path, exact_matrix_payload)

    policy_scope_path = tmp_path / "exact_and_finite_scope.json"
    artifact_script.write_json(policy_scope_path, promoted_scope)
    recommendation_library_dir = tmp_path / "recommendation_library"
    promoted_profile = recommendation_library_dir / "steady_state_harmonic_sine_1Hz_current_20_1cycle.csv"
    _write_catalog_profile(
        promoted_profile,
        waveform="sine",
        freq_hz=1.0,
        target_type="current",
        level_pp=20.0,
        cycle_count=1.0,
        request_route="exact",
    )
    catalog_entries = artifact_script.build_lut_catalog(
        scope=promoted_scope,
        validation_catalog=[],
        corrected_catalog=[],
        paths=artifact_script.ArtifactPaths(
            output_dir=tmp_path / "artifacts",
            export_validation_dir=tmp_path / "export_validation",
            policy_scope_path=policy_scope_path,
            recommendation_library_dir=recommendation_library_dir,
            recommendation_manifest_path=tmp_path / "recommendation_manifest.json",
            upload_manifest_path=tmp_path / "upload_manifest.json",
            retune_history_path=tmp_path / "validation_retune_history.json",
            retune_dirs=(),
        ),
    )
    by_id = {entry["lut_id"]: entry for entry in catalog_entries}

    canonical_10, legacy_10 = _build_dummy_transient_support(
        waveform_type="sine",
        freq_hz=1.0,
        cycle_count=1.0,
        requested_level_pp=10.0,
        test_id="promotion_alignment_10pp",
    )
    canonical_20, legacy_20 = _build_dummy_transient_support(
        waveform_type="sine",
        freq_hz=1.0,
        cycle_count=1.0,
        requested_level_pp=20.0,
        test_id="promotion_alignment_20pp",
    )
    legacy_context = _merge_transient_legacy_contexts(legacy_10, legacy_20)
    result = recommend(
        continuous_runs=[],
        transient_runs=[canonical_10, canonical_20],
        validation_runs=[],
        target=TargetRequest(
            regime="transient",
            target_waveform="sine",
            command_waveform="sine",
            freq_hz=1.0,
            commanded_cycles=1.0,
            target_type="current",
            target_level_value=20.0,
            target_level_kind="pp",
            context={
                "request_kind": "waveform_compensation",
                "frequency_mode": "exact",
                "finite_cycle_mode": True,
                "exact_matrix_artifact_path": str(exact_matrix_path),
            },
        ),
        options=RecommendationOptions(current_channel="i_sum_signed", field_channel="bz_mT"),
        legacy_context=legacy_context,
    )

    assert baseline_scope["finite_all_exact_scope"]["promotion_status"]["state"] == "provisional_only"
    assert promoted_scope["finite_all_exact_scope"]["promotion_status"]["state"] == "promoted_to_exact"
    assert baseline_exact_matrix["counts"]["finite_exact_cells"] + 1 == exact_matrix_payload["counts"]["finite_exact_cells"]
    assert baseline_exact_matrix["counts"]["provisional_cells"] == 1
    assert baseline_exact_matrix["counts"]["missing_exact_cells"] == 1
    assert exact_matrix_payload["counts"]["provisional_cells"] == 0
    assert exact_matrix_payload["counts"]["missing_exact_cells"] == 0
    assert baseline_roi["priorities"][0]["category"] == "missing_exact_promotion"
    assert promoted_roi["priorities"][0]["category"] == "continuous_exact_gap_fill"
    assert all(item["category"] != "missing_exact_promotion" for item in promoted_roi["priorities"])
    assert by_id["steady_state_harmonic_sine_1Hz_current_20_1cycle"]["status"] == "certified_exact"
    assert by_id["steady_state_harmonic_sine_1Hz_current_20_1cycle"]["source_route"] == "finite_exact"
    assert by_id["steady_state_harmonic_sine_1Hz_current_20_1cycle"]["request_route"] == "exact"
    assert by_id["steady_state_harmonic_sine_1Hz_current_20_1cycle"]["duplicate_runtime"] is False
    assert by_id["steady_state_harmonic_sine_1Hz_current_20_1cycle"]["stale_runtime"] is False
    assert result.engine_summary["support_state"] == "exact"
    assert result.engine_summary["request_route"] == "exact"
    assert result.preview_only is False
    assert result.allow_auto_download is True
    assert result.debug_info["artifact_scope_lock"]["bucket"] == "exact"


def test_artifact_scope_lock_marks_missing_without_provisional_as_unsupported(tmp_path: Path) -> None:
    exact_matrix_path = _write_exact_matrix_artifact(
        tmp_path / "exact_matrix_final.json",
        missing=[
            {
                "waveform": "sine",
                "freq_hz": 1.0,
                "cycles": 1.0,
                "level_pp_a": 20.0,
                "status": "missing_exact",
            }
        ],
        promotion_state="missing",
    )
    canonical_10, legacy_10 = _build_dummy_transient_support(
        waveform_type="sine",
        freq_hz=1.0,
        cycle_count=1.0,
        requested_level_pp=10.0,
        test_id="scope_lock_missing_10pp",
    )

    result = recommend(
        continuous_runs=[],
        transient_runs=[canonical_10],
        validation_runs=[],
        target=TargetRequest(
            regime="transient",
            target_waveform="sine",
            command_waveform="sine",
            freq_hz=1.0,
            commanded_cycles=1.0,
            target_type="current",
            target_level_value=20.0,
            target_level_kind="pp",
            context={
                "request_kind": "waveform_compensation",
                "frequency_mode": "exact",
                "finite_cycle_mode": True,
                "exact_matrix_artifact_path": str(exact_matrix_path),
            },
        ),
        options=RecommendationOptions(current_channel="i_sum_signed", field_channel="bz_mT"),
        legacy_context=legacy_10,
    )

    assert result.engine_summary["support_state"] == "unsupported"
    assert result.engine_summary["request_route"] == "unsupported"
    assert result.preview_only is True
    assert result.allow_auto_download is False
    assert result.debug_info["artifact_scope_lock"]["bucket"] == "missing_exact"


def test_artifact_scope_lock_marks_reference_only_continuous_request_as_unsupported(tmp_path: Path) -> None:
    exact_matrix_path = _write_exact_matrix_artifact(
        tmp_path / "exact_matrix_final.json",
        reference_only=[
            {
                "waveform": "sine",
                "freq_hz": 10.0,
                "level_a": 10.0,
                "status": "reference_only",
            }
        ],
        promotion_state="provisional_only",
    )
    per_test_summary, analysis = _build_dummy_analysis(freq_hz=10.0, test_id="scope_lock_ref_only")
    canonical = canonicalize_run(analysis.parsed, regime="continuous", role="train", config=CanonicalizeConfig())

    result = recommend(
        continuous_runs=[canonical],
        transient_runs=[],
        validation_runs=[],
        target=TargetRequest(
            regime="continuous",
            target_waveform="sine",
            command_waveform="sine",
            freq_hz=10.0,
            target_type="current",
            target_level_value=10.0,
            target_level_kind="pp",
            context={
                "request_kind": "waveform_compensation",
                "frequency_mode": "exact",
                "finite_cycle_mode": False,
                "preview_tail_cycles": 0.25,
                "exact_matrix_artifact_path": str(exact_matrix_path),
            },
        ),
        options=RecommendationOptions(current_channel="i_sum_signed", field_channel="bz_mT"),
        legacy_context=LegacyRecommendationContext(
            per_test_summary=per_test_summary,
            analysis_lookup={"scope_lock_ref_only": analysis},
        ),
    )

    assert result.engine_summary["support_state"] == "unsupported"
    assert result.engine_summary["request_route"] == "reference_only"
    assert result.preview_only is True
    assert result.allow_auto_download is False
    assert result.debug_info["artifact_scope_lock"]["bucket"] == "reference_only"


def test_recommend_uses_harmonic_surface_for_exact_continuous_support() -> None:
    per_test_summary, analysis = _build_dummy_analysis()
    canonical = canonicalize_run(analysis.parsed, regime="continuous", role="train", config=CanonicalizeConfig())

    result = recommend(
        continuous_runs=[canonical],
        transient_runs=[],
        validation_runs=[],
        target=TargetRequest(
            regime="continuous",
            target_waveform="sine",
            command_waveform="sine",
            freq_hz=0.5,
            target_type="current",
            target_level_value=10.0,
            target_level_kind="pp",
            context={
                "request_kind": "waveform_compensation",
                "frequency_mode": "exact",
                "finite_cycle_mode": False,
                "preview_tail_cycles": 0.25,
            },
        ),
        options=RecommendationOptions(current_channel="i_sum_signed", field_channel="bz_mT"),
        legacy_context=LegacyRecommendationContext(
            per_test_summary=per_test_summary,
            analysis_lookup={"t1": analysis},
        ),
    )

    assert result.legacy_payload is not None
    assert result.debug_info["steady_state_engine"] == "harmonic_surface"
    assert result.legacy_payload["mode"] == "harmonic_surface_inverse_exact"
    assert result.command_profile is not None
    assert "expected_current_a" in result.command_profile.columns
    assert result.legacy_payload["max_harmonics_used"] >= 1
    assert result.engine_summary["selected_engine"] == "harmonic_surface"
    assert result.engine_summary["support_state"] == "exact"
    assert result.debug_info["request_route"] == "exact"
    assert result.debug_info["plot_source"] == "exact_prediction"
    assert result.debug_info["selected_support_waveform"] == "sine"
    assert result.debug_info["field_prediction_source"] == "current_to_bz_surrogate"
    assert result.debug_info["expected_current_source"] == "exact_current_direct"
    assert result.debug_info["loss_target_type"] == "current"
    assert result.support_summary["exact_freq_match"] is True
    assert result.confidence_summary["surface_confidence"] > 0.0
    assert "gain_input_limit_margin" in result.confidence_summary
    assert "peak_input_limit_margin" in result.confidence_summary
    assert "p95_input_limit_margin" in result.confidence_summary
    assert result.confidence_summary["field_prediction_source"] == "current_to_bz_surrogate"
    assert "support_scaled_current_a" not in result.command_profile.columns
    assert "support_scaled_field_mT" not in result.command_profile.columns


def test_recommend_marks_above_5hz_current_runtime_as_reference_only_preview() -> None:
    per_test_summary, analysis = _build_dummy_analysis(freq_hz=10.0, test_id="t10")
    canonical = canonicalize_run(analysis.parsed, regime="continuous", role="train", config=CanonicalizeConfig())

    result = recommend(
        continuous_runs=[canonical],
        transient_runs=[],
        validation_runs=[],
        target=TargetRequest(
            regime="continuous",
            target_waveform="sine",
            command_waveform="sine",
            freq_hz=10.0,
            target_type="current",
            target_level_value=10.0,
            target_level_kind="pp",
            context={
                "request_kind": "waveform_compensation",
                "frequency_mode": "exact",
                "finite_cycle_mode": False,
                "preview_tail_cycles": 0.25,
            },
        ),
        options=RecommendationOptions(current_channel="i_sum_signed", field_channel="bz_mT"),
        legacy_context=LegacyRecommendationContext(
            per_test_summary=per_test_summary,
            analysis_lookup={"t10": analysis},
        ),
    )

    assert result.engine_summary["support_state"] == "unsupported"
    assert result.engine_summary["request_route"] == "reference_only"
    assert result.preview_only is True
    assert result.allow_auto_download is False
    assert result.engine_summary["artifact_scope_bucket"] == "reference_only"
    assert "official_support_band_exceeded" in result.debug_info["policy_flags"]


def test_recommend_uses_harmonic_surface_preview_only_for_interpolated_frequency() -> None:
    per_test_summary, analysis = _build_dummy_analysis(freq_hz=0.5)
    canonical = canonicalize_run(analysis.parsed, regime="continuous", role="train", config=CanonicalizeConfig())

    result = recommend(
        continuous_runs=[canonical],
        transient_runs=[],
        validation_runs=[],
        target=TargetRequest(
            regime="continuous",
            target_waveform="sine",
            command_waveform="sine",
            freq_hz=0.75,
            target_type="current",
            target_level_value=10.0,
            target_level_kind="pp",
            context={
                "request_kind": "waveform_compensation",
                "frequency_mode": "interpolate",
                "finite_cycle_mode": False,
                "preview_tail_cycles": 0.25,
            },
        ),
        options=RecommendationOptions(current_channel="i_sum_signed", field_channel="bz_mT"),
        legacy_context=LegacyRecommendationContext(
            per_test_summary=per_test_summary,
            analysis_lookup={"t1": analysis},
        ),
    )

    assert result.legacy_payload is not None
    assert result.debug_info["steady_state_engine"] == "harmonic_surface"
    assert result.preview_only is True
    assert result.allow_auto_download is False
    assert result.legacy_payload["mode"] == "harmonic_surface_inverse_interpolated_preview"
    assert result.engine_summary["support_state"] == "out_of_hull"
    assert result.debug_info["request_route"] == "preview"
    assert result.debug_info["plot_source"] == "support_blended_preview"
    assert result.debug_info["selected_support_waveform"] == "sine"
    assert result.confidence_summary["exact_frequency_match"] is False
    assert result.confidence_summary["support_run_count"] >= 1
    assert "support_scaled_current_a" in result.command_profile.columns


def test_recommend_falls_back_to_legacy_for_exact_request_without_exact_support() -> None:
    per_test_summary, analysis = _build_dummy_analysis(freq_hz=0.5)
    canonical = canonicalize_run(analysis.parsed, regime="continuous", role="train", config=CanonicalizeConfig())

    result = recommend(
        continuous_runs=[canonical],
        transient_runs=[],
        validation_runs=[],
        target=TargetRequest(
            regime="continuous",
            target_waveform="sine",
            command_waveform="sine",
            freq_hz=0.75,
            target_type="current",
            target_level_value=10.0,
            target_level_kind="pp",
            context={
                "request_kind": "waveform_compensation",
                "frequency_mode": "exact",
                "finite_cycle_mode": False,
                "preview_tail_cycles": 0.25,
            },
        ),
        options=RecommendationOptions(current_channel="i_sum_signed", field_channel="bz_mT"),
        legacy_context=LegacyRecommendationContext(
            per_test_summary=per_test_summary,
            analysis_lookup={"t1": analysis},
        ),
    )

    assert result.legacy_payload is None
    assert result.debug_info["steady_state_engine"] == "legacy"
    assert result.debug_info["harmonic_surface_debug"]["reason"] == "exact_frequency_support_required_for_exact_request"
    assert result.engine_summary["selected_engine"] == "legacy"
    assert result.engine_summary["support_state"] == "out_of_hull"
    assert result.preview_only is True


def test_policy_can_promote_interpolated_current_recommendation_to_auto() -> None:
    summary_05, analysis_05 = _build_dummy_analysis(freq_hz=0.5, test_id="t05")
    summary_10, analysis_10 = _build_dummy_analysis(freq_hz=1.0, test_id="t10")
    canonical_05 = canonicalize_run(analysis_05.parsed, regime="continuous", role="train", config=CanonicalizeConfig())
    canonical_10 = canonicalize_run(analysis_10.parsed, regime="continuous", role="train", config=CanonicalizeConfig())

    result = recommend(
        continuous_runs=[canonical_05, canonical_10],
        transient_runs=[],
        validation_runs=[],
        target=TargetRequest(
            regime="continuous",
            target_waveform="sine",
            command_waveform="sine",
            freq_hz=0.75,
            target_type="current",
            target_level_value=10.0,
            target_level_kind="pp",
            context={
                "request_kind": "waveform_compensation",
                "frequency_mode": "interpolate",
                "finite_cycle_mode": False,
                "preview_tail_cycles": 0.25,
            },
        ),
        options=RecommendationOptions(current_channel="i_sum_signed", field_channel="bz_mT"),
        legacy_context=LegacyRecommendationContext(
            per_test_summary=pd.concat([summary_05, summary_10], ignore_index=True),
            analysis_lookup={"t05": analysis_05, "t10": analysis_10},
        ),
        policy=RecommendationPolicy(
            min_surface_confidence=0.5,
            min_harmonic_fill_ratio=0.05,
            max_predicted_error_band=0.25,
            min_input_limit_margin=0.1,
            min_support_runs=2,
            allow_interpolated_auto=True,
        ),
    )

    assert result.debug_info["steady_state_engine"] == "harmonic_surface"
    assert result.engine_summary["support_state"] == "interpolated_in_hull"
    assert result.preview_only is False
    assert result.allow_auto_download is True
    assert "interpolated_auto_allowed" in result.debug_info["policy_flags"]


def test_policy_keeps_interpolated_preview_when_harmonic_fill_is_low() -> None:
    summary_05, analysis_05 = _build_dummy_analysis(freq_hz=0.5, test_id="t05")
    summary_10, analysis_10 = _build_dummy_analysis(freq_hz=1.0, test_id="t10")
    canonical_05 = canonicalize_run(analysis_05.parsed, regime="continuous", role="train", config=CanonicalizeConfig())
    canonical_10 = canonicalize_run(analysis_10.parsed, regime="continuous", role="train", config=CanonicalizeConfig())

    result = recommend(
        continuous_runs=[canonical_05, canonical_10],
        transient_runs=[],
        validation_runs=[],
        target=TargetRequest(
            regime="continuous",
            target_waveform="sine",
            command_waveform="sine",
            freq_hz=0.75,
            target_type="current",
            target_level_value=10.0,
            target_level_kind="pp",
            context={
                "request_kind": "waveform_compensation",
                "frequency_mode": "interpolate",
                "finite_cycle_mode": False,
                "preview_tail_cycles": 0.25,
            },
        ),
        options=RecommendationOptions(current_channel="i_sum_signed", field_channel="bz_mT"),
        legacy_context=LegacyRecommendationContext(
            per_test_summary=pd.concat([summary_05, summary_10], ignore_index=True),
            analysis_lookup={"t05": analysis_05, "t10": analysis_10},
        ),
        policy=RecommendationPolicy(
            min_surface_confidence=0.4,
            min_harmonic_fill_ratio=1.1,
            max_predicted_error_band=0.30,
            min_input_limit_margin=0.0,
            min_support_runs=2,
            allow_interpolated_auto=True,
        ),
    )

    assert result.preview_only is True
    assert "insufficient_harmonics" in result.debug_info["policy_flags"]


def test_policy_blocks_interpolated_auto_when_input_headroom_is_low() -> None:
    summary_05, analysis_05 = _build_dummy_analysis(freq_hz=0.5, test_id="t05")
    summary_10, analysis_10 = _build_dummy_analysis(freq_hz=1.0, test_id="t10")
    canonical_05 = canonicalize_run(analysis_05.parsed, regime="continuous", role="train", config=CanonicalizeConfig())
    canonical_10 = canonicalize_run(analysis_10.parsed, regime="continuous", role="train", config=CanonicalizeConfig())

    result = recommend(
        continuous_runs=[canonical_05, canonical_10],
        transient_runs=[],
        validation_runs=[],
        target=TargetRequest(
            regime="continuous",
            target_waveform="sine",
            command_waveform="sine",
            freq_hz=0.75,
            target_type="current",
            target_level_value=10.0,
            target_level_kind="pp",
            context={
                "request_kind": "waveform_compensation",
                "frequency_mode": "interpolate",
                "finite_cycle_mode": False,
                "preview_tail_cycles": 0.25,
            },
        ),
        options=RecommendationOptions(current_channel="i_sum_signed", field_channel="bz_mT"),
        legacy_context=LegacyRecommendationContext(
            per_test_summary=pd.concat([summary_05, summary_10], ignore_index=True),
            analysis_lookup={"t05": analysis_05, "t10": analysis_10},
        ),
        policy=RecommendationPolicy(
            min_surface_confidence=0.4,
            min_harmonic_fill_ratio=0.05,
            max_predicted_error_band=0.30,
            min_input_limit_margin=0.95,
            min_support_runs=2,
            allow_interpolated_auto=True,
        ),
    )

    assert result.preview_only is True
    assert "low_input_headroom" in result.debug_info["policy_flags"]


def test_policy_keeps_exact_auto_and_tracks_shape_risk_as_advisory() -> None:
    result = RecommendationResult(
        selected_regime="continuous",
        preview_only=False,
        allow_auto_download=True,
        recommended_time_s=np.linspace(0.0, 1.0, 16),
        recommended_input_v=np.linspace(0.0, 1.0, 16),
        predicted_current_a=None,
        predicted_bx_mT=None,
        predicted_by_mT=None,
        predicted_bz_mT=None,
        validation_report=ValidationReport(
            in_support=True,
            exact_freq_match=True,
            exact_cycle_match=True,
            shape_quality=1.0,
            expected_error_band=0.05,
            allow_auto_recommendation=True,
            reasons=[],
        ),
        engine_summary={
            "selected_engine": "harmonic_surface",
            "exact_frequency_match": True,
            "support_state": "exact",
        },
        confidence_summary={
            "predicted_shape_corr": 0.61,
            "predicted_nrmse": 0.42,
            "predicted_phase_lag_cycles": 0.18,
            "predicted_clipping": False,
        },
    )

    decision = evaluate_recommendation_policy(
        result=result,
        target=TargetRequest(
            regime="continuous",
            target_waveform="sine",
            command_waveform="sine",
            freq_hz=1.0,
            target_type="current",
            target_level_value=20.0,
            target_level_kind="pp",
        ),
        policy=RecommendationPolicy(),
    )

    assert decision.allow_auto_recommendation is True
    assert decision.preview_only is False
    assert decision.reasons == []
    assert "predicted_shape_low" in decision.policy_flags
    assert "predicted_nrmse_high" in decision.policy_flags
    assert "exact_support_shape_quality_advisory" in decision.policy_flags


def test_policy_keeps_exact_auto_for_release_candidate_quality_band() -> None:
    result = RecommendationResult(
        selected_regime="continuous",
        preview_only=False,
        allow_auto_download=True,
        recommended_time_s=np.linspace(0.0, 1.0, 16),
        recommended_input_v=np.linspace(0.0, 1.0, 16),
        predicted_current_a=None,
        predicted_bx_mT=None,
        predicted_by_mT=None,
        predicted_bz_mT=None,
        validation_report=ValidationReport(
            in_support=True,
            exact_freq_match=True,
            exact_cycle_match=True,
            shape_quality=1.0,
            expected_error_band=0.05,
            allow_auto_recommendation=True,
            reasons=[],
        ),
        engine_summary={
            "selected_engine": "harmonic_surface",
            "exact_frequency_match": True,
            "support_state": "exact",
        },
        confidence_summary={
            "predicted_shape_corr": 0.884,
            "predicted_nrmse": 0.339,
            "predicted_phase_lag_cycles": 0.071,
            "predicted_clipping": False,
        },
    )

    decision = evaluate_recommendation_policy(
        result=result,
        target=TargetRequest(
            regime="continuous",
            target_waveform="sine",
            command_waveform="sine",
            freq_hz=0.5,
            target_type="current",
            target_level_value=20.0,
            target_level_kind="pp",
        ),
        policy=RecommendationPolicy(),
    )

    assert decision.allow_auto_recommendation is True
    assert decision.preview_only is False
    assert "exact_support_auto" in decision.policy_flags


def test_policy_blocks_exact_field_auto_when_field_prediction_collapses() -> None:
    result = RecommendationResult(
        selected_regime="continuous",
        preview_only=False,
        allow_auto_download=True,
        recommended_time_s=np.linspace(0.0, 1.0, 16),
        recommended_input_v=np.linspace(0.0, 1.0, 16),
        predicted_current_a=None,
        predicted_bx_mT=None,
        predicted_by_mT=None,
        predicted_bz_mT=np.zeros(16, dtype=float),
        validation_report=ValidationReport(
            in_support=True,
            exact_freq_match=True,
            exact_cycle_match=True,
            shape_quality=1.0,
            expected_error_band=0.05,
            allow_auto_recommendation=True,
            reasons=[],
        ),
        engine_summary={
            "selected_engine": "harmonic_surface",
            "request_kind": "waveform_compensation",
            "exact_frequency_match": True,
            "support_state": "exact",
            "request_route": "exact",
            "solver_route": "harmonic_surface_inverse_exact",
            "plot_source": "exact_prediction",
        },
        confidence_summary={
            "predicted_shape_corr": 0.99,
            "predicted_nrmse": 0.01,
            "predicted_phase_lag_cycles": 0.0,
            "predicted_clipping": False,
            "field_prediction_source": "exact_field_direct",
            "field_prediction_status": "unavailable",
            "field_prediction_available": False,
            "zero_field_reason": "expected_field_near_zero",
        },
    )

    decision = evaluate_recommendation_policy(
        result=result,
        target=TargetRequest(
            regime="continuous",
            target_waveform="sine",
            command_waveform="sine",
            freq_hz=0.25,
            target_type="field",
            target_level_value=20.0,
            target_level_kind="pp",
        ),
        policy=RecommendationPolicy(),
    )

    assert decision.allow_auto_recommendation is False
    assert decision.preview_only is True
    assert "exact_field_prediction_block" in decision.policy_flags
    assert "expected_field_near_zero" in decision.reasons


def test_select_nearest_support_row_prefers_level_stable_family_for_exact_field() -> None:
    subset = pd.DataFrame(
        [
            {
                "test_id": "family_locked_low_current",
                "freq_hz": 0.25,
                "current_pp_target_a": 5.0,
                "achieved_current_pp_a_mean": 5.0,
                "achieved_bz_mT_pp_mean": 50.0,
            },
            {
                "test_id": "nearest_output_high_current",
                "freq_hz": 0.25,
                "current_pp_target_a": 30.0,
                "achieved_current_pp_a_mean": 30.0,
                "achieved_bz_mT_pp_mean": 32.0,
            },
        ]
    )

    selected, meta = _select_nearest_support_row(
        subset=subset,
        target_freq_hz=0.25,
        target_output_pp=30.0,
        output_metric="achieved_bz_mT_pp_mean",
        prefer_level_stable_family=True,
    )

    assert selected["test_id"] == "family_locked_low_current"
    assert meta["support_selection_reason"] == "exact_family_level_lock"
    assert meta["support_family_lock_applied"] is True
    assert meta["selected_support_family"] == "current_pp_target_a:5"


def test_exact_continuous_sine_and_triangle_commands_have_distinct_fft_signatures() -> None:
    summary_sine, analysis_sine = _build_dummy_analysis(freq_hz=1.0, waveform_type="sine", test_id="sine_exact")
    summary_triangle, analysis_triangle = _build_dummy_analysis(freq_hz=1.0, waveform_type="triangle", test_id="triangle_exact")
    canonical_sine = canonicalize_run(analysis_sine.parsed, regime="continuous", role="train", config=CanonicalizeConfig())
    canonical_triangle = canonicalize_run(analysis_triangle.parsed, regime="continuous", role="train", config=CanonicalizeConfig())
    legacy_summary = pd.concat([summary_sine, summary_triangle], ignore_index=True)
    legacy_lookup = {"sine_exact": analysis_sine, "triangle_exact": analysis_triangle}

    sine_result = recommend(
        continuous_runs=[canonical_sine, canonical_triangle],
        transient_runs=[],
        validation_runs=[],
        target=TargetRequest(
            regime="continuous",
            target_waveform="sine",
            command_waveform="sine",
            freq_hz=1.0,
            target_type="current",
            target_level_value=10.0,
            target_level_kind="pp",
            context={"request_kind": "waveform_compensation", "frequency_mode": "exact", "finite_cycle_mode": False},
        ),
        options=RecommendationOptions(current_channel="i_sum_signed", field_channel="bz_mT"),
        legacy_context=LegacyRecommendationContext(per_test_summary=legacy_summary, analysis_lookup=legacy_lookup),
    )
    triangle_result = recommend(
        continuous_runs=[canonical_sine, canonical_triangle],
        transient_runs=[],
        validation_runs=[],
        target=TargetRequest(
            regime="continuous",
            target_waveform="triangle",
            command_waveform="triangle",
            freq_hz=1.0,
            target_type="current",
            target_level_value=10.0,
            target_level_kind="pp",
            context={"request_kind": "waveform_compensation", "frequency_mode": "exact", "finite_cycle_mode": False},
        ),
        options=RecommendationOptions(current_channel="i_sum_signed", field_channel="bz_mT"),
        legacy_context=LegacyRecommendationContext(per_test_summary=legacy_summary, analysis_lookup=legacy_lookup),
    )

    assert sine_result.command_profile is not None
    assert triangle_result.command_profile is not None
    sine_voltage = pd.to_numeric(sine_result.command_profile["limited_voltage_v"], errors="coerce").to_numpy(dtype=float)
    triangle_voltage = pd.to_numeric(triangle_result.command_profile["limited_voltage_v"], errors="coerce").to_numpy(dtype=float)
    sine_fft = np.abs(np.fft.rfft(sine_voltage - float(np.nanmean(sine_voltage))))
    triangle_fft = np.abs(np.fft.rfft(triangle_voltage - float(np.nanmean(triangle_voltage))))
    sine_odd_energy = float(np.sum(sine_fft[[3, 5, 7]])) if len(sine_fft) > 7 else 0.0
    triangle_odd_energy = float(np.sum(triangle_fft[[3, 5, 7]])) if len(triangle_fft) > 7 else 0.0

    assert triangle_result.debug_info["plot_source"] == "exact_prediction"
    assert triangle_result.debug_info["selected_support_waveform"] == "triangle"
    assert triangle_odd_energy > sine_odd_energy * 1.5


def test_recommend_records_policy_version_and_snapshot_from_config() -> None:
    summary_05, analysis_05 = _build_dummy_analysis(freq_hz=0.5, test_id="t05")
    summary_10, analysis_10 = _build_dummy_analysis(freq_hz=1.0, test_id="t10")
    canonical_05 = canonicalize_run(analysis_05.parsed, regime="continuous", role="train", config=CanonicalizeConfig())
    canonical_10 = canonicalize_run(analysis_10.parsed, regime="continuous", role="train", config=CanonicalizeConfig())

    policy_config = RecommendationPolicyConfig(
        version="test-v7-1",
        thresholds=RecommendationPolicyThresholds(
            min_surface_confidence=0.5,
            min_harmonic_fill_ratio=0.05,
            max_predicted_error_band=0.25,
            min_input_limit_margin=0.1,
            min_support_runs=2,
        ),
        allow_interpolated_auto=True,
    )

    result = recommend(
        continuous_runs=[canonical_05, canonical_10],
        transient_runs=[],
        validation_runs=[],
        target=TargetRequest(
            regime="continuous",
            target_waveform="sine",
            command_waveform="sine",
            freq_hz=0.75,
            target_type="current",
            target_level_value=10.0,
            target_level_kind="pp",
            context={
                "request_kind": "waveform_compensation",
                "frequency_mode": "interpolate",
                "finite_cycle_mode": False,
                "preview_tail_cycles": 0.25,
            },
        ),
        options=RecommendationOptions(current_channel="i_sum_signed", field_channel="bz_mT"),
        legacy_context=LegacyRecommendationContext(
            per_test_summary=pd.concat([summary_05, summary_10], ignore_index=True),
            analysis_lookup={"t05": analysis_05, "t10": analysis_10},
        ),
        policy_config=policy_config,
    )

    assert result.debug_info["policy_version"] == "test-v7-1"
    assert result.debug_info["policy_snapshot"]["thresholds"]["min_surface_confidence"] == 0.5
    assert result.debug_info["policy_snapshot"]["margin_source"] == "gain"
    assert result.engine_summary["policy_version"] == "test-v7-1"


def test_recommend_propagates_finite_exact_route_debug_fields() -> None:
    canonical, legacy_context = _build_dummy_transient_support(
        waveform_type="triangle",
        freq_hz=1.0,
        cycle_count=1.0,
        requested_level_pp=20.0,
        test_id="tri_exact_debug",
    )

    result = recommend(
        continuous_runs=[],
        transient_runs=[canonical],
        validation_runs=[],
        target=TargetRequest(
            regime="transient",
            target_waveform="triangle",
            command_waveform="triangle",
            freq_hz=1.0,
            commanded_cycles=1.0,
            target_type="current",
            target_level_value=20.0,
            target_level_kind="pp",
            context={"request_kind": "waveform_compensation", "frequency_mode": "exact", "finite_cycle_mode": True},
        ),
        options=RecommendationOptions(current_channel="i_sum_signed", field_channel="bz_mT"),
        legacy_context=legacy_context,
    )

    assert result.preview_only is False
    assert result.debug_info["request_route"] == "exact"
    assert result.debug_info["plot_source"] == "exact_prediction"
    assert result.debug_info["selected_support_waveform"] == "triangle"
    assert str(result.debug_info["selected_support_id"]).endswith("__triangle__train__1Hz")
    assert result.debug_info["field_prediction_source"] == "exact_field_direct"
    assert result.debug_info["expected_current_source"] == "exact_current_direct"
    assert "active_window_start_s" in result.debug_info
    assert "active_window_end_s" in result.debug_info


def test_recommend_propagates_finite_preview_route_debug_fields() -> None:
    canonical, legacy_context = _build_dummy_transient_support(
        waveform_type="sine",
        freq_hz=1.0,
        cycle_count=1.0,
        requested_level_pp=10.0,
        test_id="sine_preview_debug",
    )

    result = recommend(
        continuous_runs=[],
        transient_runs=[canonical],
        validation_runs=[],
        target=TargetRequest(
            regime="transient",
            target_waveform="sine",
            command_waveform="sine",
            freq_hz=1.0,
            commanded_cycles=1.0,
            target_type="current",
            target_level_value=20.0,
            target_level_kind="pp",
            context={"request_kind": "waveform_compensation", "frequency_mode": "exact", "finite_cycle_mode": True},
        ),
        options=RecommendationOptions(current_channel="i_sum_signed", field_channel="bz_mT"),
        legacy_context=legacy_context,
    )

    assert result.preview_only is True
    assert result.debug_info["request_route"] == "provisional"
    assert result.debug_info["plot_source"] == "support_blended_preview"
    assert result.debug_info["selected_support_waveform"] == "sine"


def test_sanitize_unavailable_exact_field_prediction_replaces_zero_signal_with_nan() -> None:
    profile = pd.DataFrame(
        {
            "time_s": np.linspace(0.0, 1.0, 17),
            "expected_field_mT": np.zeros(17, dtype=float),
            "modeled_field_mT": np.zeros(17, dtype=float),
            "expected_output": np.zeros(17, dtype=float),
            "target_field_mT": np.sin(np.linspace(0.0, 2.0 * np.pi, 17)),
        }
    )
    sanitized = sanitize_unavailable_exact_field_prediction(
        profile,
        {
            "request_route": "exact",
            "field_prediction_status": "unavailable",
        },
        target_output_type="field",
    )

    assert np.isnan(pd.to_numeric(sanitized["expected_field_mT"], errors="coerce")).all()
    assert np.isnan(pd.to_numeric(sanitized["modeled_field_mT"], errors="coerce")).all()
    assert np.isnan(pd.to_numeric(sanitized["expected_output"], errors="coerce")).all()


def test_level_sensitivity_diagnosis_classifies_prediction_source_switch() -> None:
    payload = build_level_sensitivity_diagnosis(
        [
            {
                "recommendation_id": "case_low",
                "waveform_type": "triangle",
                "freq_hz": 1.0,
                "commanded_cycles": 1.0,
                "target_type": "field",
                "target_level_value": 10.0,
                "selected_support_id": "support_low",
                "solver_route": "finite_exact_direct",
                "field_prediction_source": "exact_field_direct",
                "plot_source": "exact_prediction",
                "clipping_flags": {"within_hardware_limits": True},
                "harmonic_weights": {1: 1.0, 3: 2.4},
                "expected_field_mT": np.sin(np.linspace(0.0, 2.0 * np.pi, 65)),
            },
            {
                "recommendation_id": "case_high",
                "waveform_type": "triangle",
                "freq_hz": 1.0,
                "commanded_cycles": 1.0,
                "target_type": "field",
                "target_level_value": 20.0,
                "selected_support_id": "support_high",
                "solver_route": "finite_exact_direct",
                "field_prediction_source": "support_blended_preview",
                "plot_source": "support_blended_preview",
                "clipping_flags": {"within_hardware_limits": True},
                "harmonic_weights": {1: 1.0, 3: 2.4},
                "expected_field_mT": np.sign(np.sin(np.linspace(0.0, 2.0 * np.pi, 65))),
            },
        ]
    )

    assert payload["summary"]["comparison_count"] == 1
    assert payload["summary"]["prediction_source_switch"] == 1
    assert "prediction_source_switch" in payload["comparisons"][0]["switch_types"]


def test_lcr_runtime_audits_exact_field_support_without_prior_override() -> None:
    summary_05, analysis_05 = _build_dummy_analysis(freq_hz=0.5, test_id="t05")
    summary_10, analysis_10 = _build_dummy_analysis(freq_hz=1.0, test_id="t10")
    canonical_05 = canonicalize_run(analysis_05.parsed, regime="continuous", role="train", config=CanonicalizeConfig())
    canonical_10 = canonicalize_run(analysis_10.parsed, regime="continuous", role="train", config=CanonicalizeConfig())

    result = recommend(
        continuous_runs=[canonical_05, canonical_10],
        transient_runs=[],
        validation_runs=[],
        target=TargetRequest(
            regime="continuous",
            target_waveform="sine",
            command_waveform="sine",
            freq_hz=0.5,
            target_type="field",
            target_level_value=20.0,
            target_level_kind="pp",
            context={
                "request_kind": "waveform_compensation",
                "frequency_mode": "exact",
                "finite_cycle_mode": False,
            },
        ),
        options=RecommendationOptions(
            current_channel="i_sum_signed",
            field_channel="bz_mT",
            lcr_measurements=_build_dummy_lcr_measurements(),
            lcr_blend_weight=0.6,
        ),
        legacy_context=LegacyRecommendationContext(
            per_test_summary=pd.concat([summary_05, summary_10], ignore_index=True),
            analysis_lookup={"t05": analysis_05, "t10": analysis_10},
        ),
    )

    assert result.engine_summary["request_route"] == "exact"
    assert result.debug_info["exact_field_support_present"] is True
    assert result.debug_info["lcr_usage_mode"] == "audit_only"
    assert result.debug_info["lcr_blend_weight"] == 0.0
    assert result.debug_info["used_lcr_prior"] is False
    assert result.debug_info["lcr_phase_anchor_used"] is False
    assert result.debug_info["lcr_gain_prior_used"] is False


def test_lcr_runtime_caps_preview_gap_to_weak_prior() -> None:
    summary_05, analysis_05 = _build_dummy_analysis(freq_hz=0.5, test_id="t05")
    summary_10, analysis_10 = _build_dummy_analysis(freq_hz=1.0, test_id="t10")
    canonical_05 = canonicalize_run(analysis_05.parsed, regime="continuous", role="train", config=CanonicalizeConfig())
    canonical_10 = canonicalize_run(analysis_10.parsed, regime="continuous", role="train", config=CanonicalizeConfig())

    result = recommend(
        continuous_runs=[canonical_05, canonical_10],
        transient_runs=[],
        validation_runs=[],
        target=TargetRequest(
            regime="continuous",
            target_waveform="sine",
            command_waveform="sine",
            freq_hz=0.75,
            target_type="field",
            target_level_value=20.0,
            target_level_kind="pp",
            context={
                "request_kind": "waveform_compensation",
                "frequency_mode": "interpolate",
                "finite_cycle_mode": False,
            },
        ),
        options=RecommendationOptions(
            current_channel="i_sum_signed",
            field_channel="bz_mT",
            lcr_measurements=_build_dummy_lcr_measurements(),
            lcr_blend_weight=0.6,
            frequency_mode="interpolate",
        ),
        legacy_context=LegacyRecommendationContext(
            per_test_summary=pd.concat([summary_05, summary_10], ignore_index=True),
            analysis_lookup={"t05": analysis_05, "t10": analysis_10},
        ),
    )

    assert result.engine_summary["request_route"] == "preview"
    assert result.debug_info["exact_field_support_present"] is False
    assert result.debug_info["lcr_usage_mode"] == "weak_prior"
    assert result.debug_info["lcr_blend_weight"] == 0.15
    assert result.debug_info["used_lcr_prior"] is True
    assert result.debug_info["lcr_phase_anchor_used"] is True
    assert result.debug_info["lcr_gain_prior_used"] is True


def test_lcr_influence_audit_report_passes() -> None:
    payload = lcr_audit_script.build_lcr_influence_audit()

    assert payload["success"] is True
    records = {record["scenario_id"]: record for record in payload["records"]}
    assert records["continuous_field_exact_support"]["lcr_usage_mode"] == "audit_only"
    assert records["continuous_field_exact_support"]["lcr_weight"] == 0.0
    assert records["continuous_field_preview_gap"]["lcr_usage_mode"] == "weak_prior"
    assert records["continuous_field_preview_gap"]["lcr_weight"] == 0.15
