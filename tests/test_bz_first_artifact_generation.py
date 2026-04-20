from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import check_matrix_runtime_health as healthcheck_script
import generate_bz_first_artifacts as artifact_script
import refresh_exact_supported_scope as refresh_script
import report_exact_and_finite_scope as scope_script


def _write_profile(
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


def _write_sidecar(path: Path, **payload: object) -> None:
    artifact_script.write_json(path.with_suffix(".json"), payload)


def test_scope_payload_classifies_provisional_and_reference_only(tmp_path: Path) -> None:
    continuous_dir = tmp_path / "uploads" / "continuous"
    transient_dir = tmp_path / "uploads" / "transient"
    continuous_dir.mkdir(parents=True, exist_ok=True)
    (continuous_dir / "sine_0.5_20.csv").touch()
    (continuous_dir / "tri_10_5.csv").touch()
    (transient_dir / "sinusidal" / "1hz_1cycle_20pp.csv").parent.mkdir(parents=True, exist_ok=True)
    (transient_dir / "sinusidal" / "1hz_1cycle_20pp.csv").touch()
    (transient_dir / "sinusidal" / "1hz_1cycle_10pp.csv").touch()
    (transient_dir / "triangle" / "1hz_1cycle_10pp.csv").parent.mkdir(parents=True, exist_ok=True)
    (transient_dir / "triangle" / "1hz_1cycle_10pp.csv").touch()

    payload = scope_script.build_scope_payload(
        continuous_dir=continuous_dir,
        transient_dir=transient_dir,
    )

    assert payload["continuous_reference_above_band"]["summary"][0]["freq_hz"] == 10.0
    assert payload["finite_all_exact_scope"]["official_recipe_total"] == 2
    assert payload["finite_all_exact_scope"]["provisional_preview_combinations"][0]["measured_file_present"] is True
    assert payload["finite_all_exact_scope"]["missing_exact_combinations"][0]["status"] == "missing_exact"
    assert payload["finite_all_exact_scope"]["promotion_status"]["state"] == "provisional_only"


def test_scope_payload_promotes_provisional_cell_when_exact_upload_arrives(tmp_path: Path) -> None:
    transient_dir = tmp_path / "uploads" / "transient"
    sine_dir = transient_dir / "sinusidal"
    triangle_dir = transient_dir / "triangle"
    sine_dir.mkdir(parents=True, exist_ok=True)
    triangle_dir.mkdir(parents=True, exist_ok=True)
    (sine_dir / "1hz_1cycle_10pp.csv").touch()
    (sine_dir / "1hz_1cycle_20pp.csv").touch()
    (triangle_dir / "1hz_1cycle_10pp.csv").touch()

    baseline = scope_script.build_scope_payload(
        continuous_dir=tmp_path / "uploads" / "continuous",
        transient_dir=transient_dir,
    )
    assert baseline["finite_all_exact_scope"]["official_recipe_total"] == 2
    assert len(baseline["finite_all_exact_scope"]["provisional_preview_combinations"]) == 1
    assert len(baseline["finite_all_exact_scope"]["missing_exact_combinations"]) == 1

    (sine_dir / "abcd1234_1hz_1cycle_20pp.csv").touch()
    promoted = scope_script.build_scope_payload(
        continuous_dir=tmp_path / "uploads" / "continuous",
        transient_dir=transient_dir,
    )

    assert promoted["finite_all_exact_scope"]["official_recipe_total"] == 3
    assert promoted["finite_all_exact_scope"]["promotion_status"]["state"] == "promoted_to_exact"
    assert promoted["finite_all_exact_scope"]["promotion_status"]["measured_exact_available"] is True
    assert promoted["finite_all_exact_scope"]["promotion_status"]["promoted_exact_source_files"] == [
        "sinusidal/abcd1234_1hz_1cycle_20pp.csv"
    ]
    assert promoted["finite_all_exact_scope"]["provisional_preview_combinations"] == []
    assert promoted["finite_all_exact_scope"]["missing_exact_combinations"] == []


def test_catalog_reclassifies_support_using_matrix_not_manifest(tmp_path: Path) -> None:
    continuous_dir = tmp_path / "uploads" / "continuous"
    transient_dir = tmp_path / "uploads" / "transient"
    (continuous_dir / "sine_0.5_20.csv").parent.mkdir(parents=True, exist_ok=True)
    (continuous_dir / "sine_0.5_20.csv").touch()
    (continuous_dir / "tri_10_5.csv").touch()
    (transient_dir / "sinusidal" / "1hz_1cycle_10pp.csv").parent.mkdir(parents=True, exist_ok=True)
    (transient_dir / "sinusidal" / "1hz_1cycle_10pp.csv").touch()
    (transient_dir / "sinusidal" / "1hz_1cycle_20pp.csv").touch()
    scope = scope_script.build_scope_payload(continuous_dir=continuous_dir, transient_dir=transient_dir)

    policy_scope_path = tmp_path / "exact_and_finite_scope.json"
    artifact_script.write_json(policy_scope_path, scope)

    recommendation_library_dir = tmp_path / "recommendation_library"
    _write_profile(
        recommendation_library_dir / "steady_state_harmonic_sine_1Hz_current_20_1cycle.csv",
        waveform="sine",
        freq_hz=1.0,
        target_type="current",
        cycle_count=1.0,
        level_pp=20.0,
        request_route="exact",
    )
    _write_profile(
        recommendation_library_dir / "control_formula_triangle_6Hz_current_20.csv",
        waveform="triangle",
        freq_hz=6.0,
        target_type="current",
        cycle_count=1.25,
        level_pp=20.0,
        request_route="exact",
    )
    _write_profile(
        recommendation_library_dir / "control_formula_sine_0.75Hz_current_20.csv",
        waveform="sine",
        freq_hz=0.75,
        target_type="current",
        level_pp=20.0,
        request_route="preview",
        plot_source="support_blended_preview",
    )
    _write_profile(
        recommendation_library_dir / "control_formula_sine_0.5Hz_field_20.csv",
        waveform="sine",
        freq_hz=0.5,
        target_type="field",
        level_pp=20.0,
        request_route="exact",
    )

    catalog = artifact_script.build_lut_catalog(
        scope=scope,
        validation_catalog=[],
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

    by_id = {entry["lut_id"]: entry for entry in catalog}
    assert by_id["steady_state_harmonic_sine_1Hz_current_20_1cycle"]["status"] == "provisional_experimental"
    assert by_id["steady_state_harmonic_sine_1Hz_current_20_1cycle"]["source_route"] == "provisional_preview"
    assert by_id["control_formula_triangle_6Hz_current_20"]["status"] == "deprecated"
    assert by_id["control_formula_triangle_6Hz_current_20"]["source_route"] == "reference_only"
    assert by_id["control_formula_sine_0.75Hz_current_20"]["status"] == "preview_only"
    assert by_id["control_formula_sine_0.5Hz_field_20"]["status"] == "software_ready_bench_pending"


def test_runtime_duplicate_and_stale_catalog_flags(tmp_path: Path) -> None:
    continuous_dir = tmp_path / "uploads" / "continuous"
    transient_dir = tmp_path / "uploads" / "transient"
    continuous_dir.mkdir(parents=True, exist_ok=True)
    transient_dir.mkdir(parents=True, exist_ok=True)
    (continuous_dir / "sine_0.5_20.csv").touch()
    scope = scope_script.build_scope_payload(continuous_dir=continuous_dir, transient_dir=transient_dir)

    policy_scope_path = tmp_path / "exact_and_finite_scope.json"
    artifact_script.write_json(policy_scope_path, scope)
    recommendation_library_dir = tmp_path / "recommendation_library"

    older_profile = recommendation_library_dir / "control_formula_sine_0.5Hz_current_20.csv"
    newer_profile = recommendation_library_dir / "steady_state_harmonic_sine_0.5Hz_current_20.csv"
    _write_profile(
        older_profile,
        waveform="sine",
        freq_hz=0.5,
        target_type="current",
        level_pp=20.0,
        request_route="exact",
    )
    _write_profile(
        newer_profile,
        waveform="sine",
        freq_hz=0.5,
        target_type="current",
        level_pp=20.0,
        request_route="exact",
    )
    _write_sidecar(older_profile, created_at="2026-01-01T00:00:00+00:00")
    _write_sidecar(newer_profile, created_at="2026-01-02T00:00:00+00:00")

    catalog_entries = artifact_script.build_lut_catalog(
        scope=scope,
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
    catalog_payload = artifact_script.build_lut_catalog_payload(catalog_entries)
    by_id = {entry["lut_id"]: entry for entry in catalog_entries}

    assert by_id["control_formula_sine_0.5Hz_current_20"]["duplicate_runtime"] is True
    assert by_id["control_formula_sine_0.5Hz_current_20"]["stale_runtime"] is True
    assert by_id["control_formula_sine_0.5Hz_current_20"]["runtime_preferred_lut_id"] == "steady_state_harmonic_sine_0.5Hz_current_20"
    assert by_id["steady_state_harmonic_sine_0.5Hz_current_20"]["duplicate_runtime"] is True
    assert by_id["steady_state_harmonic_sine_0.5Hz_current_20"]["stale_runtime"] is False
    assert catalog_payload["summary"]["duplicate_runtime_entries"] == 2
    assert catalog_payload["summary"]["stale_runtime_entries"] == 1
    assert by_id["control_formula_sine_0.5Hz_current_20"]["display_label"] == "current / sine / 0.5 Hz / 20 A | recommendation"
    assert by_id["steady_state_harmonic_sine_0.5Hz_current_20"]["display_label"] == "current / sine / 0.5 Hz / 20 A | recommendation"


def test_roi_priority_reorders_after_missing_cell_promotion(tmp_path: Path) -> None:
    transient_dir = tmp_path / "uploads" / "transient"
    sine_dir = transient_dir / "sinusidal"
    sine_dir.mkdir(parents=True, exist_ok=True)
    (sine_dir / "1hz_1cycle_10pp.csv").touch()
    (sine_dir / "1hz_1cycle_20pp.csv").touch()

    baseline_scope = scope_script.build_scope_payload(
        continuous_dir=tmp_path / "uploads" / "continuous",
        transient_dir=transient_dir,
    )
    baseline_roi = artifact_script.build_measurement_roi_priority(scope=baseline_scope, validation_catalog=[])
    assert baseline_roi["priorities"][0]["category"] == "missing_exact_promotion"

    (sine_dir / "abcd1234_1hz_1cycle_20pp.csv").touch()
    promoted_scope = scope_script.build_scope_payload(
        continuous_dir=tmp_path / "uploads" / "continuous",
        transient_dir=transient_dir,
    )
    promoted_roi = artifact_script.build_measurement_roi_priority(scope=promoted_scope, validation_catalog=[])

    assert promoted_scope["finite_all_exact_scope"]["promotion_status"]["state"] == "promoted_to_exact"
    assert promoted_roi["priorities"][0]["category"] == "continuous_exact_gap_fill"
    assert all(item["category"] != "missing_exact_promotion" for item in promoted_roi["priorities"])
    assert "lineage" not in " ".join(str(item.get("request") or "") for item in promoted_roi["priorities"]).lower()


def test_label_sanitization_report_and_healthcheck_pass_for_clean_payloads(tmp_path: Path) -> None:
    continuous_dir = tmp_path / "uploads" / "continuous"
    transient_dir = tmp_path / "uploads" / "transient"
    (continuous_dir / "sine_0.5_20.csv").parent.mkdir(parents=True, exist_ok=True)
    (continuous_dir / "sine_0.5_20.csv").touch()
    (transient_dir / "sinusidal" / "1hz_1cycle_10pp.csv").parent.mkdir(parents=True, exist_ok=True)
    (transient_dir / "sinusidal" / "1hz_1cycle_10pp.csv").touch()
    (transient_dir / "sinusidal" / "1hz_1cycle_20pp.csv").touch()
    scope = scope_script.build_scope_payload(continuous_dir=continuous_dir, transient_dir=transient_dir)

    policy_scope_path = tmp_path / "exact_and_finite_scope.json"
    artifact_script.write_json(policy_scope_path, scope)
    recommendation_library_dir = tmp_path / "recommendation_library"
    _write_profile(
        recommendation_library_dir / "steady_state_harmonic_sine_1Hz_current_20_1cycle.csv",
        waveform="sine",
        freq_hz=1.0,
        target_type="current",
        cycle_count=1.0,
        level_pp=20.0,
        request_route="exact",
    )

    catalog_entries = artifact_script.build_lut_catalog(
        scope=scope,
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
    catalog_payload = artifact_script.build_lut_catalog_payload(catalog_entries)
    validation_payload = {"entries": []}
    corrected_payload = {"entries": []}
    picker_payload = {"entries": catalog_entries}
    exact_matrix_payload = artifact_script.build_exact_matrix(scope)
    roi_payload = artifact_script.build_measurement_roi_priority(scope=scope, validation_catalog=[])

    label_report = artifact_script.build_label_sanitization_report(
        exact_matrix=exact_matrix_payload,
        catalog_payload=catalog_payload,
        validation_payload=validation_payload,
        corrected_payload=corrected_payload,
        picker_payload=picker_payload,
        roi_payload=roi_payload,
    )
    health_payload = healthcheck_script.build_runtime_display_label_healthcheck(
        label_report,
        output_json=tmp_path / "runtime_display_label_healthcheck.json",
        output_md=tmp_path / "runtime_display_label_healthcheck.md",
    )

    assert label_report["success"] is True
    assert label_report["summary"]["leak_violations"] == 0
    assert label_report["summary"]["display_name_mismatches"] == 0
    assert label_report["summary"]["display_label_mismatches"] == 0
    assert health_payload["success"] is True


def test_scope_promotion_smoke_passes() -> None:
    smoke = scope_script.run_provisional_promotion_smoke()

    assert smoke["pass"] is True
    assert smoke["baseline"]["promotion_state"] == "provisional_only"
    assert smoke["promoted"]["promotion_state"] == "promoted_to_exact"
    assert smoke["promoted"]["official_recipe_total"] == smoke["baseline"]["official_recipe_total"] + 1


def test_refresh_scope_pipeline_includes_bz_artifact_generator() -> None:
    script_names = [path.name for path in refresh_script.SCRIPTS]
    assert "generate_bz_first_artifacts.py" in script_names
