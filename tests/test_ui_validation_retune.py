from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIELD_ANALYSIS_SRC = ROOT.parent / "src"
if str(FIELD_ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(FIELD_ANALYSIS_SRC))

import field_analysis.ui_validation_retune as ui_retune  # noqa: E402
from field_analysis.ui_upload_state import build_upload_state_paths  # noqa: E402


def _parsed_measurement(*, test_id: str, waveform_type: str, freq_hz: float, source_file: str) -> SimpleNamespace:
    normalized_frame = pd.DataFrame(
        {
            "test_id": [test_id] * 4,
            "waveform_type": [waveform_type] * 4,
            "freq_hz": [freq_hz] * 4,
            "time_s": [0.0, 0.1, 0.2, 0.3],
        }
    )
    return SimpleNamespace(
        normalized_frame=normalized_frame,
        metadata={"waveform": waveform_type},
        source_file=source_file,
        sheet_name="main",
    )


def _preprocess_frame(column: str, values: list[float]) -> SimpleNamespace:
    return SimpleNamespace(
        corrected_frame=pd.DataFrame(
            {
                "time_s": [0.0, 0.1, 0.2, 0.3],
                column: values,
            }
        )
    )


def test_build_validation_candidate_summaries_prefers_matching_waveform_and_frequency() -> None:
    base_profile = pd.DataFrame(
        {
            "waveform_type": ["sine"],
            "freq_hz": [1.0],
            "target_current_a": [20.0],
        }
    )
    matching = _parsed_measurement(test_id="match", waveform_type="sine", freq_hz=1.0, source_file="match.csv")
    off_target = _parsed_measurement(test_id="off", waveform_type="triangle", freq_hz=5.0, source_file="off.csv")
    matching_pre = _preprocess_frame("i_sum_signed", [-10.0, 10.0, -10.0, 10.0])
    off_pre = _preprocess_frame("i_sum_signed", [-5.0, 5.0, -5.0, 5.0])

    rows = ui_retune.build_validation_candidate_summaries(
        base_profile=base_profile,
        validation_measurements=[off_target, matching],
        validation_preprocess_results=[off_pre, matching_pre],
        target_output_type="current",
        current_channel="i_sum_signed",
        field_channel="bz_mT",
    )

    assert rows[0]["test_id"] == "match"
    assert rows[0]["eligible"] is True
    assert rows[0]["output_pp"] == 20.0
    assert rows[1]["eligible"] is False


def test_refresh_catalog_artifacts_builds_picker_and_corrected_catalogs(tmp_path: Path, monkeypatch) -> None:
    artifact_dir = tmp_path / "artifacts"
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    app_paths = build_upload_state_paths(repo_root=tmp_path)
    history_entry = {
        "retune_id": "demo_retune",
        "created_at": "2026-04-20T10:00:00",
        "original_recommendation_id": "control_formula_sine_1Hz_current_20",
        "validation_run_id": "validation::demo",
        "validation_test_id": "validation_case",
        "lut_id": "control_formula_sine_1Hz_current_20",
        "source_kind": "recommendation",
        "source_lut_filename": "control_formula_sine_1Hz_current_20.csv",
        "corrected_lut_id": "control_formula_sine_1Hz_current_20__corrected_iter01",
        "iteration_index": 1,
        "exact_path": "exact_current",
        "correction_rule": "validation_residual_recommendation_loop",
        "target_output_type": "current",
        "waveform_type": "sine",
        "freq_hz": 1.0,
        "commanded_cycles": 1.0,
        "target_level_value": 20.0,
        "target_level_kind": "pp",
        "quality_label": "재현 양호",
        "quality_tone": "green",
        "quality_reasons": ["ok"],
        "acceptance_decision": {
            "decision": "improved_and_accepted",
            "label": "채택",
            "preferred_output_id": "control_formula_sine_1Hz_current_20__corrected_iter01",
            "preferred_output_kind": "corrected",
            "preferred_output_source_kind": "corrected",
            "rejection_reason": None,
        },
        "preferred_output_id": "control_formula_sine_1Hz_current_20__corrected_iter01",
        "candidate_status": "improved_and_accepted",
        "candidate_status_label": "채택",
        "rejection_reason": None,
        "artifact_paths": {
            "corrected_waveform_csv": str(tmp_path / "control_formula_sine_1Hz_current_20__corrected_iter01_waveform.csv"),
            "validation_report_json": str(tmp_path / "control_formula_sine_1Hz_current_20__corrected_iter01_validation_report.json"),
        },
    }
    app_paths.validation_retune_history_path.parent.mkdir(parents=True, exist_ok=True)
    app_paths.validation_retune_history_path.write_text(json.dumps({"retunes": [history_entry]}, ensure_ascii=False), encoding="utf-8")

    lut_catalog_path = artifact_dir / "lut_catalog.json"
    lut_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    lut_catalog_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "lut_id": "control_formula_sine_1Hz_current_20",
                        "original_recommendation_id": "control_formula_sine_1Hz_current_20",
                        "catalog_source_kind": "recommendation",
                        "display_name": "current / sine / 1 Hz / 20 pp",
                        "display_label": "current / sine / 1 Hz / 20 pp | recommendation",
                        "display_object_key": "current::sine::1::::20::pp",
                        "target_type": "current",
                        "waveform": "sine",
                        "freq_hz": 1.0,
                        "cycle_count": 1.0,
                        "created_at": "2026-04-20T09:00:00",
                        "exact_support": True,
                        "route_origin": "exact",
                        "profile_csv_path": str(tmp_path / "control_formula_sine_1Hz_current_20.csv"),
                        "file_name": "control_formula_sine_1Hz_current_20.csv",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(ui_retune, "APP_STATE_PATHS", app_paths)
    monkeypatch.setattr(ui_retune, "VALIDATION_REPORT_DIRS", [report_dir])
    monkeypatch.setattr(ui_retune, "LUT_CATALOG_PATH", lut_catalog_path)
    monkeypatch.setattr(ui_retune, "VALIDATION_CATALOG_PATH", artifact_dir / "validation_catalog.json")
    monkeypatch.setattr(ui_retune, "CORRECTED_CATALOG_PATH", artifact_dir / "corrected_lut_catalog.json")
    monkeypatch.setattr(ui_retune, "RETUNE_PICKER_PATH", artifact_dir / "retune_picker_catalog.json")

    payloads = ui_retune.refresh_catalog_artifacts()

    assert payloads["validation_catalog"]["entries"][0]["validation_run_id"] == "validation::demo"
    assert payloads["corrected_catalog"]["entries"][0]["corrected_lut_id"] == "control_formula_sine_1Hz_current_20__corrected_iter01"
    assert any(entry["source_kind"] == "corrected" for entry in payloads["picker_catalog"]["entries"])
