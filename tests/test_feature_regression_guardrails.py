from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.final_modeled_lut import build_final_modeled_voltage_lut_export


def test_final_modeled_voltage_lut_contract_is_not_fourier_or_second_correction() -> None:
    command_profile = pd.DataFrame(
        {
            "time_s": [0.0, 0.1, 0.2],
            "limited_voltage_v": [0.0, 1.0, -1.0],
            "recommended_voltage_v": [0.0, 1.2, -1.2],
        }
    )

    payload = build_final_modeled_voltage_lut_export(command_profile, freq_hz=1.0, cycle_count=1.25, waveform="sine")
    frame = payload["frame"]
    metadata = payload["metadata"]

    assert list(frame.columns) == ["sample_index", "time_s", "voltage_v"]
    assert frame["voltage_v"].tolist() == command_profile["limited_voltage_v"].tolist()
    assert metadata["voltage_source_column"] == "limited_voltage_v"
    assert metadata["time_source_column"] == "time_s"
    assert metadata["fourier_resynthesis_involved"] is False
    assert metadata["harmonic_export_involved"] is False
    assert metadata["finite_only"] is True
    assert "correction_delta_v" not in frame.columns
    assert "second_voltage_v" not in frame.columns


def test_review_ui_modules_stay_extracted_from_app_ui_snapshot() -> None:
    app_source = (REPO_ROOT / "src" / "field_analysis" / "app_ui_snapshot.py").read_text(encoding="utf-8")

    required_markers = [
        "from .ui_startup_compensation_review import render_startup_compensation_review",
        "from .ui_voltage_lut_review import",
        "render_final_voltage_lut_export_panel",
        "render_voltage_lut_review_section",
        "render_finite_actual_drive_review_section",
    ]
    missing = [marker for marker in required_markers if marker not in app_source]

    assert not missing
    assert (REPO_ROOT / "src" / "field_analysis" / "ui_startup_compensation_review.py").is_file()
    assert (REPO_ROOT / "src" / "field_analysis" / "ui_voltage_lut_review.py").is_file()
    assert (REPO_ROOT / "src" / "field_analysis" / "ui_finite_actual_drive_review.py").is_file()


def test_finite_actual_drive_phase_one_source_does_not_emit_second_correction() -> None:
    source = (REPO_ROOT / "src" / "field_analysis" / "finite_actual_drive.py").read_text(encoding="utf-8")

    required_markers = [
        "TimeMs",
        "time_s",
        "Voltage1_V",
        "first_voltage_v",
        "HallBz",
        "measured_field_mT",
    ]
    missing = [marker for marker in required_markers if marker not in source]

    assert not missing
    assert '"correction_delta_v_generated": False' in source
    assert '"second_voltage_v_generated": False' in source
    assert '"second_lut_generated": False' in source
    assert "finite_second_correction" not in source


def test_git_hygiene_document_covers_runtime_visibility_and_stale_branch_rules() -> None:
    source = (REPO_ROOT / "docs" / "GIT_AND_CODEX_WORKFLOW.md").read_text(encoding="utf-8")

    required_markers = [
        "Feature Regression Prevention Checklist",
        "stale branch",
        "source_present",
        "AppTest_visible",
        "launched_runtime_visible",
        "actual_user_path_visible",
        "generated artifacts",
        "runtime workspace SHA",
    ]
    missing = [marker for marker in required_markers if marker not in source]

    assert not missing
