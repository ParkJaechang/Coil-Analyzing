from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TEST_ROOT = REPO_ROOT / "tests"
for path in (SRC_ROOT, TEST_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import json

from field_analysis.finite_actual_drive import process_actual_drive_review_folder, review_csv_filename
from test_finite_actual_drive_response import _write_actual_drive_csv


def test_actual_drive_review_export_schema_and_batch_summary(tmp_path: Path) -> None:
    input_dir = tmp_path / "actual_drive"
    output_dir = tmp_path / "review"
    input_dir.mkdir()
    _write_actual_drive_csv(input_dir / "finite_recommended_voltage_lut_sine_1.25Hz_1.25cycle_result.csv")
    _write_actual_drive_csv(input_dir / "finite_recommended_voltage_lut_sine_2Hz_1.5cycle_result.csv")

    result = process_actual_drive_review_folder(input_dir, output_dir)

    assert result["parsed_files_count"] == 2
    assert result["expected_files_count"] == 12
    assert result["review_packet_complete"] is False
    assert "finite_recommended_voltage_lut_sine_0.5Hz_1.25cycle_result.csv" in result["missing_files"]
    summary_path = Path(result["summary_path"])
    assert summary_path.exists()
    summary = pd.read_csv(summary_path)
    assert len(summary) == 12
    assert {
        "source_file",
        "file_path",
        "parse_status",
        "alignment_status",
        "review_csv_file",
        "review_csv_path",
        "plot_paths",
        "waveform_type",
        "freq_hz",
        "cycle_count",
        "measured_active_nrmse",
        "measured_shape_corr",
        "measured_peak_error_mT",
        "possible_polarity_flip_suggested",
    }.issubset(summary.columns)
    assert set(summary["parse_status"]) == {"parsed", "missing"}

    metrics = pd.read_csv(output_dir / "finite_actual_drive_case_metrics.csv")
    assert len(metrics) == 2
    missing = pd.read_csv(output_dir / "finite_actual_drive_missing_cases.csv")
    assert len(missing) == 10
    assert "finite_recommended_voltage_lut_sine_0.5Hz_1.25cycle_result.csv" in set(missing["source_file"])

    exported = output_dir / "finite_actual_drive_review_sine_1.25Hz_1.25cycle.csv"
    assert exported.exists()
    review = pd.read_csv(exported)
    assert {
        "time_s",
        "first_voltage_v",
        "physical_target_output_mT",
        "measured_field_mT",
        "measured_residual_mT",
        "current_a",
    }.issubset(review.columns)
    assert np.allclose(
        review["measured_residual_mT"],
        review["physical_target_output_mT"] - review["measured_field_mT"],
        atol=1e-12,
    )
    assert "correction_delta_v" not in review.columns
    assert "second_voltage_v" not in review.columns
    assert not list(output_dir.glob("finite_second_correction_lut_*.csv"))
    assert (output_dir / "finite_actual_drive_review_sine_1.25Hz_1.25cycle_metadata.json").exists()
    assert (output_dir / "plots" / "actual_drive_review_sine_1.25Hz_1.25cycle_target_vs_measured.png").exists()
    assert (output_dir / "plots" / "actual_drive_review_sine_1.25Hz_1.25cycle_voltage.png").exists()
    assert (output_dir / "plots" / "actual_drive_review_sine_1.25Hz_1.25cycle_residual.png").exists()

    manifest = json.loads((output_dir / "finite_actual_drive_review_manifest.json").read_text(encoding="utf-8"))
    assert manifest["expected_files_count"] == 12
    assert manifest["parsed_files_count"] == 2
    assert manifest["review_packet_complete"] is False
    assert manifest["review_packet_status"] == "partial"
    assert manifest["correction_delta_v_generated"] is False
    assert manifest["second_voltage_v_generated"] is False
    assert manifest["second_lut_generated"] is False
    assert manifest["stale_second_correction_artifacts_ignored"] is True


def test_review_csv_filename_is_finite_phase1_standard() -> None:
    assert (
        review_csv_filename({"waveform_type": "sine", "freq_hz": 1.25, "cycle_count": 1.75})
        == "finite_actual_drive_review_sine_1.25Hz_1.75cycle.csv"
    )
