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

from field_analysis.finite_actual_drive import process_actual_drive_review_folder, review_csv_filename
from test_finite_actual_drive_response import _write_actual_drive_csv


def test_actual_drive_review_export_schema_and_batch_summary(tmp_path: Path) -> None:
    input_dir = tmp_path / "actual_drive"
    output_dir = tmp_path / "review"
    input_dir.mkdir()
    _write_actual_drive_csv(input_dir / "finite_recommended_voltage_lut_sine_1.25Hz_1.25cycle_result.csv")
    _write_actual_drive_csv(input_dir / "finite_recommended_voltage_lut_sine_2Hz_1.5cycle_result.csv")

    result = process_actual_drive_review_folder(input_dir, output_dir)

    assert result["files_parsed"] == 2
    summary_path = Path(result["summary_path"])
    assert summary_path.exists()
    summary = pd.read_csv(summary_path)
    assert len(summary) == 2
    assert {
        "source_file",
        "review_csv_file",
        "waveform_type",
        "freq_hz",
        "cycle_count",
        "measured_active_nrmse",
        "measured_shape_corr",
        "measured_peak_error_mT",
        "possible_polarity_flip_suggested",
    }.issubset(summary.columns)

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


def test_review_csv_filename_is_finite_phase1_standard() -> None:
    assert (
        review_csv_filename({"waveform_type": "sine", "freq_hz": 1.25, "cycle_count": 1.75})
        == "finite_actual_drive_review_sine_1.25Hz_1.75cycle.csv"
    )
