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

from field_analysis.finite_actual_drive import process_actual_drive_folder, second_lut_filename
from test_finite_actual_drive_response import _write_actual_drive_csv


def test_second_correction_lut_export_schema_and_batch_summary(tmp_path: Path) -> None:
    input_dir = tmp_path / "actual_drive"
    output_dir = tmp_path / "second_lut"
    input_dir.mkdir()
    _write_actual_drive_csv(input_dir / "finite_recommended_voltage_lut_sine_1.25Hz_1.25cycle_result.csv")
    _write_actual_drive_csv(input_dir / "finite_recommended_voltage_lut_sine_2Hz_1.5cycle_result.csv")

    result = process_actual_drive_folder(input_dir, output_dir)

    assert result["files_parsed"] == 2
    summary_path = Path(result["summary_path"])
    assert summary_path.exists()
    summary = pd.read_csv(summary_path)
    assert len(summary) == 2
    assert {
        "source_file",
        "second_lut_file",
        "waveform_type",
        "freq_hz",
        "cycle_count",
        "measured_active_nrmse",
        "measured_shape_corr",
        "voltage_limit_respected",
        "smoothness_preserved",
        "correction_applied",
    }.issubset(summary.columns)
    assert set(summary["waveform_type"]) == {"sine"}

    exported = output_dir / "finite_second_correction_lut_sine_1.25Hz_1.25cycle.csv"
    assert exported.exists()
    lut = pd.read_csv(exported)
    assert {
        "sample_index",
        "time_s",
        "first_voltage_v",
        "correction_delta_v",
        "second_voltage_v",
        "physical_target_output",
        "measured_field",
        "measured_residual",
        "measured_current_a",
        "second_predicted_output",
    }.issubset(lut.columns)
    assert np.allclose(lut["second_voltage_v"], lut["first_voltage_v"] + lut["correction_delta_v"], atol=1e-12)
    metadata_path = output_dir / "finite_second_correction_lut_sine_1.25Hz_1.25cycle_metadata.json"
    assert metadata_path.exists()


def test_second_lut_filename_is_finite_only_standard() -> None:
    assert (
        second_lut_filename({"waveform_type": "sine", "freq_hz": 1.25, "cycle_count": 1.75})
        == "finite_second_correction_lut_sine_1.25Hz_1.75cycle.csv"
    )
