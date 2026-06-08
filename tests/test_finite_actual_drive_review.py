from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TEST_ROOT = REPO_ROOT / "tests"
for path in (SRC_ROOT, TEST_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from field_analysis.finite_actual_drive import build_actual_drive_review_case
from field_analysis.finite_actual_drive import process_actual_drive_review_folder
from field_analysis.finite_actual_drive import read_actual_drive_result
from test_finite_actual_drive_response import _write_actual_drive_csv


def _write_large_actual_drive_csv(path: Path) -> None:
    rows = []
    time_ms = np.linspace(0.0, 1400.0, 101)
    voltage = np.zeros_like(time_ms)
    active = (time_ms >= 200.0) & (time_ms <= 1200.0)
    voltage[active] = 12.0 * np.sin(np.pi * (time_ms[active] - 200.0) / 1000.0)
    hall = 8.0 + 240.0 * np.sin(np.pi * np.clip((time_ms - 200.0) / 1000.0, 0.0, 1.0))
    for index, (t_ms, v, h) in enumerate(zip(time_ms, voltage, hall, strict=False)):
        rows.append(f"{index},{t_ms:.3f},0.0,0.0,{h:.6f},0.1,0.0,{v:.6f},0.0")
    preamble = [
        "# PreDelay(s),1.000",
        "# PostDelay(s),1.000",
        "# AutoSyncHallLag,applied 70.00ms (r=0.815)",
        "#",
        "Row,TimeMs,HallBx,HallBy,HallBz,Current1_A,Current2_A,Voltage1_V,Voltage2_V",
    ]
    path.write_text("\n".join([*preamble, *rows]), encoding="utf-8")


def test_actual_drive_review_normalizes_large_field_and_voltage_for_shape_review(tmp_path: Path) -> None:
    path = tmp_path / "finite_recommended_voltage_lut_sine_1.25Hz_1.5cycle_result.csv"
    _write_large_actual_drive_csv(path)
    record = read_actual_drive_result(path)

    review, metadata = build_actual_drive_review_case(record, intended_drive_cycle_count=1.25)

    assert np.nanmax(np.abs(review["raw_measured_field_mT"])) > 100.0
    assert np.nanmax(np.abs(review["normalized_measured_field_mT"])) == pytest.approx(50.0, abs=1e-6)
    assert np.nanmax(np.abs(review["raw_first_voltage_v"])) > 5.0
    assert np.nanmax(np.abs(review["normalized_first_voltage_v"])) <= 10.0 + 1e-9
    assert np.nanmax(np.abs(review["normalized_physical_target_output_mT"])) <= 50.0 + 1e-9
    assert np.allclose(
        review["measured_residual_normalized_mT"],
        review["normalized_physical_target_output_mT"] - review["normalized_measured_field_mT"],
        atol=1e-12,
    )
    assert metadata["field_normalization_enabled"] is True
    assert metadata["field_normalization_mode"] == "peak_to_50mT"
    assert metadata["voltage_normalization_enabled"] is True
    assert metadata["voltage_normalization_mode"] == "peak_to_10V"
    assert metadata["absolute_gain_evaluation_disabled"] is True
    assert metadata["shape_review_only"] is True
    assert metadata["modeled_cycle_count"] == 1.5
    assert metadata["intended_drive_cycle_count"] == 1.25
    assert metadata["source_filename_cycle_count"] == 1.5
    assert metadata["cycle_usage_mode"] == "modeled_cycle_as_drive_candidate"
    assert "correction_delta_v" not in review.columns
    assert "second_voltage_v" not in review.columns


def test_actual_drive_review_metrics_use_normalized_arrays(tmp_path: Path) -> None:
    path = tmp_path / "finite_recommended_voltage_lut_sine_1.25Hz_1.5cycle_result.csv"
    _write_large_actual_drive_csv(path)
    record = read_actual_drive_result(path)

    review, metadata = build_actual_drive_review_case(record)

    active = (review["time_s"] >= 0.0) & (review["time_s"] <= metadata["target_active_end_s"])
    corr = float(
        np.corrcoef(
            review.loc[active, "normalized_physical_target_output_mT"],
            review.loc[active, "normalized_measured_field_mT"],
        )[0, 1]
    )
    rmse = float(
        np.sqrt(
            np.nanmean(
                np.square(
                    review.loc[active, "normalized_physical_target_output_mT"]
                    - review.loc[active, "normalized_measured_field_mT"]
                )
            )
        )
    )
    assert metadata["normalized_shape_corr"] == pytest.approx(corr)
    target_pp = float(
        np.nanmax(review.loc[active, "normalized_physical_target_output_mT"])
        - np.nanmin(review.loc[active, "normalized_physical_target_output_mT"])
    )
    assert metadata["normalized_nrmse"] == pytest.approx(rmse / (target_pp * 0.5))
    assert metadata["measured_shape_corr"] == metadata["normalized_shape_corr"]
    assert metadata["measured_active_nrmse"] == metadata["normalized_nrmse"]
    assert np.isfinite(metadata["raw_field_peak_mT"])
    assert np.isfinite(metadata["raw_voltage_peak_v"])


def test_actual_drive_review_export_schema_contains_raw_and_normalized_columns(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    _write_actual_drive_csv(input_dir / "finite_recommended_voltage_lut_sine_1.25Hz_1.25cycle_result.csv")

    result = process_actual_drive_review_folder(input_dir, output_dir)
    exported = pd.read_csv(output_dir / "finite_actual_drive_review_sine_1.25Hz_1.25cycle.csv")
    summary = pd.read_csv(result["summary_path"])

    assert {
        "time_s",
        "raw_first_voltage_v",
        "normalized_first_voltage_v",
        "raw_measured_field_mT",
        "normalized_measured_field_mT",
        "normalized_physical_target_output_mT",
        "measured_residual_normalized_mT",
        "current_a",
        "modeled_cycle_count",
        "intended_drive_cycle_count",
        "field_normalization_scale_factor",
        "voltage_normalization_scale_factor",
    }.issubset(exported.columns)
    assert {
        "modeled_cycle_count",
        "intended_drive_cycle_count",
        "raw_field_peak_mT",
        "field_normalization_scale_factor",
        "raw_voltage_peak_v",
        "voltage_normalization_scale_factor",
        "normalized_peak_mT",
        "normalized_voltage_peak_v",
        "normalized_shape_corr",
        "normalized_nrmse",
        "terminal_peak_time_error_s",
        "shape_review_only",
    }.issubset(summary.columns)
    assert not list(output_dir.glob("finite_second_correction_lut_*.csv"))
