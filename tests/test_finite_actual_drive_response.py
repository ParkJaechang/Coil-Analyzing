from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.finite_actual_drive import (
    build_finite_actual_drive_review_dataset,
    build_actual_drive_review_case,
    expected_actual_drive_result_filenames,
    load_finite_actual_drive_result,
    parse_actual_drive_filename,
    parse_finite_actual_drive_filename,
    read_actual_drive_result,
)


def _write_actual_drive_csv(path: Path) -> None:
    rows = []
    time_ms = np.linspace(0.0, 1400.0, 101)
    voltage = np.zeros_like(time_ms)
    active = (time_ms >= 200.0) & (time_ms <= 1200.0)
    voltage[active] = 2.0 * np.sin(np.pi * (time_ms[active] - 200.0) / 1000.0)
    hall = 1.5 + 40.0 * np.sin(np.pi * np.clip((time_ms - 200.0) / 1000.0, 0.0, 1.0))
    for index, (t_ms, v, h) in enumerate(zip(time_ms, voltage, hall, strict=False)):
        rows.append(f"{index},{t_ms:.3f},0.0,0.0,{h:.6f},0.1,0.0,{v:.6f},0.0")
    preamble = [
        "# Date,2026-05-06 16:00:02",
        "# Frequency(Hz),0.000",
        "# Amplitude(V),0.000",
        "# Cycles,0.000",
        "# Repeat,1.000",
        "# PreDelay(s),1.000",
        "# PostDelay(s),1.000",
        "# HallSamples,21286",
        "# CurrentSamples,1409919",
        "# CommonRange(ms),0.00~2819.84 (span 2819.84)",
        "# Rows,5000, GridStep(ms),0.564",
        "# AutoSyncHallLag,applied 70.00ms (r=0.815)",
        "#",
        "Row,TimeMs,HallBx,HallBy,HallBz,Current1_A,Current2_A,Voltage1_V,Voltage2_V",
    ]
    path.write_text("\n".join([*preamble, *rows]), encoding="utf-8")


def test_actual_drive_filename_and_preamble_parse(tmp_path: Path) -> None:
    path = tmp_path / "finite_recommended_voltage_lut_sine_1.25Hz_1.25cycle_result.csv"
    _write_actual_drive_csv(path)

    parsed_name = parse_actual_drive_filename(path)
    record = read_actual_drive_result(path)

    assert parsed_name["waveform_type"] == "sine"
    assert parsed_name["freq_hz"] == 1.25
    assert parsed_name["cycle_count"] == 1.25
    assert parsed_name["canonical_source_filename"] == "finite_recommended_voltage_lut_sine_1.25Hz_1.25cycle_result.csv"
    assert record.metadata["pre_delay_s"] == 1.0
    assert record.metadata["post_delay_s"] == 1.0
    assert record.metadata["auto_sync_hall_lag_ms"] == 70.0
    assert record.metadata["time_unit"] == "ms"
    assert record.metadata["voltage_unit"] == "V"
    assert record.metadata["field_unit"] == "mT_inferred_from_HallBz"
    assert {"time_s_abs", "first_voltage_v", "actual_drive_voltage_v", "measured_field_raw", "hallbz_raw_mT", "current_a"}.issubset(record.frame.columns)
    assert np.isclose(float(record.frame["time_s_abs"].iloc[-1]), 1.4)
    assert np.isclose(float(record.frame["hallbz_raw_mT"].iloc[0]), 1.5)
    assert np.isclose(float(record.frame["measured_field_raw"].iloc[0]), -1.5)
    assert record.metadata["hallbz_sign_inverted"] is True


def test_actual_drive_filename_with_upload_prefix_parses_to_canonical_name() -> None:
    parsed = parse_finite_actual_drive_filename(
        "7d72d4709ef49600_finite_recommended_voltage_lut_sine_0.5Hz_1.25cycle_result.csv"
    )

    assert parsed["upload_internal_id"] == "7d72d4709ef49600"
    assert parsed["canonical_source_filename"] == "finite_recommended_voltage_lut_sine_0.5Hz_1.25cycle_result.csv"
    assert parsed["waveform"] == "sine"
    assert parsed["waveform_type"] == "sine"
    assert parsed["freq_hz"] == 0.5
    assert parsed["cycle_count"] == 1.25
    assert parsed["source_type"] == "finite_actual_drive_result"


def test_actual_drive_filename_without_prefix_still_parses() -> None:
    parsed = parse_finite_actual_drive_filename("finite_recommended_voltage_lut_sine_1.25Hz_1cycle_result.csv")

    assert parsed["upload_internal_id"] is None
    assert parsed["canonical_source_filename"] == "finite_recommended_voltage_lut_sine_1.25Hz_1cycle_result.csv"
    assert parsed["freq_hz"] == 1.25
    assert parsed["cycle_count"] == 1.0


def test_actual_drive_review_metrics_and_alignment(tmp_path: Path) -> None:
    path = tmp_path / "finite_recommended_voltage_lut_sine_1.25Hz_1.25cycle_result.csv"
    _write_actual_drive_csv(path)
    record = read_actual_drive_result(path)

    review, metadata = build_actual_drive_review_case(record)

    assert np.isclose(metadata["target_active_end_s"], 1.0)
    assert np.isclose(metadata["command_start_s"], 0.2, atol=0.02)
    assert metadata["alignment_anchor"] == "Voltage1_V_command_nonzero_start"
    assert "Voltage1_V" not in review.columns
    assert {
        "time_s",
        "first_voltage_v",
        "command_voltage_v",
        "actual_drive_voltage_v",
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
    assert np.isfinite(metadata["measured_active_nrmse"])
    assert np.isfinite(metadata["measured_shape_corr"])
    assert np.isfinite(metadata["measured_peak_error_mT"])
    assert metadata["target_pp_mT"] == 100.0
    assert metadata["correction_delta_generated"] is False
    assert metadata["second_voltage_generated"] is False
    assert metadata["second_lut_generated"] is False
    assert metadata["continuous_touched"] is False
    assert metadata["command_voltage_source"] == "Voltage1_V_no_separate_command_reference"
    assert metadata["actual_drive_voltage_source"] == "Voltage1_V"


def test_actual_drive_review_payload_contract_for_in_app_upload(tmp_path: Path) -> None:
    path = tmp_path / "upload_abc_finite_recommended_voltage_lut_sine_1.25Hz_1.25cycle_result.csv"
    _write_actual_drive_csv(path)

    case = load_finite_actual_drive_result(path)

    assert case["case_id"] == "finite_actual_drive_result:sine:1.25Hz:1.25cycle"
    assert case["display_label"] == "sine 1.25Hz 1.25cycle actual-drive result"
    assert case["source_file"] == path.name
    assert case["canonical_source_filename"] == "finite_recommended_voltage_lut_sine_1.25Hz_1.25cycle_result.csv"
    assert case["upload_internal_id"] == "upload_abc"
    assert case["parse_status"] == "parsed"
    assert case["parse_error"] is None
    assert {
        "time_s",
        "first_voltage_v",
        "command_voltage_v",
        "actual_drive_voltage_v",
        "physical_target_output_mT",
        "measured_field_mT",
        "measured_residual_mT",
        "current_a",
    }.issubset(case["time_series"].columns)
    assert "correction_delta_v" not in case["time_series"].columns
    assert "second_voltage_v" not in case["time_series"].columns
    assert np.isfinite(case["metrics"]["measured_active_nrmse"])
    assert np.isfinite(case["metrics"]["measured_shape_corr"])
    assert "baseline_ok" in case["status"]
    assert "alignment_status" in case["status"]


def test_actual_drive_review_dataset_reports_malformed_upload_error(tmp_path: Path) -> None:
    good_path = tmp_path / "finite_recommended_voltage_lut_sine_1.25Hz_1.25cycle_result.csv"
    bad_path = tmp_path / "bad_upload.csv"
    _write_actual_drive_csv(good_path)
    bad_path.write_text("not,a,result\n1,2,3\n", encoding="utf-8")

    dataset = build_finite_actual_drive_review_dataset([good_path, bad_path])

    assert len(dataset["cases"]) == 1
    assert len(dataset["errors"]) == 1
    assert dataset["errors"][0]["source_file"] == "bad_upload.csv"
    assert dataset["errors"][0]["parse_status"] == "error"
    assert "Unsupported finite actual-drive result filename" in dataset["errors"][0]["parse_error"]
    assert len(dataset["summary"]) == 2
    assert set(dataset["summary"]["parse_status"]) == {"parsed", "error"}


def test_expected_actual_drive_files_include_twelve_finite_cases() -> None:
    expected = expected_actual_drive_result_filenames()

    assert len(expected) == 12
    assert "finite_recommended_voltage_lut_sine_0.5Hz_1.25cycle_result.csv" in expected
    assert "finite_recommended_voltage_lut_sine_1.25Hz_1.25cycle_result.csv" in expected
