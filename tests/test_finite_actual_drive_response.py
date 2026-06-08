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


def _write_actual_drive_csv(path: Path, *, encoded_time_unit: str = "milliseconds") -> None:
    rows = []
    time_ms = np.linspace(0.0, 1400.0, 101)
    voltage = np.zeros_like(time_ms)
    active = (time_ms >= 200.0) & (time_ms <= 1200.0)
    voltage[active] = 2.0 * np.sin(np.pi * (time_ms[active] - 200.0) / 1000.0)
    hall = 1.5 + 40.0 * np.sin(np.pi * np.clip((time_ms - 200.0) / 1000.0, 0.0, 1.0))
    if encoded_time_unit == "seconds":
        encoded_time = time_ms / 1000.0
    elif encoded_time_unit == "microseconds":
        encoded_time = time_ms * 1000.0
    else:
        encoded_time = time_ms
    for index, (t_raw, v, h) in enumerate(zip(encoded_time, voltage, hall, strict=False)):
        rows.append(f"{index},{t_raw:.6f},0.0,0.0,{h:.6f},0.1,0.0,{v:.6f},0.0")
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
    assert record.metadata["actual_drive_time_unit_detected"] == "milliseconds"
    assert record.metadata["timebase_status"] == "ok"
    assert np.isclose(record.metadata["active_duration_ratio"], 1.0, atol=0.03)
    assert record.metadata["voltage_unit"] == "V"
    assert record.metadata["field_unit"] == "mT_inferred_from_HallBz"
    assert {
        "time_s_abs",
        "first_voltage_v",
        "actual_drive_voltage_v",
        "measured_field_raw",
        "hallbz_raw_mT",
        "raw_hallbz_mT",
        "measured_field_effective_mT",
        "current_a",
    }.issubset(record.frame.columns)
    assert np.isclose(float(record.frame["time_s_abs"].iloc[-1]), 1.4)


def test_actual_drive_timebase_detects_seconds_encoded_time_column(tmp_path: Path) -> None:
    path = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_actual_drive_csv(path, encoded_time_unit="seconds")

    record = read_actual_drive_result(path)
    review, metadata = build_actual_drive_review_case(record)

    assert record.metadata["actual_drive_time_unit_detected"] == "seconds"
    assert record.metadata["timebase_status"] == "ok"
    assert metadata["actual_drive_time_unit_detected"] == "seconds"
    assert np.isclose(float(record.frame["time_s_abs"].iloc[-1]), 1.4)
    assert np.isclose(metadata["voltage_nonzero_duration_s"], 1.0, atol=0.03)
    assert np.isclose(metadata["active_duration_ratio"], 1.0, atol=0.03)
    assert np.all(np.diff(review["time_s"].to_numpy(dtype=float)) > 0.0)


def test_actual_drive_schema_file_without_regex_uses_preamble_metadata(tmp_path: Path) -> None:
    path = tmp_path / "bench_measurement.csv"
    path.write_text(
        "\n".join(
            [
                "# Frequency(Hz),1.000",
                "# Cycles,1.000",
                "# Waveform,sine",
                "TimeMs,Voltage1_V,HallBz,Current1_A",
                "0,0,1,0.1",
                "500,2,2,0.1",
                "1000,0,1,0.1",
            ]
        ),
        encoding="utf-8",
    )

    record = read_actual_drive_result(path)

    assert record.metadata["metadata_source"] == "preamble"
    assert record.freq_hz == 1.0
    assert record.cycle_count == 1.0


def test_actual_drive_schema_file_without_metadata_uses_current_selection_fallback(tmp_path: Path) -> None:
    path = tmp_path / "uploaded_feedback.csv"
    rows = [
        "TimeMs,Voltage1_V,HallBz,Current1_A",
        "0,0,1,0.1",
        "500,2,2,0.1",
        "1000,0,1,0.1",
    ]
    path.write_text("\n".join(rows), encoding="utf-8")

    record = read_actual_drive_result(path, waveform_type="sine", freq_hz=1.0, cycle_count=1.0)

    assert record.metadata["metadata_source"] == "current_quick_lut_selection"
    assert record.waveform_type == "sine"
    assert record.freq_hz == 1.0
    assert record.cycle_count == 1.0
    assert np.isclose(float(record.frame["hallbz_raw_mT"].iloc[0]), 1.0)
    assert np.isclose(float(record.frame["measured_field_raw"].iloc[0]), 1.0)
    assert np.isclose(float(record.frame["measured_field_effective_mT"].iloc[0]), -1.0)
    assert record.metadata["hallbz_sign_inverted"] is True


def test_actual_drive_non_result_filename_with_schema_reads_filename_metadata(tmp_path: Path) -> None:
    path = tmp_path / "finite_recommended_voltage_lut_sine_1.5Hz_1.5cycle.csv"
    rows = [
        "# Frequency(Hz),0.000",
        "# Cycles,0.000",
        "TimeMs,Voltage1_V,HallBz,Current1_A",
        "0,0,1,0.1",
        "500,2,2,0.1",
        "1000,0,1,0.1",
    ]
    path.write_text("\n".join(rows), encoding="utf-8")

    record = read_actual_drive_result(path)

    assert record.metadata["metadata_source"] == "filename"
    assert record.freq_hz == 1.5
    assert record.cycle_count == 1.5
    assert "hallbz_raw_mT" in record.frame.columns


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


def test_actual_drive_default_triangle_hz_cycle_filename_parses() -> None:
    parsed = parse_finite_actual_drive_filename("0.25hz_1.5cycle.csv")

    assert parsed["upload_internal_id"] is None
    assert parsed["canonical_source_filename"] == "0.25hz_1.5cycle.csv"
    assert parsed["waveform_type"] == "triangle"
    assert parsed["freq_hz"] == 0.25
    assert parsed["cycle_count"] == 1.5


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
        "raw_hallbz_mT",
        "measured_field_effective_mT",
        "baseline_removed_effective_field_mT",
        "measured_field_baseline_removed_mT",
        "normalized_effective_field_mT",
        "measured_field_smoothed_mT",
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
    assert metadata["actual_drive_time_unit"] == "milliseconds"
    assert metadata["source_time_monotonic"] is True
    assert metadata["duplicate_time_count"] == 0
    assert metadata["hallbz_sign_selection_status"] == "fixed_project_convention_negative_hallbz"
    assert metadata["effective_field_convention"] == "effective_field_mT = -HallBz_raw"
    assert np.isfinite(metadata["hallbz_negative_convention_corr"])
    assert np.isfinite(metadata["hallbz_positive_convention_corr"])
    assert metadata["actual_drive_review_smoothing_enabled"] is True
    assert metadata["actual_drive_review_smoothing_method"] == "median_then_rolling"
    sign = float(metadata["hallbz_effective_sign"])
    assert np.allclose(review["measured_field_effective_mT"], sign * review["raw_hallbz_mT"], atol=1e-12)
    assert np.isfinite(review["measured_field_smoothed_mT"].to_numpy(dtype=float)).all()
    assert np.allclose(
        review["measured_field_baseline_removed_mT"],
        review["baseline_removed_effective_field_mT"],
        atol=1e-12,
    )


def test_hallbz_raw_effective_normalized_shape_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "finite_recommended_voltage_lut_sine_1.25Hz_1cycle_result.csv"
    _write_actual_drive_csv(path)
    record = read_actual_drive_result(path)

    review, metadata = build_actual_drive_review_case(record)

    active = review["time_s"].between(0.0, metadata["target_active_end_s"])
    raw = review.loc[active, "raw_hallbz_mT"].to_numpy(dtype=float)
    effective = review.loc[active, "measured_field_effective_mT"].to_numpy(dtype=float)
    baseline_removed = review.loc[active, "baseline_removed_effective_field_mT"].to_numpy(dtype=float)
    normalized = review.loc[active, "normalized_effective_field_mT"].to_numpy(dtype=float)

    assert np.allclose(effective, float(metadata["hallbz_effective_sign"]) * raw, atol=1e-12)
    assert np.corrcoef(effective[np.isfinite(effective)], baseline_removed[np.isfinite(baseline_removed)])[0, 1] > 0.99
    assert np.corrcoef(baseline_removed[np.isfinite(baseline_removed)], normalized[np.isfinite(normalized)])[0, 1] > 0.99


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
    assert "actual-drive result table header" in dataset["errors"][0]["parse_error"]
    assert len(dataset["summary"]) == 2
    assert set(dataset["summary"]["parse_status"]) == {"parsed", "error"}


def test_expected_actual_drive_files_include_twelve_finite_cases() -> None:
    expected = expected_actual_drive_result_filenames()

    assert len(expected) == 12
    assert "finite_recommended_voltage_lut_sine_0.5Hz_1.25cycle_result.csv" in expected
    assert "finite_recommended_voltage_lut_sine_1.25Hz_1.25cycle_result.csv" in expected
