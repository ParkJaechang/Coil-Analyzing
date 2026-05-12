from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.finite_second_modeling import generate_second_modeled_voltage_lut
from field_analysis.final_modeled_lut import build_final_modeled_voltage_lut_export
from tests.test_finite_actual_drive_response import _write_actual_drive_csv


def _first_profile() -> pd.DataFrame:
    time_s = np.linspace(0.0, 1.0, 101)
    target = 50.0 * np.sin(np.pi * time_s)
    voltage = 3.0 * np.sin(np.pi * time_s)
    return pd.DataFrame(
        {
            "time_s": time_s,
            "physical_target_output_mT": target,
            "limited_voltage_v": voltage,
            "recommended_voltage_v": voltage,
        }
    )


def test_second_modeling_generates_limited_voltage_for_one_cycle(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_actual_drive_csv(actual)

    frame, metadata = generate_second_modeled_voltage_lut(
        _first_profile(),
        actual,
        freq_hz=1.0,
        cycle_count=1.0,
        correction_gain=0.25,
    )

    assert metadata["second_modeling_available"] is True
    assert metadata["second_modeling_status"] == "ok"
    assert metadata["hallbz_sign_applied"] is True
    assert metadata["final_export_voltage_source_column"] == "second_limited_voltage_v"
    assert np.nanmax(np.abs(frame["second_limited_voltage_v"])) <= 5.0 + 1e-12
    assert np.isclose(np.nanmax(np.abs(frame["measured_field_normalized_mT"])), 50.0)
    assert {"second_correction_delta_v", "second_modeled_voltage_v", "final_voltage_v"}.issubset(frame.columns)
    assert bool(frame["target_unchanged"].iloc[0]) is True
    assert np.allclose(frame["measured_field_effective_mT"], -frame["raw_hallbz_mT"], equal_nan=True)
    assert metadata["double_sign_flip_detected"] is False
    assert metadata["source_time_monotonic"] is True


def test_second_modeling_generates_limited_voltage_for_one_point_five_cycle(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1.5Hz_1.5cycle_result.csv"
    _write_actual_drive_csv(actual)
    profile = _first_profile()

    frame, metadata = generate_second_modeled_voltage_lut(
        profile,
        actual,
        freq_hz=1.5,
        cycle_count=1.5,
        correction_gain=0.25,
    )

    assert metadata["second_modeling_available"] is True
    assert metadata["second_modeling_status"] == "ok"
    assert metadata["production_cycle_policy"] == "1p0_1p5_cycles"
    assert np.nanmax(np.abs(frame["second_limited_voltage_v"])) <= 5.0 + 1e-12


def test_second_modeling_rejects_unsupported_cycle_without_delta(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1.25Hz_1.25cycle_result.csv"
    _write_actual_drive_csv(actual)

    frame, metadata = generate_second_modeled_voltage_lut(_first_profile(), actual, freq_hz=1.0, cycle_count=1.25)

    assert frame.equals(_first_profile())
    assert metadata["second_modeling_available"] is False
    assert metadata["second_modeling_status"] == "unsupported_cycle_policy_1p0_1p5_only"
    assert metadata["second_correction_delta_v_generated"] is False
    assert "second_correction_delta_v" not in frame.columns
    assert "second_limited_voltage_v" not in frame.columns


def test_second_modeling_blocks_actual_drive_freq_cycle_mismatch(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_actual_drive_csv(actual)

    frame, metadata = generate_second_modeled_voltage_lut(_first_profile(), actual, freq_hz=1.5, cycle_count=1.5)

    assert frame.equals(_first_profile())
    assert metadata["second_modeling_available"] is False
    assert metadata["second_modeling_status"] == "actual_drive_target_mismatch"
    assert metadata["second_correction_delta_v_generated"] is False
    assert "second_limited_voltage_v" not in frame.columns


def test_second_modeling_blocks_when_actual_drive_time_range_does_not_cover_target(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_actual_drive_csv(actual)
    profile = _first_profile()
    profile["time_s"] = profile["time_s"] + 2.0

    frame, metadata = generate_second_modeled_voltage_lut(profile, actual, freq_hz=1.0, cycle_count=1.0)

    assert frame.equals(profile)
    assert metadata["second_modeling_available"] is False
    assert metadata["second_modeling_status"] == "actual_drive_time_range_insufficient"
    assert metadata["interpolation_status"] == "source_time_range_insufficient_no_extrapolation"
    assert "second_limited_voltage_v" not in frame.columns


def test_final_lut_export_prefers_second_limited_voltage_when_available(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_actual_drive_csv(actual)
    frame, _metadata = generate_second_modeled_voltage_lut(_first_profile(), actual, freq_hz=1.0, cycle_count=1.0)
    frame["second_modeling_available"] = True
    frame["second_modeling_status"] = "ok"

    export = build_final_modeled_voltage_lut_export(frame, freq_hz=1.0, cycle_count=1.0, waveform="sine")

    assert export["metadata"]["voltage_source_column"] == "second_limited_voltage_v"
    assert np.allclose(export["frame"]["voltage_v"], frame["second_limited_voltage_v"])
    assert list(export["frame"].columns) == ["sample_index", "time_s", "voltage_v"]
    assert export["metadata"]["fourier_resynthesis_involved"] is False
    assert export["metadata"]["harmonic_export_involved"] is False


def test_second_modeling_ui_uses_actual_cycle_for_final_export_and_korean_plot_labels() -> None:
    from field_analysis.ui_second_modeling import build_native_actual_drive_raw_plot_frame, build_second_modeling_plot_frames

    source = (SRC_ROOT / "field_analysis" / "ui_second_modeling.py").read_text(encoding="utf-8")

    assert "cycle_count=cycle_count" in source
    assert "cycle_count=1.0" not in source
    assert "목표 자기장" in source
    assert "실측 자기장" in source
    assert "오차 (목표 - 실측)" in source
    assert "1차 모델링 전압" in source
    assert "2차 모델링 전압" in source
    assert "보정 전압 변화량" in source
    assert "Raw HallBz" in source
    assert "부호 보정 자기장 (-HallBz)" in source
    assert "기준선 제거 후 자기장" in source
    assert "1차 실구동 데이터 원본 확인" in source
    assert "quick_lut_second_model_dirty" in source
    assert "native relative timebase" in source
    assert "command/target grid interpolation을 사용하지 않습니다" in source

    frames = build_second_modeling_plot_frames(
        pd.DataFrame(
            {
                "time_s": [0.0],
                "physical_target_output_mT": [0.0],
                "measured_field_normalized_mT": [0.0],
                "first_model_residual_mT": [0.0],
                "first_modeled_voltage_v": [0.0],
                "actual_drive_voltage_normalized_v": [0.0],
                "raw_hallbz_mT": [1.0],
                "measured_field_effective_mT": [-1.0],
                "baseline_removed_effective_field_mT": [0.0],
                "measured_field_baseline_removed_mT": [0.0],
                "second_limited_voltage_v": [0.0],
                "second_correction_delta_v": [0.0],
            }
        )
    )
    assert "physical_target_output_mT" not in frames["field"].columns
    assert "목표 자기장" in frames["field"].columns
    assert "second_limited_voltage_v" not in frames["voltage"].columns
    assert "2차 모델링 전압" in frames["voltage"].columns
    assert np.isclose(frames["raw"]["Raw HallBz"].iloc[0], 1.0)
    assert np.isclose(frames["raw"]["부호 보정 자기장 (-HallBz)"].iloc[0], -1.0)
    native = build_native_actual_drive_raw_plot_frame(
        pd.DataFrame(
            {
                "time_s": [0.0, 0.2, 0.5],
                "raw_hallbz_mT": [1.0, 2.0, 3.0],
                "measured_field_effective_mT": [-1.0, -2.0, -3.0],
                "measured_field_baseline_removed_mT": [0.0, -1.0, -2.0],
                "measured_field_normalized_mT": [0.0, -25.0, -50.0],
                "raw_first_voltage_v": [0.0, 2.0, 0.0],
                "normalized_first_voltage_v": [0.0, 5.0, 0.0],
            }
        )
    )
    assert native["time_s"].tolist() == [0.0, 0.2, 0.5]
    assert native["Raw HallBz"].tolist() == [1.0, 2.0, 3.0]
