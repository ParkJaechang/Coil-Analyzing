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


def _write_delayed_actual_drive_csv(path: Path, *, delay_s: float) -> None:
    rows = []
    time_ms = np.linspace(0.0, 1400.0, 201)
    relative_s = (time_ms - 200.0) / 1000.0
    voltage = np.zeros_like(time_ms)
    active = (relative_s >= 0.0) & (relative_s <= 1.0)
    voltage[active] = 2.0 * np.sin(np.pi * relative_s[active])
    effective_field = 40.0 * np.sin(np.pi * np.clip(relative_s - delay_s, 0.0, 1.0))
    hallbz = -effective_field
    for index, (t_raw, v, h) in enumerate(zip(time_ms, voltage, hallbz, strict=False)):
        rows.append(f"{index},{t_raw:.6f},0.0,0.0,{h:.6f},0.1,0.0,{v:.6f},0.0")
    preamble = [
        "# Date,2026-05-06 16:00:02",
        "# Frequency(Hz),1.000",
        "# Amplitude(V),0.000",
        "# Cycles,1.000",
        "# Repeat,1.000",
        "# PreDelay(s),1.000",
        "# PostDelay(s),1.000",
        "# HallSamples,201",
        "# CurrentSamples,201",
        "# CommonRange(ms),0.00~1400.00 (span 1400.00)",
        "# Rows,201, GridStep(ms),7.000",
        "# AutoSyncHallLag,applied 0.00ms (r=1.000)",
        "#",
        "Row,TimeMs,HallBx,HallBy,HallBz,Current1_A,Current2_A,Voltage1_V,Voltage2_V",
    ]
    path.write_text("\n".join([*preamble, *rows]), encoding="utf-8")


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


def test_second_modeling_uses_smoothed_measured_field_for_residual(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_actual_drive_csv(actual)
    text = actual.read_text(encoding="utf-8")
    lines = text.splitlines()
    header_index = next(index for index, line in enumerate(lines) if line.startswith("Row,TimeMs"))
    noisy_rows = []
    for row_index, line in enumerate(lines[header_index + 1 :]):
        parts = line.split(",")
        if len(parts) >= 9:
            parts[4] = f"{float(parts[4]) + (8.0 if row_index % 2 else -8.0):.6f}"
        noisy_rows.append(",".join(parts))
    actual.write_text("\n".join([*lines[: header_index + 1], *noisy_rows]), encoding="utf-8")

    frame, metadata = generate_second_modeled_voltage_lut(
        _first_profile(),
        actual,
        freq_hz=1.0,
        cycle_count=1.0,
        correction_gain=0.25,
    )

    assert metadata["measured_field_smoothing_enabled"] is True
    assert metadata["measured_field_smoothing_status"] == "ok"
    assert metadata["residual_source_for_second_modeling"] == "first_peak_aligned_smoothed_measured_field"
    assert metadata["correction_delta_source"] == "first_model_residual_for_second_mT"
    assert {
        "measured_field_smoothed_mT",
        "second_modeling_measured_field_mT",
        "first_model_residual_raw_mT",
        "first_model_residual_smoothed_mT",
    }.issubset(frame.columns)
    assert np.allclose(frame["first_model_residual_mT"], frame["first_model_residual_for_second_mT"], equal_nan=True)
    assert not np.allclose(frame["first_model_residual_raw_mT"], frame["first_model_residual_smoothed_mT"], equal_nan=True)


def test_second_modeling_defaults_to_first_peak_aligned_stabilized_residual(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_delayed_actual_drive_csv(actual, delay_s=0.12)

    frame, metadata = generate_second_modeled_voltage_lut(
        _first_profile(),
        actual,
        freq_hz=1.0,
        cycle_count=1.0,
        correction_gain=0.25,
    )

    assert metadata["residual_alignment_mode"] == "first_peak_aligned_stabilized"
    assert metadata["phase_alignment_enabled"] is True
    assert metadata["phase_alignment_method"] == "first_positive_peak"
    assert metadata["phase_alignment_status"] == "ok"
    assert metadata["correction_stabilization_enabled"] is True
    assert metadata["correction_zero_start_guard_enabled"] is True
    assert metadata["correction_ramp_in_enabled"] is True
    assert metadata["correction_taper_out_enabled"] is False
    assert metadata["correction_polarity_guard_enabled"] is True
    assert metadata["correction_envelope_applied"] is True
    assert metadata["correction_delta_smoothing_enabled"] is True
    assert metadata["measured_field_smoothing_scope"] == "native_window_until_zero_return"
    assert np.isclose(metadata["phase_alignment_shift_s"], 0.12, atol=0.035)
    assert np.isclose(metadata["phase_alignment_shift_cycles"], 0.12, atol=0.035)
    assert {
        "measured_field_aligned_mT",
        "first_model_residual_pointwise_mT",
        "first_model_residual_aligned_mT",
        "first_model_residual_for_second_mT",
        "raw_second_correction_delta_v",
        "second_correction_delta_v_smooth",
        "correction_envelope",
        "measured_field_smoothed_full_mT",
    }.issubset(frame.columns)
    target_peak_time = float(metadata["target_first_peak_time_s"])
    aligned_peak_time = float(frame.loc[frame["measured_field_aligned_mT"].idxmax(), "time_s"])
    assert np.isclose(aligned_peak_time, target_peak_time, atol=0.04)
    assert np.allclose(frame["first_model_residual_for_second_mT"], frame["first_model_residual_aligned_mT"], equal_nan=True)
    active = frame["active_window_mask"].astype(bool).to_numpy()
    active_indices = np.flatnonzero(active)
    start_index = int(active_indices[0])
    assert abs(float(frame.loc[start_index, "second_correction_delta_v"])) <= 1e-9
    assert np.isclose(
        frame.loc[start_index, "second_limited_voltage_v"],
        frame.loc[start_index, "first_modeled_voltage_v"],
        atol=1e-9,
    )
    first_positive_lobe = active & (frame["first_modeled_voltage_v"].to_numpy(dtype=float) >= 0.05)
    assert np.nanmin(frame.loc[first_positive_lobe, "second_limited_voltage_v"]) >= -1e-12
    assert frame.loc[start_index, "correction_envelope"] == 0.0
    assert frame.loc[active_indices[-1], "correction_envelope"] > 0.99


def test_second_modeling_stabilization_uses_start_gate_without_global_offset(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_delayed_actual_drive_csv(actual, delay_s=0.12)

    frame, metadata = generate_second_modeled_voltage_lut(
        _first_profile(),
        actual,
        freq_hz=1.0,
        cycle_count=1.0,
        correction_gain=0.25,
        residual_alignment_mode="first_peak_aligned_stabilized",
    )

    source = (SRC_ROOT / "field_analysis" / "finite_second_modeling_stabilization.py").read_text(encoding="utf-8")
    assert "- raw[start_index]" not in source
    assert "correction_start_gate_duration_s" in metadata
    assert np.isclose(metadata["correction_start_gate_duration_s"], 0.25, atol=1e-12)
    assert metadata["correction_start_gate_applied_only_to_initial_segment"] is True
    assert metadata["polarity_guard_mode"] == "start_segment_only"
    assert metadata["polarity_guard_discontinuity_risk"] == "low_start_segment_only"
    assert {
        "raw_correction_delta_v",
        "smoothed_correction_delta_v",
        "stabilized_correction_delta_v",
        "start_gate",
        "taper_gate",
        "correction_envelope",
        "second_voltage_before_polarity_guard_v",
        "second_voltage_after_polarity_guard_v",
        "correction_nan_mask",
        "correction_active_mask",
        "source_range_valid_mask",
        "polarity_guard_applied_mask",
    }.issubset(frame.columns)
    active = frame["active_window_mask"].astype(bool).to_numpy()
    active_indices = np.flatnonzero(active)
    start_index = int(active_indices[0])
    after_gate = int(np.flatnonzero(frame["time_s"].to_numpy(dtype=float) >= 0.25)[0])
    mid_index = int(active_indices[len(active_indices) // 2])
    assert np.isclose(frame.loc[start_index, "start_gate"], 0.0)
    assert frame.loc[after_gate, "start_gate"] >= 0.999
    assert np.isclose(frame.loc[mid_index, "correction_envelope"], 1.0, atol=0.02)
    assert np.isclose(
        frame.loc[mid_index, "stabilized_correction_delta_v"],
        frame.loc[mid_index, "smoothed_correction_delta_v"],
        atol=0.10,
    )
    assert np.allclose(frame["second_correction_delta_v"], frame["stabilized_correction_delta_v"], equal_nan=True)


def test_second_modeling_preserves_active_end_correction_and_adds_zero_return_tail(tmp_path: Path) -> None:
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

    active = frame["active_window_mask"].astype(bool).to_numpy()
    tail = frame["tail_window_mask"].astype(bool).to_numpy()
    active_indices = np.flatnonzero(active)
    tail_indices = np.flatnonzero(tail)
    active_end_index = int(active_indices[-1])
    before_end_index = int(active_indices[-2])

    assert metadata["active_taper_out_enabled"] is False
    assert metadata["active_end_correction_preserved"] is True
    assert metadata["post_cycle_zero_tail_enabled"] is True
    assert metadata["post_cycle_zero_tail_cycle_count"] >= 0.25
    assert np.isclose(
        metadata["post_cycle_zero_tail_duration_s"],
        metadata["post_cycle_zero_tail_cycle_count"] / 1.5,
    )
    assert len(frame) > len(profile)
    assert tail.any()
    assert np.isclose(frame.loc[active_end_index, "taper_gate"], 1.0)
    assert np.isclose(frame.loc[before_end_index, "taper_gate"], 1.0)
    assert np.isclose(
        frame.loc[active_end_index, "stabilized_correction_delta_v"],
        frame.loc[active_end_index, "smoothed_correction_delta_v"],
        atol=0.12,
    )
    assert np.allclose(frame.loc[tail, "tail_target_field_mT"], 0.0)
    assert np.isclose(frame.loc[tail_indices[-1], "second_limited_voltage_v"], 0.0, atol=1e-9)
    assert np.isclose(frame.loc[tail_indices[-1], "tail_voltage_v"], 0.0, atol=1e-9)
    assert "active_to_tail_voltage_jump_v" in metadata
    assert metadata["active_to_tail_continuity_status"] in {"ok", "warning_jump_detected"}


def test_second_modeling_discontinuity_diagnostics_exist(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_delayed_actual_drive_csv(actual, delay_s=0.12)

    _frame, metadata = generate_second_modeled_voltage_lut(
        _first_profile(),
        actual,
        freq_hz=1.0,
        cycle_count=1.0,
        correction_gain=0.25,
    )

    assert "correction_discontinuity_detected" in metadata
    assert "correction_discontinuity_time_s" in metadata
    assert "correction_discontinuity_source" in metadata
    assert "max_abs_delta_step_v" in metadata
    assert "max_abs_second_voltage_step_v" in metadata
    assert "discontinuity_threshold_v" in metadata


def test_second_modeling_first_peak_aligned_without_stabilization_is_preserved(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_delayed_actual_drive_csv(actual, delay_s=0.12)

    frame, metadata = generate_second_modeled_voltage_lut(
        _first_profile(),
        actual,
        freq_hz=1.0,
        cycle_count=1.0,
        correction_gain=0.25,
        residual_alignment_mode="first_peak_aligned",
    )

    assert metadata["residual_alignment_mode"] == "first_peak_aligned"
    assert metadata["correction_stabilization_enabled"] is False
    expected_delta = frame["first_model_residual_for_second_mT"] / 50.0 * 5.0 * 0.25
    active = frame["active_window_mask"].astype(bool).to_numpy()
    interior = active & np.isfinite(expected_delta.to_numpy())
    assert np.nanmean(np.abs(frame.loc[interior, "second_correction_delta_v"] - expected_delta[interior])) < 0.08


def test_second_modeling_pointwise_residual_mode_is_preserved(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_delayed_actual_drive_csv(actual, delay_s=0.12)

    frame, metadata = generate_second_modeled_voltage_lut(
        _first_profile(),
        actual,
        freq_hz=1.0,
        cycle_count=1.0,
        correction_gain=0.25,
        residual_alignment_mode="pointwise",
    )

    assert metadata["residual_alignment_mode"] == "pointwise"
    assert metadata["phase_alignment_enabled"] is False
    assert metadata["phase_alignment_status"] == "disabled_pointwise"
    assert np.allclose(frame["measured_field_aligned_mT"], frame["measured_field_smoothed_mT"], equal_nan=True)
    assert np.allclose(frame["first_model_residual_for_second_mT"], frame["first_model_residual_pointwise_mT"], equal_nan=True)


def test_second_modeling_first_peak_alignment_falls_back_when_shift_too_large(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_delayed_actual_drive_csv(actual, delay_s=0.5)

    frame, metadata = generate_second_modeled_voltage_lut(
        _first_profile(),
        actual,
        freq_hz=1.0,
        cycle_count=1.0,
        correction_gain=0.25,
        residual_alignment_mode="first_peak_aligned",
    )

    assert metadata["residual_alignment_mode"] == "first_peak_aligned"
    assert metadata["phase_alignment_status"] == "shift_too_large"
    assert np.allclose(frame["first_model_residual_for_second_mT"], frame["first_model_residual_pointwise_mT"], equal_nan=True)


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


def test_second_modeling_preserves_first_command_profile_and_limited_voltage(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_actual_drive_csv(actual)
    profile = _first_profile()
    original = profile.copy(deep=True)

    frame, metadata = generate_second_modeled_voltage_lut(profile, actual, freq_hz=1.0, cycle_count=1.0)

    assert metadata["second_modeling_status"] == "ok"
    pd.testing.assert_frame_equal(profile, original)
    first_slice = frame.iloc[: len(original)]
    assert np.allclose(first_slice["first_modeled_voltage_v"], original["limited_voltage_v"])
    assert np.allclose(first_slice["limited_voltage_v"], original["limited_voltage_v"])
    assert not np.allclose(first_slice["second_limited_voltage_v"], original["limited_voltage_v"])


def test_second_modeling_uses_first_command_not_final_voltage_as_input(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_actual_drive_csv(actual)
    profile = _first_profile()
    profile["final_voltage_v"] = 4.75

    frame, metadata = generate_second_modeled_voltage_lut(profile, actual, freq_hz=1.0, cycle_count=1.0)

    assert metadata["second_modeling_status"] == "ok"
    first_slice = frame.iloc[: len(profile)]
    assert np.allclose(first_slice["first_modeled_voltage_v"], profile["limited_voltage_v"])
    assert not np.allclose(first_slice["first_modeled_voltage_v"], profile["final_voltage_v"])


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


def test_second_lut_export_includes_zero_return_tail_samples(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1.5Hz_1.5cycle_result.csv"
    _write_actual_drive_csv(actual)
    frame, metadata = generate_second_modeled_voltage_lut(_first_profile(), actual, freq_hz=1.5, cycle_count=1.5)
    frame["second_modeling_available"] = True
    frame["second_modeling_status"] = "ok"

    export = build_final_modeled_voltage_lut_export(frame, freq_hz=1.5, cycle_count=1.5, waveform="sine")

    assert metadata["post_cycle_zero_tail_enabled"] is True
    assert len(export["frame"]) == len(frame)
    assert list(export["frame"].columns) == ["sample_index", "time_s", "voltage_v"]
    assert np.isclose(export["frame"]["voltage_v"].iloc[-1], 0.0, atol=1e-9)
    assert np.allclose(export["frame"]["voltage_v"], frame["second_limited_voltage_v"])


def test_second_modeling_ui_uses_actual_cycle_for_final_export_and_korean_plot_labels() -> None:
    from field_analysis.ui_second_modeling import build_native_actual_drive_raw_plot_frame, build_second_modeling_plot_frames

    source = (SRC_ROOT / "field_analysis" / "ui_second_modeling.py").read_text(encoding="utf-8")

    assert "cycle_count=cycle_count" in source
    assert "cycle_count=1.0" not in source
    assert "freq_hz=freq_hz" in source
    assert "waveform_type=waveform_type" in source
    assert "목표 자기장" in source
    assert "실측 자기장" in source
    assert "오차 (목표 - 보정 계산용 실측)" in source
    assert "1차 모델링 전압" in source
    assert "2차 보정 전압" in source
    assert "전압 제한 후 2차 command" in source
    assert "2차 보정 command" in source
    assert "사용 중인 1차 실구동 데이터" in source
    assert "보정 전압 변화량" in source
    assert "Raw HallBz" in source
    assert "부호 보정 자기장 (-HallBz)" in source
    assert "기준선 제거 후 자기장" in source
    assert "1차 실구동 데이터 원본 확인" in source
    assert "quick_lut_second_model_dirty" in source
    assert "quick_lut_second_model_result" in source
    assert "quick_lut_actual_drive_review_result" in source
    assert "2차 보정 command runtime trace" in source
    assert "단계별 trace" in source
    assert "실측 자기장 smoothing" in source
    assert "오차 (목표 - 보정 계산용 실측)" in source
    assert "보정 전압 변화량 = gain × 오차 / 50mT × 5V" in source
    assert "2차 command = 1차 command + 안정화된 보정 전압 변화량" in source
    assert "2차 보정 residual 계산 방식" in source
    assert "첫 피크 정렬 + 안정화" in source
    assert "첫 피크 정렬 residual" in source
    assert "피크 정렬 실측 자기장" in source
    assert "보정 계산용 실측 자기장" in source
    assert "피크 정렬 확인" in source
    assert "raw 보정 전압 변화량" in source
    assert "smoothing 후 보정 전압 변화량" in source
    assert "안정화 후 보정 전압 변화량" in source
    assert "tail 자기장 0 복귀 전압" in source
    assert "active 구간 끝에서는 보정을 강제로 0으로 줄이지 않습니다" in source
    assert "자동 추천 gain" in source
    assert "수동 gain" in source
    assert "보정 전압 불연속 진단" in source
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
                "second_modeled_voltage_v": [0.1],
                "second_limited_voltage_v": [0.0],
                "second_correction_delta_v": [0.0],
                "raw_second_correction_delta_v": [0.2],
                "smoothed_correction_delta_v": [0.1],
                "second_correction_delta_v_smooth": [0.0],
                "tail_voltage_v": [0.0],
                "active_correction_delta_v": [0.0],
            }
        )
    )
    assert "physical_target_output_mT" not in frames["field"].columns
    assert "목표 자기장" in frames["field"].columns
    assert "second_limited_voltage_v" not in frames["voltage"].columns
    assert "2차 보정 전압" in frames["voltage"].columns
    assert "전압 제한 후 2차 command" in frames["voltage"].columns
    assert "실측 자기장 smoothing" in frames["field"].columns
    assert "보정 계산용 실측 자기장" in frames["field"].columns
    assert "raw 보정 전압 변화량" in frames["voltage"].columns
    assert "smoothing 후 보정 전압 변화량" in frames["voltage"].columns
    assert "안정화 후 보정 전압 변화량" in frames["voltage"].columns
    assert "tail 자기장 0 복귀 전압" in frames["voltage"].columns
    assert "active 보정 전압 변화량" in frames["voltage"].columns
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


def test_second_modeling_ui_does_not_auto_switch_final_export_source() -> None:
    source = (SRC_ROOT / "field_analysis" / "ui_second_modeling.py").read_text(encoding="utf-8")
    export_source = (SRC_ROOT / "field_analysis" / "ui_final_voltage_lut_export.py").read_text(encoding="utf-8")

    assert 'quick_lut_final_export_source"] = "second_model"' not in source
    assert "2차 결과가 생겨도 최종 LUT 추출 대상은 자동으로 바뀌지 않습니다." in source
    assert "index=0" in export_source
    assert 'index=1 if second_available' not in export_source
