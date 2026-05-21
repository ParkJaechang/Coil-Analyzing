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
from field_analysis.finite_second_modeling_stabilization import stabilize_correction_delta
from field_analysis.finite_second_modeling_tail import resolve_finite_tail_policy
from tests.test_finite_actual_drive_response import _write_actual_drive_csv
from tests.test_finite_second_modeling import _first_profile, _write_delayed_actual_drive_csv


def test_resolve_finite_tail_policy_auto_and_manual_modes() -> None:
    assert resolve_finite_tail_policy(1.0, "auto", 2.0)["finite_tail_effective_enabled"] is True
    assert resolve_finite_tail_policy(1.5, "auto", 2.0)["finite_tail_effective_enabled"] is True
    assert resolve_finite_tail_policy(2.0, "auto", 2.0)["finite_tail_effective_enabled"] is False
    assert resolve_finite_tail_policy(3.0, "auto", 2.0)["finite_tail_effective_enabled"] is False
    assert resolve_finite_tail_policy(3.0, "on", 2.0)["finite_tail_effective_enabled"] is True
    assert resolve_finite_tail_policy(1.0, "off", 2.0)["finite_tail_effective_enabled"] is False
    assert resolve_finite_tail_policy(2.0, "auto", 2.0)["finite_tail_disabled_reason"] == "auto_disabled_high_frequency"


def test_second_modeling_tail_off_has_active_only_samples(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_actual_drive_csv(actual)
    profile = _first_profile()
    tail_rows = profile.tail(10).copy()
    tail_rows["time_s"] = np.linspace(1.01, 1.20, len(tail_rows))
    profile = pd.concat([profile, tail_rows], ignore_index=True)

    frame, metadata = generate_second_modeled_voltage_lut(
        profile,
        actual,
        freq_hz=1.0,
        cycle_count=1.0,
        post_cycle_zero_tail_enabled=False,
    )

    assert metadata["post_cycle_zero_tail_enabled"] is False
    assert metadata["finite_tail_effective_enabled"] is False
    assert metadata["tail_return_mode"] == "disabled"
    assert metadata["tail_voltage_generated"] is False
    assert metadata["tail_window_sample_count"] == 0
    assert metadata["total_command_duration_s"] == 1.0
    assert frame["time_s"].max() <= 1.0 + 1e-12
    assert not frame["tail_window_mask"].astype(bool).any()


def test_second_modeling_one_point_five_tail_off_has_no_tail_samples(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1.5Hz_1.5cycle_result.csv"
    _write_actual_drive_csv(actual)

    frame, metadata = generate_second_modeled_voltage_lut(
        _first_profile(),
        actual,
        freq_hz=1.5,
        cycle_count=1.5,
        post_cycle_zero_tail_enabled=False,
    )

    assert metadata["post_cycle_zero_tail_enabled"] is False
    assert metadata["tail_window_sample_count"] == 0
    assert not frame["tail_window_mask"].astype(bool).any()


def test_finite_tail_threshold_message_is_dynamic() -> None:
    policy = resolve_finite_tail_policy(3.0, "auto", 3.0)

    assert policy["finite_tail_effective_enabled"] is False
    assert policy["finite_tail_warning_message_dynamic"] is True
    assert "3" in policy["finite_tail_status_message"]
    assert "3" in policy["finite_tail_warning_message"]
    assert "2Hz 이상" not in (SRC_ROOT / "field_analysis" / "ui_finite_tail_policy.py").read_text(encoding="utf-8")


def test_second_modeling_tail_timebase_is_monotonic_and_continuous(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1.5Hz_1.5cycle_result.csv"
    _write_actual_drive_csv(actual)

    frame, metadata = generate_second_modeled_voltage_lut(_first_profile(), actual, freq_hz=1.5, cycle_count=1.5)

    time_s = frame["time_s"].to_numpy(dtype=float)
    tail = frame["tail_window_mask"].astype(bool).to_numpy()
    active = frame["active_window_mask"].astype(bool).to_numpy()
    active_end_index = int(np.flatnonzero(active)[-1])
    tail_start_index = int(np.flatnonzero(tail)[0])

    assert metadata["second_command_time_monotonic"] is True
    assert metadata["second_command_duplicate_time_count"] == 0
    assert metadata["active_tail_duplicate_time_removed"] == 0
    assert np.all(np.diff(time_s) > 0)
    assert time_s[tail_start_index] > time_s[active_end_index]
    assert metadata["active_to_tail_voltage_jump_v"] <= 0.5
    assert np.isclose(
        frame.loc[tail_start_index, "tail_start_voltage_v"],
        frame.loc[active_end_index, "second_limited_voltage_v"],
        atol=1e-9,
    )
    assert metadata["tail_continuity_blend_enabled"] is True


def test_second_modeling_tail_start_does_not_reset_to_zero_when_active_end_is_nonzero(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_slow_zero_return_actual_drive_csv(actual)
    profile = _first_profile()
    active_end_row = int(profile["time_s"].idxmax())
    profile.loc[active_end_row, "limited_voltage_v"] = -0.8
    profile.loc[active_end_row, "recommended_voltage_v"] = -0.8

    frame, metadata = generate_second_modeled_voltage_lut(profile, actual, freq_hz=1.0, cycle_count=1.0)

    active = frame["active_window_mask"].astype(bool).to_numpy()
    tail = frame["tail_window_mask"].astype(bool).to_numpy()
    active_end_index = int(np.flatnonzero(active)[-1])
    tail_start_index = int(np.flatnonzero(tail)[0])

    assert metadata["tail_voltage_generated_independently_from_first_voltage"] is True
    assert metadata["tail_start_reset_to_zero_detected"] is False
    assert metadata["active_to_tail_zero_reset_detected"] is False
    assert metadata["active_to_tail_voltage_jump_v"] <= 1e-9
    assert np.isclose(
        frame.loc[tail_start_index, "second_limited_voltage_v"],
        frame.loc[active_end_index, "second_limited_voltage_v"],
        atol=1e-9,
    )
    assert not np.isclose(frame.loc[tail_start_index, "second_limited_voltage_v"], 0.0, atol=1e-9)
    assert np.isclose(frame.loc[tail_start_index, "first_modeled_voltage_v"], 0.0, atol=1e-12)


def test_second_modeling_seconds_tail_duration_uses_exact_user_seconds(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1.5Hz_1.5cycle_result.csv"
    _write_actual_drive_csv(actual)

    frame, metadata = generate_second_modeled_voltage_lut(
        _first_profile(),
        actual,
        freq_hz=1.5,
        cycle_count=1.5,
        tail_duration_mode="seconds",
        post_cycle_zero_tail_duration_s=0.2,
    )

    assert metadata["tail_duration_mode"] == "seconds"
    assert np.isclose(metadata["tail_duration_s"], 0.2)
    assert np.isclose(metadata["tail_cycle_count"], 0.3)
    assert np.isclose(metadata["total_command_duration_s"], 1.2, atol=0.01)
    assert np.isclose(frame["time_s"].max(), 1.2, atol=0.02)


def test_second_modeling_cycle_tail_duration_mode_is_preserved(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1.5Hz_1.5cycle_result.csv"
    _write_actual_drive_csv(actual)

    _frame, metadata = generate_second_modeled_voltage_lut(
        _first_profile(),
        actual,
        freq_hz=1.5,
        cycle_count=1.5,
        tail_duration_mode="cycle",
        post_cycle_zero_tail_cycle_count=0.3,
    )

    assert metadata["tail_duration_mode"] == "cycle"
    assert np.isclose(metadata["tail_cycle_count"], 0.3)
    assert np.isclose(metadata["tail_duration_s"], 0.2)


def test_second_modeling_tail_duration_ui_markers_exist() -> None:
    source = (SRC_ROOT / "field_analysis" / "ui_second_modeling.py").read_text(encoding="utf-8")

    assert "자기장 0 복귀 tail 방식" in source
    assert "자기장 0 복귀 시간 (s)" in source
    assert "이 시간 안에 tail 전압이 자기장을 0으로 보내도록 계산합니다." in source
    assert "tail 길이 (cycle)" not in source


def test_second_modeling_tail_b0_uses_native_actual_measured_field_at_active_end(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_delayed_actual_drive_csv(actual, delay_s=0.08)

    frame, metadata = generate_second_modeled_voltage_lut(_first_profile(), actual, freq_hz=1.0, cycle_count=1.0)

    active = frame["active_window_mask"].astype(bool).to_numpy()
    active_end_index = int(np.flatnonzero(active)[-1])
    assert metadata["tail_B0_source"] == "native_smoothed_measured_at_active_end"
    assert np.isclose(
        metadata["tail_B0_mT"],
        frame.loc[active_end_index, "measured_field_smoothed_native_mT"],
        atol=1e-9,
        equal_nan=True,
    )


def test_second_modeling_uses_unified_active_tail_residual_and_correction(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_slow_zero_return_actual_drive_csv(actual)

    frame, metadata = generate_second_modeled_voltage_lut(
        _first_profile(),
        actual,
        freq_hz=1.0,
        cycle_count=1.0,
        tail_return_mode="residual",
    )

    active = frame["active_window_mask"].astype(bool).to_numpy()
    tail = frame["tail_window_mask"].astype(bool).to_numpy()
    correction_mask = active | tail
    assert metadata["second_command_synthesis_mode"] == "unified_active_tail_residual"
    assert {"target_field_for_second_mT", "measured_field_for_second_mT", "residual_for_second_mT"}.issubset(frame.columns)
    assert {"measured_field_smoothed_native_mT", "measured_field_aligned_native_mT"}.issubset(frame.columns)
    assert metadata["raw_delta_zeroed_outside_active"] is False
    assert metadata["correction_valid_mask_source"] == "active_plus_tail_measured_support"
    assert np.allclose(frame.loc[tail, "target_field_for_second_mT"], 0.0)
    assert np.allclose(
        frame.loc[correction_mask, "residual_for_second_mT"],
        frame.loc[correction_mask, "target_field_for_second_mT"] - frame.loc[correction_mask, "measured_field_for_second_mT"],
        equal_nan=True,
    )
    assert np.allclose(
        frame.loc[correction_mask, "correction_delta_v"],
        frame.loc[correction_mask, "correction_delta_v_smoothed"] * frame.loc[correction_mask, "correction_start_gate"] * frame.loc[correction_mask, "correction_tail_taper_gate"],
        atol=1e-9,
        equal_nan=True,
    )
    assert np.allclose(
        frame["second_voltage_before_clip_v"],
        frame["first_modeled_voltage_v"] + frame["correction_delta_v"],
        equal_nan=True,
    )


def test_second_modeling_does_not_last_value_hold_missing_tail_measured_data(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_actual_drive_csv(actual)
    lines = actual.read_text(encoding="utf-8").splitlines()
    header_index = next(index for index, line in enumerate(lines) if line.startswith("Row,TimeMs"))
    rows = [line for line in lines[header_index + 1 :] if float(line.split(",")[1]) <= 1219.0]
    actual.write_text("\n".join([*lines[: header_index + 1], *rows]), encoding="utf-8")

    frame, metadata = generate_second_modeled_voltage_lut(
        _first_profile(),
        actual,
        freq_hz=1.0,
        cycle_count=1.0,
        tail_duration_mode="seconds",
        post_cycle_zero_tail_duration_s=0.25,
    )

    tail = frame["tail_window_mask"].astype(bool).to_numpy()
    missing_tail = tail & ~frame["measured_support_valid_mask"].astype(bool).to_numpy()
    assert missing_tail.any()
    assert frame.loc[missing_tail, "measured_field_for_second_mT"].isna().all()
    assert metadata["measured_tail_synthetic_fill_used"] is False
    assert metadata["measured_tail_last_value_hold_used"] is False
    assert metadata["measured_tail_fake_decay_used"] is False
    assert metadata["measured_tail_actual_data_only"] is True


def test_second_modeling_residual_tail_blocks_when_phase_shifted_tail_data_missing(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_actual_drive_csv(actual)
    lines = actual.read_text(encoding="utf-8").splitlines()
    header_index = next(index for index, line in enumerate(lines) if line.startswith("Row,TimeMs"))
    rows = [line for line in lines[header_index + 1 :] if float(line.split(",")[1]) <= 1219.0]
    actual.write_text("\n".join([*lines[: header_index + 1], *rows]), encoding="utf-8")

    frame, metadata = generate_second_modeled_voltage_lut(
        _first_profile(),
        actual,
        freq_hz=1.0,
        cycle_count=1.0,
        tail_return_mode="residual",
        tail_duration_mode="seconds",
        post_cycle_zero_tail_duration_s=0.25,
    )

    assert "second_limited_voltage_v" not in frame.columns
    assert metadata["second_modeling_available"] is False
    assert metadata["second_modeling_status"] == "residual_tail_measured_data_unavailable"
    assert metadata["residual_tail_available"] is False
    assert metadata["residual_tail_unavailable_reason"] == "phase_shifted_tail_source_range_insufficient"


def test_second_modeling_does_not_overlay_separate_tail_voltage(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1.5Hz_1.5cycle_result.csv"
    _write_actual_drive_csv(actual)

    frame, metadata = generate_second_modeled_voltage_lut(_first_profile(), actual, freq_hz=1.5, cycle_count=1.5)

    tail = frame["tail_window_mask"].astype(bool).to_numpy()
    assert metadata["tail_voltage_overlay_used"] is False
    assert np.allclose(frame.loc[tail, "tail_voltage_v"], frame.loc[tail, "correction_delta_v"], equal_nan=True)
    assert np.allclose(frame["second_limited_voltage_v"], np.clip(frame["second_voltage_before_clip_v"], -5.0, 5.0))


def test_second_modeling_measured_source_stays_consistent_across_active_tail_boundary(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_delayed_actual_drive_csv(actual, delay_s=0.05)

    frame, metadata = generate_second_modeled_voltage_lut(_first_profile(), actual, freq_hz=1.0, cycle_count=1.0)

    active = frame["active_window_mask"].astype(bool).to_numpy()
    tail = frame["tail_window_mask"].astype(bool).to_numpy()
    active_end_index = int(np.flatnonzero(active)[-1])
    tail_start_index = int(np.flatnonzero(tail)[0])

    assert metadata["measured_field_source_switch_at_active_end"] is False
    assert np.isclose(
        frame.loc[tail_start_index, "measured_field_for_second_mT"],
        frame.loc[tail_start_index, "measured_field_aligned_mT"],
        atol=1e-9,
        equal_nan=True,
    )
    boundary_jump = abs(
        float(frame.loc[tail_start_index, "measured_field_for_second_mT"])
        - float(frame.loc[active_end_index, "measured_field_for_second_mT"])
    )
    local_step = np.nanmedian(np.abs(np.diff(frame["measured_field_for_second_mT"].to_numpy(dtype=float))))
    assert boundary_jump <= max(20.0, 8.0 * float(local_step))


def _write_slow_zero_return_actual_drive_csv(path: Path) -> None:
    rows = []
    time_ms = np.linspace(0.0, 2200.0, 315)
    relative_s = (time_ms - 200.0) / 1000.0
    voltage = np.zeros_like(time_ms)
    active = (relative_s >= 0.0) & (relative_s <= 1.0)
    voltage[active] = 2.0 * np.sin(np.pi * relative_s[active])
    effective = np.zeros_like(time_ms)
    effective[active] = 40.0 * np.sin(np.pi * relative_s[active])
    post = (relative_s > 1.0) & (relative_s <= 1.75)
    effective[post] = 30.0 * (1.0 - (relative_s[post] - 1.0) / 0.75)
    hallbz = -effective
    for index, (t_raw, v, h) in enumerate(zip(time_ms, voltage, hallbz, strict=False)):
        rows.append(f"{index},{t_raw:.6f},0.0,0.0,{h:.6f},0.1,0.0,{v:.6f},0.0")
    preamble = [
        "# Date,2026-05-06 16:00:02",
        "# Frequency(Hz),1.000",
        "# Cycles,1.000",
        "# Rows,315",
        "Row,TimeMs,HallBx,HallBy,HallBz,Current1_A,Current2_A,Voltage1_V,Voltage2_V",
    ]
    path.write_text("\n".join([*preamble, *rows]), encoding="utf-8")


def test_second_modeling_measured_support_extends_to_detected_zero_return(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_slow_zero_return_actual_drive_csv(actual)

    frame, metadata = generate_second_modeled_voltage_lut(_first_profile(), actual, freq_hz=1.0, cycle_count=1.0)

    assert metadata["zero_return_detection_enabled"] is True
    assert metadata["measured_zero_return_status"] == "detected_zero_return"
    assert metadata["measured_support_end_mode"] == "detected_zero_return"
    assert np.isclose(metadata["measured_zero_return_time_s"], 1.75, atol=0.06)
    assert np.isclose(metadata["post_cycle_zero_tail_cycle_count"], 0.25)
    assert metadata["tail_return_mode"] == "finite_time_zero_return"
    assert metadata["measured_support_end_s"] >= float(metadata["measured_zero_return_time_s"]) - 0.02
    assert np.allclose(frame.loc[frame["tail_window_mask"].astype(bool), "target_field_for_second_mT"], 0.0)


def test_second_modeling_required_source_end_uses_zero_return_time_and_phase_shift(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_delayed_actual_drive_csv(actual, delay_s=0.05)

    frame, metadata = generate_second_modeled_voltage_lut(_first_profile(), actual, freq_hz=1.0, cycle_count=1.0)

    expected_required_end = float(metadata["measured_support_end_s"]) + abs(float(metadata["phase_alignment_shift_s"]))
    assert np.isclose(metadata["required_measured_support_end_s"], expected_required_end, atol=1e-9)
    assert metadata["actual_drive_source_time_end_s"] >= 1.19
    assert metadata["measured_support_coverage_status"] in {"ok", "insufficient_for_zero_return"}
    assert metadata["aligned_measured_source_range_status"] == metadata["measured_support_coverage_status"]
    assert metadata["measured_field_smoothing_scope"] == "native_window_until_zero_return"
    assert "measured_support_valid_mask" in frame.columns
    assert bool(frame["measured_support_valid_mask"].any())


def test_second_modeling_measured_support_reports_ok_when_source_covers_required_end(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_delayed_actual_drive_csv(actual, delay_s=0.05)

    lines = actual.read_text(encoding="utf-8").splitlines()
    header_index = next(index for index, line in enumerate(lines) if line.startswith("Row,TimeMs"))
    rows = list(lines[header_index + 1 :])
    for row_index in range(202, 321):
        t_ms = 1400.0 + (row_index - 201) * 7.0
        rows.append(f"{row_index},{t_ms:.6f},0.0,0.0,0.000000,0.1,0.0,0.000000,0.0")
    actual.write_text("\n".join([*lines[: header_index + 1], *rows]), encoding="utf-8")

    frame, metadata = generate_second_modeled_voltage_lut(_first_profile(), actual, freq_hz=1.0, cycle_count=1.0)

    assert metadata["measured_support_coverage_status"] == "ok"
    assert metadata["measured_support_covers_aligned_tail"] is True
    assert metadata["measured_field_source_switch_at_active_end"] is False
    assert metadata["measured_support_end_mode"] == "detected_zero_return"
    tail = frame["tail_window_mask"].astype(bool).to_numpy()
    invalid_tail = tail & ~frame["measured_support_valid_mask"].astype(bool).to_numpy()
    assert frame.loc[invalid_tail, "measured_field_for_second_mT"].isna().all()
    assert metadata["measured_tail_actual_data_only"] is True


def test_second_modeling_zero_flat_diagnostic_exists_without_mid_segment_mask_flat(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_delayed_actual_drive_csv(actual, delay_s=0.12)

    frame, metadata = generate_second_modeled_voltage_lut(_first_profile(), actual, freq_hz=1.0, cycle_count=1.0)

    mask = frame["correction_zero_flat_segment_mask"].astype(bool).to_numpy()
    active_or_tail = (
        frame["active_window_mask"].astype(bool).to_numpy()
        | frame["tail_window_mask"].astype(bool).to_numpy()
    )
    assert "correction_zero_flat_segment_detected" in metadata
    assert "correction_zero_flat_segment_time_ranges" in metadata
    assert "correction_invalid_policy" in metadata
    assert not bool(np.any(mask & active_or_tail))


def test_second_modeling_has_no_active_only_zeroing_or_global_start_offset_source() -> None:
    modeling_source = (SRC_ROOT / "field_analysis" / "finite_second_modeling.py").read_text(encoding="utf-8")
    stabilization_source = (SRC_ROOT / "field_analysis" / "finite_second_modeling_stabilization.py").read_text(encoding="utf-8")

    assert "raw_delta[~active_mask" not in modeling_source
    assert "smoothed_delta[~active_mask" not in modeling_source
    assert "raw[start_index]" not in stabilization_source
    assert "- raw[start_index]" not in stabilization_source
    assert "anchored" not in stabilization_source


def test_second_modeling_tapers_only_at_tail_end(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1.5Hz_1.5cycle_result.csv"
    _write_actual_drive_csv(actual)

    frame, metadata = generate_second_modeled_voltage_lut(_first_profile(), actual, freq_hz=1.5, cycle_count=1.5)

    active = frame["active_window_mask"].astype(bool).to_numpy()
    tail = frame["tail_window_mask"].astype(bool).to_numpy()
    active_end_index = int(np.flatnonzero(active)[-1])
    tail_start_index = int(np.flatnonzero(tail)[0])
    tail_end_index = int(np.flatnonzero(tail)[-1])

    assert metadata["active_taper_out_enabled"] is False
    assert metadata["tail_end_taper_out_enabled"] is True
    assert metadata["active_end_correction_preserved"] is True
    assert np.isclose(frame.loc[active_end_index, "correction_tail_taper_gate"], 1.0)
    assert np.isclose(frame.loc[tail_start_index, "correction_tail_taper_gate"], 1.0)
    assert np.isclose(frame.loc[tail_end_index, "correction_tail_taper_gate"], 0.0)
    assert metadata["tail_end_voltage_zero_status"] == "ok"
    assert abs(float(metadata["second_command_final_voltage_v"])) <= 1e-9


def test_second_modeling_polarity_guard_does_not_create_flat_zero_segment(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_delayed_actual_drive_csv(actual, delay_s=0.12)

    _frame, metadata = generate_second_modeled_voltage_lut(_first_profile(), actual, freq_hz=1.0, cycle_count=1.0)

    assert metadata["polarity_guard_mode"] in {"start_segment_only", "none"}
    assert metadata["polarity_guard_flat_zero_segment_detected"] is False


def test_second_modeling_tail_uses_post_active_measured_field_when_available(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_delayed_actual_drive_csv(actual, delay_s=0.05)

    frame, metadata = generate_second_modeled_voltage_lut(_first_profile(), actual, freq_hz=1.0, cycle_count=1.0)

    tail = frame["tail_window_mask"].astype(bool).to_numpy()
    assert tail.any()
    assert metadata["tail_field_source"] == "measured_post_active"
    assert metadata["tail_extrapolation_used"] is False
    assert np.nanmax(np.abs(frame.loc[tail, "raw_tail_correction_delta_v"])) > 0.0


def test_second_modeling_warns_when_tail_support_is_missing_without_fake_line_to_zero(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_actual_drive_csv(actual)
    lines = actual.read_text(encoding="utf-8").splitlines()
    header_index = next(index for index, line in enumerate(lines) if line.startswith("Row,TimeMs"))
    rows = []
    for line in lines[header_index + 1 :]:
        parts = line.split(",")
        if len(parts) >= 2 and float(parts[1]) <= 1219.0:
            rows.append(line)
    actual.write_text("\n".join([*lines[: header_index + 1], *rows]), encoding="utf-8")

    frame, metadata = generate_second_modeled_voltage_lut(_first_profile(), actual, freq_hz=1.0, cycle_count=1.0)

    assert "second_limited_voltage_v" in frame.columns
    assert metadata["second_modeling_available"] is True
    assert metadata["measured_support_coverage_status"] == "insufficient_for_zero_return"
    assert metadata["measured_tail_fake_line_to_zero_used"] is False
    assert metadata["fake_line_to_zero_fallback_used"] is False
    assert metadata["tail_field_source_status"] == "warning_insufficient_tail_support_no_fake_zero_return"
    assert metadata["second_correction_delta_v_generated"] is True
    assert metadata["second_voltage_v_generated"] is True
    assert metadata["tail_extrapolation_used"] is False


def test_second_modeling_has_no_fake_line_to_zero_source() -> None:
    modeling_source = (SRC_ROOT / "field_analysis" / "finite_second_modeling.py").read_text(encoding="utf-8")
    tail_source = (SRC_ROOT / "field_analysis" / "finite_second_modeling_tail_controller.py").read_text(encoding="utf-8")

    combined = modeling_source + tail_source
    assert "np.linspace(1.0, 0.0" not in combined
    assert "active_end_fallback" not in combined
    assert "fake_line_to_zero_fallback_used" in combined


def test_stabilization_accepts_read_only_raw_delta() -> None:
    raw = np.linspace(-0.2, 0.3, 80, dtype=float)
    raw.setflags(write=False)
    first = np.zeros_like(raw)
    time_s = np.linspace(0.0, 1.0, raw.size)
    active = np.ones(raw.size, dtype=bool)

    delta, metadata, arrays = stabilize_correction_delta(
        raw,
        first,
        time_s,
        active,
        freq_hz=1.0,
        cycle_count=1.0,
        enabled=True,
    )

    assert metadata["correction_stabilization_enabled"] is True
    assert np.isfinite(delta).all()
    assert arrays["smoothed_correction_delta_v"].flags.writeable is True


def test_second_modeling_auto_gain_is_default_and_respects_headroom(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_delayed_actual_drive_csv(actual, delay_s=0.12)
    high_headroom = _first_profile()
    low_headroom = _first_profile()
    low_headroom["limited_voltage_v"] = np.clip(low_headroom["limited_voltage_v"] + 1.8, -4.8, 4.8)

    _frame_high, meta_high = generate_second_modeled_voltage_lut(high_headroom, actual, freq_hz=1.0, cycle_count=1.0)
    _frame_low, meta_low = generate_second_modeled_voltage_lut(low_headroom, actual, freq_hz=1.0, cycle_count=1.0)

    assert meta_high["correction_gain_mode"] == "auto"
    assert 0.05 <= float(meta_high["correction_gain_auto"]) <= 0.50
    assert meta_high["correction_gain_used"] == meta_high["correction_gain_auto"]
    assert meta_low["correction_gain_auto"] <= meta_high["correction_gain_auto"]
    assert "auto_gain_unit_delta_peak_v" in meta_high
    assert "auto_gain_headroom_safe_v" in meta_high
    assert "auto_gain_target_delta_peak_v" in meta_high
    assert "tail_gain_used" in meta_high


def test_second_modeling_manual_gain_mode_is_preserved(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_delayed_actual_drive_csv(actual, delay_s=0.12)

    _frame, metadata = generate_second_modeled_voltage_lut(
        _first_profile(),
        actual,
        freq_hz=1.0,
        cycle_count=1.0,
        correction_gain=0.33,
        correction_gain_mode="manual",
    )

    assert metadata["correction_gain_mode"] == "manual"
    assert np.isclose(metadata["correction_gain_used"], 0.33)
    assert np.isclose(metadata["correction_gain_manual"], 0.33)
