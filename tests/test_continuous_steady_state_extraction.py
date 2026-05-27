from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.continuous_steady_state_extraction import (
    adapt_continuous_source_frame,
    build_continuous_steady_state_modeling_case,
    build_continuous_phase_aligned_command_profile,
    evaluate_cycle_stability,
    extract_steady_state_one_cycle_window,
)


def _continuous_frame(*, freq_hz: float = 2.0, cycles: int = 8, samples_per_cycle: int = 80) -> pd.DataFrame:
    period = 1.0 / freq_hz
    dt = period / samples_per_cycle
    time_s = np.arange(0, cycles * samples_per_cycle, dtype=float) * dt
    phase = 2.0 * np.pi * freq_hz * time_s
    startup_scale = 1.0 - 0.35 * np.exp(-np.arange(len(time_s)) / (1.2 * samples_per_cycle))
    voltage = 3.0 * np.sin(phase)
    effective_field = 50.0 * startup_scale * np.sin(phase - 0.05)
    hallbz_raw = -effective_field + 1.5
    return pd.DataFrame(
        {
            "time_s": time_s,
            "Voltage1_V": voltage,
            "HallBz": hallbz_raw,
        }
    )


def _continuous_frame_with_stop(*, freq_hz: float = 2.0, cycles: int = 8, samples_per_cycle: int = 80, stop_cycles: float = 7.0) -> pd.DataFrame:
    frame = _continuous_frame(freq_hz=freq_hz, cycles=cycles, samples_per_cycle=samples_per_cycle)
    stop_s = stop_cycles / freq_hz
    stopped = frame["time_s"] >= stop_s
    frame.loc[stopped, "Voltage1_V"] = 0.0
    return frame


def test_extract_steady_state_one_cycle_excludes_startup_and_returns_one_cycle_contract() -> None:
    source = _continuous_frame(freq_hz=2.0)

    window, metadata = extract_steady_state_one_cycle_window(
        source,
        waveform_type="sine",
        freq_hz=2.0,
        min_discard_cycles=2,
    )

    assert metadata["modeling_input_mode"] == "continuous_steady_state"
    assert metadata["continuous_production_cycle_count"] == 1.0
    assert metadata["continuous_repeating_lut"] is True
    assert metadata["startup_transient_excluded"] is True
    assert metadata["steady_state_cycle_extraction_used"] is True
    assert metadata["zero_return_tail_enabled"] is False
    assert metadata["selected_cycle_index"] >= 2
    assert metadata["representative_cycle_mode"] == "last_stable_non_terminal_cycle"
    assert metadata["exclude_terminal_cycles"] is True
    assert metadata["terminal_guard_cycle_count"] == 1
    assert metadata["selected_cycle_is_terminal"] is False
    assert metadata["selected_cycle_count"] == 1.0
    assert metadata["selected_cycle_duration_status"] == "ok"
    assert abs(float(metadata["selected_cycle_duration_ratio"]) - 1.0) <= 0.05
    assert metadata["continuous_window_cycle_count"] == 1.0
    assert window["cycle_phase_s"].min() >= 0.0
    assert window["cycle_phase_s"].max() < 0.5
    assert np.isclose(window["time_s"].min(), 0.0)
    assert np.nanmax(np.abs(window["measured_field_normalized_mT"])) <= 50.0 + 1e-9
    assert np.nanmax(np.abs(window["voltage_normalized_v"])) <= 5.0 + 1e-9
    assert np.allclose(window["measured_field_effective_mT"], -window["raw_hallbz_mT"])
    assert {
        "time_s",
        "time_s_abs",
        "cycle_index",
        "cycle_phase_s",
        "raw_hallbz_mT",
        "measured_field_effective_mT",
        "measured_field_baseline_removed_mT",
        "measured_field_normalized_mT",
        "raw_voltage_v",
        "voltage_normalized_v",
        "physical_target_output_mT",
        "normalized_physical_target_output_mT",
        "steady_state_selected_mask",
        "steady_state_cycle_index",
    }.issubset(window.columns)


def test_continuous_extraction_excludes_terminal_stop_influenced_cycle() -> None:
    source = _continuous_frame_with_stop(freq_hz=2.0, cycles=8, stop_cycles=7.0)

    window, metadata = extract_steady_state_one_cycle_window(
        source,
        waveform_type="sine",
        freq_hz=2.0,
        min_discard_cycles=2,
    )

    assert not window.empty
    assert metadata["representative_cycle_mode"] == "last_stable_non_terminal_cycle"
    assert metadata["command_stop_detection_method"] == "voltage_nonzero_window"
    assert metadata["selected_cycle_is_terminal"] is False
    assert metadata["selected_cycle_stop_influence_status"] == "ok"
    assert metadata["selected_cycle_phase_support_clear_of_stop"] is True
    assert metadata["field_support_uses_post_stop_data"] is False
    assert metadata["field_support_end_s"] <= metadata["command_stop_s"]
    assert metadata["selected_cycle_index"] < 6
    assert metadata["stop_influenced_cycle_rejected_count"] >= 1


def test_two_hz_continuous_selected_duration_is_full_one_cycle() -> None:
    window, metadata = extract_steady_state_one_cycle_window(
        _continuous_frame(freq_hz=2.0, cycles=8),
        waveform_type="sine",
        freq_hz=2.0,
        min_discard_cycles=2,
    )

    assert metadata["expected_period_s"] == 0.5
    assert metadata["selected_cycle_duration_status"] == "ok"
    assert abs(float(metadata["selected_cycle_duration_s"]) - 0.5) <= 0.025
    assert 0.95 <= float(metadata["selected_cycle_duration_ratio"]) <= 1.05
    assert window["time_s"].max() <= 0.5


def test_three_hz_continuous_selected_duration_is_full_one_cycle() -> None:
    window, metadata = extract_steady_state_one_cycle_window(
        _continuous_frame(freq_hz=3.0, cycles=8),
        waveform_type="sine",
        freq_hz=3.0,
        min_discard_cycles=2,
    )

    assert metadata["expected_period_s"] == 1.0 / 3.0
    assert metadata["selected_cycle_duration_status"] == "ok"
    assert abs(float(metadata["selected_cycle_duration_s"]) - (1.0 / 3.0)) <= 0.02
    assert 0.95 <= float(metadata["selected_cycle_duration_ratio"]) <= 1.05
    assert window["time_s"].max() <= (1.0 / 3.0)


def test_source_frequency_mismatch_blocks_continuous_extraction() -> None:
    source = _continuous_frame(freq_hz=2.0, cycles=8)
    source.attrs["continuous_source_freq_hz"] = 2.0
    source.attrs["continuous_source_file"] = "continuous_sine_2Hz.csv"

    window, metadata = extract_steady_state_one_cycle_window(
        source,
        waveform_type="sine",
        freq_hz=3.0,
        min_discard_cycles=2,
    )

    assert window.empty
    assert metadata["steady_state_extraction_status"] == "unavailable_frequency_mismatch"
    assert metadata["frequency_match_status"] == "mismatch"
    assert metadata["frequency_mismatch_blocked"] is True
    assert metadata["continuous_source_freq_hz"] == 2.0
    assert metadata["quick_lut_target_freq_hz"] == 3.0


def test_duration_validation_rejects_half_cycle_window() -> None:
    source = _continuous_frame(freq_hz=3.0, cycles=8)

    window, metadata = extract_steady_state_one_cycle_window(
        source,
        waveform_type="sine",
        freq_hz=2.0,
        min_discard_cycles=2,
    )

    assert window.empty
    assert metadata["steady_state_extraction_status"] == "unavailable_invalid_cycle_duration"
    assert metadata["selected_cycle_duration_status"] in {"rejected_half_cycle_window", "rejected_invalid_duration"}
    assert float(metadata["selected_cycle_duration_ratio"]) < 0.75


def test_cycle_stability_metrics_identify_stable_late_cycles() -> None:
    source = _continuous_frame(freq_hz=1.5, cycles=7)

    metrics, metadata = evaluate_cycle_stability(source, freq_hz=1.5)

    assert not metrics.empty
    assert {"cycle_index", "positive_peak_mT", "negative_peak_mT", "peak_drift_pct", "baseline_drift_mT"}.issubset(metrics.columns)
    assert metadata["steady_state_detection_method"] in {"cycle_metrics", "fixed_period_segmentation"}
    late = metrics.loc[metrics["cycle_index"] >= 3, "peak_drift_pct"].dropna()
    assert float(late.tail(3).max()) < 8.0


def test_cycle_boundary_uses_positive_voltage_zero_crossing_when_available() -> None:
    freq_hz = 2.0
    period = 1.0 / freq_hz
    time_s = np.linspace(0.0, 4.0, 800, endpoint=False)
    phase_offset_s = 0.08
    phase = 2.0 * np.pi * freq_hz * (time_s - phase_offset_s)
    frame = pd.DataFrame(
        {
            "time_s": time_s,
            "Voltage1_V": 3.0 * np.sin(phase),
            "HallBz": -(45.0 * np.sin(phase - 0.03)),
        }
    )

    metrics, metadata = evaluate_cycle_stability(frame, freq_hz=freq_hz)

    assert metadata["cycle_boundary_method"] == "voltage_positive_zero_crossing"
    assert metadata["zero_cross_detection_status"] == "ok"
    assert metadata["positive_going_zero_crossing_count"] >= 6
    assert metadata["negative_going_zero_crossing_count"] >= 6
    assert metadata["half_cycle_boundary_rejected"] is True
    assert metadata["fixed_period_fallback_used"] is False
    assert metadata["detected_cycle_count"] >= 6
    first_start = float(metadata["cycle_start_times_s"][0])
    assert abs(first_start - phase_offset_s) < 0.02
    assert abs(float(metrics.iloc[0]["cycle_start_s"]) - phase_offset_s) < 0.02
    assert abs(float(metrics.iloc[1]["cycle_start_s"]) - (phase_offset_s + period)) < 0.02


def test_cycle_boundary_falls_back_to_fixed_period_when_zero_crossing_unavailable() -> None:
    frame = _continuous_frame(freq_hz=1.0, cycles=5)
    frame["Voltage1_V"] = 2.0

    metrics, metadata = evaluate_cycle_stability(frame, freq_hz=1.0)

    assert metadata["cycle_boundary_method"] == "fixed_period"
    assert metadata["zero_cross_detection_status"] == "insufficient_crossings"
    assert metadata["fixed_period_fallback_used"] is True
    assert metadata["detected_cycle_count"] >= 4
    assert not metrics.empty


def test_continuous_schema_adapter_accepts_alias_columns() -> None:
    frame = pd.DataFrame(
        {
            "Time": [0.0, 0.01, 0.02],
            "command_voltage_v": [0.0, 1.0, 0.0],
            "HallZ": [1.0, -2.0, 1.0],
        }
    )

    adapted, metadata = adapt_continuous_source_frame(frame)

    assert metadata["continuous_schema_status"] == "ok"
    assert metadata["continuous_schema_time_column"] == "Time"
    assert metadata["continuous_schema_voltage_column"] == "command_voltage_v"
    assert metadata["continuous_schema_hall_column"] == "HallZ"
    assert {
        "time_s",
        "time_s_abs",
        "raw_hallbz_mT",
        "measured_field_effective_mT",
        "measured_field_baseline_removed_mT",
        "measured_field_normalized_mT",
        "raw_voltage_v",
        "voltage_normalized_v",
    }.issubset(adapted.columns)
    assert np.allclose(adapted["measured_field_effective_mT"], -adapted["raw_hallbz_mT"])


def test_continuous_schema_adapter_accepts_extended_alias_and_normalized_field_only() -> None:
    frame = pd.DataFrame(
        {
            "Time_ms": [0.0, 10.0, 20.0],
            "normalized_actual_drive_voltage_v": [0.0, 2.5, 0.0],
            "normalized_measured_field_mT": [0.0, 40.0, 0.0],
        }
    )

    adapted, metadata = adapt_continuous_source_frame(frame)

    assert metadata["continuous_schema_status"] == "ok"
    assert metadata["continuous_schema_time_column"] == "Time_ms"
    assert metadata["continuous_schema_voltage_column"] == "normalized_actual_drive_voltage_v"
    assert metadata["continuous_schema_hall_or_field_column"] == "normalized_measured_field_mT"
    assert metadata["raw_hallbz_available"] is False
    assert metadata["normalized_field_available"] is True
    assert np.allclose(adapted["time_s_abs"], [0.0, 0.01, 0.02])
    assert adapted["raw_hallbz_mT"].isna().all()
    assert np.allclose(adapted["measured_field_normalized_mT"], [0.0, 40.0, 0.0])
    assert np.allclose(adapted["voltage_normalized_v"], [0.0, 2.5, 0.0])


def test_continuous_schema_adapter_rejects_final_voltage_lut() -> None:
    frame = pd.DataFrame({"sample_index": [0, 1], "time_s": [0.0, 0.1], "voltage_v": [0.0, 1.0]})

    try:
        adapt_continuous_source_frame(frame)
    except ValueError as exc:
        assert "final_voltage_lut_not_measured_input" in str(exc)
    else:
        raise AssertionError("final LUT schema must be rejected as continuous measured input")


def test_continuous_phase_aligned_command_profile_uses_shared_kernel_without_tail() -> None:
    case = build_continuous_steady_state_modeling_case(
        _continuous_frame(freq_hz=2.0),
        waveform_type="sine",
        freq_hz=2.0,
        min_discard_cycles=2,
    )

    command, metadata = build_continuous_phase_aligned_command_profile(
        case["steady_state_one_cycle_frame"],
        support_frame=case["steady_state_support_frame"],
        freq_hz=2.0,
        waveform_type="sine",
    )

    assert metadata["continuous_first_modeling_uses_phase_aligned_kernel"] is True
    assert metadata["continuous_modeling_kernel_source"] == "finite_second_modeling_shared_kernel"
    assert metadata["continuous_first_modeling_tail_disabled"] is True
    assert metadata["continuous_first_modeling_cycle_count"] == 1.0
    assert metadata["continuous_loop_output"] is True
    assert metadata["loop_endpoint_policy"] == "period_exclusive"
    assert metadata["continuous_base_voltage_source"] == "raw_input_voltage_scaled_by_field_peak"
    assert metadata["continuous_input_voltage_field_scale_source"] == "extraction_field_normalization_scale_to_target"
    assert metadata["continuous_source_voltage_column"] == "raw_voltage_v"
    assert metadata["continuous_base_voltage_peak_v"] == pytest.approx(
        metadata["source_voltage_raw_peak_v"] * metadata["continuous_input_voltage_field_scale"],
        rel=0.02,
    )
    assert metadata["continuous_final_voltage_limit_v"] == 5.0
    assert metadata["source_voltage_base_normalized_peak_v"] == pytest.approx(metadata["continuous_base_voltage_peak_v"], rel=0.02)
    assert "continuous_clipping_fraction" in metadata
    assert {
        "first_modeled_voltage_v",
        "limited_voltage_v",
        "correction_delta_v",
        "measured_field_smoothed_mT",
        "measured_field_aligned_mT",
        "residual_for_modeling_mT",
    }.issubset(command.columns)
    assert "tail_voltage_v" not in command.columns
    assert command["time_s"].min() == 0.0
    assert command["time_s"].max() < 0.5


def test_continuous_first_modeling_prefers_raw_voltage_over_legacy_normalized_voltage() -> None:
    case = build_continuous_steady_state_modeling_case(
        _continuous_frame(freq_hz=2.0),
        waveform_type="sine",
        freq_hz=2.0,
        min_discard_cycles=2,
    )
    window = case["steady_state_one_cycle_frame"].copy()
    window["voltage_normalized_v"] = 5.0 * np.sin(2.0 * np.pi * window["time_s"] / 0.5)

    command, metadata = build_continuous_phase_aligned_command_profile(
        window,
        support_frame=case["steady_state_support_frame"],
        freq_hz=2.0,
        waveform_type="sine",
        base_voltage_peak_v=2.5,
    )

    assert metadata["continuous_source_voltage_column"] == "raw_voltage_v"
    assert np.nanmax(np.abs(command["source_voltage_v"])) == pytest.approx(3.0, rel=0.02)
    assert np.nanmax(np.abs(command["base_voltage_v"])) == pytest.approx(
        3.0 * metadata["continuous_input_voltage_field_scale"],
        rel=0.03,
    )
    assert np.nanmax(np.abs(command["limited_voltage_v"])) <= 5.0 + 1e-9
    assert metadata["source_voltage_raw_peak_v"] == pytest.approx(3.0, rel=0.02)
    assert metadata["source_voltage_base_normalized_peak_v"] == pytest.approx(np.nanmax(np.abs(command["base_voltage_v"])), rel=0.02)


def test_continuous_first_modeling_reports_clipping_warning_for_manual_high_gain() -> None:
    case = build_continuous_steady_state_modeling_case(
        _continuous_frame(freq_hz=2.0),
        waveform_type="sine",
        freq_hz=2.0,
        min_discard_cycles=2,
    )
    window = case["steady_state_one_cycle_frame"].copy()
    window["normalized_physical_target_output_mT"] = 50.0
    window["measured_field_normalized_mT"] = -50.0

    _command, metadata = build_continuous_phase_aligned_command_profile(
        window,
        support_frame=case["steady_state_support_frame"],
        freq_hz=2.0,
        waveform_type="sine",
        base_voltage_peak_v=5.0,
        correction_gain_mode="manual",
        correction_gain=1.0,
    )

    assert metadata["continuous_voltage_clip_sample_count"] > 0
    assert metadata["continuous_voltage_clip_fraction"] > 0.0
    assert metadata["continuous_voltage_clip_status"] in {"warning", "severe"}
    assert metadata["continuous_clipping_warning"]


def test_continuous_support_window_extends_to_field_cycle_end_for_phase_delay() -> None:
    freq_hz = 2.0
    period = 1.0 / freq_hz
    delay_s = 0.08
    time_s = np.linspace(0.0, period * 8 + delay_s + 0.1, 900, endpoint=False)
    frame = pd.DataFrame(
        {
            "time_s": time_s,
            "Voltage1_V": 3.0 * np.sin(2.0 * np.pi * freq_hz * time_s),
            "HallBz": -(45.0 * np.sin(2.0 * np.pi * freq_hz * (time_s - delay_s))),
        }
    )

    case = build_continuous_steady_state_modeling_case(frame, waveform_type="sine", freq_hz=freq_hz)
    metadata = case["metadata"]

    assert metadata["phase_support_status"] == "ok"
    assert metadata["field_support_end_s"] > metadata["voltage_model_end_s"]
    assert metadata["estimated_phase_delay_s"] > 0.04
    assert isinstance(case["steady_state_support_frame"], pd.DataFrame)
    assert not case["steady_state_support_frame"].empty


def test_continuous_phase_alignment_uses_voltage_peak_reference() -> None:
    freq_hz = 2.0
    period = 1.0 / freq_hz
    delay_s = 0.06
    time_s = np.linspace(0.0, period * 8 + delay_s + 0.1, 1000, endpoint=False)
    frame = pd.DataFrame(
        {
            "time_s": time_s,
            "Voltage1_V": 4.0 * np.sin(2.0 * np.pi * freq_hz * time_s),
            "HallBz": -(42.0 * np.sin(2.0 * np.pi * freq_hz * (time_s - delay_s))),
        }
    )

    case = build_continuous_steady_state_modeling_case(frame, waveform_type="sine", freq_hz=freq_hz)
    command, metadata = build_continuous_phase_aligned_command_profile(
        case["steady_state_one_cycle_frame"],
        support_frame=case["steady_state_support_frame"],
        freq_hz=freq_hz,
        waveform_type="sine",
    )

    assert metadata["continuous_phase_alignment_method"] == "field_peak_to_voltage_peak"
    assert metadata["continuous_first_modeling_phase_reference"] == "voltage_peak"
    assert abs(float(metadata["continuous_phase_delay_s"]) - delay_s) < 0.03
    assert abs(float(metadata["measured_aligned_first_peak_time_s"]) - float(metadata["voltage_first_peak_time_s"])) < 0.03
    assert command["measured_field_aligned_mT"].notna().all()


def test_continuous_phase_alignment_requires_support_beyond_output_cycle() -> None:
    freq_hz = 2.0
    period = 1.0 / freq_hz
    delay_s = 0.08
    output_time = np.linspace(0.0, period, 200, endpoint=False)
    window = pd.DataFrame(
        {
            "time_s": output_time,
            "raw_voltage_v": 4.0 * np.sin(2.0 * np.pi * freq_hz * output_time),
            "voltage_normalized_v": 4.0 * np.sin(2.0 * np.pi * freq_hz * output_time),
            "measured_field_normalized_mT": 45.0 * np.sin(2.0 * np.pi * freq_hz * (output_time - delay_s)),
            "normalized_physical_target_output_mT": 50.0 * np.sin(2.0 * np.pi * freq_hz * output_time),
        }
    )
    support_time = np.linspace(0.0, period + delay_s + 0.03, 260, endpoint=False)
    support = pd.DataFrame(
        {
            "time_s": support_time,
            "raw_voltage_v": 4.0 * np.sin(2.0 * np.pi * freq_hz * support_time),
            "voltage_normalized_v": 4.0 * np.sin(2.0 * np.pi * freq_hz * support_time),
            "measured_field_normalized_mT": 45.0 * np.sin(2.0 * np.pi * freq_hz * (support_time - delay_s)),
        }
    )

    command, metadata = build_continuous_phase_aligned_command_profile(
        window,
        support_frame=support,
        freq_hz=freq_hz,
        waveform_type="sine",
    )

    assert metadata["measurement_support_grid_separate_from_output_grid"] is True
    assert metadata["aligned_measured_support_status"] == "ok"
    assert metadata["aligned_measured_finite_ratio"] == 1.0
    assert metadata["continuous_nan_to_zero_used"] is False
    assert command["measured_field_aligned_mT"].notna().all()
    assert command["residual_for_modeling_mT"].notna().all()


def test_continuous_phase_alignment_blocks_when_support_tail_missing() -> None:
    freq_hz = 2.0
    period = 1.0 / freq_hz
    delay_s = 0.08
    output_time = np.linspace(0.0, period, 200, endpoint=False)
    window = pd.DataFrame(
        {
            "time_s": output_time,
            "raw_voltage_v": 4.0 * np.sin(2.0 * np.pi * freq_hz * output_time),
            "voltage_normalized_v": 4.0 * np.sin(2.0 * np.pi * freq_hz * output_time),
            "measured_field_normalized_mT": 45.0 * np.sin(2.0 * np.pi * freq_hz * (output_time - delay_s)),
            "normalized_physical_target_output_mT": 50.0 * np.sin(2.0 * np.pi * freq_hz * output_time),
        }
    )

    command, metadata = build_continuous_phase_aligned_command_profile(
        window,
        support_frame=window,
        freq_hz=freq_hz,
        waveform_type="sine",
    )

    assert command.empty
    assert metadata["continuous_first_modeling_status"] == "unavailable_phase_support_incomplete"
    assert metadata["continuous_phase_support_incomplete_blocked"] is True
    assert metadata["continuous_nan_to_zero_used"] is False


def test_continuous_modeling_case_is_finite_like_one_cycle_and_loop_safe() -> None:
    case = build_continuous_steady_state_modeling_case(
        _continuous_frame(freq_hz=2.0),
        waveform_type="sine",
        freq_hz=2.0,
        min_discard_cycles=2,
    )

    frame = case["steady_state_one_cycle_frame"]
    metadata = case["metadata"]

    assert metadata["finite_like_window_from_continuous"] is True
    assert metadata["continuous_loop_output"] is True
    assert metadata["continuous_export_cycle_count"] == 1.0
    assert metadata["loop_endpoint_policy"] == "period_exclusive"
    assert frame["time_s"].iloc[0] == 0.0
    assert frame["time_s"].iloc[-1] < 0.5
    assert metadata["continuous_steady_state_window_support_status"] == "ok"
    assert metadata["continuous_target_shape"] == "fixed_rounded_triangle"
    assert metadata["continuous_target_cycle_count"] == 1.0
