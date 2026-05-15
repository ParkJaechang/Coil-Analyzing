from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

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
    assert metadata["selected_cycle_count"] == 1.0
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
    assert adapted.columns.tolist() == ["time_s_abs", "raw_hallbz_mT", "measured_field_effective_mT", "raw_voltage_v"]
    assert np.allclose(adapted["measured_field_effective_mT"], -adapted["raw_hallbz_mT"])


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
        freq_hz=2.0,
        waveform_type="sine",
    )

    assert metadata["continuous_first_modeling_uses_phase_aligned_kernel"] is True
    assert metadata["continuous_modeling_kernel_source"] == "finite_second_modeling_shared_kernel"
    assert metadata["continuous_first_modeling_tail_disabled"] is True
    assert metadata["continuous_first_modeling_cycle_count"] == 1.0
    assert metadata["continuous_loop_output"] is True
    assert metadata["loop_endpoint_policy"] == "period_exclusive"
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
