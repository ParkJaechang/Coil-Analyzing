from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.compensation import _startup_metrics_worsened, synthesize_current_waveform_compensation
from field_analysis.models import (
    CycleDetectionResult,
    DatasetAnalysis,
    ParsedMeasurement,
    PreprocessResult,
    SheetPreview,
)

TEST_ROOT = Path(__file__).resolve().parent
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

import test_finite_empirical_field_route as finite_fixture


def _scaled_waveform(values: np.ndarray, target_pp: float) -> np.ndarray:
    centered = values - float(np.mean(values))
    peak_to_peak = float(np.max(centered) - np.min(centered))
    if peak_to_peak <= 1e-12:
        return centered
    return centered * float(target_pp) / peak_to_peak


def _build_startup_offset_analysis(
    *,
    test_id: str = "startup_5hz",
    freq_hz: float = 5.0,
    first_cycle_field_offset_mT: float = 16.0,
    cycles: int = 6,
) -> DatasetAnalysis:
    points_per_cycle = 128
    cycle_progress = np.linspace(0.0, 1.0, points_per_cycle)
    radians = 2.0 * np.pi * cycle_progress
    voltage = _scaled_waveform(np.sin(radians), 10.0)
    current = _scaled_waveform(np.sin(radians - np.deg2rad(10.0)), 8.0)
    base_field = _scaled_waveform(np.sin(radians + np.deg2rad(20.0)), 80.0)
    period_s = 1.0 / float(freq_hz)

    annotated_rows: list[dict[str, float | int]] = []
    cycle_rows: list[dict[str, float | int]] = []
    for cycle_index in range(cycles):
        offset = first_cycle_field_offset_mT if cycle_index == 0 else 0.0
        for progress, command_v, current_a, field_mt in zip(cycle_progress, voltage, current, base_field, strict=False):
            annotated_rows.append(
                {
                    "cycle_index": cycle_index,
                    "cycle_progress": float(progress),
                    "cycle_time_s": float(progress * period_s),
                    "freq_hz": float(freq_hz),
                    "daq_input_v": float(command_v),
                    "i_sum_signed": float(current_a),
                    "bz_mT": float(field_mt + offset),
                }
            )
        cycle_rows.append(
            {
                "cycle_index": cycle_index,
                "achieved_current_pp_a": 8.0,
                "achieved_bz_mT_pp": 80.0,
                "daq_input_v_pp": 10.0,
            }
        )

    annotated_frame = pd.DataFrame(annotated_rows)
    per_cycle_summary = pd.DataFrame(cycle_rows)
    per_test_summary = pd.DataFrame(
        [
            {
                "test_id": test_id,
                "waveform_type": "sine",
                "freq_hz": float(freq_hz),
                "current_pp_target_a": 8.0,
                "achieved_current_pp_a_mean": 8.0,
                "daq_input_v_pp_mean": 10.0,
                "amp_gain_setting_mean": 100.0,
                "achieved_bz_mT_pp_mean": 80.0,
            }
        ]
    )
    preview = SheetPreview(
        sheet_name="Sheet1",
        row_count=len(annotated_frame),
        column_count=len(annotated_frame.columns),
        columns=list(annotated_frame.columns),
        header_row_index=0,
        metadata={},
        preview_rows=[],
        recommended_mapping={},
    )
    parsed = ParsedMeasurement(
        source_file=f"{test_id}.csv",
        file_type="csv",
        sheet_name="Sheet1",
        structure_preview=preview,
        metadata={"test_id": test_id},
        mapping={},
        raw_frame=annotated_frame.copy(),
        normalized_frame=annotated_frame.copy(),
    )
    preprocess = PreprocessResult(corrected_frame=annotated_frame.copy(), offsets={}, lags=[])
    cycle_detection = CycleDetectionResult(
        annotated_frame=annotated_frame,
        boundaries=[],
        estimated_period_s=period_s,
        estimated_frequency_hz=float(freq_hz),
        reference_channel="i_sum_signed",
    )
    return DatasetAnalysis(
        parsed=parsed,
        preprocess=preprocess,
        cycle_detection=cycle_detection,
        per_cycle_summary=per_cycle_summary,
        per_test_summary=per_test_summary,
    )


def _run_compensation(
    *,
    analysis: DatasetAnalysis | None = None,
    finite_support_entries: list[dict[str, object]] | None = None,
    finite_cycle_mode: bool = False,
    target_cycle_count: float | None = None,
    freq_hz: float = 5.0,
) -> dict[str, object]:
    analysis = analysis or _build_startup_offset_analysis(freq_hz=freq_hz)
    result = synthesize_current_waveform_compensation(
        per_test_summary=analysis.per_test_summary,
        analyses_by_test_id={str(analysis.per_test_summary["test_id"].iloc[0]): analysis},
        waveform_type="sine",
        freq_hz=freq_hz,
        target_current_pp_a=45.0,
        target_output_type="field",
        target_output_pp=45.0,
        finite_cycle_mode=finite_cycle_mode,
        target_cycle_count=target_cycle_count,
        preview_tail_cycles=0.25,
        points_per_cycle=128,
        max_daq_voltage_pp=1000.0,
        amp_gain_at_100_pct=1.0,
        amp_gain_limit_pct=100.0,
        amp_max_output_pk_v=1000.0,
        finite_support_entries=finite_support_entries,
    )
    assert result is not None
    return result


def _early_late_mean_delta(profile: pd.DataFrame, column: str, *, period_s: float) -> float:
    time_s = pd.to_numeric(profile["time_s"], errors="coerce")
    values = pd.to_numeric(profile[column], errors="coerce")
    early = values[(time_s >= 0.0) & (time_s <= period_s * 0.35)].mean()
    late = values[(time_s >= period_s * 0.75) & (time_s <= period_s * 1.25)].mean()
    return float(early - late)


def _early_late_residual_delta(profile: pd.DataFrame, column: str, *, period_s: float) -> float:
    time_s = pd.to_numeric(profile["time_s"], errors="coerce")
    values = pd.to_numeric(profile[column], errors="coerce")
    target = pd.to_numeric(profile["physical_target_output_mT"], errors="coerce")
    residual = values - target
    early = residual[(time_s >= 0.0) & (time_s <= period_s * 0.35)].mean()
    late = residual[(time_s >= period_s * 0.75) & (time_s <= period_s * 1.25)].mean()
    return float(early - late)


def _with_unstable_start(frame: pd.DataFrame) -> pd.DataFrame:
    mutated = frame.copy()
    mutated.loc[mutated.index[:8], "bz_mT"] = [40.0, -40.0] * 4
    return mutated


def test_continuous_startup_component_is_separated_and_compensated() -> None:
    analysis = _build_startup_offset_analysis(first_cycle_field_offset_mT=18.0, freq_hz=5.0)
    result = _run_compensation(analysis=analysis, freq_hz=5.0)
    profile = result["command_profile"]

    assert result["startup_transient_applied"] is True
    assert result["startup_source_type"] == "continuous_early_cycles"
    assert result["startup_source_file"] == "startup_5hz.csv"
    assert result["startup_transient_source"] == "continuous_early_cycles"
    assert result["startup_transient_status"] == "ok"
    assert result["startup_rejected_reason"] in (None, [])
    assert result["startup_data_quality_ok"] is True
    assert result["startup_source_waveform_family"] == "sine"
    assert result["startup_cycle_count_used"] == 1
    assert result["steady_cycle_count_used"] >= 3
    assert result["startup_initial_field_offset_mT"] > 0.0
    assert result["startup_bias_mT"] > 0.0
    assert result["startup_residual_pp_mT"] > 0.0
    assert result["startup_residual_rms_mT"] > 0.0
    assert result["startup_source_freq_hz"] == 5.0
    assert result["startup_target_freq_hz"] == 5.0
    assert result["startup_frequency_distance_hz"] == 0.0
    assert result["startup_frequency_fallback_used"] is False
    assert np.isfinite(float(result["early_cycle_residual_before"]))
    assert np.isfinite(float(result["early_cycle_residual_after"]))
    assert np.isfinite(float(result["active_nrmse_before"]))
    assert np.isfinite(float(result["active_nrmse_after"]))
    assert np.isfinite(float(result["active_shape_corr_before"]))
    assert np.isfinite(float(result["active_shape_corr_after"]))
    assert float(result["active_nrmse_after"]) <= float(result["active_nrmse_before"]) + 0.05
    assert float(result["active_shape_corr_after"]) >= float(result["active_shape_corr_before"]) - 0.05
    for column in (
        "open_loop_predicted_field_mT",
        "startup_transient_component_mT",
        "compensated_predicted_field_mT",
        "baseline_recommended_voltage_v",
        "compensated_recommended_voltage_v",
        "startup_compensation_command_delta_v",
        "recommended_voltage_v",
    ):
        assert column in profile.columns
    assert bool(profile["physical_target_output_mT"].equals(profile["target_field_mT"])) is True
    startup_component = pd.to_numeric(profile["startup_transient_component_mT"], errors="coerce")
    assert float(startup_component.abs().max()) > 10.0
    open_loop_residual = abs(float(profile["open_loop_predicted_field_mT"].iloc[0] - profile["physical_target_output_mT"].iloc[0]))
    compensated_residual = abs(float(profile["compensated_predicted_field_mT"].iloc[0] - profile["physical_target_output_mT"].iloc[0]))
    assert compensated_residual < open_loop_residual
    assert abs(float(result["startup_residual_after_mT"])) < abs(float(result["startup_residual_before_mT"]))
    assert result["startup_compensation_applied"] is True
    assert result["startup_compensation_reject_reason"] in (None, [])
    assert result["voltage_limit_respected"] is True
    assert result["startup_compensated_prediction_source"] == "compensated_command_forward_approximation"
    assert np.isfinite(float(result["command_smoothness_before"]))
    assert np.isfinite(float(result["command_smoothness_after"]))
    assert float(result["command_smoothness_after"]) <= float(result["command_smoothness_before"]) + 0.05
    assert "terminal_peak_error_before_mT" in result
    assert "terminal_peak_error_after_mT" in result
    assert "tail_residual_before" in result
    assert "tail_residual_after" in result
    baseline_command = pd.to_numeric(profile["baseline_recommended_voltage_v"], errors="coerce")
    compensated_command = pd.to_numeric(profile["compensated_recommended_voltage_v"], errors="coerce")
    command_delta = pd.to_numeric(profile["startup_compensation_command_delta_v"], errors="coerce")
    assert float(command_delta.abs().max()) > 1e-9
    assert not np.allclose(baseline_command, compensated_command)
    assert np.allclose(profile["recommended_voltage_v"], compensated_command)
    assert bool(profile["startup_candidate_forward_prediction_available"].all()) is True
    assert result["forward_prediction_source"] == "compensated_recommended_voltage_v"
    assert result["plotted_command_source"] == "compensated_recommended_voltage_v"
    assert result["plotted_predicted_source"] == "compensated_predicted_field_mT"
    assert result["forward_prediction_available"] is True
    assert result["predicted_from_plotted_command"] is True
    assert result["displayed_predicted_valid"] is True
    assert result["command_prediction_consistency_status"] == "ok"
    assert np.allclose(profile["displayed_predicted_field_mT"], profile["compensated_predicted_field_mT"], equal_nan=True)


def test_finite_startup_component_is_separated_and_compensated_without_target_stretch() -> None:
    entry = finite_fixture._build_finite_entry(
        test_id="finite_startup_exact",
        waveform_type="sine",
        freq_hz=5.0,
        cycle_count=1.5,
        field_pp=100.0,
    )
    frame = entry["frame"].copy()
    first_cycle_mask = frame["time_s"] <= 0.2 + 1e-12
    frame.loc[first_cycle_mask, "bz_mT"] = frame.loc[first_cycle_mask, "bz_mT"] + 20.0
    entry["frame"] = frame

    result = _run_compensation(
        finite_support_entries=[entry],
        finite_cycle_mode=True,
        target_cycle_count=1.5,
        freq_hz=5.0,
    )
    profile = result["command_profile"]

    assert result["finite_route_mode"] == "finite_empirical_field_support"
    assert result["startup_transient_applied"] is True
    assert result["startup_source_type"] == "finite_support"
    assert result["startup_source_file"] == "finite_startup_exact.csv"
    assert result["startup_transient_source"] == "finite_support"
    assert result["startup_transient_status"] == "ok"
    assert result["startup_rejected_reason"] in (None, [])
    assert result["startup_data_quality_ok"] is True
    assert result["startup_source_waveform_family"] == "sine"
    assert result["startup_source_cycle_count"] == 1.5
    assert result["startup_initial_field_offset_mT"] > 0.0
    assert np.isfinite(float(result["early_cycle_residual_before"]))
    assert np.isfinite(float(result["early_cycle_residual_after"]))
    assert np.isfinite(float(result["active_nrmse_before"]))
    assert np.isfinite(float(result["active_nrmse_after"]))
    assert np.isfinite(float(result["active_shape_corr_before"]))
    assert np.isfinite(float(result["active_shape_corr_after"]))
    assert np.isclose(float(result["target_active_end_s"]), 0.3, atol=0.002)
    for column in (
        "open_loop_predicted_field_mT",
        "startup_transient_component_mT",
        "compensated_predicted_field_mT",
        "baseline_recommended_voltage_v",
        "compensated_recommended_voltage_v",
        "startup_compensation_command_delta_v",
        "recommended_voltage_v",
    ):
        assert column in profile.columns
    assert bool(profile["physical_target_output_mT"].equals(profile["target_field_mT"])) is True
    assert _early_late_residual_delta(profile, "open_loop_predicted_field_mT", period_s=0.2) > 2.0
    assert abs(_early_late_residual_delta(profile, "compensated_predicted_field_mT", period_s=0.2)) < abs(
        _early_late_residual_delta(profile, "open_loop_predicted_field_mT", period_s=0.2)
    )
    assert abs(float(result["startup_residual_after_mT"])) < abs(float(result["startup_residual_before_mT"]))
    assert result["support_reference_trace_status"] == "ok"
    assert result["support_reference_plotted_column"] == "support_reference_output_mT"
    assert result["support_reference_source_label"] == "selected_support_trace"
    assert result["support_reference_selected_support_id"] == "finite_startup_exact"
    assert result["support_reference_used_for_command"] is False
    assert result["command_generation_target"] == "physical_target"
    assert result["forward_prediction_source"] == "compensated_recommended_voltage_v"
    assert result["plotted_command_source"] == "compensated_recommended_voltage_v"
    assert result["plotted_predicted_source"] == "compensated_predicted_field_mT"
    assert result["forward_prediction_available"] is True
    assert result["predicted_from_plotted_command"] is True
    assert result["displayed_predicted_valid"] is True
    support_reference = pd.to_numeric(profile["support_reference_output_mT"], errors="coerce")
    support_scaled = pd.to_numeric(profile["support_scaled_field_mT"], errors="coerce")
    open_loop = pd.to_numeric(profile["open_loop_predicted_field_mT"], errors="coerce")
    compensated = pd.to_numeric(profile["compensated_predicted_field_mT"], errors="coerce")
    assert np.allclose(support_reference, support_scaled, equal_nan=True)
    assert not np.allclose(support_reference, open_loop, equal_nan=True)
    assert not np.allclose(support_reference, compensated, equal_nan=True)
    baseline_command = pd.to_numeric(profile["baseline_recommended_voltage_v"], errors="coerce")
    compensated_command = pd.to_numeric(profile["compensated_recommended_voltage_v"], errors="coerce")
    assert not np.allclose(baseline_command, compensated_command)
    assert np.allclose(profile["recommended_voltage_v"], compensated_command)
    assert result["startup_compensation_applied"] is True
    assert result["startup_compensated_prediction_source"] == "compensated_command_forward_approximation"
    assert abs(float(result["terminal_peak_error_after_mT"])) <= abs(float(result["terminal_peak_error_before_mT"])) + 1e-9
    assert float(result["tail_residual_after"]) <= float(result["tail_residual_before"]) + 1e-9
    assert bool(profile["startup_candidate_forward_prediction_available"].all()) is True
    assert float(result["predicted_jump_ratio"]) <= 0.20


def test_nonzero_start_support_is_rejected_as_startup_source() -> None:
    entry = finite_fixture._build_finite_entry(
        test_id="finite_nonzero_start",
        waveform_type="sine",
        freq_hz=5.0,
        cycle_count=1.5,
        field_pp=100.0,
    )
    frame = entry["frame"].copy()
    frame.loc[0, "daq_input_v"] = 6.0
    frame.loc[0, "bz_mT"] = 25.0
    entry["frame"] = frame

    result = _run_compensation(
        finite_support_entries=[entry],
        finite_cycle_mode=True,
        target_cycle_count=1.5,
        freq_hz=5.0,
    )

    assert result["startup_transient_applied"] is False
    assert result["startup_source_type"] == "finite_support"
    assert result["startup_source_file"] == "finite_nonzero_start.csv"
    assert result["startup_transient_status"] == "source_nonzero_start"
    rejected_reason = result["startup_rejected_reason"]
    rejected_reasons = rejected_reason if isinstance(rejected_reason, list) else [rejected_reason]
    assert "source_nonzero_start" in rejected_reasons
    assert result["startup_data_quality_ok"] is False
    assert result["startup_transient_source"] == "finite_support"


def test_startup_source_with_quality_gate_violations_is_rejected() -> None:
    cases = [
        ("missing_prebaseline", lambda frame: frame.iloc[4:].reset_index(drop=True), "missing_prebaseline"),
        (
            "source_spike_detected",
            lambda frame: frame.assign(bz_mT=frame["bz_mT"].mask(frame.index == 20, 1000.0)),
            "source_spike_detected",
        ),
        (
            "truncated_active_window",
            lambda frame: frame.iloc[:60].reset_index(drop=True),
            "truncated_active_window",
        ),
        (
            "non_monotonic_time",
            lambda frame: frame.assign(time_s=frame["time_s"].mask(frame.index == 10, frame["time_s"].iloc[9] - 0.01)),
            "non_monotonic_time",
        ),
        (
            "duplicated_timestamp",
            lambda frame: frame.assign(time_s=frame["time_s"].mask(frame.index == 10, frame["time_s"].iloc[9])),
            "duplicated_timestamp",
        ),
        (
            "severe_sampling_irregularity",
            lambda frame: frame.assign(time_s=frame["time_s"].mask(frame.index >= 80, frame["time_s"] + 0.2)),
            "severe_sampling_irregularity",
        ),
        (
            "missing_key_columns",
            lambda frame: frame.drop(columns=["bz_mT"]),
            "missing_key_columns",
        ),
        (
            "clipping",
            lambda frame: frame.assign(daq_input_v=frame["daq_input_v"].mask(frame.index.to_series().between(30, 36), 5.0)),
            "clipping",
        ),
        (
            "unstable_baseline",
            _with_unstable_start,
            "unstable_baseline",
        ),
        (
            "source_coverage_insufficient",
            lambda frame: frame[frame["time_s"] <= 0.1].reset_index(drop=True),
            "source_coverage_insufficient",
        ),
    ]

    for test_id, mutate_frame, expected_reason in cases:
        entry = finite_fixture._build_finite_entry(
            test_id=test_id,
            waveform_type="sine",
            freq_hz=5.0,
            cycle_count=1.5,
            field_pp=100.0,
        )
        entry["frame"] = mutate_frame(entry["frame"].copy())

        result = _run_compensation(
            finite_support_entries=[entry],
            finite_cycle_mode=True,
            target_cycle_count=1.5,
            freq_hz=5.0,
        )

        assert result["startup_transient_applied"] is False, test_id
        assert result["startup_data_quality_ok"] is False, test_id
        assert result["startup_source_type"] == "finite_support"
        rejected_reason = result["startup_rejected_reason"]
        rejected_reasons = rejected_reason if isinstance(rejected_reason, list) else [rejected_reason]
        assert expected_reason in rejected_reasons, test_id


def test_startup_metric_gate_rejects_active_shape_regression() -> None:
    before = {"active_nrmse": 0.10, "active_shape_corr": 0.98}

    assert _startup_metrics_worsened(before, {"active_nrmse": 0.20, "active_shape_corr": 0.98}) is True
    assert _startup_metrics_worsened(before, {"active_nrmse": 0.10, "active_shape_corr": 0.85}) is True
    assert _startup_metrics_worsened(before, {"active_nrmse": 0.11, "active_shape_corr": 0.97}) is False


def test_one_point_seven_five_without_exact_support_remains_unavailable() -> None:
    result = _run_compensation(
        finite_support_entries=[
            finite_fixture._build_finite_entry(
                test_id="finite_1p5_only",
                waveform_type="sine",
                freq_hz=5.0,
                cycle_count=1.5,
                field_pp=100.0,
            )
        ],
        finite_cycle_mode=True,
        target_cycle_count=1.75,
        freq_hz=5.0,
    )

    assert result["finite_route_mode"] == "finite_unavailable_no_exact_1_75_support"
    assert result["finite_prediction_available"] is False
    assert result["exact_cycle_support_used"] is False
