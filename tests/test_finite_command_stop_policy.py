from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.compensation import synthesize_current_waveform_compensation
from field_analysis.models import (
    CycleDetectionResult,
    DatasetAnalysis,
    ParsedMeasurement,
    PreprocessResult,
    SheetPreview,
)


def _scaled_waveform(values: np.ndarray, target_pp: float) -> np.ndarray:
    centered = values - float(np.mean(values))
    peak_to_peak = float(np.max(centered) - np.min(centered))
    if peak_to_peak <= 1e-12:
        return centered
    return centered * float(target_pp) / peak_to_peak


def _build_analysis(
    *,
    test_id: str,
    freq_hz: float,
    current_pp: float,
    field_pp: float,
    field_phase_deg: float,
) -> DatasetAnalysis:
    cycle_progress = np.linspace(0.0, 1.0, 128)
    radians = 2.0 * np.pi * cycle_progress
    voltage = _scaled_waveform(np.sin(radians), 10.0)
    current = _scaled_waveform(np.sin(radians - np.deg2rad(10.0)), current_pp)
    field = _scaled_waveform(np.sin(radians + np.deg2rad(field_phase_deg)) + 0.18 * np.sin(3.0 * radians), field_pp)

    period_s = 1.0 / float(freq_hz)
    annotated_rows: list[dict[str, float | int]] = []
    cycle_rows: list[dict[str, float | int]] = []
    for cycle_index in range(4):
        for progress, command_v, current_a, field_mt in zip(cycle_progress, voltage, current, field, strict=False):
            annotated_rows.append(
                {
                    "cycle_index": cycle_index,
                    "cycle_progress": float(progress),
                    "cycle_time_s": float(progress * period_s),
                    "freq_hz": float(freq_hz),
                    "daq_input_v": float(command_v),
                    "i_sum_signed": float(current_a),
                    "bz_mT": float(field_mt),
                }
            )
        cycle_rows.append(
            {
                "cycle_index": cycle_index,
                "achieved_current_pp_a": float(current_pp),
                "achieved_bz_mT_pp": float(field_pp),
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
                "current_pp_target_a": float(current_pp),
                "achieved_current_pp_a_mean": float(current_pp),
                "daq_input_v_pp_mean": 10.0,
                "amp_gain_setting_mean": 100.0,
                "achieved_bz_mT_pp_mean": float(field_pp),
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


def _build_support_context() -> tuple[pd.DataFrame, dict[str, DatasetAnalysis]]:
    low = _build_analysis(
        test_id="lagged_2hz",
        freq_hz=2.0,
        current_pp=6.0,
        field_pp=24.0,
        field_phase_deg=-32.0,
    )
    high = _build_analysis(
        test_id="lagged_4hz",
        freq_hz=4.0,
        current_pp=16.0,
        field_pp=68.0,
        field_phase_deg=-48.0,
    )
    summary = pd.concat([low.per_test_summary, high.per_test_summary], ignore_index=True)
    return summary, {"lagged_2hz": low, "lagged_4hz": high}


def _run_fallback_finite_compensation() -> dict[str, object]:
    summary, analyses = _build_support_context()
    result = synthesize_current_waveform_compensation(
        per_test_summary=summary,
        analyses_by_test_id=analyses,
        waveform_type="sine",
        freq_hz=3.0,
        target_current_pp_a=45.0,
        target_output_type="field",
        target_output_pp=45.0,
        finite_cycle_mode=True,
        target_cycle_count=1.5,
        preview_tail_cycles=0.5,
        max_daq_voltage_pp=1000.0,
        amp_gain_at_100_pct=1.0,
        amp_gain_limit_pct=100.0,
        amp_max_output_pk_v=1000.0,
        finite_support_entries=[],
    )
    assert result is not None
    return result


def test_fallback_command_does_not_stop_before_target_end() -> None:
    result = _run_fallback_finite_compensation()
    profile = result["command_profile"]
    summary = result["finite_signal_consistency"]

    assert result["finite_support_used"] is False
    assert result["finite_route_mode"] == "steady_state_harmonic_expanded"
    assert float(result["estimated_output_lag_seconds"]) > 0.0
    assert float(result["command_nonzero_end_s"]) >= float(result["target_active_end_s"]) - 0.02
    assert bool(result["command_extends_through_target_end"]) is True
    assert abs(float(profile["finite_command_nonzero_end_s"].iloc[0]) - float(summary["command_nonzero_end_s"])) <= 1e-12
    assert bool(profile["finite_command_covers_target_end"].iloc[0]) is True


def test_phase_lead_is_sampling_only_for_fallback_finite_route() -> None:
    result = _run_fallback_finite_compensation()
    profile = result["command_profile"]

    assert bool(result["phase_lead_applied_to_sampling_only"]) is True
    assert str(result["finite_command_stop_policy"]) == "phase_lead_sampling_only_preserve_target_window"
    assert "command_sampling_phase_total" in profile.columns
    assert float(result["phase_lead_seconds_applied"]) > 0.0


def test_post_target_tail_metadata_is_reported() -> None:
    result = _run_fallback_finite_compensation()

    assert result["post_target_command_tail_s"] is not None
    assert float(result["post_target_command_tail_s"]) >= 0.0
    assert bool(result["command_extends_through_target_end"]) is True
    assert float(result["command_early_stop_s"]) == 0.0


def test_old_target_end_minus_lag_cutoff_regression_is_avoided() -> None:
    result = _run_fallback_finite_compensation()

    target_end = float(result["target_active_end_s"])
    lag_seconds = float(result["phase_lead_seconds_applied"])
    old_cutoff = target_end - lag_seconds

    assert lag_seconds > 0.0
    assert old_cutoff < target_end
    assert float(result["command_nonzero_end_s"]) > old_cutoff + 0.01
    assert bool(result["early_command_cutoff_warning"]) is False
