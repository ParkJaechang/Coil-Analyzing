from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.compensation import FIELD_ROUTE_NORMALIZED_TARGET_PP, synthesize_current_waveform_compensation
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


def _normalized_shape(values: pd.Series | np.ndarray) -> np.ndarray:
    signal = np.asarray(values, dtype=float)
    centered = signal - float(np.nanmean(signal))
    peak_to_peak = float(np.nanmax(centered) - np.nanmin(centered))
    if peak_to_peak <= 1e-12:
        return centered
    return centered / (peak_to_peak / 2.0)


def _build_analysis(
    *,
    test_id: str,
    freq_hz: float,
    current_pp: float,
    field_pp: float,
    field_phase_deg: float,
    field_h3: float,
    voltage_pp: float = 10.0,
    cycles: int = 4,
    points_per_cycle: int = 128,
) -> DatasetAnalysis:
    cycle_progress = np.linspace(0.0, 1.0, points_per_cycle)
    radians = 2.0 * np.pi * cycle_progress
    voltage = _scaled_waveform(np.sin(radians), voltage_pp)
    current = _scaled_waveform(np.sin(radians - np.deg2rad(12.0)), current_pp)
    field_shape = np.sin(radians + np.deg2rad(field_phase_deg)) + field_h3 * np.sin(3.0 * radians)
    field = _scaled_waveform(field_shape, field_pp)

    period_s = 1.0 / float(freq_hz)
    annotated_rows: list[dict[str, float | int]] = []
    cycle_rows: list[dict[str, float | int]] = []
    for cycle_index in range(cycles):
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
                "daq_input_v_pp": float(voltage_pp),
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
                "daq_input_v_pp_mean": float(voltage_pp),
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
    low_freq = _build_analysis(
        test_id="support_2hz",
        freq_hz=2.0,
        current_pp=6.0,
        field_pp=26.0,
        field_phase_deg=10.0,
        field_h3=0.10,
    )
    high_freq = _build_analysis(
        test_id="support_4hz",
        freq_hz=4.0,
        current_pp=18.0,
        field_pp=70.0,
        field_phase_deg=18.0,
        field_h3=0.22,
    )
    summary = pd.concat(
        [low_freq.per_test_summary, high_freq.per_test_summary],
        ignore_index=True,
    )
    analyses = {
        "support_2hz": low_freq,
        "support_4hz": high_freq,
    }
    return summary, analyses


def _run_field_compensation(
    per_test_summary: pd.DataFrame,
    analyses_by_test_id: dict[str, DatasetAnalysis],
    *,
    freq_hz: float,
    target_output_pp: float,
    lcr_measurements: pd.DataFrame | None = None,
    lcr_blend_weight: float = 0.0,
) -> dict[str, object]:
    result = synthesize_current_waveform_compensation(
        per_test_summary=per_test_summary,
        analyses_by_test_id=analyses_by_test_id,
        waveform_type="sine",
        freq_hz=float(freq_hz),
        target_current_pp_a=float(target_output_pp),
        target_output_type="field",
        target_output_pp=float(target_output_pp),
        points_per_cycle=128,
        max_daq_voltage_pp=1000.0,
        amp_gain_at_100_pct=1.0,
        amp_gain_limit_pct=100.0,
        amp_max_output_pk_v=1000.0,
        lcr_measurements=lcr_measurements,
        lcr_blend_weight=lcr_blend_weight,
    )
    assert result is not None
    return result


def test_field_route_blends_neighboring_supports_smoothly_below_5hz() -> None:
    summary, analyses = _build_support_context()

    below = _run_field_compensation(summary, analyses, freq_hz=2.9, target_output_pp=45.0)
    above = _run_field_compensation(summary, analyses, freq_hz=3.1, target_output_pp=45.0)

    below_profile = below["command_profile"]
    above_profile = above["command_profile"]
    below_shape = _normalized_shape(below_profile["predicted_field_mT"])
    above_shape = _normalized_shape(above_profile["predicted_field_mT"])

    assert below["mode"] == "harmonic_inverse_field_only_freq_blend"
    assert above["mode"] == "harmonic_inverse_field_only_freq_blend"
    assert below["field_support_freq_count"] == 2
    assert above["field_support_freq_count"] == 2
    assert below["field_support_test_ids"] == "support_2hz|support_4hz"
    assert above["field_support_test_ids"] == "support_2hz|support_4hz"
    assert float(np.corrcoef(below_shape, above_shape)[0, 1]) > 0.995
    assert float(np.max(np.abs(below_shape - above_shape))) < 0.05


def test_field_route_target_scale_changes_do_not_change_normalized_prediction_shape() -> None:
    summary, analyses = _build_support_context()

    small_target = _run_field_compensation(summary, analyses, freq_hz=3.0, target_output_pp=25.0)
    large_target = _run_field_compensation(summary, analyses, freq_hz=3.0, target_output_pp=250.0)

    small_profile = small_target["command_profile"]
    large_profile = large_target["command_profile"]
    small_shape = _normalized_shape(small_profile["predicted_field_mT"])
    large_shape = _normalized_shape(large_profile["predicted_field_mT"])

    assert float(small_profile["target_output_pp"].iloc[0]) == FIELD_ROUTE_NORMALIZED_TARGET_PP
    assert float(large_profile["target_output_pp"].iloc[0]) == FIELD_ROUTE_NORMALIZED_TARGET_PP
    assert small_target["shape_target_output_pp"] == FIELD_ROUTE_NORMALIZED_TARGET_PP
    assert large_target["shape_target_output_pp"] == FIELD_ROUTE_NORMALIZED_TARGET_PP
    assert np.allclose(small_shape, large_shape, atol=1e-6)


def test_field_route_ignores_current_and_lcr_branches_for_shape_selection() -> None:
    summary, analyses = _build_support_context()
    mutated_summary = summary.copy()
    mutated_summary["current_pp_target_a"] = [3.0, 900.0]
    mutated_summary["achieved_current_pp_a_mean"] = [1.0, 1500.0]
    lcr_measurements = pd.DataFrame(
        {
            "freq_hz": [1.0, 3.0, 10.0],
            "impedance_ohm": [5.0, 7.0, 10.0],
            "phase_deg": [5.0, 15.0, 25.0],
        }
    )

    baseline = _run_field_compensation(summary, analyses, freq_hz=3.0, target_output_pp=40.0)
    mutated = _run_field_compensation(
        mutated_summary,
        analyses,
        freq_hz=3.0,
        target_output_pp=40.0,
        lcr_measurements=lcr_measurements,
        lcr_blend_weight=0.9,
    )

    baseline_profile = baseline["command_profile"]
    mutated_profile = mutated["command_profile"]
    assert baseline["support_selection_reason"] == "nearest_frequency_support"
    assert mutated["support_selection_reason"] == "nearest_frequency_support"
    assert baseline["selected_support_id"] == mutated["selected_support_id"]
    assert baseline["used_lcr_prior"] is False
    assert mutated["used_lcr_prior"] is False
    assert mutated["lcr_usage_mode"] == "disabled_for_field_shape_route"
    assert np.allclose(
        _normalized_shape(baseline_profile["predicted_field_mT"]),
        _normalized_shape(mutated_profile["predicted_field_mT"]),
        atol=1e-6,
    )
    assert np.allclose(
        pd.to_numeric(baseline_profile["recommended_voltage_v"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(mutated_profile["recommended_voltage_v"], errors="coerce").to_numpy(dtype=float),
        atol=1e-6,
    )
