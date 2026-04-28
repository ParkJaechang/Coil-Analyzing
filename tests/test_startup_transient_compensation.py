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


def test_continuous_prediction_reflects_startup_initial_field_offset() -> None:
    analysis = _build_startup_offset_analysis(first_cycle_field_offset_mT=18.0, freq_hz=5.0)
    result = _run_compensation(analysis=analysis, freq_hz=5.0)
    profile = result["command_profile"]

    assert result["startup_transient_applied"] is True
    assert result["startup_initial_field_offset_mT"] > 0.0
    assert bool(profile["physical_target_output_mT"].equals(profile["target_field_mT"])) is True
    first_residual = float(profile["predicted_field_mT"].iloc[0] - profile["physical_target_output_mT"].iloc[0])
    assert first_residual > 10.0


def test_finite_prediction_reflects_startup_offset_without_target_stretch() -> None:
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
    assert result["startup_initial_field_offset_mT"] > 0.0
    assert np.isclose(float(result["target_active_end_s"]), 0.3, atol=0.002)
    assert bool(profile["physical_target_output_mT"].equals(profile["target_field_mT"])) is True
    assert _early_late_residual_delta(profile, "predicted_field_mT", period_s=0.2) > 2.0
    assert float(result["predicted_jump_ratio"]) <= 0.20


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
