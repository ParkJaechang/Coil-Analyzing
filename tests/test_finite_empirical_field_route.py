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
) -> DatasetAnalysis:
    cycle_progress = np.linspace(0.0, 1.0, 128)
    radians = 2.0 * np.pi * cycle_progress
    voltage = _scaled_waveform(np.sin(radians), 10.0)
    current = _scaled_waveform(np.sin(radians - np.deg2rad(12.0)), current_pp)
    field = _scaled_waveform(np.sin(radians + np.deg2rad(18.0)) + 0.2 * np.sin(3.0 * radians), field_pp)

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
    low = _build_analysis(test_id="steady_2hz", freq_hz=2.0, current_pp=6.0, field_pp=26.0)
    high = _build_analysis(test_id="steady_4hz", freq_hz=4.0, current_pp=18.0, field_pp=70.0)
    summary = pd.concat([low.per_test_summary, high.per_test_summary], ignore_index=True)
    return summary, {"steady_2hz": low, "steady_4hz": high}


def _build_finite_entry(
    *,
    test_id: str,
    waveform_type: str,
    freq_hz: float,
    cycle_count: float,
    field_pp: float,
    voltage_pp: float = 6.0,
) -> dict[str, object]:
    active_duration_s = cycle_count / freq_hz
    total_duration_s = active_duration_s + 0.2
    time_s = np.linspace(0.0, total_duration_s, 240)
    active_mask = time_s <= active_duration_s + 1e-12
    phase = np.clip(time_s / max(active_duration_s, 1e-9), 0.0, 1.0)
    if waveform_type == "triangle":
        base = 2.0 * np.abs(2.0 * phase - 1.0) - 1.0
    else:
        base = np.sin(np.pi * phase)
    taper = np.where(active_mask, 1.0, np.exp(-(time_s - active_duration_s) * 12.0))
    field = _scaled_waveform(base * taper, field_pp)
    current = _scaled_waveform(base * taper, 8.0)
    voltage = _scaled_waveform(base * taper, voltage_pp)
    frame = pd.DataFrame(
        {
            "time_s": time_s,
            "daq_input_v": voltage,
            "i_sum_signed": current,
            "bz_mT": field,
        }
    )
    return {
        "test_id": test_id,
        "waveform_type": waveform_type,
        "freq_hz": float(freq_hz),
        "approx_cycle_span": float(cycle_count),
        "field_pp": float(field_pp),
        "current_pp": 8.0,
        "daq_voltage_pp": float(voltage_pp),
        "frame": frame,
    }


def _run_field_compensation(
    *,
    finite_support_entries: list[dict[str, object]] | None,
    target_cycle_count: float = 1.5,
) -> dict[str, object]:
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
        target_cycle_count=target_cycle_count,
        preview_tail_cycles=0.5,
        max_daq_voltage_pp=1000.0,
        amp_gain_at_100_pct=1.0,
        amp_gain_limit_pct=100.0,
        amp_max_output_pk_v=1000.0,
        finite_support_entries=finite_support_entries,
    )
    assert result is not None
    return result


def test_exact_finite_support_route_is_selected_and_nonzero() -> None:
    result = _run_field_compensation(
        finite_support_entries=[
            _build_finite_entry(
                test_id="finite_exact",
                waveform_type="sine",
                freq_hz=3.0,
                cycle_count=1.5,
                field_pp=100.0,
            )
        ]
    )

    profile = result["command_profile"]
    assert result["mode"] == "finite_empirical_field_support"
    assert result["finite_support_used"] is True
    assert result["request_route"] == "exact"
    assert result["plot_source"] == "exact_prediction"
    assert float(np.nanmax(np.abs(pd.to_numeric(profile["support_scaled_field_mT"], errors="coerce").to_numpy(dtype=float)))) > 1e-9
    assert float(np.nanmax(np.abs(pd.to_numeric(profile["predicted_field_mT"], errors="coerce").to_numpy(dtype=float)))) > 1e-9
    assert float(result["command_nonzero_end_s"]) >= float(result["target_active_end_s"]) - 0.02


def test_nearest_finite_support_preview_route_keeps_metadata() -> None:
    result = _run_field_compensation(
        finite_support_entries=[
            _build_finite_entry(
                test_id="finite_near",
                waveform_type="sine",
                freq_hz=3.0,
                cycle_count=1.25,
                field_pp=38.0,
            )
        ]
    )

    assert result["mode"] == "finite_empirical_weighted_support"
    assert result["finite_support_used"] is True
    assert result["request_route"] == "preview"
    assert result["support_point_count"] > 0
    assert result["zero_padded_fraction"] is not None
    assert float(result["shape_target_output_pp"]) == 100.0


def test_no_finite_support_falls_back_to_harmonic_route_with_reason() -> None:
    result = _run_field_compensation(finite_support_entries=[])

    assert result["finite_support_used"] is False
    assert result["finite_support_fallback_reason"] == "no_finite_support_entries"
    assert str(result["mode"]).startswith("harmonic_inverse_field_only")


def test_support_family_sensitivity_metadata_is_present_for_cross_family_preview() -> None:
    result = _run_field_compensation(
        finite_support_entries=[
            _build_finite_entry(
                test_id="triangle_only",
                waveform_type="triangle",
                freq_hz=3.0,
                cycle_count=1.25,
                field_pp=42.0,
            ),
            _build_finite_entry(
                test_id="triangle_alt",
                waveform_type="triangle",
                freq_hz=3.1,
                cycle_count=1.5,
                field_pp=39.0,
            ),
        ]
    )

    assert result["finite_support_used"] is True
    assert result["support_waveform_role"] == "input_support_family"
    assert "support_family_sensitivity_flag" in result
    assert "support_family_sensitivity_reason" in result


def test_support_blended_output_zero_bug_is_guarded() -> None:
    result = _run_field_compensation(
        finite_support_entries=[
            _build_finite_entry(
                test_id="finite_guard",
                waveform_type="triangle",
                freq_hz=3.0,
                cycle_count=1.25,
                field_pp=55.0,
            )
        ]
    )

    support_scaled = pd.to_numeric(result["command_profile"]["support_scaled_field_mT"], errors="coerce").to_numpy(dtype=float)
    assert float(np.nanmax(np.abs(support_scaled))) > 1e-9
