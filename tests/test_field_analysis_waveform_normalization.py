from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIELD_ANALYSIS_SRC = ROOT.parent / "src"
if str(FIELD_ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(FIELD_ANALYSIS_SRC))

from field_analysis.compensation import (
    run_validation_recommendation_loop,
    synthesize_current_waveform_compensation,
    synthesize_finite_empirical_compensation,
)
from field_analysis.lut import recommend_voltage_waveform
from field_analysis.models import (
    CycleDetectionResult,
    DatasetAnalysis,
    ParsedMeasurement,
    PreprocessResult,
    SheetPreview,
)
from field_analysis.parser import parse_measurement_file
from field_analysis.plotting import plot_output_compensation_waveforms
from field_analysis.schema_config import build_default_schema
from field_analysis.utils import (
    canonicalize_waveform_type,
    infer_conditions_from_filename,
    infer_current_from_text,
    infer_waveform_from_text,
)


def _normalized_waveform(phase: np.ndarray | float, waveform_type: str) -> np.ndarray:
    waveform_type = canonicalize_waveform_type(waveform_type) or "sine"
    phase_array = np.mod(np.asarray(phase, dtype=float), 1.0)
    if waveform_type == "triangle":
        waveform = np.empty_like(phase_array)
        rising = phase_array < 0.25
        falling = (phase_array >= 0.25) & (phase_array < 0.75)
        waveform[rising] = 4.0 * phase_array[rising]
        waveform[falling] = 2.0 - 4.0 * phase_array[falling]
        waveform[~(rising | falling)] = -4.0 + 4.0 * phase_array[~(rising | falling)]
        return waveform
    return np.sin(2.0 * np.pi * phase_array)


def _build_dummy_analysis(
    phase_shift_cycles: float = 0.0,
    waveform_type: str = "sine",
    freq_hz: float = 0.5,
    test_id: str = "t1",
) -> tuple[pd.DataFrame, DatasetAnalysis]:
    waveform_type = canonicalize_waveform_type(waveform_type) or "sine"
    per_test_summary = pd.DataFrame(
        [
            {
                "test_id": test_id,
                "waveform_type": waveform_type,
                "freq_hz": freq_hz,
                "current_pp_target_a": 10.0,
                "achieved_current_pp_a_mean": 10.0,
                "achieved_bz_mT_pp_mean": 20.0,
                "achieved_bmag_mT_pp_mean": 20.0,
                "daq_input_v_pp_mean": 4.0,
                "amp_gain_setting_mean": 50.0,
            }
        ]
    )

    phase = np.linspace(0.0, 1.0, 32)
    rows: list[dict[str, float]] = []
    for cycle_index in range(4):
        cycle_time = phase * 2.0
        for cycle_progress, cycle_time_s in zip(phase, cycle_time, strict=False):
            shifted_phase = cycle_progress + phase_shift_cycles
            rows.append(
                {
                    "cycle_index": float(cycle_index),
                    "cycle_progress": float(cycle_progress),
                    "cycle_time_s": float(cycle_time_s),
                    "time_s": float(cycle_time_s + cycle_index * 2.0),
                    "freq_hz": freq_hz,
                    "daq_input_v": 2.0 * _normalized_waveform(shifted_phase, waveform_type),
                    "i_sum_signed": 5.0 * _normalized_waveform(shifted_phase, waveform_type),
                    "bz_mT": 10.0 * _normalized_waveform(shifted_phase - 0.1, waveform_type),
                }
            )
    annotated_frame = pd.DataFrame(rows)
    per_cycle_summary = pd.DataFrame(
        [
            {
                "cycle_index": cycle_index,
                "achieved_current_pp_a": 10.0,
                "achieved_bz_mT_pp": 20.0,
                "daq_input_v_pp": 4.0,
            }
            for cycle_index in range(4)
        ]
    )

    preview = SheetPreview("main", 0, 0, [], 0, {}, [], {}, [], [])
    parsed = ParsedMeasurement(
        source_file="file.csv",
        file_type="csv",
        sheet_name="main",
        structure_preview=preview,
        metadata={"waveform": waveform_type},
        mapping={},
        raw_frame=annotated_frame.copy(),
        normalized_frame=annotated_frame.copy(),
        warnings=[],
        logs=[],
    )
    preprocess = PreprocessResult(
        corrected_frame=annotated_frame.copy(),
        offsets={},
        lags=[],
        warnings=[],
        logs=[],
    )
    cycle_detection = CycleDetectionResult(
        annotated_frame=annotated_frame.copy(),
        boundaries=[],
        estimated_period_s=2.0,
        estimated_frequency_hz=0.5,
        reference_channel="daq_input_v",
        warnings=[],
        logs=[],
    )
    analysis = DatasetAnalysis(
        parsed=parsed,
        preprocess=preprocess,
        cycle_detection=cycle_detection,
        per_cycle_summary=per_cycle_summary,
        per_test_summary=per_test_summary.copy(),
        warnings=[],
    )
    return per_test_summary, analysis


def test_canonicalize_waveform_type_accepts_ui_labels() -> None:
    assert canonicalize_waveform_type("Sine") == "sine"
    assert canonicalize_waveform_type("Triangle") == "triangle"
    assert canonicalize_waveform_type("사인파") == "sine"
    assert canonicalize_waveform_type("삼각파") == "triangle"


def test_waveform_and_level_inference_accepts_nested_transient_paths() -> None:
    sample = "sinusidal/00777c78a89e3efe_2hz_1cycle_20pp.csv"
    assert infer_waveform_from_text(sample) == "sine"
    assert infer_current_from_text(sample) == 20.0


def test_transient_filename_inference_preserves_pp_mode_for_triangle_paths() -> None:
    sample = "triangle/1.25hz_1.25cycle_10pp.csv"
    inferred = infer_conditions_from_filename(sample)
    assert inferred["freq_hz"] == 1.25
    assert inferred["cycle_count"] == 1.25
    assert inferred["current_target_a"] == 10.0
    assert inferred["current_target_mode"] == "pp"


def test_transient_filename_inference_accepts_p_decimal_notation() -> None:
    sample = "sinusidal/abcd1234_1p25hz_1p5cycle_20pp.csv"
    inferred = infer_conditions_from_filename(sample)
    assert inferred["waveform_type"] is None
    assert inferred["freq_hz"] == 1.25
    assert inferred["cycle_count"] == 1.5
    assert inferred["current_target_a"] == 20.0
    assert inferred["current_target_mode"] == "pp"


def test_parser_uses_filename_cycle_count_when_metadata_is_missing() -> None:
    csv_content = "\n".join(
        [
            "Time,Voltage1,Current1_A,Current2_A,HallBz",
            "0.0,0.0,0.0,0.1,0.0",
            "0.1,1.0,0.0,0.2,1.0",
            "0.2,0.0,0.0,0.1,0.0",
            "0.3,-1.0,0.0,0.2,-1.0",
        ]
    )
    parsed = parse_measurement_file(
        file_name="sinusidal/abcd1234_1p25hz_1p5cycle_20pp.csv",
        file_bytes=csv_content.encode("utf-8"),
        schema=build_default_schema(),
    )[0]
    normalized = parsed.normalized_frame
    assert np.isclose(float(normalized["cycle_total_expected"].iloc[0]), 1.5)
    assert normalized["target_current_mode_inferred"].iloc[0] == "pp"


def test_recommend_voltage_waveform_accepts_title_case_waveform() -> None:
    per_test_summary, _ = _build_dummy_analysis()

    result = recommend_voltage_waveform(
        per_test_summary=per_test_summary,
        analyses_by_test_id={},
        waveform_type="Sine",
        freq_hz=0.5,
        target_metric="achieved_bz_mT_pp_mean",
        target_value=20.0,
    )

    assert result is not None
    assert result["waveform_type"] == "sine"
    assert not result["command_waveform"].empty


def test_recommend_voltage_waveform_reanchors_shifted_template_to_zero_start() -> None:
    per_test_summary, analysis = _build_dummy_analysis(phase_shift_cycles=0.2)

    result = recommend_voltage_waveform(
        per_test_summary=per_test_summary,
        analyses_by_test_id={"t1": analysis},
        waveform_type="Sine",
        freq_hz=0.5,
        target_metric="achieved_bz_mT_pp_mean",
        target_value=20.0,
    )

    assert result is not None
    waveform = result["command_waveform"]
    peak = float(np.nanmax(np.abs(waveform["recommended_voltage_v"].to_numpy(dtype=float))))
    assert peak > 0
    assert abs(float(waveform["recommended_voltage_v"].iloc[0])) <= peak * 0.05


def test_current_compensation_accepts_title_case_waveform() -> None:
    per_test_summary, analysis = _build_dummy_analysis()

    result = synthesize_current_waveform_compensation(
        per_test_summary=per_test_summary,
        analyses_by_test_id={"t1": analysis},
        waveform_type="Sine",
        freq_hz=0.5,
        target_current_pp_a=10.0,
    )

    assert result is not None
    assert result["waveform_type"] == "sine"
    assert not result["command_profile"].empty


def test_triangle_current_compensation_uses_extended_harmonics() -> None:
    per_test_summary, analysis = _build_dummy_analysis(waveform_type="triangle")

    result = synthesize_current_waveform_compensation(
        per_test_summary=per_test_summary,
        analyses_by_test_id={"t1": analysis},
        waveform_type="Triangle",
        freq_hz=0.5,
        target_current_pp_a=10.0,
        points_per_cycle=256,
    )

    assert result is not None
    assert result["waveform_type"] == "triangle"
    assert result["max_harmonics_used"] == 31
    target_current = result["command_profile"]["target_current_a"].to_numpy(dtype=float)
    quarter = max(len(target_current) // 4, 2)
    assert abs(float(target_current[0])) <= 1e-9
    assert np.all(np.diff(target_current[:quarter]) >= -1e-9)


def test_current_compensation_reanchors_command_to_zero_start() -> None:
    per_test_summary, analysis = _build_dummy_analysis(phase_shift_cycles=0.2)

    result = synthesize_current_waveform_compensation(
        per_test_summary=per_test_summary,
        analyses_by_test_id={"t1": analysis},
        waveform_type="Sine",
        freq_hz=0.5,
        target_current_pp_a=10.0,
    )

    assert result is not None
    profile = result["command_profile"]
    peak = float(np.nanmax(np.abs(profile["limited_voltage_v"].to_numpy(dtype=float))))
    assert peak > 0
    assert abs(float(profile["limited_voltage_v"].iloc[0])) <= peak * 0.05


def test_current_compensation_keeps_target_reference_at_zero_and_separates_field_preview() -> None:
    per_test_summary, analysis = _build_dummy_analysis(phase_shift_cycles=0.2)

    result = synthesize_current_waveform_compensation(
        per_test_summary=per_test_summary,
        analyses_by_test_id={"t1": analysis},
        waveform_type="Sine",
        freq_hz=0.5,
        target_current_pp_a=10.0,
    )

    assert result is not None
    profile = result["command_profile"]
    assert "aligned_target_current_a" in profile.columns
    assert np.allclose(
        profile["aligned_target_current_a"].to_numpy(dtype=float),
        profile["target_current_a"].to_numpy(dtype=float),
        atol=1e-9,
    )
    assert "support_scaled_field_mT" in profile.columns
    assert "expected_field_mT" not in profile.columns


def test_current_compensation_prefers_exact_frequency_bucket_when_available() -> None:
    summary_a, analysis_a = _build_dummy_analysis(freq_hz=0.5, test_id="t05")
    summary_b, analysis_b = _build_dummy_analysis(freq_hz=1.0, test_id="t10")
    per_test_summary = pd.concat([summary_a, summary_b], ignore_index=True)

    result = synthesize_current_waveform_compensation(
        per_test_summary=per_test_summary,
        analyses_by_test_id={"t05": analysis_a, "t10": analysis_b},
        waveform_type="Sine",
        freq_hz=0.5,
        target_current_pp_a=10.0,
    )

    assert result is not None
    assert result["frequency_bucket_mode"] == "exact_frequency_bucket"
    assert np.allclose(result["support_table"]["freq_hz"].to_numpy(dtype=float), np.array([0.5]))


def test_validation_recommendation_loop_reanchors_refined_command_to_zero_start() -> None:
    phase = np.linspace(0.0, 1.0, 129)
    time_s = phase * 2.0
    base_profile = pd.DataFrame(
        {
            "time_s": time_s,
            "limited_voltage_v": 2.0 * np.sin(2.0 * np.pi * phase),
            "target_output": 10.0 * np.sin(2.0 * np.pi * phase),
            "finite_cycle_mode": False,
            "is_active_target": True,
        }
    )
    validation_frame = pd.DataFrame(
        {
            "time_s": time_s,
            "daq_input_v": 2.0 * np.sin(2.0 * np.pi * phase),
            "i_sum_signed": 10.0 * np.sin(2.0 * np.pi * phase - (np.pi / 2.0)),
            "bz_mT": 20.0 * np.sin(2.0 * np.pi * phase - (np.pi / 2.0)),
        }
    )

    result = run_validation_recommendation_loop(
        command_profile=base_profile,
        validation_frame=validation_frame,
        target_output_type="current",
        correction_gain=1.0,
        max_iterations=1,
        max_daq_voltage_pp=20.0,
        amp_gain_at_100_pct=20.0,
        support_amp_gain_pct=100.0,
        amp_gain_limit_pct=100.0,
        amp_max_output_pk_v=180.0,
    )

    assert result is not None
    assert np.isfinite(float(result["predicted_nrmse_final"]))
    profile = result["command_profile"]
    peak = float(np.nanmax(np.abs(profile["limited_voltage_v"].to_numpy(dtype=float))))
    assert peak > 0
    assert abs(float(profile["limited_voltage_v"].iloc[0])) <= peak * 0.05


def test_finite_empirical_prefers_exact_cycle_bucket() -> None:
    def build_frame(cycle_span: float) -> pd.DataFrame:
        duration_s = cycle_span
        time_s = np.linspace(0.0, duration_s, 257)
        phase = time_s / max(duration_s, 1e-9)
        return pd.DataFrame(
            {
                "time_s": time_s,
                "daq_input_v": 2.0 * np.sin(2.0 * np.pi * phase),
                "i_sum_signed": 5.0 * np.sin(2.0 * np.pi * phase),
                "bz_mT": 10.0 * np.sin(2.0 * np.pi * phase),
            }
        )

    finite_support_entries = [
        {
            "test_id": "cycle_075",
            "source_file": "c075.csv",
            "sheet_name": "main",
            "waveform_type": "sine",
            "freq_hz": 1.0,
            "duration_s": 0.75,
            "approx_cycle_span": 0.75,
            "target_current_a": 5.0,
            "notes": "",
            "current_pp": 10.0,
            "field_pp": 20.0,
            "daq_voltage_pp": 4.0,
            "frame": build_frame(0.75),
        },
        {
            "test_id": "cycle_100",
            "source_file": "c100.csv",
            "sheet_name": "main",
            "waveform_type": "sine",
            "freq_hz": 1.0,
            "duration_s": 1.0,
            "approx_cycle_span": 1.0,
            "target_current_a": 5.0,
            "notes": "",
            "current_pp": 10.0,
            "field_pp": 20.0,
            "daq_voltage_pp": 4.0,
            "frame": build_frame(1.0),
        },
    ]

    result = synthesize_finite_empirical_compensation(
        finite_support_entries=finite_support_entries,
        waveform_type="sine",
        freq_hz=1.0,
        target_cycle_count=0.75,
        target_output_type="current",
        target_output_pp=10.0,
        max_support_count=2,
    )

    assert result is not None
    assert result["cycle_bucket_mode"] == "exact_cycle_bucket"
    assert result["support_count_used"] == 1
    assert np.isclose(result["support_cycle_count"], 0.75)
    assert np.allclose(result["support_table"]["approx_cycle_span"].to_numpy(dtype=float), np.array([0.75]))


def test_finite_empirical_supports_triangle_waveform() -> None:
    time_s = np.linspace(0.0, 0.75, 257)
    phase = time_s / 1.0
    triangle = _normalized_waveform(phase, "triangle")
    finite_support_entries = [
        {
            "test_id": "tri_075",
            "source_file": "tri.csv",
            "sheet_name": "main",
            "waveform_type": "triangle",
            "freq_hz": 1.0,
            "duration_s": 0.75,
            "approx_cycle_span": 0.75,
            "target_current_a": 5.0,
            "notes": "",
            "current_pp": 10.0,
            "field_pp": 20.0,
            "daq_voltage_pp": 4.0,
            "frame": pd.DataFrame(
                {
                    "time_s": time_s,
                    "daq_input_v": 2.0 * triangle,
                    "i_sum_signed": 5.0 * triangle,
                    "bz_mT": 10.0 * triangle,
                }
            ),
        }
    ]

    result = synthesize_finite_empirical_compensation(
        finite_support_entries=finite_support_entries,
        waveform_type="triangle",
        freq_hz=1.0,
        target_cycle_count=0.75,
        target_output_type="current",
        target_output_pp=10.0,
        max_support_count=1,
    )

    assert result is not None
    profile = result["command_profile"]
    assert result["support_tests_used"] == ["tri_075"]
    assert abs(float(profile["target_output"].iloc[0])) <= 1e-9
    quarter = max(len(profile) // 4, 2)
    assert np.all(np.diff(profile["target_output"].to_numpy(dtype=float)[:quarter]) >= -1e-9)


def test_finite_empirical_does_not_fallback_to_other_waveform_support() -> None:
    time_s = np.linspace(0.0, 0.75, 257)
    phase = time_s / 1.0
    sine = np.sin(2.0 * np.pi * phase)
    finite_support_entries = [
        {
            "test_id": "sin_075",
            "source_file": "sin.csv",
            "sheet_name": "main",
            "waveform_type": "sine",
            "freq_hz": 1.0,
            "duration_s": 0.75,
            "approx_cycle_span": 0.75,
            "target_current_a": 5.0,
            "notes": "",
            "current_pp": 10.0,
            "field_pp": 20.0,
            "daq_voltage_pp": 4.0,
            "frame": pd.DataFrame(
                {
                    "time_s": time_s,
                    "daq_input_v": 2.0 * sine,
                    "i_sum_signed": 5.0 * sine,
                    "bz_mT": 10.0 * sine,
                }
            ),
        }
    ]

    result = synthesize_finite_empirical_compensation(
        finite_support_entries=finite_support_entries,
        waveform_type="triangle",
        freq_hz=1.0,
        target_cycle_count=0.75,
        target_output_type="current",
        target_output_pp=10.0,
        max_support_count=1,
    )

    assert result is None


def test_finite_empirical_prefers_exact_frequency_bucket() -> None:
    time_s = np.linspace(0.0, 1.0, 257)
    phase = time_s / 1.0
    sine = np.sin(2.0 * np.pi * phase)
    finite_support_entries = [
        {
            "test_id": "exact_freq",
            "source_file": "f1.csv",
            "sheet_name": "main",
            "waveform_type": "sine",
            "freq_hz": 1.0,
            "duration_s": 1.0,
            "approx_cycle_span": 1.0,
            "target_current_a": 5.0,
            "notes": "",
            "current_pp": 10.0,
            "field_pp": 20.0,
            "daq_voltage_pp": 4.0,
            "frame": pd.DataFrame(
                {
                    "time_s": time_s,
                    "daq_input_v": 2.0 * sine,
                    "i_sum_signed": 5.0 * sine,
                    "bz_mT": 10.0 * sine,
                }
            ),
        },
        {
            "test_id": "other_freq",
            "source_file": "f05.csv",
            "sheet_name": "main",
            "waveform_type": "sine",
            "freq_hz": 0.5,
            "duration_s": 2.0,
            "approx_cycle_span": 1.0,
            "target_current_a": 5.0,
            "notes": "",
            "current_pp": 10.0,
            "field_pp": 20.0,
            "daq_voltage_pp": 4.0,
            "frame": pd.DataFrame(
                {
                    "time_s": np.linspace(0.0, 2.0, 257),
                    "daq_input_v": 2.0 * sine,
                    "i_sum_signed": 5.0 * sine,
                    "bz_mT": 10.0 * sine,
                }
            ),
        },
    ]

    result = synthesize_finite_empirical_compensation(
        finite_support_entries=finite_support_entries,
        waveform_type="sine",
        freq_hz=1.0,
        target_cycle_count=1.0,
        target_output_type="current",
        target_output_pp=10.0,
        max_support_count=2,
    )

    assert result is not None
    assert result["frequency_bucket_mode"] == "exact_frequency_bucket"
    assert result["support_tests_used"] == ["exact_freq"]


def test_finite_exact_route_uses_direct_prediction_without_support_scaled_columns() -> None:
    time_s = np.linspace(0.0, 1.0, 257)
    phase = time_s / 1.0
    triangle = _normalized_waveform(phase, "triangle")
    finite_support_entries = [
        {
            "test_id": "tri_exact",
            "source_file": "tri_exact.csv",
            "sheet_name": "main",
            "waveform_type": "triangle",
            "freq_hz": 1.0,
            "duration_s": 1.0,
            "approx_cycle_span": 1.0,
            "requested_cycle_count": 1.0,
            "requested_current_pp": 20.0,
            "current_pp": 20.0,
            "field_pp": 40.0,
            "daq_voltage_pp": 8.0,
            "frame": pd.DataFrame(
                {
                    "time_s": time_s,
                    "daq_input_v": 4.0 * triangle,
                    "i_sum_signed": 10.0 * triangle,
                    "bz_mT": 20.0 * triangle,
                }
            ),
            "active_frame": pd.DataFrame(
                {
                    "time_s": time_s,
                    "daq_input_v": 4.0 * triangle,
                    "i_sum_signed": 10.0 * triangle,
                    "bz_mT": 20.0 * triangle,
                }
            ),
        }
    ]

    result = synthesize_finite_empirical_compensation(
        finite_support_entries=finite_support_entries,
        waveform_type="triangle",
        freq_hz=1.0,
        target_cycle_count=1.0,
        target_output_type="current",
        target_output_pp=20.0,
        max_support_count=3,
    )

    assert result is not None
    assert result["request_route"] == "exact"
    assert result["plot_source"] == "exact_prediction"
    profile = result["command_profile"]
    assert "support_scaled_current_a" not in profile.columns
    assert "support_scaled_field_mT" not in profile.columns
    assert profile["plot_source"].iloc[0] == "exact_prediction"


def test_finite_exact_sine_and_triangle_routes_produce_distinct_voltage_waveforms() -> None:
    time_s = np.linspace(0.0, 1.0, 257)
    phase = time_s / 1.0
    sine = _normalized_waveform(phase, "sine")
    triangle = _normalized_waveform(phase, "triangle")
    finite_support_entries = [
        {
            "test_id": "sin_exact",
            "source_file": "sin_exact.csv",
            "sheet_name": "main",
            "waveform_type": "sine",
            "freq_hz": 1.0,
            "duration_s": 1.0,
            "approx_cycle_span": 1.0,
            "requested_cycle_count": 1.0,
            "requested_current_pp": 20.0,
            "current_pp": 20.0,
            "field_pp": 40.0,
            "daq_voltage_pp": 8.0,
            "frame": pd.DataFrame({"time_s": time_s, "daq_input_v": 4.0 * sine, "i_sum_signed": 10.0 * sine, "bz_mT": 20.0 * sine}),
            "active_frame": pd.DataFrame({"time_s": time_s, "daq_input_v": 4.0 * sine, "i_sum_signed": 10.0 * sine, "bz_mT": 20.0 * sine}),
        },
        {
            "test_id": "tri_exact",
            "source_file": "tri_exact.csv",
            "sheet_name": "main",
            "waveform_type": "triangle",
            "freq_hz": 1.0,
            "duration_s": 1.0,
            "approx_cycle_span": 1.0,
            "requested_cycle_count": 1.0,
            "requested_current_pp": 20.0,
            "current_pp": 20.0,
            "field_pp": 40.0,
            "daq_voltage_pp": 8.0,
            "frame": pd.DataFrame({"time_s": time_s, "daq_input_v": 4.0 * triangle, "i_sum_signed": 10.0 * triangle, "bz_mT": 20.0 * triangle}),
            "active_frame": pd.DataFrame({"time_s": time_s, "daq_input_v": 4.0 * triangle, "i_sum_signed": 10.0 * triangle, "bz_mT": 20.0 * triangle}),
        },
    ]

    sine_result = synthesize_finite_empirical_compensation(
        finite_support_entries=finite_support_entries,
        waveform_type="sine",
        freq_hz=1.0,
        target_cycle_count=1.0,
        target_output_type="current",
        target_output_pp=20.0,
    )
    triangle_result = synthesize_finite_empirical_compensation(
        finite_support_entries=finite_support_entries,
        waveform_type="triangle",
        freq_hz=1.0,
        target_cycle_count=1.0,
        target_output_type="current",
        target_output_pp=20.0,
    )

    assert sine_result is not None
    assert triangle_result is not None
    sine_voltage = sine_result["command_profile"]["limited_voltage_v"].to_numpy(dtype=float)
    triangle_voltage = triangle_result["command_profile"]["limited_voltage_v"].to_numpy(dtype=float)
    assert not np.allclose(sine_voltage, triangle_voltage)


def test_finite_preview_uses_active_window_instead_of_full_frame_normalization() -> None:
    full_time = np.linspace(0.0, 1.5, 301)
    active_mask = (full_time >= 0.5) & (full_time <= 1.25)
    active_phase = (full_time[active_mask] - 0.5) / 0.75
    active_wave = _normalized_waveform(active_phase * 0.75, "sine")
    voltage = np.zeros_like(full_time)
    current = np.zeros_like(full_time)
    field = np.zeros_like(full_time)
    voltage[active_mask] = 2.0 * active_wave
    current[active_mask] = 5.0 * active_wave
    field[active_mask] = 10.0 * active_wave
    full_frame = pd.DataFrame({"time_s": full_time, "daq_input_v": voltage, "i_sum_signed": current, "bz_mT": field})
    active_frame = full_frame.loc[active_mask].copy()

    result = synthesize_finite_empirical_compensation(
        finite_support_entries=[
            {
                "test_id": "preview_shifted",
                "source_file": "preview_shifted.csv",
                "sheet_name": "main",
                "waveform_type": "sine",
                "freq_hz": 1.0,
                "duration_s": 1.5,
                "approx_cycle_span": 0.75,
                "requested_cycle_count": 0.75,
                "requested_current_pp": 10.0,
                "current_pp": 10.0,
                "field_pp": 20.0,
                "daq_voltage_pp": 4.0,
                "frame": full_frame,
                "active_frame": active_frame,
            }
        ],
        waveform_type="sine",
        freq_hz=1.0,
        target_cycle_count=0.75,
        target_output_type="current",
        target_output_pp=20.0,
        preview_tail_cycles=0.25,
    )

    assert result is not None
    assert result["request_route"] == "preview"
    assert result["plot_source"] == "support_blended_preview"
    assert np.isclose(float(result["active_window_start_s"]), 0.5)
    assert np.isclose(float(result["active_window_end_s"]), 1.25)
    assert np.isclose(float(result["active_duration_s"]), 0.75)
    assert float(result["zero_padded_fraction"]) > 0.10
    profile = result["command_profile"]
    assert np.isclose(float(profile["zero_padded_fraction"].iloc[0]), float(result["zero_padded_fraction"]))
    early_active = profile.loc[
        (profile["time_s"] > 0.05) & (profile["time_s"] < 0.20),
        "expected_current_a",
    ].to_numpy(dtype=float)
    assert np.nanmax(np.abs(early_active)) > 1.0


def test_exact_finite_plot_omits_support_blended_trace() -> None:
    command_profile = pd.DataFrame(
        {
            "time_s": np.linspace(0.0, 1.0, 8),
            "target_output": np.linspace(0.0, 1.0, 8),
            "expected_output": np.linspace(0.0, 1.0, 8),
            "is_active_target": [True] * 8,
        }
    )

    figure = plot_output_compensation_waveforms(
        command_profile=command_profile,
        nearest_profile=None,
        nearest_column="measured_current_a",
        title="Exact Finite",
        yaxis_title="Current (A)",
        predicted_label="Exact Predicted Output",
    )

    trace_names = [trace.name for trace in figure.data]
    assert "Exact Predicted Output" in trace_names
    assert "Support-Blended Output" not in trace_names
    assert "Nearest Measured Output" not in trace_names


def test_preview_finite_plot_keeps_support_blended_trace() -> None:
    command_profile = pd.DataFrame(
        {
            "time_s": np.linspace(0.0, 1.0, 8),
            "target_output": np.linspace(0.0, 1.0, 8),
            "expected_output": np.linspace(0.0, 1.0, 8),
            "is_active_target": [True] * 8,
        }
    )
    nearest_profile = pd.DataFrame(
        {
            "time_s": np.linspace(0.0, 1.0, 8),
            "measured_current_a": np.linspace(1.0, 0.0, 8),
        }
    )

    figure = plot_output_compensation_waveforms(
        command_profile=command_profile,
        nearest_profile=nearest_profile,
        nearest_column="measured_current_a",
        title="Preview Finite",
        yaxis_title="Current (A)",
    )

    trace_names = [trace.name for trace in figure.data]
    assert "Predicted Output" in trace_names
    assert "Support-Blended Output" in trace_names
