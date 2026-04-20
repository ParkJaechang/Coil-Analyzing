from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
FIELD_ANALYSIS_SRC = ROOT.parent / "src"
if str(FIELD_ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(FIELD_ANALYSIS_SRC))

from field_analysis.canonicalize import CanonicalizeConfig, canonicalize_run
from field_analysis.models import (
    CycleDetectionResult,
    DatasetAnalysis,
    ParsedMeasurement,
    PreprocessResult,
    SheetPreview,
)
from field_analysis.recommendation_service import (
    LegacyRecommendationContext,
    RecommendationOptions,
    TargetRequest,
    recommend,
)
from field_analysis.shape_amplitude_split import (
    build_amplitude_lut_audit,
    build_shape_engine_audit,
    build_support_route_level_influence,
)


def _normalized_waveform(phase: np.ndarray | float, waveform_type: str) -> np.ndarray:
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


def _build_analysis(
    *,
    test_id: str,
    level_pp: float,
    waveform_type: str = "sine",
    freq_hz: float = 0.5,
    field_gain: float = 2.0,
) -> tuple[pd.DataFrame, DatasetAnalysis]:
    phase = np.linspace(0.0, 1.0, 32)
    rows: list[dict[str, float]] = []
    for cycle_index in range(4):
        cycle_time = phase * (1.0 / freq_hz)
        waveform = _normalized_waveform(phase, waveform_type)
        for cycle_progress, cycle_time_s, sample in zip(phase, cycle_time, waveform, strict=False):
            rows.append(
                {
                    "cycle_index": float(cycle_index),
                    "cycle_progress": float(cycle_progress),
                    "cycle_time_s": float(cycle_time_s),
                    "time_s": float(cycle_time_s + cycle_index * (1.0 / freq_hz)),
                    "freq_hz": freq_hz,
                    "daq_input_v": (level_pp / 4.0) * sample,
                    "i_sum_signed": (level_pp / 2.0) * sample,
                    "bz_mT": (level_pp * field_gain / 2.0) * sample,
                    "current_pp_target_a": level_pp,
                    "current_pk_target_a": level_pp / 2.0,
                    "waveform_type": waveform_type,
                    "cycle_total_expected": 4.0,
                    "source_cycle_no": np.nan,
                    "test_id": test_id,
                }
            )
    frame = pd.DataFrame(rows)
    per_test_summary = pd.DataFrame(
        [
            {
                "test_id": test_id,
                "waveform_type": waveform_type,
                "freq_hz": freq_hz,
                "current_pp_target_a": level_pp,
                "achieved_current_pp_a_mean": level_pp,
                "achieved_bz_mT_pp_mean": level_pp * field_gain,
                "achieved_bmag_mT_pp_mean": level_pp * field_gain,
                "daq_input_v_pp_mean": level_pp / 2.0,
                "amp_gain_setting_mean": 50.0,
            }
        ]
    )
    preview = SheetPreview("main", 0, 0, [], 0, {}, [], {}, [], [])
    parsed = ParsedMeasurement(
        source_file=f"{test_id}.csv",
        file_type="csv",
        sheet_name="main",
        structure_preview=preview,
        metadata={"waveform": waveform_type, "cycle": "4"},
        mapping={},
        raw_frame=frame.copy(),
        normalized_frame=frame.copy(),
        warnings=[],
        logs=[],
    )
    preprocess = PreprocessResult(
        corrected_frame=frame.copy(),
        offsets={},
        lags=[],
        warnings=[],
        logs=[],
    )
    cycle_detection = CycleDetectionResult(
        annotated_frame=frame.copy(),
        boundaries=[],
        estimated_period_s=1.0 / freq_hz,
        estimated_frequency_hz=freq_hz,
        reference_channel="daq_input_v",
        warnings=[],
        logs=[],
    )
    analysis = DatasetAnalysis(
        parsed=parsed,
        preprocess=preprocess,
        cycle_detection=cycle_detection,
        per_cycle_summary=pd.DataFrame(),
        per_test_summary=per_test_summary.copy(),
        warnings=[],
    )
    return per_test_summary, analysis


def test_size_lut_debug_payload_exposes_shape_amplitude_split_fields() -> None:
    summary_10, analysis_10 = _build_analysis(test_id="size_lut_10", level_pp=10.0)
    summary_20, analysis_20 = _build_analysis(test_id="size_lut_20", level_pp=20.0)
    canonical = canonicalize_run(analysis_10.parsed, regime="continuous", role="train", config=CanonicalizeConfig())

    result = recommend(
        continuous_runs=[canonical],
        transient_runs=[],
        validation_runs=[],
        target=TargetRequest(
            regime="continuous",
            target_waveform="sine",
            command_waveform="sine",
            freq_hz=0.5,
            target_type="field",
            target_level_value=20.0,
            target_level_kind="pp",
            context={
                "request_kind": "size_lut",
                "target_metric": "achieved_bz_mT_pp_mean",
                "target_value": 40.0,
                "frequency_mode": "exact",
                "finite_cycle_mode": False,
            },
        ),
        options=RecommendationOptions(current_channel="i_sum_signed", field_channel="bz_mT"),
        legacy_context=LegacyRecommendationContext(
            per_test_summary=pd.concat([summary_10, summary_20], ignore_index=True),
            analysis_lookup={"size_lut_10": analysis_10, "size_lut_20": analysis_20},
        ),
    )

    assert result.debug_info["amplitude_lut_meaning"] == "mixed"
    assert result.debug_info["shape_engine_source"] == "measured_template_waveform"
    assert result.debug_info["amplitude_engine_source"] == "scalar_voltage_lut"
    assert result.debug_info["pp_affects_shape"] is True


def test_amplitude_lut_audit_reports_mixed_implementation_but_amplitude_only_observation() -> None:
    summary_10, analysis_10 = _build_analysis(test_id="lut_audit_10", level_pp=10.0)
    summary_20, analysis_20 = _build_analysis(test_id="lut_audit_20", level_pp=20.0)
    audit = build_amplitude_lut_audit(
        per_test_summary=pd.concat([summary_10, summary_20], ignore_index=True),
        analyses_by_test_id={"lut_audit_10": analysis_10, "lut_audit_20": analysis_20},
    )

    assert audit["amplitude_lut_meaning"] == "mixed"
    assert audit["observed_behavior_classification"] == "amplitude_only_candidate"
    assert audit["groups"]
    assert audit["groups"][0]["implementation_meaning"] == "mixed"
    assert audit["groups"][0]["observed_shape_meaning"] == "amplitude_only"


def test_shape_engine_audit_marks_identical_bz_shapes_as_pp_stable() -> None:
    summary_10, analysis_10 = _build_analysis(test_id="shape_10", level_pp=10.0)
    summary_20, analysis_20 = _build_analysis(test_id="shape_20", level_pp=20.0)
    audit = build_shape_engine_audit(
        analyses_by_test_id={"shape_10": analysis_10, "shape_20": analysis_20},
        per_test_summary=pd.concat([summary_10, summary_20], ignore_index=True),
        finite_entries=[],
    )

    assert audit["groups"]
    assert audit["groups"][0]["shape_engine_source"] == "normalized_exact_support_mean"
    assert audit["groups"][0]["pp_affects_shape"] is False


def test_support_route_level_influence_detects_prediction_source_switch() -> None:
    payload = build_support_route_level_influence(
        probe_groups={
            "continuous_field_exact": [
                {
                    "target_level_pp": 20.0,
                    "selected_support_id": "support_a",
                    "field_prediction_source": "current_to_bz_surrogate",
                    "field_prediction_status": "available",
                    "shape_engine_source": "harmonic_surface_model",
                    "amplitude_engine_source": "inverse_voltage_scaling",
                    "predicted_shape_corr": 0.95,
                    "predicted_clipping": False,
                    "within_hardware_limits": True,
                },
                {
                    "target_level_pp": 40.0,
                    "selected_support_id": "support_a",
                    "field_prediction_source": "exact_field_direct",
                    "field_prediction_status": "available",
                    "shape_engine_source": "harmonic_surface_model",
                    "amplitude_engine_source": "inverse_voltage_scaling",
                    "predicted_shape_corr": 0.94,
                    "predicted_clipping": False,
                    "within_hardware_limits": True,
                },
            ]
        }
    )

    assert payload["probe_groups"][0]["pp_affects_shape"] is True
    assert "prediction_source_switch" in payload["probe_groups"][0]["reason_codes"]
