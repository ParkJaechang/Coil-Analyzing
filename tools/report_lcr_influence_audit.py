from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIELD_ANALYSIS_SRC = ROOT.parent / "src"
if str(FIELD_ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(FIELD_ANALYSIS_SRC))

from field_analysis.canonicalize import CanonicalizeConfig, canonicalize_run
from field_analysis.models import (  # noqa: E402
    CycleDetectionResult,
    DatasetAnalysis,
    ParsedMeasurement,
    PreprocessResult,
    SheetPreview,
)
from field_analysis.recommendation_service import (  # noqa: E402
    LegacyRecommendationContext,
    RecommendationOptions,
    TargetRequest,
    recommend,
)

ARTIFACT_DIR = ROOT / "artifacts" / "bz_first_exact_matrix"
OUTPUT_JSON = ARTIFACT_DIR / "lcr_influence_audit.json"
OUTPUT_MD = ARTIFACT_DIR / "lcr_influence_audit.md"


def _normalized_waveform(phase: np.ndarray, waveform_type: str) -> np.ndarray:
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


def _build_dummy_analysis(*, freq_hz: float, test_id: str) -> tuple[DatasetAnalysis, object]:
    waveform_type = "sine"
    phase = np.linspace(0.0, 1.0, 64)
    rows: list[dict[str, float]] = []
    for cycle_index in range(4):
        cycle_time = phase * (1.0 / freq_hz)
        for cycle_progress, cycle_time_s in zip(phase, cycle_time, strict=False):
            drive = _normalized_waveform(np.asarray([cycle_progress]), waveform_type)[0]
            rows.append(
                {
                    "cycle_index": float(cycle_index),
                    "cycle_progress": float(cycle_progress),
                    "cycle_time_s": float(cycle_time_s),
                    "time_s": float(cycle_time_s + cycle_index * (1.0 / freq_hz)),
                    "freq_hz": float(freq_hz),
                    "daq_input_v": 2.0 * drive,
                    "i_sum_signed": 5.0 * drive,
                    "bz_mT": 10.0 * _normalized_waveform(np.asarray([cycle_progress - 0.08]), waveform_type)[0],
                    "current_pp_target_a": 10.0,
                    "current_pk_target_a": 5.0,
                    "waveform_type": waveform_type,
                    "cycle_total_expected": 4.0,
                    "source_cycle_no": np.nan,
                }
            )
    annotated_frame = pd.DataFrame(rows)
    per_cycle_summary = pd.DataFrame(
        [
            {"cycle_index": idx, "achieved_current_pp_a": 10.0, "achieved_bz_mT_pp": 20.0, "daq_input_v_pp": 4.0}
            for idx in range(4)
        ]
    )
    per_test_summary = pd.DataFrame(
        [
            {
                "test_id": test_id,
                "waveform_type": waveform_type,
                "freq_hz": float(freq_hz),
                "current_pp_target_a": 10.0,
                "achieved_current_pp_a_mean": 10.0,
                "achieved_bz_mT_pp_mean": 20.0,
                "achieved_bmag_mT_pp_mean": 20.0,
                "daq_input_v_pp_mean": 4.0,
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
        per_cycle_summary=per_cycle_summary,
        per_test_summary=per_test_summary.copy(),
        warnings=[],
    )
    canonical = canonicalize_run(parsed, regime="continuous", role="train", config=CanonicalizeConfig())
    return analysis, canonical


def _build_context() -> tuple[list[object], LegacyRecommendationContext]:
    analyses: dict[str, DatasetAnalysis] = {}
    continuous_runs: list[object] = []
    summaries: list[pd.DataFrame] = []
    for freq_hz, test_id in ((0.5, "exact_0p5"), (1.0, "exact_1p0")):
        analysis, canonical = _build_dummy_analysis(freq_hz=freq_hz, test_id=test_id)
        analyses[test_id] = analysis
        continuous_runs.append(canonical)
        summaries.append(analysis.per_test_summary)
    return continuous_runs, LegacyRecommendationContext(
        per_test_summary=pd.concat(summaries, ignore_index=True),
        analysis_lookup=analyses,
    )


def _build_lcr_measurements() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "freq_hz": [0.25, 0.5, 1.0, 2.0, 3.0],
            "rs_ohm": [2.4, 2.5, 2.7, 3.0, 3.3],
            "ls_h": [0.012, 0.012, 0.011, 0.010, 0.009],
            "cs_f": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )


def _peak_to_peak(values: np.ndarray | None) -> float:
    if values is None:
        return float("nan")
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.nanmax(finite) - np.nanmin(finite))


def _run_case(*, freq_hz: float, frequency_mode: str) -> dict[str, object]:
    continuous_runs, legacy_context = _build_context()
    lcr_measurements = _build_lcr_measurements()
    target = TargetRequest(
        regime="continuous",
        target_waveform="sine",
        command_waveform="sine",
        freq_hz=float(freq_hz),
        target_type="field",
        target_level_value=20.0,
        target_level_kind="pp",
        context={
            "request_kind": "waveform_compensation",
            "frequency_mode": frequency_mode,
        },
    )
    common_options = dict(
        current_channel="i_sum_signed",
        field_channel="bz_mT",
        frequency_mode=frequency_mode,
        max_daq_voltage_pp=20.0,
        amp_gain_at_100_pct=20.0,
        amp_gain_limit_pct=100.0,
        amp_max_output_pk_v=180.0,
        default_support_amp_gain_pct=100.0,
    )
    result_without = recommend(
        continuous_runs=continuous_runs,
        transient_runs=[],
        validation_runs=[],
        target=target,
        options=RecommendationOptions(**common_options),
        legacy_context=legacy_context,
    )
    result_with = recommend(
        continuous_runs=continuous_runs,
        transient_runs=[],
        validation_runs=[],
        target=target,
        options=RecommendationOptions(
            **common_options,
            lcr_measurements=lcr_measurements,
            lcr_blend_weight=0.6,
        ),
        legacy_context=legacy_context,
    )
    predicted_pp_without = _peak_to_peak(result_without.predicted_bz_mT)
    predicted_pp_with = _peak_to_peak(result_with.predicted_bz_mT)
    target_pp = float(target.target_level_value or np.nan)
    debug_with = dict(result_with.debug_info)
    return {
        "request_freq_hz": float(freq_hz),
        "frequency_mode": frequency_mode,
        "request_route": result_with.engine_summary.get("request_route"),
        "solver_route": result_with.engine_summary.get("solver_route"),
        "lcr_usage_mode": debug_with.get("lcr_usage_mode"),
        "lcr_weight": debug_with.get("lcr_blend_weight"),
        "requested_lcr_weight": debug_with.get("requested_lcr_weight"),
        "exact_field_support_present": bool(debug_with.get("exact_field_support_present")),
        "lcr_phase_anchor_used": bool(debug_with.get("lcr_phase_anchor_used")),
        "lcr_gain_prior_used": bool(debug_with.get("lcr_gain_prior_used")),
        "predicted_metric_name": "predicted_bz_pp_delta_mT",
        "predicted_metric_delta_with_lcr": (
            float(predicted_pp_with - target_pp) if np.isfinite(predicted_pp_with) and np.isfinite(target_pp) else None
        ),
        "predicted_metric_delta_without_lcr": (
            float(predicted_pp_without - target_pp) if np.isfinite(predicted_pp_without) and np.isfinite(target_pp) else None
        ),
        "predicted_bz_pp_with_lcr": float(predicted_pp_with) if np.isfinite(predicted_pp_with) else None,
        "predicted_bz_pp_without_lcr": float(predicted_pp_without) if np.isfinite(predicted_pp_without) else None,
        "used_lcr_prior": bool(debug_with.get("used_lcr_prior", False)),
    }


def build_lcr_influence_audit() -> dict[str, object]:
    exact_case = _run_case(freq_hz=0.5, frequency_mode="exact")
    preview_case = _run_case(freq_hz=0.75, frequency_mode="interpolate")
    records = [
        {"scenario_id": "continuous_field_exact_support", **exact_case},
        {"scenario_id": "continuous_field_preview_gap", **preview_case},
    ]
    success = (
        exact_case["lcr_usage_mode"] == "audit_only"
        and float(exact_case["lcr_weight"] or 0.0) == 0.0
        and preview_case["lcr_usage_mode"] == "weak_prior"
        and float(preview_case["lcr_weight"] or 0.0) <= 0.15
    )
    return {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "success": bool(success),
        "records": records,
        "summary": {
            "exact_field_mode": exact_case["lcr_usage_mode"],
            "exact_field_weight": exact_case["lcr_weight"],
            "preview_gap_mode": preview_case["lcr_usage_mode"],
            "preview_gap_weight": preview_case["lcr_weight"],
        },
    }


def render_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# LCR Influence Audit",
        "",
        f"- success: `{payload.get('success')}`",
        f"- exact field mode: `{payload.get('summary', {}).get('exact_field_mode')}`",
        f"- preview gap mode: `{payload.get('summary', {}).get('preview_gap_mode')}`",
        "",
    ]
    for record in payload.get("records", []):
        lines.extend(
            [
                f"## {record['scenario_id']}",
                "",
                f"- route: `{record['request_route']}` / `{record['solver_route']}`",
                f"- lcr_usage_mode: `{record['lcr_usage_mode']}`",
                f"- lcr_weight: `{record['lcr_weight']}` (requested `{record['requested_lcr_weight']}`)",
                f"- exact_field_support_present: `{record['exact_field_support_present']}`",
                f"- lcr_phase_anchor_used: `{record['lcr_phase_anchor_used']}`",
                f"- lcr_gain_prior_used: `{record['lcr_gain_prior_used']}`",
                f"- predicted_metric_delta_with_lcr: `{record['predicted_metric_delta_with_lcr']}`",
                f"- predicted_metric_delta_without_lcr: `{record['predicted_metric_delta_without_lcr']}`",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_lcr_influence_audit()
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
