from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.analysis import analyze_measurements, combine_analysis_frames
from field_analysis.canonicalize import CanonicalizeConfig, canonicalize_batch
from field_analysis.models import CycleDetectionConfig, PreprocessConfig
from field_analysis.parser import parse_measurement_file
from field_analysis.recommendation_service import (
    DEFAULT_RECOMMENDATION_POLICY_CONFIG,
    LegacyRecommendationContext,
    RecommendationOptions,
    TargetRequest,
    build_finite_support_entries,
    recommend,
)
from field_analysis.schema_config import load_schema_config


CURRENT_CHANNEL = "i_sum_signed"
FIELD_CHANNEL = "bz_mT"
CONTINUOUS_DIR = REPO_ROOT.parent / "outputs" / "field_analysis_app_state" / "uploads" / "continuous"
TRANSIENT_DIR = REPO_ROOT.parent / "outputs" / "field_analysis_app_state" / "uploads" / "transient"
OUTPUT_JSON = REPO_ROOT / "artifacts" / "policy_eval" / "finite_generalization_stage2.json"
OUTPUT_MD = REPO_ROOT / "artifacts" / "policy_eval" / "finite_generalization_stage2.md"


@dataclass(slots=True)
class DatasetBundle:
    parsed: list[Any]
    canonical_runs: list[Any]
    analyses: list[Any]
    per_test_summary: pd.DataFrame
    analysis_lookup: dict[str, Any]


def _load_dataset(directory: Path, *, regime: str, expected_cycles: int = 10) -> DatasetBundle:
    schema = load_schema_config(None)
    parsed_measurements = []
    for path in sorted(directory.rglob("*.csv")):
        source_name = path.relative_to(directory).as_posix()
        parsed_measurements.extend(
            parse_measurement_file(
                source_name,
                path.read_bytes(),
                schema=schema,
                expected_cycles=expected_cycles,
                target_current_mode="auto",
            )
        )
    canonical_runs = canonicalize_batch(
        parsed_measurements,
        regime=regime,
        role="train",
        config=CanonicalizeConfig(preferred_field_axis=FIELD_CHANNEL),
    )
    analyses = analyze_measurements(
        parsed_measurements,
        canonical_runs=canonical_runs,
        preprocess_config=PreprocessConfig(),
        cycle_config=CycleDetectionConfig(expected_cycles=expected_cycles, reference_channel=CURRENT_CHANNEL),
        current_channel=CURRENT_CHANNEL,
        main_field_axis=FIELD_CHANNEL,
    )
    _, per_test_summary, _ = combine_analysis_frames(analyses, field_axis=FIELD_CHANNEL)
    analysis_lookup: dict[str, Any] = {}
    for analysis in analyses:
        if analysis.per_test_summary.empty:
            continue
        test_id = str(analysis.per_test_summary.iloc[0]["test_id"])
        analysis_lookup[test_id] = analysis
    return DatasetBundle(
        parsed=parsed_measurements,
        canonical_runs=canonical_runs,
        analyses=analyses,
        per_test_summary=per_test_summary,
        analysis_lookup=analysis_lookup,
    )


def _centered_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    a_centered = a - np.nanmean(a)
    b_centered = b - np.nanmean(b)
    denom = np.linalg.norm(a_centered) * np.linalg.norm(b_centered)
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    return float(np.dot(a_centered, b_centered) / denom)


def _phase_lag_seconds(reference: np.ndarray, candidate: np.ndarray, dt: float) -> float:
    if len(reference) < 2 or len(candidate) < 2 or not np.isfinite(dt) or dt <= 0:
        return float("nan")
    ref = reference - np.nanmean(reference)
    cand = candidate - np.nanmean(candidate)
    corr = np.correlate(cand, ref, mode="full")
    lag_index = int(np.argmax(corr) - (len(reference) - 1))
    return float(lag_index * dt)


def _signal_metrics(actual: np.ndarray, predicted: np.ndarray, dt: float) -> dict[str, float]:
    finite_mask = np.isfinite(actual) & np.isfinite(predicted)
    if finite_mask.sum() < 8:
        return {"shape_corr": float("nan"), "nrmse": float("nan"), "phase_lag_s": float("nan")}
    actual = actual[finite_mask]
    predicted = predicted[finite_mask]
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    span = float(np.nanmax(actual) - np.nanmin(actual))
    nrmse = float(rmse / span) if span > 0 else float("nan")
    return {
        "shape_corr": _centered_corr(actual, predicted),
        "nrmse": nrmse,
        "phase_lag_s": _phase_lag_seconds(actual, predicted, dt),
    }


def _resample_actual_signal(active_frame: pd.DataFrame, *, target_time_s: np.ndarray) -> np.ndarray:
    actual_time = pd.to_numeric(active_frame["time_s"], errors="coerce").to_numpy(dtype=float)
    actual_signal = pd.to_numeric(active_frame[CURRENT_CHANNEL], errors="coerce").to_numpy(dtype=float)
    finite_mask = np.isfinite(actual_time) & np.isfinite(actual_signal)
    actual_time = actual_time[finite_mask]
    actual_signal = actual_signal[finite_mask]
    if len(actual_time) < 2:
        return np.full_like(target_time_s, np.nan, dtype=float)
    actual_time = actual_time - float(np.nanmin(actual_time))
    overlap_end = min(float(np.nanmax(target_time_s)), float(np.nanmax(actual_time)))
    valid_target = target_time_s <= overlap_end
    resampled = np.full_like(target_time_s, np.nan, dtype=float)
    if np.any(valid_target):
        resampled[valid_target] = np.interp(target_time_s[valid_target], actual_time, actual_signal)
    return resampled


def _build_transient_case_records(dataset: DatasetBundle) -> list[dict[str, Any]]:
    support_entries = build_finite_support_entries(
        transient_measurements=dataset.parsed,
        transient_preprocess_results=[analysis.preprocess for analysis in dataset.analyses],
        transient_canonical_runs=dataset.canonical_runs,
        current_channel=CURRENT_CHANNEL,
        field_channel=FIELD_CHANNEL,
    )
    records: list[dict[str, Any]] = []
    for parsed, canonical_run, analysis, support_entry in zip(
        dataset.parsed,
        dataset.canonical_runs,
        dataset.analyses,
        support_entries,
        strict=False,
    ):
        waveform = str(support_entry.get("waveform_type") or canonical_run.command_waveform or "")
        freq_hz = float(support_entry.get("freq_hz", np.nan))
        cycle_count = float(support_entry.get("requested_cycle_count", np.nan))
        current_pp = float(support_entry.get("requested_current_pp", support_entry.get("current_pp", np.nan)))
        if waveform.lower() != "sine":
            continue
        if not (np.isfinite(freq_hz) and np.isfinite(cycle_count) and np.isfinite(current_pp)):
            continue
        records.append(
            {
                "source_file": parsed.source_file,
                "freq_hz": freq_hz,
                "cycle_count": cycle_count,
                "current_pp": current_pp,
                "support_entry": support_entry,
                "canonical_run": canonical_run,
                "analysis": analysis,
            }
        )
    return records


def _build_reduced_legacy_context(
    continuous: DatasetBundle,
    transient: DatasetBundle,
    *,
    holdout_source_file: str,
) -> tuple[list[Any], LegacyRecommendationContext]:
    keep_mask = [parsed.source_file != holdout_source_file for parsed in transient.parsed]
    reduced_parsed = [parsed for parsed, keep in zip(transient.parsed, keep_mask, strict=False) if keep]
    reduced_analyses = [analysis for analysis, keep in zip(transient.analyses, keep_mask, strict=False) if keep]
    reduced_canonical = [run for run, keep in zip(transient.canonical_runs, keep_mask, strict=False) if keep]
    legacy_context = LegacyRecommendationContext(
        per_test_summary=continuous.per_test_summary,
        analysis_lookup=continuous.analysis_lookup,
        transient_measurements=reduced_parsed,
        transient_preprocess_results=[analysis.preprocess for analysis in reduced_analyses],
        transient_canonical_runs=reduced_canonical,
        validation_measurements=[],
        validation_preprocess_results=[],
    )
    return reduced_canonical, legacy_context


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate finite sine preview quality via leave-one-recipe-out.")
    parser.add_argument("--max-cases", type=int, default=0, help="Optional max case count for faster local runs.")
    args = parser.parse_args()

    continuous = _load_dataset(CONTINUOUS_DIR, regime="continuous", expected_cycles=10)
    transient = _load_dataset(TRANSIENT_DIR, regime="transient", expected_cycles=10)
    transient_records = _build_transient_case_records(transient)
    if args.max_cases > 0:
        transient_records = transient_records[: args.max_cases]

    options = RecommendationOptions(current_channel=CURRENT_CHANNEL, field_channel=FIELD_CHANNEL)
    policy_config = DEFAULT_RECOMMENDATION_POLICY_CONFIG

    case_rows: list[dict[str, Any]] = []
    for record in transient_records:
        reduced_transient_runs, legacy_context = _build_reduced_legacy_context(
            continuous,
            transient,
            holdout_source_file=str(record["source_file"]),
        )
        target = TargetRequest(
            regime="transient",
            target_waveform="sine",
            command_waveform="sine",
            freq_hz=float(record["freq_hz"]),
            commanded_cycles=float(record["cycle_count"]),
            target_type="current",
            target_level_value=float(record["current_pp"]),
            target_level_kind="pp",
            context={
                "request_kind": "waveform_compensation",
                "frequency_mode": "exact",
                "finite_cycle_mode": True,
                "preview_tail_cycles": 0.25,
            },
        )
        result = recommend(
            continuous_runs=continuous.canonical_runs,
            transient_runs=reduced_transient_runs,
            validation_runs=[],
            target=target,
            options=options,
            legacy_context=legacy_context,
            policy_config=policy_config,
        )

        command_profile = getattr(result, "command_profile", None)
        metrics = {"shape_corr": float("nan"), "nrmse": float("nan"), "phase_lag_s": float("nan")}
        if command_profile is not None and not command_profile.empty:
            time_s = pd.to_numeric(command_profile["time_s"], errors="coerce").to_numpy(dtype=float)
            modeled = pd.to_numeric(
                command_profile["modeled_current_a"]
                if "modeled_current_a" in command_profile.columns
                else command_profile["modeled_output"],
                errors="coerce",
            ).to_numpy(dtype=float)
            actual = _resample_actual_signal(record["support_entry"]["active_frame"], target_time_s=time_s)
            dt = float(np.nanmedian(np.diff(time_s))) if len(time_s) >= 2 else float("nan")
            metrics = _signal_metrics(actual, modeled, dt)

        case_rows.append(
            {
                "source_file": str(record["source_file"]),
                "freq_hz": float(record["freq_hz"]),
                "cycle_count": float(record["cycle_count"]),
                "target_current_pp": float(record["current_pp"]),
                "preview_only": bool(getattr(result, "preview_only", False)),
                "allow_auto_download": bool(getattr(result, "allow_auto_download", False)),
                "support_state": str(getattr(result, "engine_summary", {}).get("support_state", "unknown")),
                "policy_reasons": list(getattr(result, "validation_report", None).reasons if getattr(result, "validation_report", None) else []),
                "shape_corr": metrics["shape_corr"],
                "nrmse": metrics["nrmse"],
                "phase_lag_s": metrics["phase_lag_s"],
            }
        )

    frame = pd.DataFrame(case_rows)
    preview_frame = frame[frame["preview_only"]].copy()
    payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "policy_version": policy_config.version,
        "case_count": int(len(frame)),
        "preview_case_count": int(len(preview_frame)),
        "summary": {
            "mean_shape_corr": float(pd.to_numeric(preview_frame["shape_corr"], errors="coerce").mean()) if not preview_frame.empty else float("nan"),
            "mean_nrmse": float(pd.to_numeric(preview_frame["nrmse"], errors="coerce").mean()) if not preview_frame.empty else float("nan"),
            "mean_phase_lag_s": float(pd.to_numeric(preview_frame["phase_lag_s"], errors="coerce").mean()) if not preview_frame.empty else float("nan"),
        },
        "cases": case_rows,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Finite Generalization Stage 2",
        "",
        "- mode: `leave-one-recipe-out finite sine preview quality`",
        f"- cases: `{payload['case_count']}`",
        f"- preview cases: `{payload['preview_case_count']}`",
        f"- mean shape corr: `{payload['summary']['mean_shape_corr']:.4f}`",
        f"- mean NRMSE: `{payload['summary']['mean_nrmse']:.4f}`",
        f"- mean phase lag: `{payload['summary']['mean_phase_lag_s']:.6f}` s",
        "",
        "| source_file | freq_hz | cycle_count | target_current_pp | preview_only | shape_corr | nrmse | phase_lag_s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in case_rows[:20]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["source_file"]),
                    f"{row['freq_hz']:.3f}",
                    f"{row['cycle_count']:.2f}",
                    f"{row['target_current_pp']:.2f}",
                    str(row["preview_only"]),
                    f"{row['shape_corr']:.4f}" if np.isfinite(row["shape_corr"]) else "nan",
                    f"{row['nrmse']:.4f}" if np.isfinite(row["nrmse"]) else "nan",
                    f"{row['phase_lag_s']:.6f}" if np.isfinite(row["phase_lag_s"]) else "nan",
                ]
            )
            + " |"
        )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
