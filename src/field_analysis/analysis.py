from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np
import pandas as pd

from .canonical_runs import CanonicalRun
from .compensation import build_representative_cycle_profile
from .cycle_detection import detect_cycles
from .metrics import FIELD_AXES, apply_reference_normalization, build_coverage_matrix, compute_cycle_and_test_metrics
from .models import CycleDetectionConfig, DatasetAnalysis, ParsedMeasurement, PreprocessConfig
from .preprocessing import apply_preprocessing
from .segmentation import build_analysis_frame_from_canonical, segment_canonical_run
from .utils import canonicalize_waveform_type


def analyze_measurements(
    parsed_measurements: Iterable[ParsedMeasurement],
    preprocess_config: PreprocessConfig,
    cycle_config: CycleDetectionConfig,
    current_channel: str,
    main_field_axis: str,
    canonical_runs: Iterable[CanonicalRun] | None = None,
) -> list[DatasetAnalysis]:
    """Run the full analysis pipeline for each parsed measurement."""

    parsed_measurements = list(parsed_measurements)
    canonical_run_list = list(canonical_runs) if canonical_runs is not None else []
    analyses: list[DatasetAnalysis] = []
    for index, parsed in enumerate(parsed_measurements):
        canonical_run = canonical_run_list[index] if index < len(canonical_run_list) else None
        analysis_frame = (
            build_analysis_frame_from_canonical(parsed, canonical_run)
            if canonical_run is not None
            else parsed.normalized_frame
        )
        preprocess_result = apply_preprocessing(analysis_frame, preprocess_config)
        if canonical_run is not None:
            cycle_result = segment_canonical_run(
                canonical_run,
                preprocess_result.corrected_frame,
                replace(cycle_config, reference_channel=cycle_config.reference_channel or current_channel),
            )
        else:
            cycle_result = detect_cycles(
                preprocess_result.corrected_frame,
                replace(cycle_config, reference_channel=cycle_config.reference_channel or current_channel),
            )
        per_cycle, per_test = compute_cycle_and_test_metrics(
            annotated_frame=cycle_result.annotated_frame,
            current_channel=current_channel,
            main_field_axis=main_field_axis,
        )

        analysis_warnings = parsed.warnings + preprocess_result.warnings + cycle_result.warnings
        analyses.append(
            DatasetAnalysis(
                parsed=parsed,
                preprocess=preprocess_result,
                cycle_detection=cycle_result,
                per_cycle_summary=per_cycle,
                per_test_summary=per_test,
                warnings=list(dict.fromkeys(analysis_warnings)),
            )
        )
    return analyses


def combine_analysis_frames(
    analyses: Iterable[DatasetAnalysis],
    reference_test_id: str | None = None,
    field_axis: str = "bz_mT",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stack all cycle/test summaries and derive the coverage matrix."""

    analyses = list(analyses)
    per_cycle_frames = [analysis.per_cycle_summary for analysis in analyses if not analysis.per_cycle_summary.empty]
    per_test_frames = [analysis.per_test_summary for analysis in analyses if not analysis.per_test_summary.empty]

    per_cycle = pd.concat(per_cycle_frames, ignore_index=True) if per_cycle_frames else pd.DataFrame()
    per_test = pd.concat(per_test_frames, ignore_index=True) if per_test_frames else pd.DataFrame()
    per_test = apply_reference_normalization(per_test, reference_test_id=reference_test_id, field_axis=field_axis)
    coverage = build_coverage_matrix(per_test)
    return per_cycle, per_test, coverage


def build_warning_table(analyses: Iterable[DatasetAnalysis]) -> pd.DataFrame:
    """Collect file/test-level warnings into one table for the UI."""

    rows: list[dict[str, str]] = []
    for analysis in analyses:
        for warning in analysis.warnings:
            rows.append(
                {
                    "source_file": analysis.parsed.source_file,
                    "sheet_name": analysis.parsed.sheet_name,
                    "test_id": analysis.parsed.normalized_frame["test_id"].iloc[0],
                    "warning": warning,
                }
            )
    return pd.DataFrame(rows)


def build_shape_phase_comparison(
    analyses: Iterable[DatasetAnalysis],
    waveform_type: str,
    freq_hz: float,
    signal_channel: str,
    reference_test_id: str | None = None,
    current_channel: str = "i_sum_signed",
    main_field_axis: str = "bz_mT",
    normalization_mode: str = "peak_to_peak",
    points_per_cycle: int = 256,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare normalized one-cycle shapes and relative phase at one waveform/frequency."""

    waveform_type = canonicalize_waveform_type(waveform_type)
    if waveform_type is None:
        return pd.DataFrame(), pd.DataFrame()

    candidate_entries: list[dict[str, object]] = []
    for analysis in analyses:
        if analysis.per_test_summary.empty:
            continue
        summary_row = analysis.per_test_summary.iloc[0]
        summary_waveform = canonicalize_waveform_type(summary_row.get("waveform_type"))
        if summary_waveform != waveform_type:
            continue
        if not np.isclose(float(summary_row.get("freq_hz", np.nan)), float(freq_hz), equal_nan=False):
            continue

        profile, value_column = _build_comparison_profile(
            analysis=analysis,
            signal_channel=signal_channel,
            current_channel=current_channel,
            main_field_axis=main_field_axis,
            points_per_cycle=points_per_cycle,
        )
        if profile.empty or value_column not in profile.columns:
            continue

        signal = pd.to_numeric(profile[value_column], errors="coerce").to_numpy(dtype=float)
        normalized_signal, centered_signal, signal_mean, signal_scale, signal_pp = _normalize_cycle_signal(
            signal,
            normalization_mode=normalization_mode,
        )
        if not np.isfinite(normalized_signal).any():
            continue

        period_s = float(profile["time_s"].max()) if "time_s" in profile.columns and len(profile) > 1 else (
            1.0 / float(freq_hz) if float(freq_hz) > 0 else np.nan
        )
        current_target = float(summary_row.get("current_pp_target_a", np.nan))
        current_achieved = float(summary_row.get("achieved_current_pp_a_mean", np.nan))
        legend_label = _build_shape_legend_label(summary_row)

        candidate_entries.append(
            {
                "test_id": str(summary_row["test_id"]),
                "legend_label": legend_label,
                "waveform_type": summary_waveform or str(summary_row["waveform_type"]),
                "freq_hz": float(summary_row["freq_hz"]),
                "current_pp_target_a": current_target,
                "achieved_current_pp_a_mean": current_achieved,
                "profile": profile[["cycle_progress", "time_s"]].copy(),
                "signal": signal,
                "signal_centered": centered_signal,
                "normalized_signal": normalized_signal,
                "signal_mean": signal_mean,
                "signal_scale": signal_scale,
                "signal_pp": signal_pp,
                "period_s": period_s,
            }
        )

    if not candidate_entries:
        return pd.DataFrame(), pd.DataFrame()

    candidate_entries.sort(
        key=lambda entry: (
            np.nan_to_num(float(entry["current_pp_target_a"]), nan=np.inf),
            np.nan_to_num(float(entry["achieved_current_pp_a_mean"]), nan=np.inf),
            str(entry["test_id"]),
        )
    )

    reference_entry = _select_reference_entry(candidate_entries, reference_test_id)
    reference_signal = np.asarray(reference_entry["normalized_signal"], dtype=float)
    reference_test_id = str(reference_entry["test_id"])

    overlay_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for entry in candidate_entries:
        normalized_signal = np.asarray(entry["normalized_signal"], dtype=float)
        alignment_shift = _best_alignment_shift(reference_signal, normalized_signal)
        aligned_signal = np.roll(normalized_signal, alignment_shift)
        phase_lag_samples = int(-alignment_shift)
        phase_lag_fraction = phase_lag_samples / max(len(normalized_signal), 1)
        phase_lag_degrees = phase_lag_fraction * 360.0
        phase_lag_seconds = phase_lag_fraction * float(entry["period_s"]) if np.isfinite(entry["period_s"]) else np.nan

        raw_corr = _safe_corr(reference_signal, normalized_signal)
        aligned_corr = _safe_corr(reference_signal, aligned_signal)
        raw_nrmse = _normalized_rmse(reference_signal, normalized_signal)
        aligned_nrmse = _normalized_rmse(reference_signal, aligned_signal)

        profile_frame = entry["profile"].copy()
        profile_frame["test_id"] = entry["test_id"]
        profile_frame["legend_label"] = entry["legend_label"]
        profile_frame["waveform_type"] = entry["waveform_type"]
        profile_frame["freq_hz"] = entry["freq_hz"]
        profile_frame["current_pp_target_a"] = entry["current_pp_target_a"]
        profile_frame["achieved_current_pp_a_mean"] = entry["achieved_current_pp_a_mean"]
        profile_frame["signal_raw"] = entry["signal"]
        profile_frame["signal_centered"] = entry["signal_centered"]
        profile_frame["normalized_signal_raw"] = normalized_signal
        profile_frame["normalized_signal_aligned"] = aligned_signal
        profile_frame["phase_lag_deg"] = phase_lag_degrees
        profile_frame["phase_lag_seconds"] = phase_lag_seconds
        profile_frame["shape_corr_aligned"] = aligned_corr
        profile_frame["shape_nrmse_aligned"] = aligned_nrmse
        profile_frame["is_reference"] = entry["test_id"] == reference_test_id
        overlay_rows.append(profile_frame)

        summary_rows.append(
            {
                "test_id": entry["test_id"],
                "legend_label": entry["legend_label"],
                "waveform_type": entry["waveform_type"],
                "freq_hz": entry["freq_hz"],
                "current_pp_target_a": entry["current_pp_target_a"],
                "achieved_current_pp_a_mean": entry["achieved_current_pp_a_mean"],
                "signal_channel": signal_channel,
                "normalization_mode": normalization_mode,
                "reference_test_id": reference_test_id,
                "is_reference": entry["test_id"] == reference_test_id,
                "signal_mean": entry["signal_mean"],
                "signal_scale": entry["signal_scale"],
                "signal_pp": entry["signal_pp"],
                "phase_lag_samples": phase_lag_samples,
                "phase_lag_fraction": phase_lag_fraction,
                "phase_lag_deg": phase_lag_degrees,
                "phase_lag_seconds": phase_lag_seconds,
                "shape_corr_raw": raw_corr,
                "shape_corr_aligned": aligned_corr,
                "shape_nrmse_raw": raw_nrmse,
                "shape_nrmse_aligned": aligned_nrmse,
            }
        )

    overlay_frame = pd.concat(overlay_rows, ignore_index=True) if overlay_rows else pd.DataFrame()
    summary_frame = pd.DataFrame(summary_rows).sort_values(
        ["current_pp_target_a", "achieved_current_pp_a_mean", "test_id"],
        na_position="last",
    ).reset_index(drop=True)
    return overlay_frame, summary_frame


def _build_comparison_profile(
    analysis: DatasetAnalysis,
    signal_channel: str,
    current_channel: str,
    main_field_axis: str,
    points_per_cycle: int,
) -> tuple[pd.DataFrame, str]:
    field_channels = set(FIELD_AXES)
    voltage_channel = "daq_input_v"
    comparison_current = current_channel
    comparison_field = main_field_axis
    profile_value_column = "measured_current_a"

    if signal_channel in field_channels:
        comparison_field = signal_channel
        profile_value_column = "measured_field_mT"
    elif signal_channel == "daq_input_v":
        voltage_channel = signal_channel
        profile_value_column = "command_voltage_v"
    else:
        comparison_current = signal_channel
        profile_value_column = "measured_current_a"

    profile = build_representative_cycle_profile(
        analysis=analysis,
        current_channel=comparison_current,
        voltage_channel=voltage_channel,
        field_channel=comparison_field,
        points_per_cycle=points_per_cycle,
    )
    return profile, profile_value_column


def _normalize_cycle_signal(
    signal: np.ndarray,
    normalization_mode: str,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    centered = np.asarray(signal, dtype=float) - float(np.nanmean(signal))
    signal_pp = float(np.nanmax(centered) - np.nanmin(centered)) if np.isfinite(centered).any() else np.nan

    if normalization_mode == "rms":
        scale = float(np.sqrt(np.nanmean(np.square(centered)))) if np.isfinite(centered).any() else np.nan
    elif normalization_mode == "peak":
        scale = float(np.nanmax(np.abs(centered))) if np.isfinite(centered).any() else np.nan
    else:
        scale = signal_pp / 2.0 if np.isfinite(signal_pp) else np.nan

    if not np.isfinite(scale) or scale <= 0:
        normalized = np.full_like(centered, np.nan, dtype=float)
    else:
        normalized = centered / scale
    return normalized, centered, float(np.nanmean(signal)), scale, signal_pp


def _select_reference_entry(
    candidate_entries: list[dict[str, object]],
    reference_test_id: str | None,
) -> dict[str, object]:
    if reference_test_id:
        for entry in candidate_entries:
            if entry["test_id"] == reference_test_id:
                return entry
    return candidate_entries[0]


def _build_shape_legend_label(summary_row: pd.Series) -> str:
    target_value = summary_row.get("current_pp_target_a", np.nan)
    if pd.notna(target_value):
        return f"{float(target_value):g} App | {summary_row['test_id']}"
    return str(summary_row["test_id"])


def _best_alignment_shift(reference: np.ndarray, candidate: np.ndarray) -> int:
    valid = np.isfinite(reference) & np.isfinite(candidate)
    if valid.sum() < 8:
        return 0
    reference_valid = reference[valid]
    candidate_valid = candidate[valid]
    max_shift = max(1, len(reference_valid) // 4)
    best_shift = 0
    best_score = -np.inf
    for shift in range(-max_shift, max_shift + 1):
        shifted = np.roll(candidate_valid, shift)
        score = _safe_corr(reference_valid, shifted)
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_shift = shift
    return int(best_shift)


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return float("nan")
    left_valid = left[valid]
    right_valid = right[valid]
    if np.allclose(np.nanstd(left_valid), 0.0) or np.allclose(np.nanstd(right_valid), 0.0):
        return float("nan")
    return float(np.corrcoef(left_valid, right_valid)[0, 1])


def _normalized_rmse(reference: np.ndarray, candidate: np.ndarray) -> float:
    valid = np.isfinite(reference) & np.isfinite(candidate)
    if valid.sum() < 3:
        return float("nan")
    reference_valid = reference[valid]
    candidate_valid = candidate[valid]
    rmse = float(np.sqrt(np.mean(np.square(reference_valid - candidate_valid))))
    denominator = max(float(np.nanmax(np.abs(reference_valid))), 1e-12)
    return rmse / denominator
