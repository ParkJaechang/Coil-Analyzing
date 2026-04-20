from __future__ import annotations

import argparse
import glob
import json
import sys
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
from field_analysis.compensation import build_representative_cycle_profile
from field_analysis.models import CycleDetectionConfig, PreprocessConfig
from field_analysis.parser import parse_measurement_file
from field_analysis.recommendation_service import (
    DEFAULT_RECOMMENDATION_POLICY_CONFIG,
    LegacyRecommendationContext,
    RecommendationOptions,
    RecommendationPolicyConfig,
    RecommendationPolicyThresholds,
    TargetRequest,
    recommend,
)
from field_analysis.schema_config import load_schema_config
from field_analysis.utils import canonicalize_waveform_type


def _load_lcr_measurements(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        workbook = pd.ExcelFile(path)
        sheet_name = "all_bands_surrogate" if "all_bands_surrogate" in workbook.sheet_names else workbook.sheet_names[0]
        frame = workbook.parse(sheet_name)

    normalized = frame.copy()
    column_map: dict[str, str] = {}
    lowered = {str(column).strip().lower(): str(column) for column in normalized.columns}
    for candidates, target in (
        (("freq_hz", "frequency_hz", "frequency"), "freq_hz"),
        (("rs_ohm", "r_ohm", "r_ohm_calc", "resistance_ohm", "resistance", "r_ohm"), "rs_ohm"),
        (("ls_h", "l_h", "inductance_h", "l_h"), "ls_h"),
        (("cs_f", "c_f", "capacitance_f"), "cs_f"),
    ):
        for candidate in candidates:
            if candidate in lowered:
                column_map[lowered[candidate]] = target
                break
    normalized = normalized.rename(columns=column_map)
    if "freq_hz" not in normalized.columns and "freq_khz" in lowered:
        normalized["freq_hz"] = pd.to_numeric(normalized[lowered["freq_khz"]], errors="coerce") * 1000.0
    if "rs_ohm" not in normalized.columns and "r_ohm" in lowered:
        normalized["rs_ohm"] = pd.to_numeric(normalized[lowered["r_ohm"]], errors="coerce")
    if "ls_h" not in normalized.columns and "l_h" in lowered:
        normalized["ls_h"] = pd.to_numeric(normalized[lowered["l_h"]], errors="coerce")
    for column in ("freq_hz", "rs_ohm", "ls_h"):
        if column not in normalized.columns:
            return pd.DataFrame()
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    if "cs_f" in normalized.columns:
        normalized["cs_f"] = pd.to_numeric(normalized["cs_f"], errors="coerce")
    else:
        normalized["cs_f"] = 0.0
    return normalized.dropna(subset=["freq_hz", "rs_ohm", "ls_h"]).reset_index(drop=True)


def _first_numeric_value(frame: pd.DataFrame | None, column: str) -> float:
    if frame is None or column not in frame.columns or frame.empty:
        return float("nan")
    return float(pd.to_numeric(frame[column], errors="coerce").iloc[0])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate interpolated current-target auto recommendation with leave-one-frequency-out validation.",
    )
    parser.add_argument(
        "--input-glob",
        default=str(REPO_ROOT / ".coil_analyzer" / "uploads" / "*.csv"),
        help="Glob for continuous CSV uploads.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=str(REPO_ROOT / "artifacts" / "policy_eval"),
        help="Directory for evaluation CSV/JSON outputs.",
    )
    parser.add_argument("--expected-cycles", type=int, default=10)
    parser.add_argument("--current-channel", default="i_sum_signed")
    parser.add_argument("--field-channel", default="bz_mT")
    parser.add_argument("--points-per-cycle", type=int, default=256)
    parser.add_argument("--waveform", default=None, help="Optional waveform filter, e.g. sine or triangle.")
    parser.add_argument("--max-freq-hz", type=float, default=None, help="Optional maximum frequency cutoff for the evaluation corpus.")
    parser.add_argument("--lcr-file", type=Path, default=None, help="Optional LCR CSV/XLSX file for harmonic-surface prior evaluation.")
    parser.add_argument("--lcr-blend-weight", type=float, default=0.0, help="LCR prior blend weight used during evaluation.")
    parser.add_argument("--policy-version", default=None, help="Override policy version label.")
    parser.add_argument("--min-surface-confidence", type=float, default=None)
    parser.add_argument("--min-harmonic-fill-ratio", type=float, default=None)
    parser.add_argument("--max-predicted-error-band", type=float, default=None)
    parser.add_argument("--min-input-limit-margin", type=float, default=None)
    parser.add_argument("--min-support-runs", type=int, default=None)
    parser.add_argument(
        "--policy-margin-source",
        choices=("gain", "peak", "p95"),
        default=None,
        help="Policy input margin source for offline evaluation.",
    )
    parser.add_argument(
        "--allow-interpolated-auto",
        dest="allow_interpolated_auto",
        action="store_true",
        help="Explicitly enable interpolated auto recommendation for the evaluation policy.",
    )
    parser.add_argument(
        "--disable-interpolated-auto",
        dest="allow_interpolated_auto",
        action="store_false",
        help="Explicitly disable interpolated auto recommendation for the evaluation policy.",
    )
    parser.set_defaults(allow_interpolated_auto=None)
    parser.add_argument(
        "--include-edge",
        action="store_true",
        help="Include interpolated_edge cases in addition to interpolated_in_hull.",
    )
    parser.add_argument(
        "--include-out-of-hull",
        action="store_true",
        help="Include out_of_hull cases as blocked references.",
    )
    parser.add_argument("--shape-corr-threshold", type=float, default=0.95)
    parser.add_argument("--nrmse-threshold", type=float, default=0.15)
    parser.add_argument("--current-pp-error-threshold", type=float, default=10.0)
    parser.add_argument(
        "--holdout-mode",
        choices=("frequency", "file"),
        default="frequency",
        help="Holdout strategy for support removal. 'frequency' removes every exact support at the requested frequency.",
    )
    return parser.parse_args()


def _build_policy_config(args: argparse.Namespace) -> RecommendationPolicyConfig:
    base = DEFAULT_RECOMMENDATION_POLICY_CONFIG
    base_thresholds = base.thresholds
    thresholds = RecommendationPolicyThresholds(
        min_surface_confidence=(
            float(args.min_surface_confidence)
            if args.min_surface_confidence is not None
            else float(base_thresholds.min_surface_confidence)
        ),
        min_harmonic_fill_ratio=(
            float(args.min_harmonic_fill_ratio)
            if args.min_harmonic_fill_ratio is not None
            else float(base_thresholds.min_harmonic_fill_ratio)
        ),
        max_predicted_error_band=(
            float(args.max_predicted_error_band)
            if args.max_predicted_error_band is not None
            else float(base_thresholds.max_predicted_error_band)
        ),
        min_input_limit_margin=(
            float(args.min_input_limit_margin)
            if args.min_input_limit_margin is not None
            else float(base_thresholds.min_input_limit_margin)
        ),
        min_support_runs=(
            int(args.min_support_runs)
            if args.min_support_runs is not None
            else int(base_thresholds.min_support_runs)
        ),
    )
    return RecommendationPolicyConfig(
        version=str(args.policy_version or base.version),
        thresholds=thresholds,
        allow_interpolated_auto=(
            bool(args.allow_interpolated_auto)
            if args.allow_interpolated_auto is not None
            else bool(base.allow_interpolated_auto)
        ),
        margin_source=str(args.policy_margin_source or base.margin_source),
    )


def _load_dataset(
    paths: list[Path],
    *,
    expected_cycles: int,
    current_channel: str,
    field_channel: str,
) -> list[dict[str, Any]]:
    schema = load_schema_config(None)
    parsed_measurements = []
    for path in paths:
        parsed_measurements.extend(
            parse_measurement_file(
                path.name,
                path.read_bytes(),
                schema=schema,
                expected_cycles=expected_cycles,
                target_current_mode="auto",
            )
        )
    canonical_runs = canonicalize_batch(
        parsed_measurements,
        regime="continuous",
        role="train",
        config=CanonicalizeConfig(preferred_field_axis=field_channel),
    )
    analyses = analyze_measurements(
        parsed_measurements,
        canonical_runs=canonical_runs,
        preprocess_config=PreprocessConfig(),
        cycle_config=CycleDetectionConfig(
            expected_cycles=expected_cycles,
            reference_channel=current_channel,
        ),
        current_channel=current_channel,
        main_field_axis=field_channel,
    )

    records: list[dict[str, Any]] = []
    for parsed, canonical_run, analysis in zip(parsed_measurements, canonical_runs, analyses, strict=False):
        if analysis.per_test_summary.empty:
            continue
        summary = analysis.per_test_summary.iloc[0].to_dict()
        waveform_type = canonicalize_waveform_type(summary.get("waveform_type"))
        freq_hz = pd.to_numeric(pd.Series([summary.get("freq_hz")]), errors="coerce").iloc[0]
        achieved_current_pp = pd.to_numeric(pd.Series([summary.get("achieved_current_pp_a_mean")]), errors="coerce").iloc[0]
        if waveform_type is None or pd.isna(freq_hz) or pd.isna(achieved_current_pp):
            continue
        test_id = str(summary.get("test_id") or parsed.source_file)
        records.append(
            {
                "test_id": test_id,
                "waveform_type": waveform_type,
                "freq_hz": float(freq_hz),
                "achieved_current_pp_a_mean": float(achieved_current_pp),
                "parsed": parsed,
                "canonical_run": canonical_run,
                "analysis": analysis,
                "summary": summary,
            }
        )
    return records


def _periodic_resample(signal: np.ndarray, phase: np.ndarray, points_per_cycle: int) -> np.ndarray:
    base_phase = np.linspace(0.0, 1.0, points_per_cycle, endpoint=False, dtype=float)
    if signal.size == 0 or phase.size == 0:
        return np.full(points_per_cycle, np.nan, dtype=float)
    finite_mask = np.isfinite(signal) & np.isfinite(phase)
    if int(finite_mask.sum()) < 2:
        return np.full(points_per_cycle, np.nan, dtype=float)
    phase_valid = phase[finite_mask]
    signal_valid = signal[finite_mask]
    sort_order = np.argsort(phase_valid)
    phase_valid = phase_valid[sort_order]
    signal_valid = signal_valid[sort_order]
    if phase_valid[0] > 0.0:
        phase_valid = np.concatenate([[0.0], phase_valid])
        signal_valid = np.concatenate([[signal_valid[0]], signal_valid])
    if phase_valid[-1] < 1.0:
        phase_valid = np.concatenate([phase_valid, [1.0]])
        signal_valid = np.concatenate([signal_valid, [signal_valid[0]]])
    return np.interp(base_phase, phase_valid, signal_valid)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size or a.size < 2:
        return float("nan")
    a_std = float(np.nanstd(a))
    b_std = float(np.nanstd(b))
    if not np.isfinite(a_std) or not np.isfinite(b_std) or a_std <= 0 or b_std <= 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _best_alignment_metrics(actual_signal: np.ndarray, predicted_signal: np.ndarray) -> dict[str, float]:
    actual_centered = actual_signal - float(np.nanmean(actual_signal))
    predicted_centered = predicted_signal - float(np.nanmean(predicted_signal))
    best_shift = 0
    best_corr = -np.inf
    best_aligned = predicted_centered
    for shift in range(len(predicted_centered)):
        shifted = np.roll(predicted_centered, shift)
        corr = _safe_corr(actual_centered, shifted)
        if np.isfinite(corr) and corr > best_corr:
            best_corr = corr
            best_shift = shift
            best_aligned = shifted
    scale = max(float(np.nanmax(actual_centered) - np.nanmin(actual_centered)), 1e-9)
    nrmse = float(np.sqrt(np.nanmean((best_aligned - actual_centered) ** 2)) / scale)
    phase_shift_cycles = float(-best_shift / max(len(predicted_centered), 1))
    return {
        "shape_corr": float(best_corr) if np.isfinite(best_corr) else float("nan"),
        "nrmse": nrmse,
        "phase_lag_deg": phase_shift_cycles * 360.0,
        "phase_lag_cycles": phase_shift_cycles,
    }


def _peak_to_peak(signal: np.ndarray) -> float:
    finite = signal[np.isfinite(signal)]
    if finite.size == 0:
        return float("nan")
    return float(np.nanmax(finite) - np.nanmin(finite))


def _evaluate_case(
    holdout: dict[str, Any],
    support_records: list[dict[str, Any]],
    *,
    current_channel: str,
    field_channel: str,
    points_per_cycle: int,
    policy_config: RecommendationPolicyConfig,
    shape_corr_threshold: float,
    nrmse_threshold: float,
    current_pp_error_threshold: float,
    lcr_measurements: pd.DataFrame | None,
    lcr_blend_weight: float,
) -> dict[str, Any]:
    support_analyses = [record["analysis"] for record in support_records]
    support_canonical_runs = [record["canonical_run"] for record in support_records]
    _, support_per_test_summary, _ = combine_analysis_frames(support_analyses, field_axis=field_channel)
    analysis_lookup = {str(record["summary"].get("test_id") or record["test_id"]): record["analysis"] for record in support_records}
    support_freqs = sorted(float(record["freq_hz"]) for record in support_records)

    result = recommend(
        continuous_runs=support_canonical_runs,
        transient_runs=[],
        validation_runs=[],
        target=TargetRequest(
            regime="continuous",
            target_waveform=str(holdout["waveform_type"]),
            command_waveform=str(holdout["waveform_type"]),
            freq_hz=float(holdout["freq_hz"]),
            target_type="current",
            target_level_value=float(holdout["achieved_current_pp_a_mean"]),
            target_level_kind="pp",
            context={
                "request_kind": "waveform_compensation",
                "frequency_mode": "interpolate",
                "finite_cycle_mode": False,
                "preview_tail_cycles": 0.25,
            },
        ),
        options=RecommendationOptions(
            current_channel=current_channel,
            field_channel=field_channel,
            lcr_measurements=lcr_measurements,
            lcr_blend_weight=float(lcr_blend_weight),
        ),
        legacy_context=LegacyRecommendationContext(
            per_test_summary=support_per_test_summary,
            analysis_lookup=analysis_lookup,
        ),
        policy_config=policy_config,
    )

    support_state = str(result.engine_summary.get("support_state", "unsupported"))
    payload = result.legacy_payload or {}
    service_allow_auto_recommendation = bool(result.allow_auto_download)
    service_preview_only = bool(result.preview_only)
    case_row: dict[str, Any] = {
        "holdout_test_id": holdout["test_id"],
        "holdout_mode": "",
        "waveform_type": holdout["waveform_type"],
        "target_freq_hz": float(holdout["freq_hz"]),
        "target_current_pp_a": float(holdout["achieved_current_pp_a_mean"]),
        "support_freqs_hz": ",".join(f"{value:g}" for value in support_freqs),
        "support_run_count": int(len(support_records)),
        "in_hull": support_state == "interpolated_in_hull",
        "selected_engine": str(result.engine_summary.get("selected_engine", "unknown")),
        "support_state": support_state,
        "service_allow_auto_recommendation": service_allow_auto_recommendation,
        "service_preview_only": service_preview_only,
        "allow_auto_recommendation": bool(result.allow_auto_download),
        "preview_only": bool(result.preview_only),
        "policy_version": str(result.debug_info.get("policy_version", "")),
        "policy_snapshot": json.dumps(result.debug_info.get("policy_snapshot", {}), ensure_ascii=False, sort_keys=True),
        "policy_margin_source": str(policy_config.margin_source),
        "lcr_blend_weight": float(lcr_blend_weight),
        "surface_confidence": result.confidence_summary.get("surface_confidence"),
        "harmonics_used": result.confidence_summary.get("harmonics_used"),
        "harmonic_cap": result.confidence_summary.get("harmonic_cap"),
        "harmonic_fill_ratio": result.confidence_summary.get("harmonic_fill_ratio"),
        "predicted_error_band": result.confidence_summary.get("predicted_error_band"),
        "input_limit_margin": result.confidence_summary.get("input_limit_margin"),
        "gain_input_limit_margin": result.confidence_summary.get("gain_input_limit_margin"),
        "peak_input_limit_margin": result.confidence_summary.get("peak_input_limit_margin"),
        "p95_input_limit_margin": result.confidence_summary.get("p95_input_limit_margin"),
        "lcr_consistency_score": result.confidence_summary.get("lcr_consistency_score"),
        "lcr_gain_mismatch_log_abs": result.confidence_summary.get("lcr_gain_mismatch_log_abs"),
        "lcr_phase_mismatch_rad": result.confidence_summary.get("lcr_phase_mismatch_rad"),
        "lcr_weight_mean": result.confidence_summary.get("lcr_weight_mean"),
        "lcr_prior_fraction": result.confidence_summary.get("lcr_prior_fraction"),
        "service_policy_reasons": " | ".join(str(reason) for reason in result.debug_info.get("policy_reasons", [])),
        "policy_reasons": " | ".join(str(reason) for reason in result.debug_info.get("policy_reasons", [])),
    }

    command_profile = result.command_profile
    if command_profile is None or command_profile.empty or "expected_current_a" not in command_profile.columns:
        case_row["evaluation_status"] = "no_prediction_payload"
        case_row["confusion_label"] = "blocked_no_payload"
        return case_row

    holdout_profile = build_representative_cycle_profile(
        analysis=holdout["analysis"],
        current_channel=current_channel,
        voltage_channel="daq_input_v",
        field_channel=field_channel,
        points_per_cycle=points_per_cycle,
    )
    if holdout_profile.empty or "measured_current_a" not in holdout_profile.columns:
        case_row["evaluation_status"] = "missing_holdout_profile"
        case_row["confusion_label"] = "skipped_missing_profile"
        return case_row

    actual_signal = _periodic_resample(
        pd.to_numeric(holdout_profile["measured_current_a"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(holdout_profile["cycle_progress"], errors="coerce").to_numpy(dtype=float),
        points_per_cycle,
    )
    predicted_signal = _periodic_resample(
        pd.to_numeric(command_profile["expected_current_a"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(command_profile["cycle_progress"], errors="coerce").to_numpy(dtype=float),
        points_per_cycle,
    )
    alignment = _best_alignment_metrics(actual_signal, predicted_signal)
    actual_pp = _peak_to_peak(actual_signal)
    predicted_pp = _peak_to_peak(predicted_signal)
    pp_error_pct = (
        float(abs(predicted_pp - actual_pp) / max(abs(actual_pp), 1e-9) * 100.0)
        if np.isfinite(actual_pp) and np.isfinite(predicted_pp)
        else float("nan")
    )
    recommended_input_pp = _first_numeric_value(command_profile, "limited_voltage_pp")
    if not np.isfinite(recommended_input_pp):
        recommended_input_pp = float(pd.to_numeric(pd.Series([payload.get("limited_voltage_pp")]), errors="coerce").iloc[0])
    max_daq_voltage_pp = _first_numeric_value(command_profile, "max_daq_voltage_pp")
    if not np.isfinite(max_daq_voltage_pp):
        max_daq_voltage_pp = float(pd.to_numeric(pd.Series([payload.get("max_daq_voltage_pp")]), errors="coerce").iloc[0])
    if "within_hardware_limits" in command_profile.columns and not command_profile.empty:
        within_hardware_limits = bool(command_profile["within_hardware_limits"].iloc[0])
    else:
        within_hardware_limits = bool(payload.get("within_hardware_limits", True))
    input_limit_utilization = (
        float(recommended_input_pp / max(max_daq_voltage_pp, 1e-9))
        if np.isfinite(recommended_input_pp) and np.isfinite(max_daq_voltage_pp)
        else float("nan")
    )
    command_was_limited = False
    if {"recommended_voltage_v", "limited_voltage_v"}.issubset(command_profile.columns):
        recommended_voltage = pd.to_numeric(command_profile["recommended_voltage_v"], errors="coerce").to_numpy(dtype=float)
        limited_voltage = pd.to_numeric(command_profile["limited_voltage_v"], errors="coerce").to_numpy(dtype=float)
        command_was_limited = bool(np.nanmax(np.abs(recommended_voltage - limited_voltage)) > 1e-6)

    actual_safe_auto = bool(
        np.isfinite(alignment["shape_corr"])
        and alignment["shape_corr"] >= float(shape_corr_threshold)
        and np.isfinite(alignment["nrmse"])
        and alignment["nrmse"] <= float(nrmse_threshold)
        and np.isfinite(pp_error_pct)
        and pp_error_pct <= float(current_pp_error_threshold)
        and within_hardware_limits
    )
    if result.allow_auto_download and not actual_safe_auto:
        confusion_label = "false_auto"
    elif not result.allow_auto_download and actual_safe_auto:
        confusion_label = "false_block"
    elif result.allow_auto_download and actual_safe_auto:
        confusion_label = "correct_auto"
    else:
        confusion_label = "correct_block"

    case_row.update(
        {
            "evaluation_status": "ok",
            "realized_shape_corr": alignment["shape_corr"],
            "realized_nrmse": alignment["nrmse"],
            "realized_phase_lag_deg": alignment["phase_lag_deg"],
            "realized_phase_lag_cycles": alignment["phase_lag_cycles"],
            "actual_current_pp_a": actual_pp,
            "predicted_current_pp_a": predicted_pp,
            "current_pp_error_pct": pp_error_pct,
            "within_hardware_limits_model": within_hardware_limits,
            "recommended_input_limit_utilization": input_limit_utilization,
            "command_was_limited": command_was_limited,
            "actual_safe_auto": actual_safe_auto,
            "confusion_label": confusion_label,
        }
    )
    return case_row


def _build_summary(frame: pd.DataFrame, *, policy_config: RecommendationPolicyConfig) -> dict[str, Any]:
    confusion_counts = (
        frame["confusion_label"].value_counts(dropna=False).to_dict()
        if not frame.empty and "confusion_label" in frame.columns
        else {}
    )
    support_state_counts = (
        frame["support_state"].value_counts(dropna=False).to_dict()
        if not frame.empty and "support_state" in frame.columns
        else {}
    )
    evaluation_status_counts = (
        frame["evaluation_status"].value_counts(dropna=False).to_dict()
        if not frame.empty and "evaluation_status" in frame.columns
        else {}
    )
    false_auto_count = int(confusion_counts.get("false_auto", 0))
    auto_count = int(pd.to_numeric(frame.get("allow_auto_recommendation"), errors="coerce").fillna(False).astype(bool).sum()) if not frame.empty else 0
    evaluated_mask = frame.get("evaluation_status", pd.Series(dtype=object)).eq("ok") if not frame.empty else pd.Series(dtype=bool)
    return {
        "policy_version": policy_config.version,
        "max_freq_hz": (
            float(pd.to_numeric(frame["max_freq_hz"], errors="coerce").dropna().iloc[0])
            if not frame.empty and "max_freq_hz" in frame.columns and pd.to_numeric(frame["max_freq_hz"], errors="coerce").notna().any()
            else None
        ),
        "policy_allow_interpolated_auto": bool(policy_config.allow_interpolated_auto),
        "policy_margin_source": str(frame["policy_margin_source"].iloc[0]) if not frame.empty and "policy_margin_source" in frame.columns else "current",
        "holdout_mode": str(frame["holdout_mode"].iloc[0]) if not frame.empty and "holdout_mode" in frame.columns else "unknown",
        "policy_thresholds": {
            "min_surface_confidence": float(policy_config.thresholds.min_surface_confidence),
            "min_harmonic_fill_ratio": float(policy_config.thresholds.min_harmonic_fill_ratio),
            "max_predicted_error_band": float(policy_config.thresholds.max_predicted_error_band),
            "min_input_limit_margin": float(policy_config.thresholds.min_input_limit_margin),
            "min_support_runs": int(policy_config.thresholds.min_support_runs),
        },
        "case_count": int(len(frame)),
        "evaluated_case_count": int(evaluated_mask.sum()) if not frame.empty else 0,
        "auto_count": auto_count,
        "preview_or_block_count": int(len(frame) - auto_count),
        "false_auto_count": false_auto_count,
        "false_auto_rate_within_auto": float(false_auto_count / max(auto_count, 1)),
        "mean_realized_shape_corr": float(pd.to_numeric(frame.get("realized_shape_corr"), errors="coerce").mean()) if not frame.empty else float("nan"),
        "mean_realized_nrmse": float(pd.to_numeric(frame.get("realized_nrmse"), errors="coerce").mean()) if not frame.empty else float("nan"),
        "mean_predicted_error_band": float(pd.to_numeric(frame.get("predicted_error_band"), errors="coerce").mean()) if not frame.empty else float("nan"),
        "mean_input_limit_margin": float(pd.to_numeric(frame.get("input_limit_margin"), errors="coerce").mean()) if not frame.empty else float("nan"),
        "mean_gain_input_limit_margin": float(pd.to_numeric(frame.get("gain_input_limit_margin"), errors="coerce").mean()) if not frame.empty else float("nan"),
        "mean_peak_input_limit_margin": float(pd.to_numeric(frame.get("peak_input_limit_margin"), errors="coerce").mean()) if not frame.empty else float("nan"),
        "mean_p95_input_limit_margin": float(pd.to_numeric(frame.get("p95_input_limit_margin"), errors="coerce").mean()) if not frame.empty else float("nan"),
        "support_state_counts": support_state_counts,
        "evaluation_status_counts": evaluation_status_counts,
        "confusion_counts": confusion_counts,
    }


def main() -> int:
    args = _parse_args()
    input_paths = [Path(path) for path in sorted(glob.glob(args.input_glob))]
    if not input_paths:
        print(f"No files matched: {args.input_glob}", file=sys.stderr)
        return 1

    policy_config = _build_policy_config(args)
    lcr_measurements = _load_lcr_measurements(args.lcr_file)
    records = _load_dataset(
        input_paths,
        expected_cycles=int(args.expected_cycles),
        current_channel=str(args.current_channel),
        field_channel=str(args.field_channel),
    )
    if args.waveform:
        waveform_filter = canonicalize_waveform_type(args.waveform)
        records = [record for record in records if record["waveform_type"] == waveform_filter]
    if args.max_freq_hz is not None:
        max_freq_hz = float(args.max_freq_hz)
        records = [record for record in records if float(record["freq_hz"]) <= max_freq_hz]
    if not records:
        print("No usable continuous records after loading.", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = []
    for holdout in records:
        if args.holdout_mode == "frequency":
            support_records = [
                record
                for record in records
                if record["waveform_type"] == holdout["waveform_type"]
                and not np.isclose(float(record["freq_hz"]), float(holdout["freq_hz"]), atol=1e-9)
            ]
        else:
            support_records = [
                record
                for record in records
                if record["test_id"] != holdout["test_id"] and record["waveform_type"] == holdout["waveform_type"]
            ]
        if not support_records:
            continue
        support_freqs = np.array([float(record["freq_hz"]) for record in support_records], dtype=float)
        lower_exists = bool(np.any(support_freqs < float(holdout["freq_hz"])))
        upper_exists = bool(np.any(support_freqs > float(holdout["freq_hz"])))
        in_hull = lower_exists and upper_exists
        out_of_hull = not in_hull and (
            float(holdout["freq_hz"]) < float(np.min(support_freqs)) or float(holdout["freq_hz"]) > float(np.max(support_freqs))
        )
        edge_case = not in_hull and not out_of_hull
        if in_hull:
            pass
        elif out_of_hull and not args.include_out_of_hull:
            continue
        elif edge_case and not args.include_edge:
            continue

        row = _evaluate_case(
            holdout,
            support_records,
            current_channel=str(args.current_channel),
            field_channel=str(args.field_channel),
            points_per_cycle=int(args.points_per_cycle),
            policy_config=policy_config,
            shape_corr_threshold=float(args.shape_corr_threshold),
            nrmse_threshold=float(args.nrmse_threshold),
            current_pp_error_threshold=float(args.current_pp_error_threshold),
            lcr_measurements=lcr_measurements if not lcr_measurements.empty else None,
            lcr_blend_weight=float(args.lcr_blend_weight),
        )
        row["holdout_mode"] = str(args.holdout_mode)
        row["max_freq_hz"] = float(args.max_freq_hz) if args.max_freq_hz is not None else np.nan
        rows.append(row)

    case_frame = pd.DataFrame(rows)
    summary = _build_summary(case_frame, policy_config=policy_config)

    output_dir = Path(args.artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"policy_eval_{policy_config.version}.csv"
    json_path = output_dir / f"policy_eval_{policy_config.version}.json"
    case_frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved case report: {csv_path}")
    print(f"Saved summary: {json_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
