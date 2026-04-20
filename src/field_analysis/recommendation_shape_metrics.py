from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .validation import ValidationReport


def _resolve_prediction_target_column(command_profile: pd.DataFrame) -> str | None:
    for column in ("aligned_used_target_output", "used_target_output", "aligned_target_output", "target_output"):
        if column in command_profile.columns:
            return column
    return None


def _resolve_prediction_output_column(command_profile: pd.DataFrame) -> str | None:
    for column in (
        "expected_output",
        "aligned_expected_output",
        "modeled_output",
        "expected_current_a",
        "expected_field_mT",
        "modeled_current_a",
        "modeled_field_mT",
    ):
        if column in command_profile.columns:
            return column
    return None


def _signal_peak_to_peak(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan")
    return float(np.nanmax(finite) - np.nanmin(finite))


def _signal_correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    valid = np.isfinite(reference) & np.isfinite(candidate)
    if int(valid.sum()) < 3:
        return float("nan")
    ref = np.asarray(reference[valid], dtype=float) - float(np.nanmean(reference[valid]))
    comp = np.asarray(candidate[valid], dtype=float) - float(np.nanmean(candidate[valid]))
    ref_std = float(np.nanstd(ref))
    comp_std = float(np.nanstd(comp))
    if ref_std <= 1e-12 or comp_std <= 1e-12:
        return float("nan")
    return float(np.clip(np.corrcoef(ref, comp)[0, 1], -1.0, 1.0))


def _estimate_phase_lag_seconds(
    reference: np.ndarray,
    candidate: np.ndarray,
    time_grid: np.ndarray,
) -> float:
    valid = np.isfinite(reference) & np.isfinite(candidate) & np.isfinite(time_grid)
    if int(valid.sum()) < 4:
        return float("nan")
    ref = np.asarray(reference[valid], dtype=float) - float(np.nanmean(reference[valid]))
    comp = np.asarray(candidate[valid], dtype=float) - float(np.nanmean(candidate[valid]))
    if np.nanstd(ref) <= 1e-12 or np.nanstd(comp) <= 1e-12:
        return float("nan")
    correlation = np.correlate(comp, ref, mode="full")
    lag_index = int(np.argmax(correlation) - (len(ref) - 1))
    dt = float(np.nanmedian(np.diff(np.asarray(time_grid[valid], dtype=float))))
    if not np.isfinite(dt) or dt <= 0:
        return float("nan")
    return float(lag_index * dt)


def _detect_plateau_clipping(values: np.ndarray, repeat_ratio_threshold: float = 0.05) -> bool:
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size < 8:
        return False
    rounded = np.round(finite_values, 6)
    max_repeat_ratio = float(np.mean(np.isclose(rounded, np.nanmax(rounded), atol=1e-6)))
    min_repeat_ratio = float(np.mean(np.isclose(rounded, np.nanmin(rounded), atol=1e-6)))
    return bool(max(max_repeat_ratio, min_repeat_ratio) >= float(repeat_ratio_threshold))


def _compute_prediction_shape_metrics(command_profile: pd.DataFrame) -> dict[str, Any]:
    if command_profile.empty or "time_s" not in command_profile.columns:
        return {
            "predicted_shape_corr": float("nan"),
            "predicted_nrmse": float("nan"),
            "predicted_phase_lag": float("nan"),
            "predicted_phase_lag_cycles": float("nan"),
            "predicted_pp_error": float("nan"),
            "predicted_peak_error": float("nan"),
            "predicted_clipping": False,
        }

    target_column = _resolve_prediction_target_column(command_profile)
    expected_column = _resolve_prediction_output_column(command_profile)
    if target_column is None or expected_column is None:
        return {
            "predicted_shape_corr": float("nan"),
            "predicted_nrmse": float("nan"),
            "predicted_phase_lag": float("nan"),
            "predicted_phase_lag_cycles": float("nan"),
            "predicted_pp_error": float("nan"),
            "predicted_peak_error": float("nan"),
            "predicted_clipping": False,
        }

    time_s = pd.to_numeric(command_profile["time_s"], errors="coerce").to_numpy(dtype=float)
    target_values = pd.to_numeric(command_profile[target_column], errors="coerce").to_numpy(dtype=float)
    expected_values = pd.to_numeric(command_profile[expected_column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(time_s) & np.isfinite(target_values) & np.isfinite(expected_values)
    if int(valid.sum()) < 4:
        return {
            "predicted_shape_corr": float("nan"),
            "predicted_nrmse": float("nan"),
            "predicted_phase_lag": float("nan"),
            "predicted_phase_lag_cycles": float("nan"),
            "predicted_pp_error": float("nan"),
            "predicted_peak_error": float("nan"),
            "predicted_clipping": False,
        }

    target_values = np.asarray(target_values[valid], dtype=float)
    expected_values = np.asarray(expected_values[valid], dtype=float)
    time_values = np.asarray(time_s[valid], dtype=float)
    error = expected_values - target_values
    target_pp = float(np.nanmax(target_values) - np.nanmin(target_values)) if len(target_values) else float("nan")
    rmse = float(np.sqrt(np.nanmean(np.square(error)))) if len(error) else float("nan")
    denom = max(target_pp / 2.0, 1e-12) if np.isfinite(target_pp) and target_pp > 0 else float("nan")
    nrmse = rmse / denom if np.isfinite(denom) else float("nan")
    phase_lag_s = _estimate_phase_lag_seconds(target_values, expected_values, time_values)
    freq_hz = float(pd.to_numeric(command_profile.get("freq_hz", pd.Series([np.nan])), errors="coerce").iloc[0])
    phase_lag_cycles = (
        float(phase_lag_s * freq_hz)
        if np.isfinite(phase_lag_s) and np.isfinite(freq_hz) and freq_hz > 0
        else float("nan")
    )
    within_hardware_limits = True
    if "within_hardware_limits" in command_profile.columns:
        within_hardware_limits = bool(pd.to_numeric(command_profile["within_hardware_limits"], errors="coerce").fillna(1).astype(bool).iloc[0])
    voltage_values = (
        pd.to_numeric(command_profile["limited_voltage_v"], errors="coerce").to_numpy(dtype=float)
        if "limited_voltage_v" in command_profile.columns
        else np.array([], dtype=float)
    )
    predicted_clipping = bool((not within_hardware_limits) or _detect_plateau_clipping(voltage_values))
    return {
        "predicted_shape_corr": _signal_correlation(target_values, expected_values),
        "predicted_nrmse": float(nrmse) if np.isfinite(nrmse) else float("nan"),
        "predicted_phase_lag": float(phase_lag_s) if np.isfinite(phase_lag_s) else float("nan"),
        "predicted_phase_lag_cycles": float(phase_lag_cycles) if np.isfinite(phase_lag_cycles) else float("nan"),
        "predicted_pp_error": float((np.nanmax(expected_values) - np.nanmin(expected_values)) - target_pp)
        if np.isfinite(target_pp)
        else float("nan"),
        "predicted_peak_error": float(np.nanmax(np.abs(error))) if len(error) else float("nan"),
        "predicted_clipping": predicted_clipping,
    }


def _build_surface_confidence_summary(
    *,
    exact_frequency_match: bool,
    interpolation_requested: bool,
    support_run_count: int,
    harmonic_cap_value: int,
    harmonics_used: int,
    phase_clamp_fraction: float,
    validation_report: ValidationReport,
    command_profile: pd.DataFrame,
    frequency_geometry: dict[str, Any] | None = None,
    inverse_debug_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    prediction_metrics = _compute_prediction_shape_metrics(command_profile)
    gain_limit_margin = float("nan")
    if not command_profile.empty and {"available_amp_gain_pct", "required_amp_gain_pct"}.issubset(command_profile.columns):
        available_gain = float(pd.to_numeric(command_profile["available_amp_gain_pct"], errors="coerce").iloc[0])
        required_gain = float(pd.to_numeric(command_profile["required_amp_gain_pct"], errors="coerce").iloc[0])
        if np.isfinite(available_gain) and available_gain > 0 and np.isfinite(required_gain):
            gain_limit_margin = float(np.clip((available_gain - required_gain) / available_gain, 0.0, 1.0))
    peak_input_limit_margin = float("nan")
    p95_input_limit_margin = float("nan")
    if not command_profile.empty:
        if "peak_input_limit_margin" in command_profile.columns:
            peak_input_limit_margin = float(pd.to_numeric(command_profile["peak_input_limit_margin"], errors="coerce").iloc[0])
        if "p95_input_limit_margin" in command_profile.columns:
            p95_input_limit_margin = float(pd.to_numeric(command_profile["p95_input_limit_margin"], errors="coerce").iloc[0])

    input_limit_margin = gain_limit_margin

    support_score = min(max(int(support_run_count), 0), 4) / 4.0
    harmonic_fill_ratio = (
        float(np.clip(harmonics_used / max(int(harmonic_cap_value), 1), 0.0, 1.0))
        if harmonic_cap_value > 0
        else 0.0
    )
    clamp_penalty = float(np.clip(phase_clamp_fraction, 0.0, 1.0))
    margin_score = float(input_limit_margin) if np.isfinite(input_limit_margin) else 0.0
    geometry = dict(frequency_geometry or {})
    nearest_distance_log = float(pd.to_numeric(pd.Series([geometry.get("nearest_support_distance_log")]), errors="coerce").iloc[0])
    bracket_span_log = float(pd.to_numeric(pd.Series([geometry.get("bracket_span_log")]), errors="coerce").iloc[0])
    bracket_position_log = float(pd.to_numeric(pd.Series([geometry.get("bracket_position_log")]), errors="coerce").iloc[0])

    if exact_frequency_match:
        interpolation_geometry_score = 1.0
    else:
        nearest_norm = (
            float(np.clip(nearest_distance_log / max(np.log(2.0), 1e-9), 0.0, 1.0))
            if np.isfinite(nearest_distance_log)
            else 1.0
        )
        span_norm = (
            float(np.clip(bracket_span_log / max(np.log(4.0), 1e-9), 0.0, 1.0))
            if np.isfinite(bracket_span_log)
            else 1.0
        )
        position_norm = (
            float(np.clip(abs(bracket_position_log - 0.5) / 0.5, 0.0, 1.0))
            if np.isfinite(bracket_position_log)
            else 1.0
        )
        interpolation_geometry_score = float(
            np.clip(1.0 - (0.45 * nearest_norm + 0.35 * span_norm + 0.20 * position_norm), 0.0, 1.0)
        )

    lcr_consistency_score = float("nan")
    lcr_gain_mismatch_log_abs = float("nan")
    lcr_phase_mismatch_rad = float("nan")
    lcr_weight_mean = float("nan")
    lcr_prior_fraction = float("nan")
    if inverse_debug_frame is not None and not inverse_debug_frame.empty and "lcr_prior_available" in inverse_debug_frame.columns:
        lcr_rows = inverse_debug_frame[
            pd.to_numeric(inverse_debug_frame["lcr_prior_available"], errors="coerce").fillna(0).astype(bool)
        ].copy()
        if not lcr_rows.empty:
            phase_residual_std_rad = float(
                pd.to_numeric(lcr_rows.get("phase_residual_std_rad"), errors="coerce").dropna().mean()
            ) if "phase_residual_std_rad" in lcr_rows.columns else float("nan")
            lcr_gain_mismatch_log_abs = float(
                pd.to_numeric(lcr_rows.get("lcr_gain_mismatch_log_abs"), errors="coerce").dropna().mean()
            )
            lcr_phase_mismatch_rad = float(
                pd.to_numeric(lcr_rows.get("lcr_phase_mismatch_rad"), errors="coerce").dropna().mean()
            )
            lcr_weight_mean = float(
                pd.to_numeric(lcr_rows.get("lcr_weight_used"), errors="coerce").dropna().mean()
            )
            lcr_prior_fraction = float(len(lcr_rows) / max(len(inverse_debug_frame), 1))
            residual_penalty = (
                float(np.clip(phase_residual_std_rad / (np.pi / 2.0), 0.0, 1.0))
                if np.isfinite(phase_residual_std_rad)
                else np.nan
            )
            gain_penalty = (
                float(np.clip(lcr_gain_mismatch_log_abs / max(np.log(2.0), 1e-9), 0.0, 1.0))
                if np.isfinite(lcr_gain_mismatch_log_abs)
                else np.nan
            )
            phase_penalty = (
                float(np.clip(lcr_phase_mismatch_rad / np.pi, 0.0, 1.0))
                if np.isfinite(lcr_phase_mismatch_rad)
                else np.nan
            )
            if np.isfinite(residual_penalty):
                lcr_consistency_score = float(
                    np.clip(
                        1.0
                        - (
                            0.60 * residual_penalty
                            + 0.25 * (phase_penalty if np.isfinite(phase_penalty) else 0.5)
                            + 0.15 * (gain_penalty if np.isfinite(gain_penalty) else 0.5)
                        ),
                        0.0,
                        1.0,
                    )
                )
            else:
                lcr_consistency_score = float(
                    np.clip(
                        1.0
                        - (
                            0.55 * (phase_penalty if np.isfinite(phase_penalty) else 0.5)
                            + 0.45 * (gain_penalty if np.isfinite(gain_penalty) else 0.5)
                        ),
                        0.0,
                        1.0,
                    )
                )
        else:
            phase_residual_std_rad = float("nan")
    else:
        phase_residual_std_rad = float("nan")

    if exact_frequency_match:
        confidence = (
            0.45
            + 0.20 * support_score
            + 0.20 * harmonic_fill_ratio
            + 0.15 * margin_score
            - 0.20 * clamp_penalty
        )
    else:
        confidence = (
            0.05
            + 0.15 * support_score
            + 0.20 * harmonic_fill_ratio
            + 0.10 * margin_score
            + 0.25 * interpolation_geometry_score
            + 0.15 * (lcr_consistency_score if np.isfinite(lcr_consistency_score) else 0.0)
            - 0.15 * clamp_penalty
        )
    confidence = float(np.clip(confidence, 0.0, 1.0))
    predicted_error_band = float(validation_report.expected_error_band)
    if not exact_frequency_match and interpolation_requested:
        predicted_error_band = float(
            np.clip(
                0.10
                + 0.10 * (1.0 - interpolation_geometry_score)
                + 0.08 * (1.0 - (lcr_consistency_score if np.isfinite(lcr_consistency_score) else 0.5))
                + 0.08 * clamp_penalty
                + 0.05 * (1.0 - harmonic_fill_ratio)
                + 0.04 * (1.0 - support_score),
                0.10,
                0.35,
            )
        )

    return {
        "surface_confidence": confidence,
        "exact_frequency_match": exact_frequency_match,
        "interpolation_requested": interpolation_requested,
        "support_run_count": int(support_run_count),
        "harmonics_used": int(harmonics_used),
        "harmonic_cap": int(harmonic_cap_value),
        "harmonic_fill_ratio": harmonic_fill_ratio,
        "phase_clamp_fraction": clamp_penalty,
        "predicted_error_band": predicted_error_band,
        "input_limit_margin": input_limit_margin,
        "gain_input_limit_margin": gain_limit_margin,
        "peak_input_limit_margin": peak_input_limit_margin,
        "p95_input_limit_margin": p95_input_limit_margin,
        "interpolation_geometry_score": interpolation_geometry_score,
        "nearest_support_distance_hz": geometry.get("nearest_support_distance_hz"),
        "nearest_support_distance_log": geometry.get("nearest_support_distance_log"),
        "lower_support_hz": geometry.get("lower_support_hz"),
        "upper_support_hz": geometry.get("upper_support_hz"),
        "bracket_span_hz": geometry.get("bracket_span_hz"),
        "bracket_span_log": geometry.get("bracket_span_log"),
        "bracket_position_log": geometry.get("bracket_position_log"),
        "support_density_hz": geometry.get("support_density_hz"),
        "lcr_consistency_score": lcr_consistency_score,
        "lcr_gain_mismatch_log_abs": lcr_gain_mismatch_log_abs,
        "lcr_phase_mismatch_rad": lcr_phase_mismatch_rad,
        "lcr_weight_mean": lcr_weight_mean,
        "lcr_prior_fraction": lcr_prior_fraction,
        "phase_residual_std_rad": phase_residual_std_rad,
        **prediction_metrics,
    }
