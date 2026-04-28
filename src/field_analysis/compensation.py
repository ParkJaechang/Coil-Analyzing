from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .finite_cycle_metrics import (
    FiniteCycleMetrics,
    attach_finite_cycle_metrics,
    build_finite_metric_improvement_summary,
    evaluate_finite_cycle_metrics,
)
from .hardware import apply_command_hardware_model
from .lcr import build_lcr_harmonic_prior, build_lcr_impedance_table
from .lut import _theoretical_template
from .models import DatasetAnalysis, ParsedMeasurement, PreprocessResult
from .recommendation_lcr_runtime import resolve_lcr_runtime_policy
from .utils import canonicalize_waveform_type

FIELD_ROUTE_NORMALIZED_TARGET_PP = 100.0
FIELD_ROUTE_ALLOWED_FINITE_CYCLE_COUNTS = (1.0, 1.25, 1.5, 1.75)
FIELD_ROUTE_SHAPE_SELECTION_EXCLUDES = ("current", "gain", "hardware", "lcr")
FINITE_SIGNAL_JUMP_RATIO_LIMIT = 0.20


def _default_harmonic_count(waveform_type: str, points_per_cycle: int) -> int:
    max_available = max(int(points_per_cycle) // 2 - 1, 1)
    waveform_type = canonicalize_waveform_type(waveform_type) or "sine"
    if waveform_type == "triangle":
        return max(3, min(max_available, 31))
    return max(1, min(max_available, 11))


def _rounded_triangle_normalized(phase: np.ndarray) -> np.ndarray:
    phase_values = np.asarray(phase, dtype=float)
    signal = np.zeros_like(phase_values, dtype=float)
    for harmonic in (1, 3, 5):
        sign = 1.0 if harmonic % 4 == 1 else -1.0
        signal += sign * np.sin(2.0 * np.pi * harmonic * phase_values) / float(harmonic * harmonic)
    peak = float(np.nanmax(np.abs(signal))) if len(signal) else float("nan")
    if not np.isfinite(peak) or peak <= 1e-12:
        return signal
    return signal / peak


def _build_target_template(
    waveform_type: str,
    freq_hz: float,
    points_per_cycle: int,
    *,
    force_rounded_triangle: bool = False,
) -> pd.DataFrame:
    if not force_rounded_triangle:
        return _theoretical_template(
            waveform_type=waveform_type,
            freq_hz=freq_hz,
            points_per_cycle=points_per_cycle,
        ).rename(columns={"voltage_normalized": "target_output_normalized"})

    phase_grid = np.linspace(0.0, 1.0, int(points_per_cycle))
    period_s = 1.0 / float(freq_hz) if np.isfinite(freq_hz) and float(freq_hz) > 0 else 1.0
    return pd.DataFrame(
        {
            "cycle_progress": phase_grid,
            "time_s": phase_grid * period_s,
            "target_output_normalized": _rounded_triangle_normalized(phase_grid),
        }
    )


def _normalize_field_finite_cycle_count(target_cycle_count: float | None) -> float | None:
    if target_cycle_count is None or not np.isfinite(target_cycle_count):
        return None
    requested = float(target_cycle_count)
    if abs(requested - 0.75) <= 0.125:
        return 0.75
    if abs(requested - 1.75) <= 0.125:
        return 1.75
    nearest = min(
        FIELD_ROUTE_ALLOWED_FINITE_CYCLE_COUNTS,
        key=lambda value: (abs(float(value) - requested), float(value)),
    )
    return float(nearest)


def synthesize_current_waveform_compensation(
    per_test_summary: pd.DataFrame,
    analyses_by_test_id: dict[str, DatasetAnalysis],
    waveform_type: str,
    freq_hz: float,
    target_current_pp_a: float,
    current_metric: str = "achieved_current_pp_a_mean",
    current_channel: str = "i_sum_signed",
    field_channel: str = "bz_mT",
    target_output_type: str = "current",
    target_output_pp: float | None = None,
    frequency_mode: str = "interpolate",
    voltage_channel: str = "daq_input_v",
    finite_cycle_mode: bool = False,
    target_cycle_count: float | None = None,
    preview_tail_cycles: float = 0.25,
    points_per_cycle: int = 256,
    max_daq_voltage_pp: float = 20.0,
    amp_gain_at_100_pct: float = 20.0,
    amp_gain_limit_pct: float = 100.0,
    amp_max_output_pk_v: float = 180.0,
    default_support_amp_gain_pct: float = 100.0,
    allow_output_extrapolation: bool = True,
    finite_support_entries: list[dict[str, Any]] | None = None,
    lcr_measurements: pd.DataFrame | None = None,
    lcr_blend_weight: float = 0.0,
    apply_startup_correction: bool = False,
    startup_transition_cycles: float = 1.5,
    startup_correction_strength: float = 1.0,
    startup_preview_cycle_count: int = 3,
) -> dict[str, Any] | None:
    """Build a phase-wise inverse command waveform to approach a target output shape."""

    waveform_type = canonicalize_waveform_type(waveform_type)
    if waveform_type is None:
        return None
    finite_support_entries = finite_support_entries or []

    output_context = _resolve_output_context(
        target_output_type=target_output_type,
        field_channel=field_channel,
        current_metric=current_metric,
    )
    requested_target_output_pp = float(target_output_pp if target_output_pp is not None else target_current_pp_a)
    target_output_pp = requested_target_output_pp
    field_only_route = output_context["output_type"] == "field"
    shape_target_output_pp = (
        float(FIELD_ROUTE_NORMALIZED_TARGET_PP)
        if field_only_route
        else requested_target_output_pp
    )
    max_harmonics = _default_harmonic_count(waveform_type, points_per_cycle)

    if per_test_summary.empty or output_context["output_metric"] not in per_test_summary.columns:
        return None

    requested_freq_hz = float(freq_hz)
    waveform_subset = per_test_summary[
        per_test_summary["waveform_type"].map(canonicalize_waveform_type) == waveform_type
    ].copy()
    waveform_subset = waveform_subset.dropna(subset=[output_context["output_metric"], "freq_hz"]).copy()
    if waveform_subset.empty:
        return None

    available_freq_values = waveform_subset["freq_hz"].to_numpy(dtype=float)
    available_freq_min = float(np.nanmin(available_freq_values))
    available_freq_max = float(np.nanmax(available_freq_values))
    used_freq_hz = (
        requested_freq_hz
        if frequency_mode == "exact"
        else float(np.clip(requested_freq_hz, available_freq_min, available_freq_max))
    )

    exact_freq_subset = waveform_subset[
        np.isclose(waveform_subset["freq_hz"], requested_freq_hz, atol=1e-6, equal_nan=False)
    ].copy()
    if frequency_mode == "exact":
        subset = exact_freq_subset.copy()
        frequency_bucket_mode = "exact_frequency_bucket" if not subset.empty else "no_exact_frequency_support"
    elif not exact_freq_subset.empty:
        subset = exact_freq_subset.copy()
        frequency_bucket_mode = "exact_frequency_bucket"
    else:
        subset = waveform_subset.copy()
        frequency_bucket_mode = "interpolated_frequency_blend"

    subset = subset.sort_values(["freq_hz", output_context["output_metric"]]).reset_index(drop=True)
    if subset.empty:
        return None

    representative_cycle_selection_mode = "warm_tail"
    support_profiles: list[dict[str, Any]] = []
    for row in subset.to_dict(orient="records"):
        analysis = analyses_by_test_id.get(str(row["test_id"]))
        if analysis is None:
            continue
        selection_info = _select_representative_cycle_indices(
            analysis=analysis,
            cycle_selection_mode=representative_cycle_selection_mode,
        )
        profile = build_representative_cycle_profile(
            analysis=analysis,
            current_channel=current_channel,
            voltage_channel=voltage_channel,
            field_channel=field_channel,
            points_per_cycle=points_per_cycle,
            cycle_indices=selection_info["selected_cycle_indices"],
        )
        if profile.empty:
            continue
        support_profiles.append(
            {
                "meta": row,
                "profile": profile,
                "cycle_selection": selection_info,
                "startup_diagnostics": _build_startup_diagnostics(analysis=analysis, field_channel=field_channel),
            }
        )

    if not support_profiles:
        return None

    target_cycle_count = (
        max(float(target_cycle_count), 0.25)
        if target_cycle_count is not None
        else None
    )
    if field_only_route and finite_cycle_mode:
        target_cycle_count = _normalize_field_finite_cycle_count(target_cycle_count)
        if target_cycle_count is None:
            return None
    preview_tail_cycles = max(float(preview_tail_cycles), 0.0)

    target_profile = _build_target_template(
        waveform_type=waveform_type,
        freq_hz=requested_freq_hz,
        points_per_cycle=points_per_cycle,
        force_rounded_triangle=field_only_route,
    )
    target_profile["target_output"] = target_profile["target_output_normalized"] * shape_target_output_pp / 2.0
    target_profile["used_target_output"] = target_profile["target_output"]
    if output_context["target_column"] == "target_current_a":
        target_profile["target_current_a"] = target_profile["target_output"]
        target_profile["used_target_current_a"] = target_profile["used_target_output"]
    else:
        target_profile[output_context["target_column"]] = target_profile["target_output"]
        target_profile[output_context["used_target_column"]] = target_profile["used_target_output"]

    support_output_values = subset[output_context["output_metric"]].to_numpy(dtype=float)
    support_min = float(np.nanmin(support_output_values))
    support_max = float(np.nanmax(support_output_values))
    if field_only_route:
        nearest_row, support_selection_meta = _select_nearest_frequency_support_row(
            subset=subset,
            target_freq_hz=used_freq_hz,
        )
    else:
        nearest_row, support_selection_meta = _select_nearest_support_row(
            subset=subset,
            target_freq_hz=used_freq_hz,
            target_output_pp=target_output_pp,
            output_metric=output_context["output_metric"],
        )
    nearest_test_id = str(nearest_row["test_id"])
    nearest_support = next(
        item for item in support_profiles if str(item["meta"]["test_id"]) == nearest_test_id
    )
    field_support_weights = (
        _build_frequency_support_weight_table(
            support_profiles=support_profiles,
            target_freq_hz=used_freq_hz,
        )
        if field_only_route
        else pd.DataFrame()
    )
    support_amp_gain_pct = float(nearest_row.get("amp_gain_setting_mean", np.nan))
    if not np.isfinite(support_amp_gain_pct) or support_amp_gain_pct <= 0:
        support_amp_gain_pct = float(default_support_amp_gain_pct)
    output_scale = 1.0
    if target_output_type == "field":
        nearest_current_pp = float(nearest_row.get(current_metric, np.nan))
        nearest_output_pp = float(nearest_row.get(output_context["output_metric"], np.nan))
        if np.isfinite(nearest_current_pp) and nearest_current_pp > 0 and np.isfinite(nearest_output_pp):
            output_scale = nearest_output_pp / nearest_current_pp
    if field_only_route:
        lcr_policy = {
            "lcr_usage_mode": "disabled_for_field_shape_route",
            "lcr_weight": 0.0,
            "requested_lcr_weight": float(np.clip(lcr_blend_weight, 0.0, 1.0)),
            "exact_field_support_present": bool(not exact_freq_subset.empty),
            "lcr_phase_anchor_used": False,
            "lcr_gain_prior_used": False,
        }
    else:
        lcr_policy = resolve_lcr_runtime_policy(
            requested_lcr_weight=lcr_blend_weight,
            lcr_prior_available=bool(lcr_measurements is not None and not lcr_measurements.empty),
            exact_field_support_present=bool(output_context["output_type"] == "field" and not exact_freq_subset.empty),
            support_point_count=len(support_profiles),
            waveform_type=waveform_type,
            official_band_applied=bool(requested_freq_hz <= 5.0),
        )
    lcr_prior_table = pd.DataFrame()
    if (
        not field_only_route
        and lcr_measurements is not None
        and not lcr_measurements.empty
        and float(lcr_policy["lcr_weight"]) > 0.0
    ):
        lcr_prior_table = build_lcr_harmonic_prior(
            lcr_impedance_table=build_lcr_impedance_table(lcr_measurements),
            base_freq_hz=used_freq_hz,
            harmonics=range(1, max_harmonics + 1),
            daq_to_amp_gain=float(amp_gain_at_100_pct) * float(support_amp_gain_pct) / 100.0,
            output_scale=output_scale,
        )
    lcr_prior_used = bool(not lcr_prior_table.empty and float(lcr_policy["lcr_weight"]) > 0.0)
    finite_empirical_model = (
        synthesize_finite_empirical_compensation(
            finite_support_entries=finite_support_entries,
            waveform_type=waveform_type,
            freq_hz=requested_freq_hz,
            target_cycle_count=target_cycle_count,
            target_output_type=output_context["output_type"],
            target_output_pp=shape_target_output_pp,
            current_channel=current_channel,
            field_channel=field_channel,
            max_daq_voltage_pp=float(max_daq_voltage_pp),
            amp_gain_at_100_pct=float(amp_gain_at_100_pct),
            amp_gain_limit_pct=float(amp_gain_limit_pct),
            amp_max_output_pk_v=float(amp_max_output_pk_v),
            default_support_amp_gain_pct=float(default_support_amp_gain_pct),
            preview_tail_cycles=preview_tail_cycles,
        )
        if field_only_route and finite_cycle_mode and finite_support_entries
        else None
    )
    use_finite_empirical_route = bool(finite_empirical_model)

    if use_finite_empirical_route:
        command_profile = finite_empirical_model["command_profile"].copy()
        transfer_model = {}
        mode = str(finite_empirical_model["mode"])
        clamped_ratio = float("nan")
        clamp_fraction = 0.0
        estimated_output_lag_seconds = 0.0
    elif field_only_route:
        command_profile, transfer_model = _harmonic_inverse_field_only_compensation(
            support_profiles=support_profiles,
            target_profile=target_profile,
            target_freq_hz=used_freq_hz,
            output_signal_column=output_context["signal_column"],
            nearest_support=nearest_support,
            points_per_cycle=points_per_cycle,
            support_weight_table=field_support_weights,
            max_harmonics=max_harmonics,
            normalized_target_pp=shape_target_output_pp,
        )
        support_frequency_count = int(field_support_weights["freq_hz"].dropna().nunique()) if not field_support_weights.empty else 1
        if support_frequency_count > 1:
            mode = (
                "harmonic_inverse_field_only_freq_blend"
                if available_freq_min <= requested_freq_hz <= available_freq_max
                else "harmonic_inverse_field_only_freq_clamped"
            )
        else:
            mode = "harmonic_inverse_field_only_single_support"
        clamped_ratio = float("nan")
        clamp_fraction = 0.0
    else:
        if len(support_profiles) == 1:
            command_profile, transfer_model = _harmonic_inverse_compensation(
                support_profiles=support_profiles,
                target_profile=target_profile,
                target_output_pp=target_output_pp,
                target_freq_hz=used_freq_hz,
                output_metric=output_context["output_metric"],
                output_signal_column=output_context["signal_column"],
                nearest_support=nearest_support,
                points_per_cycle=points_per_cycle,
                allow_output_extrapolation=allow_output_extrapolation,
                max_harmonics=max_harmonics,
                lcr_prior_table=lcr_prior_table,
                lcr_blend_weight=float(lcr_policy["lcr_weight"]),
            )
            mode = "harmonic_inverse_single_support"
            nearest_output_pp = float(nearest_row[output_context["output_metric"]])
            clamped_ratio = (
                target_output_pp / nearest_output_pp
                if np.isfinite(nearest_output_pp) and nearest_output_pp != 0
                else float("nan")
            )
            clamp_fraction = (
                0.0
                if allow_output_extrapolation
                else (1.0 if target_output_pp != float(nearest_row[output_context["output_metric"]]) else 0.0)
            )
        else:
            command_profile, transfer_model = _harmonic_inverse_compensation(
                support_profiles=support_profiles,
                target_profile=target_profile,
                target_output_pp=target_output_pp,
                target_freq_hz=used_freq_hz,
                output_metric=output_context["output_metric"],
                output_signal_column=output_context["signal_column"],
                nearest_support=nearest_support,
                points_per_cycle=points_per_cycle,
                allow_output_extrapolation=allow_output_extrapolation,
                max_harmonics=max_harmonics,
                lcr_prior_table=lcr_prior_table,
                lcr_blend_weight=float(lcr_policy["lcr_weight"]),
            )
            support_frequency_count = int(subset["freq_hz"].dropna().nunique())
            if support_frequency_count > 1:
                mode = "harmonic_inverse_freq_interpolated" if available_freq_min <= requested_freq_hz <= available_freq_max else "harmonic_inverse_freq_clamped"
            else:
                mode = "harmonic_inverse"
            clamped_ratio = float("nan")
            clamp_fraction = (
                0.0
                if allow_output_extrapolation
                else (0.0 if support_min <= target_output_pp <= support_max else 1.0)
            )

    if not use_finite_empirical_route:
        estimated_output_lag_seconds = _estimate_weighted_output_lag_seconds(
            support_profiles=support_profiles,
            output_signal_column=output_context["signal_column"],
            output_metric=output_context["output_metric"],
            target_freq_hz=used_freq_hz,
            target_output_pp=target_output_pp,
            prefer_frequency_only=field_only_route,
        )
        period_s = 1.0 / requested_freq_hz if requested_freq_hz > 0 else 1.0
        max_reasonable_lag = 0.45 * period_s
        estimated_output_lag_seconds = float(
            np.clip(estimated_output_lag_seconds, -max_reasonable_lag, max_reasonable_lag)
        )

    if use_finite_empirical_route:
        pass
    elif finite_cycle_mode and target_cycle_count is not None:
        command_profile = _expand_command_profile_to_finite_run(
            command_cycle_profile=command_profile,
            waveform_type=waveform_type,
            freq_hz=requested_freq_hz,
            target_output_pp=shape_target_output_pp,
            target_cycle_count=target_cycle_count,
            preview_tail_cycles=preview_tail_cycles,
            output_context=output_context,
            phase_lead_seconds=estimated_output_lag_seconds,
            points_per_cycle=points_per_cycle,
            force_rounded_triangle_target=field_only_route,
        )
        command_profile = apply_command_hardware_model(
            command_waveform=command_profile,
            max_daq_voltage_pp=float(max_daq_voltage_pp),
            amp_gain_at_100_pct=float(amp_gain_at_100_pct),
            support_amp_gain_pct=support_amp_gain_pct,
            amp_gain_limit_pct=float(amp_gain_limit_pct),
            amp_max_output_pk_v=float(amp_max_output_pk_v),
            preserve_start_voltage=True,
        )
    else:
        command_profile = apply_command_hardware_model(
            command_waveform=command_profile,
            max_daq_voltage_pp=float(max_daq_voltage_pp),
            amp_gain_at_100_pct=float(amp_gain_at_100_pct),
            support_amp_gain_pct=support_amp_gain_pct,
            amp_gain_limit_pct=float(amp_gain_limit_pct),
            amp_max_output_pk_v=float(amp_max_output_pk_v),
            preserve_start_voltage=False,
        )
        command_profile["aligned_target_output"] = command_profile["target_output"]
        command_profile["aligned_used_target_output"] = command_profile["used_target_output"]
        if output_context["target_column"] == "target_current_a":
            command_profile["aligned_target_current_a"] = command_profile["target_current_a"]
            command_profile["aligned_used_target_current_a"] = command_profile["used_target_current_a"]
        else:
            aligned_target_column = output_context["target_column"].replace("target_", "aligned_target_")
            aligned_used_target_column = output_context["used_target_column"].replace("used_target_", "aligned_used_target_")
            command_profile[aligned_target_column] = command_profile[output_context["target_column"]]
            command_profile[aligned_used_target_column] = command_profile[output_context["used_target_column"]]

    command_profile["target_output_pp"] = shape_target_output_pp
    command_profile["shape_target_output_pp"] = shape_target_output_pp
    command_profile["requested_target_output_pp"] = requested_target_output_pp
    command_profile["field_only_target_shape"] = "rounded_triangle" if field_only_route else waveform_type
    command_profile["target_shape_locked"] = bool(field_only_route)
    command_profile["target_pp_locked"] = bool(field_only_route)
    command_profile["waveform_type"] = waveform_type
    command_profile["freq_hz"] = float(freq_hz)
    command_profile["finite_cycle_mode"] = finite_cycle_mode
    command_profile["target_cycle_count"] = target_cycle_count if finite_cycle_mode else np.nan
    command_profile["preview_tail_cycles"] = preview_tail_cycles if finite_cycle_mode else np.nan
    command_profile["estimated_output_lag_seconds"] = estimated_output_lag_seconds
    command_profile["estimated_output_lag_cycles"] = (
        estimated_output_lag_seconds * requested_freq_hz if requested_freq_hz > 0 else float("nan")
    )

    support_table = subset[
        [
            column
            for column in dict.fromkeys(
                (
                "test_id",
                "waveform_type",
                "freq_hz",
                "current_pp_target_a",
                output_context["output_metric"],
                "achieved_current_pp_a_mean",
                "daq_input_v_pp_mean",
                "amp_gain_setting_mean",
                "achieved_bz_mT_pp_mean",
                "achieved_bmag_mT_pp_mean",
                )
            )
            if column in subset.columns
        ]
    ].reset_index(drop=True)
    support_table = support_table.assign(
        freq_distance_hz=(support_table["freq_hz"] - used_freq_hz).abs()
        if "freq_hz" in support_table.columns
        else np.nan,
        output_distance=(
            support_table[output_context["output_metric"]] - target_output_pp
        ).abs()
        if output_context["output_metric"] in support_table.columns
        else np.nan,
    )
    if field_only_route and "freq_distance_hz" in support_table.columns:
        support_table = support_table.sort_values(["freq_distance_hz", "freq_hz"]).reset_index(drop=True)
    elif "freq_distance_hz" in support_table.columns and "output_distance" in support_table.columns:
        freq_range = max(float(subset["freq_hz"].max()) - float(subset["freq_hz"].min()), 1e-9)
        output_range = max(
            float(subset[output_context["output_metric"]].max()) - float(subset[output_context["output_metric"]].min()),
            1e-9,
        )
        support_table["combined_distance"] = np.sqrt(
            np.square(support_table["freq_distance_hz"] / freq_range)
            + np.square(support_table["output_distance"] / output_range)
        )
        support_table = support_table.sort_values(["combined_distance", "freq_distance_hz", "output_distance"]).reset_index(drop=True)
    if use_finite_empirical_route:
        support_table = finite_empirical_model["support_table"].copy()

    nearest_profile = nearest_support["profile"].copy()
    nearest_profile["nearest_test_id"] = nearest_test_id
    nearest_profile["nearest_current_pp_a"] = float(nearest_row.get(current_metric, np.nan))
    nearest_profile["nearest_output_pp"] = float(nearest_row.get(output_context["output_metric"], np.nan))
    if field_only_route:
        support_profile = _build_field_only_support_profile_preview(
            support_profiles=support_profiles,
            target_freq_hz=used_freq_hz,
            points_per_cycle=points_per_cycle,
            output_freq_hz=requested_freq_hz,
            output_signal_column=output_context["signal_column"],
            support_weight_table=field_support_weights,
            normalized_target_pp=shape_target_output_pp,
        )
    else:
        support_profile = _build_weighted_support_profile_preview(
            support_profiles=support_profiles,
            target_freq_hz=used_freq_hz,
            target_output_pp=target_output_pp,
            output_metric=output_context["output_metric"],
            points_per_cycle=points_per_cycle,
            output_freq_hz=requested_freq_hz,
        )
    nearest_profile_preview = (
        _expand_measured_profile_preview(
            profile=nearest_profile,
            waveform_type=waveform_type,
            freq_hz=requested_freq_hz,
            target_cycle_count=target_cycle_count,
            preview_tail_cycles=preview_tail_cycles,
            output_signal_column=output_context["signal_column"],
            phase_lead_seconds=estimated_output_lag_seconds,
            points_per_cycle=points_per_cycle,
        )
        if finite_cycle_mode and target_cycle_count is not None
        else nearest_profile.copy()
    )
    support_profile_preview = (
        _expand_measured_profile_preview(
            profile=support_profile,
            waveform_type=waveform_type,
            freq_hz=requested_freq_hz,
            target_cycle_count=target_cycle_count,
            preview_tail_cycles=preview_tail_cycles,
            output_signal_column=output_context["signal_column"],
            phase_lead_seconds=estimated_output_lag_seconds,
            points_per_cycle=points_per_cycle,
        )
        if finite_cycle_mode and target_cycle_count is not None
        else support_profile.copy()
    )
    if use_finite_empirical_route:
        nearest_profile = command_profile.copy()
        nearest_profile_preview = command_profile.copy()
        support_profile = command_profile.copy()
        support_profile_preview = command_profile.copy()
    else:
        command_profile = _attach_expected_response_columns(
            command_profile=command_profile,
            support_profile_preview=support_profile_preview,
            target_output_type=target_output_type,
            finite_cycle_mode=finite_cycle_mode,
        )
    if not finite_cycle_mode:
        command_profile = _apply_forward_harmonic_prediction(
            command_profile=command_profile,
            transfer_model=transfer_model,
            target_output_type=target_output_type,
        )
        command_profile = _phase_register_command_profile(
            command_profile=command_profile,
            voltage_column="limited_voltage_v",
        )
    if field_only_route:
        command_profile = _attach_field_prediction_metrics(
            command_profile=command_profile,
            support_weight_table=field_support_weights,
            finite_cycle_mode=finite_cycle_mode,
        )
    finite_cycle_metrics: dict[str, Any] = {}
    finite_cycle_metrics_before: dict[str, Any] = {}
    finite_cycle_metrics_after: dict[str, Any] = {}
    finite_metric_improvement_summary: dict[str, Any] = {}
    finite_terminal_correction_applied = False
    finite_terminal_correction_reason = ""
    finite_terminal_correction_gain = 1.0
    if field_only_route and finite_cycle_mode:
        (
            command_profile,
            finite_metrics_before,
            finite_metrics_after,
            finite_metric_improvement_summary,
        ) = _apply_finite_terminal_tail_correction(
            command_profile=command_profile,
            freq_hz=requested_freq_hz,
            max_daq_voltage_pp=float(max_daq_voltage_pp),
            amp_gain_at_100_pct=float(amp_gain_at_100_pct),
            support_amp_gain_pct=support_amp_gain_pct,
            amp_gain_limit_pct=float(amp_gain_limit_pct),
            amp_max_output_pk_v=float(amp_max_output_pk_v),
        )
        command_profile = _apply_finite_active_shape_fit_correction(command_profile)
        command_profile = _apply_finite_output_continuity_guard(command_profile)
        command_profile = _attach_field_prediction_metrics(
            command_profile=command_profile,
            support_weight_table=field_support_weights,
            finite_cycle_mode=finite_cycle_mode,
        )
        finite_metrics = evaluate_finite_cycle_metrics(command_profile)
        command_profile = attach_finite_cycle_metrics(
            command_profile=command_profile,
            metrics=finite_metrics,
        )
        finite_cycle_metrics = finite_metrics.to_dict()
        finite_cycle_metrics_before = finite_metrics_before or finite_cycle_metrics
        finite_cycle_metrics_after = finite_metrics_after or finite_cycle_metrics
        finite_terminal_correction_applied = bool(_first_boolish(command_profile.get("finite_terminal_correction_applied")))
        finite_terminal_correction_reason = _first_text(command_profile.get("finite_terminal_correction_reason"))
        finite_terminal_correction_gain = _first_numeric(command_profile.get("finite_terminal_correction_gain"))
    finite_support_used = bool(use_finite_empirical_route)
    finite_support_fallback_reason = None
    request_route = finite_empirical_model.get("request_route") if use_finite_empirical_route else "fallback"
    plot_source = finite_empirical_model.get("plot_source") if use_finite_empirical_route else "steady_state_harmonic_preview"
    support_tests_used = finite_empirical_model.get("support_tests_used", []) if use_finite_empirical_route else []
    support_count_used = int(finite_empirical_model.get("support_count_used", 0)) if use_finite_empirical_route else 0
    support_cycle_count = finite_empirical_model.get("support_cycle_count") if use_finite_empirical_route else None
    support_freq_hz = finite_empirical_model.get("support_freq_hz") if use_finite_empirical_route else None
    selected_support_waveform = finite_empirical_model.get("selected_support_waveform") if use_finite_empirical_route else None
    zero_padded_fraction = finite_empirical_model.get("zero_padded_fraction") if use_finite_empirical_route else None
    support_waveform_role = finite_empirical_model.get("support_waveform_role") if use_finite_empirical_route else None
    support_family_sensitivity_flag = finite_empirical_model.get("support_family_sensitivity_flag") if use_finite_empirical_route else False
    support_family_sensitivity_reason = finite_empirical_model.get("support_family_sensitivity_reason") if use_finite_empirical_route else None
    support_family_selection_mode = finite_empirical_model.get("support_family_selection_mode") if use_finite_empirical_route else None
    user_requested_support_family = finite_empirical_model.get("user_requested_support_family") if use_finite_empirical_route else None
    candidate_support_families = finite_empirical_model.get("candidate_support_families", []) if use_finite_empirical_route else []
    support_family_warning = finite_empirical_model.get("support_family_warning") if use_finite_empirical_route else None
    support_family_sensitivity_level = finite_empirical_model.get("support_family_sensitivity_level") if use_finite_empirical_route else None
    support_family_override_applied = finite_empirical_model.get("support_family_override_applied") if use_finite_empirical_route else False
    support_family_override_reason = finite_empirical_model.get("support_family_override_reason") if use_finite_empirical_route else None
    support_blended_output_nonzero = finite_empirical_model.get("support_blended_output_nonzero") if use_finite_empirical_route else None
    support_blended_zero_guard_applied = finite_empirical_model.get("support_blended_zero_guard_applied") if use_finite_empirical_route else False
    support_spike_filtered_count = int(finite_empirical_model.get("support_spike_filtered_count", 0) or 0) if use_finite_empirical_route else 0
    support_source_spike_detected = bool(finite_empirical_model.get("support_source_spike_detected", False)) if use_finite_empirical_route else False
    support_blend_boundary_count = int(finite_empirical_model.get("support_blend_boundary_count", 0) or 0) if use_finite_empirical_route else 0
    command_extension_applied = finite_empirical_model.get("command_extension_applied") if use_finite_empirical_route else False
    command_extension_reason = finite_empirical_model.get("command_extension_reason") if use_finite_empirical_route else None
    command_stop_policy = finite_empirical_model.get("command_stop_policy") if use_finite_empirical_route else None
    predicted_extension_applied = finite_empirical_model.get("predicted_extension_applied") if use_finite_empirical_route else False
    support_extension_applied = finite_empirical_model.get("support_extension_applied") if use_finite_empirical_route else False
    support_coverage_mode = finite_empirical_model.get("support_coverage_mode") if use_finite_empirical_route else None
    partial_support_coverage = finite_empirical_model.get("partial_support_coverage") if use_finite_empirical_route else False
    support_observed_end_s = finite_empirical_model.get("support_observed_end_s") if use_finite_empirical_route else None
    support_observed_coverage_ratio = finite_empirical_model.get("support_observed_coverage_ratio") if use_finite_empirical_route else None
    support_padding_gap_s = finite_empirical_model.get("support_padding_gap_s") if use_finite_empirical_route else None
    support_resampled_to_target_window = finite_empirical_model.get("support_resampled_to_target_window") if use_finite_empirical_route else False
    hybrid_fill_applied = finite_empirical_model.get("hybrid_fill_applied") if use_finite_empirical_route else False
    hybrid_fill_start_s = finite_empirical_model.get("hybrid_fill_start_s") if use_finite_empirical_route else None
    hybrid_fill_end_s = finite_empirical_model.get("hybrid_fill_end_s") if use_finite_empirical_route else None
    finite_prediction_source = finite_empirical_model.get("finite_prediction_source") if use_finite_empirical_route else None
    predicted_cover_reason = finite_empirical_model.get("predicted_cover_reason") if use_finite_empirical_route else None
    support_cover_reason = finite_empirical_model.get("support_cover_reason") if use_finite_empirical_route else None
    finite_cycle_decomposition = _finite_cycle_decomposition_metadata(
        target_cycle_count,
        finite_support_used=use_finite_empirical_route,
        selected_support_id=str(finite_empirical_model.get("selected_support_id")) if use_finite_empirical_route else None,
        selected_support_cycle_count=finite_empirical_model.get("support_cycle_count") if use_finite_empirical_route else None,
    )
    if field_only_route:
        if "physical_target_output_mT" not in command_profile.columns and "target_field_mT" in command_profile.columns:
            command_profile["physical_target_output_mT"] = command_profile["target_field_mT"]
        if "support_reference_output_mT" not in command_profile.columns and "support_scaled_field_mT" in command_profile.columns:
            command_profile["support_reference_output_mT"] = command_profile["support_scaled_field_mT"]
        command_profile["target_shape_family"] = "rounded_triangle"
        command_profile["target_pp_fixed"] = float(FIELD_ROUTE_NORMALIZED_TARGET_PP)
        command_profile["support_family_used"] = selected_support_waveform
        command_profile["support_family_requested"] = user_requested_support_family or waveform_type
    if (
        field_only_route
        and finite_cycle_mode
        and not use_finite_empirical_route
        and finite_cycle_decomposition["finite_cycle_decomposition_mode"] in {
            "fallback_no_exact_1_75_support",
            "unsupported_fractional_cycle_request",
        }
    ):
        command_profile = _suppress_unsafe_finite_prediction(
            command_profile,
            reason=(
                "unsupported_cycle_count"
                if finite_cycle_decomposition["finite_cycle_decomposition_mode"] == "unsupported_fractional_cycle_request"
                else "no_exact_1_75_support"
            ),
        )
    target_active_end_s = (
        float(np.nanmax(pd.to_numeric(command_profile.loc[command_profile["is_active_target"] == True, "time_s"], errors="coerce").to_numpy(dtype=float)))
        if finite_cycle_mode and "is_active_target" in command_profile.columns and bool(pd.Series(command_profile["is_active_target"]).fillna(False).any())
        else None
    )
    if use_finite_empirical_route and target_active_end_s is not None and np.isfinite(target_active_end_s):
        command_profile, final_extension_metadata = _extend_finite_active_window_signals(
            command_profile,
            active_end_s=float(target_active_end_s),
            command_columns=("recommended_voltage_v", "limited_voltage_v"),
            predicted_columns=("expected_field_mT", "expected_output", "predicted_field_mT"),
            support_columns=("support_scaled_field_mT",),
        )
        command_extension_applied = bool(command_extension_applied or final_extension_metadata["command_extension_applied"])
        command_extension_reason = command_extension_reason or final_extension_metadata["command_extension_reason"]
        command_stop_policy = final_extension_metadata["command_stop_policy"] or command_stop_policy
        predicted_extension_applied = bool(predicted_extension_applied or final_extension_metadata["predicted_extension_applied"])
        support_extension_applied = bool(support_extension_applied or final_extension_metadata["support_extension_applied"])
        support_coverage_mode = final_extension_metadata["support_coverage_mode"] or support_coverage_mode
        partial_support_coverage = bool(partial_support_coverage or final_extension_metadata["partial_support_coverage"])
        support_observed_end_s = final_extension_metadata["support_observed_end_s"]
        support_observed_coverage_ratio = final_extension_metadata["support_observed_coverage_ratio"]
        support_padding_gap_s = final_extension_metadata["support_padding_gap_s"]
        support_resampled_to_target_window = bool(
            support_resampled_to_target_window or final_extension_metadata["support_resampled_to_target_window"]
        )
        hybrid_fill_applied = bool(hybrid_fill_applied or final_extension_metadata["hybrid_fill_applied"])
        hybrid_fill_start_s = final_extension_metadata["hybrid_fill_start_s"]
        hybrid_fill_end_s = final_extension_metadata["hybrid_fill_end_s"]
        finite_prediction_source = final_extension_metadata["finite_prediction_source"] or finite_prediction_source
        predicted_cover_reason = final_extension_metadata["predicted_cover_reason"] or predicted_cover_reason
        support_cover_reason = final_extension_metadata["support_cover_reason"] or support_cover_reason
    startup_diagnostics = dict(
        finite_empirical_model.get("startup_diagnostics", {})
        if use_finite_empirical_route
        else nearest_support.get("startup_diagnostics", {})
    )
    startup_diagnostics.setdefault("source_test_id", nearest_test_id)
    startup_transient_metadata: dict[str, Any] = {"startup_transient_applied": False}
    if field_only_route:
        command_profile, startup_transient_metadata = _apply_startup_transient_prediction(
            command_profile=command_profile,
            startup_diagnostics=startup_diagnostics,
            target_output_type=output_context["output_type"],
            freq_hz=requested_freq_hz,
            transition_cycles=startup_transition_cycles,
            correction_strength=startup_correction_strength,
        )
    finite_route_mode = None
    finite_route_reason = None
    finite_route_warning = None
    if field_only_route and finite_cycle_mode and not use_finite_empirical_route:
        finite_support_fallback_reason = (
            "no_finite_support_entries"
            if not finite_support_entries
            else "finite_empirical_route_unavailable_using_steady_state_fallback"
        )
        finite_route_mode = "steady_state_harmonic_expanded"
        finite_route_reason = "finite_support_unavailable"
        finite_route_warning = "finite transient data not used"
        if bool(_first_boolish(command_profile.get("unsafe_fallback_suppressed"))):
            suppressed_reason = _first_text(command_profile.get("finite_prediction_unavailable_reason"))
            if suppressed_reason == "unsupported_cycle_count":
                finite_support_fallback_reason = "unsupported_cycle_count"
                finite_route_mode = "finite_unavailable_unsupported_cycle_count"
                finite_route_reason = "unsupported_cycle_count"
                finite_route_warning = "unsupported finite cycle request suppressed"
            else:
                finite_support_fallback_reason = "no_exact_1_75_support"
                finite_route_mode = "finite_unavailable_no_exact_1_75_support"
                finite_route_reason = "no_exact_1_75_support"
                finite_route_warning = "unsafe fallback prediction suppressed"
    elif use_finite_empirical_route:
        finite_route_mode = str(finite_empirical_model.get("mode") or "finite_empirical_field_support")
        finite_route_reason = (
            "exact_finite_support_match"
            if str(finite_empirical_model.get("request_route")) == "exact"
            else "nearest_finite_support_blend"
        )
    if use_finite_empirical_route:
        nearest_test_id = str(finite_empirical_model.get("support_test_id") or nearest_test_id)
        support_selection_meta = {
            "selected_support_id": finite_empirical_model.get("selected_support_id") or nearest_test_id,
            "selected_support_family": finite_empirical_model.get("selected_support_family"),
            "support_selection_reason": finite_empirical_model.get("support_selection_reason"),
            "support_family_metric": finite_empirical_model.get("support_family_metric"),
            "support_family_value": finite_empirical_model.get("support_family_value"),
            "estimated_family_level": None,
            "support_family_lock_applied": bool(finite_empirical_model.get("support_family_lock_applied", False)),
            "support_bz_to_current_ratio": finite_empirical_model.get("support_bz_to_current_ratio"),
        }
        support_min = min(support_min, float(finite_empirical_model.get("support_output_pp", support_min)))
        support_max = max(support_max, float(finite_empirical_model.get("support_output_pp", support_max)))
    finite_stop_policy = _summarize_finite_command_stop_policy(
        command_profile=command_profile,
        target_active_end_s=target_active_end_s,
        phase_lead_seconds=estimated_output_lag_seconds if (field_only_route and finite_cycle_mode and not use_finite_empirical_route) else 0.0,
        finite_command_stop_policy=(
            "finite_empirical_support_timing"
            if use_finite_empirical_route
            else (
                "phase_lead_sampling_only_preserve_target_window"
                if field_only_route and finite_cycle_mode
                else "not_applicable"
            )
        ),
        phase_lead_applied_to_sampling_only=bool(field_only_route and finite_cycle_mode),
    )
    finite_signal_consistency: dict[str, Any] = {}
    if field_only_route and finite_cycle_mode:
        finite_signal_consistency = build_finite_signal_consistency_summary(
            command_profile,
            finite_support_used=finite_support_used,
            support_input_field_pp=(
                finite_empirical_model.get("support_output_pp")
                if use_finite_empirical_route
                else None
            ),
            target_active_end_s=finite_stop_policy["target_active_end_s"],
            command_nonzero_end_s=finite_stop_policy["command_nonzero_end_s"],
        )
        finite_stop_policy = {**finite_stop_policy, **{
            "target_active_end_s": finite_signal_consistency["target_active_end_s"],
            "command_nonzero_end_s": finite_signal_consistency["command_nonzero_end_s"],
            "command_early_stop_s": finite_signal_consistency["command_early_stop_s"],
            "command_extends_through_target_end": finite_signal_consistency["command_covers_target_end"],
            "early_command_cutoff_warning": "command_early_stop" in str(finite_signal_consistency["finite_signal_consistency_status"]).split("|"),
        }}
        support_spike_filtered_count = int(finite_signal_consistency.get("support_spike_filtered_count", support_spike_filtered_count) or 0)
        support_source_spike_detected = bool(finite_signal_consistency.get("support_source_spike_detected", support_source_spike_detected))
        support_blend_boundary_count = int(finite_signal_consistency.get("support_blend_boundary_count", support_blend_boundary_count) or 0)
    command_nonzero_end_s = finite_stop_policy["command_nonzero_end_s"]
    target_active_end_s = finite_stop_policy["target_active_end_s"]
    early_command_cutoff_warning = finite_stop_policy["early_command_cutoff_warning"]
    if field_only_route and finite_cycle_mode:
        command_profile["finite_route_mode"] = finite_route_mode
        command_profile["finite_route_reason"] = finite_route_reason
        command_profile["finite_support_used"] = bool(finite_support_used)
        command_profile["finite_support_fallback_reason"] = finite_support_fallback_reason
        command_profile["finite_route_warning"] = finite_route_warning
        command_profile["request_route"] = request_route
        command_profile["plot_source"] = plot_source
        command_profile["support_count_used"] = int(support_count_used)
        command_profile["support_tests_used"] = "|".join(str(item) for item in support_tests_used)
        command_profile["requested_cycle_count"] = target_cycle_count
        command_profile["support_cycle_count"] = support_cycle_count
        command_profile["selected_support_cycle_count"] = support_cycle_count
        command_profile["support_freq_hz"] = support_freq_hz
        command_profile["selected_support_waveform"] = selected_support_waveform
        command_profile["exact_cycle_support_used"] = bool(
            use_finite_empirical_route and support_cycle_count is not None and np.isfinite(float(support_cycle_count))
            and target_cycle_count is not None and np.isfinite(float(target_cycle_count))
            and abs(float(support_cycle_count) - float(target_cycle_count)) <= 0.05
        )
        command_profile["support_family_selection_mode"] = support_family_selection_mode
        command_profile["user_requested_support_family"] = user_requested_support_family
        command_profile["candidate_support_families"] = "|".join(str(item) for item in candidate_support_families)
        command_profile["support_family_warning"] = support_family_warning
        command_profile["support_family_sensitivity_level"] = support_family_sensitivity_level
        command_profile["support_family_override_applied"] = bool(support_family_override_applied)
        command_profile["support_family_override_reason"] = support_family_override_reason
        command_profile["zero_padded_fraction"] = zero_padded_fraction
        command_profile["support_blended_output_nonzero"] = support_blended_output_nonzero
        command_profile["support_blended_zero_guard_applied"] = bool(support_blended_zero_guard_applied)
        command_profile = command_profile.assign(
            finite_cycle_decomposition_mode=finite_cycle_decomposition["finite_cycle_decomposition_mode"],
            target_integer_cycle_count=finite_cycle_decomposition["target_integer_cycle_count"],
            target_terminal_fraction=finite_cycle_decomposition["target_terminal_fraction"],
            interior_support_id=finite_cycle_decomposition["interior_support_id"],
            terminal_tail_support_id=finite_cycle_decomposition["terminal_tail_support_id"],
            terminal_tail_fraction=finite_cycle_decomposition["terminal_tail_fraction"],
            cycle_semantics_warning=finite_cycle_decomposition["cycle_semantics_warning"],
            whole_support_substitution_used=finite_cycle_decomposition["whole_support_substitution_used"],
            whole_support_substitution_valid=finite_cycle_decomposition["whole_support_substitution_valid"],
            finite_cycle_policy_version=finite_cycle_decomposition["finite_cycle_policy_version"],
            supported_cycle_counts=str(finite_cycle_decomposition["supported_cycle_counts"]),
        )
        command_profile["command_extension_applied"] = bool(command_extension_applied)
        command_profile["command_extension_reason"] = command_extension_reason
        command_profile["command_stop_policy"] = command_stop_policy
        command_profile["predicted_extension_applied"] = bool(predicted_extension_applied)
        command_profile["support_extension_applied"] = bool(support_extension_applied)
        command_profile["support_coverage_mode"] = support_coverage_mode
        command_profile["partial_support_coverage"] = bool(partial_support_coverage)
        command_profile["support_observed_end_s"] = support_observed_end_s
        command_profile["support_observed_coverage_ratio"] = support_observed_coverage_ratio
        command_profile["support_padding_gap_s"] = support_padding_gap_s
        command_profile["support_resampled_to_target_window"] = bool(support_resampled_to_target_window)
        command_profile["hybrid_fill_applied"] = bool(hybrid_fill_applied)
        command_profile["hybrid_fill_start_s"] = hybrid_fill_start_s
        command_profile["hybrid_fill_end_s"] = hybrid_fill_end_s
        command_profile["finite_prediction_source"] = finite_prediction_source
        command_profile["predicted_cover_reason"] = predicted_cover_reason
        command_profile["support_cover_reason"] = support_cover_reason
        command_profile["finite_command_stop_policy"] = finite_stop_policy["finite_command_stop_policy"]
        command_profile["command_nonzero_end_s"] = finite_stop_policy["command_nonzero_end_s"]
        command_profile["target_active_end_s"] = finite_stop_policy["target_active_end_s"]
        command_profile["command_early_stop_s"] = finite_stop_policy["command_early_stop_s"]
        command_profile["command_extends_through_target_end"] = finite_stop_policy["command_extends_through_target_end"]
        command_profile["post_target_command_tail_s"] = finite_stop_policy["post_target_command_tail_s"]
        command_profile["phase_lead_seconds_applied"] = finite_stop_policy["phase_lead_seconds_applied"]
        command_profile["phase_lead_applied_to_sampling_only"] = finite_stop_policy["phase_lead_applied_to_sampling_only"]
        command_profile["early_command_cutoff_warning"] = finite_stop_policy["early_command_cutoff_warning"]
        command_profile["finite_command_nonzero_end_s"] = finite_signal_consistency.get("command_nonzero_end_s")
        command_profile["finite_predicted_nonzero_end_s"] = finite_signal_consistency.get("predicted_nonzero_end_s")
        command_profile["finite_support_nonzero_end_s"] = finite_signal_consistency.get("support_nonzero_end_s")
        command_profile["finite_command_covers_target_end"] = finite_signal_consistency.get("command_covers_target_end")
        command_profile["finite_predicted_covers_target_end"] = finite_signal_consistency.get("predicted_covers_target_end")
        command_profile["finite_support_covers_target_end"] = finite_signal_consistency.get("support_covers_target_end")
        command_profile["finite_signal_consistency_status"] = finite_signal_consistency.get("finite_signal_consistency_status")
        command_profile = command_profile.assign(
            predicted_jump_ratio=finite_signal_consistency.get("predicted_jump_ratio"),
            support_jump_ratio=finite_signal_consistency.get("support_jump_ratio"),
            max_predicted_jump_mT=finite_signal_consistency.get("max_predicted_jump_mT"),
            max_support_jump_mT=finite_signal_consistency.get("max_support_jump_mT"),
            max_jump_time_s=finite_signal_consistency.get("max_jump_time_s"),
            support_continuity_status=finite_signal_consistency.get("support_continuity_status"),
            support_splice_discontinuity_detected=finite_signal_consistency.get("support_splice_discontinuity_detected"),
            support_spike_filtered_count=finite_signal_consistency.get("support_spike_filtered_count"),
            support_source_spike_detected=finite_signal_consistency.get("support_source_spike_detected"),
            support_blend_boundary_count=finite_signal_consistency.get("support_blend_boundary_count"),
            finite_prediction_available=finite_signal_consistency.get("finite_prediction_available"),
            finite_prediction_unavailable_reason=finite_signal_consistency.get("finite_prediction_unavailable_reason"),
            support_prediction_masked=finite_signal_consistency.get("support_prediction_masked"),
            unsafe_fallback_suppressed=finite_signal_consistency.get("unsafe_fallback_suppressed"),
            user_warning_key=finite_signal_consistency.get("user_warning_key"),
            active_shape_corr=finite_signal_consistency.get("active_shape_corr"),
            active_shape_nrmse=finite_signal_consistency.get("active_shape_nrmse"),
            target_predicted_frequency_proxy_mismatch=finite_signal_consistency.get("target_predicted_frequency_proxy_mismatch"),
            predicted_spike_detected=finite_signal_consistency.get("predicted_spike_detected"),
            predicted_kink_detected=finite_signal_consistency.get("predicted_kink_detected"),
            max_slope_jump_ratio=finite_signal_consistency.get("max_slope_jump_ratio"),
        )
    nearest_cycle_selection = nearest_support.get("cycle_selection", {})
    startup_preview_profile = pd.DataFrame()
    startup_correction_applied = False
    startup_correction_factor = float("nan")
    startup_observed_output_ratio = float("nan")
    validation_base_profile = command_profile
    if not finite_cycle_mode and apply_startup_correction:
        startup_preview_profile, startup_metadata = _build_startup_corrected_preview(
            command_profile=command_profile,
            startup_diagnostics=startup_diagnostics,
            target_output_type=target_output_type,
            freq_hz=requested_freq_hz,
            max_daq_voltage_pp=float(max_daq_voltage_pp),
            amp_gain_at_100_pct=float(amp_gain_at_100_pct),
            support_amp_gain_pct=support_amp_gain_pct,
            amp_gain_limit_pct=float(amp_gain_limit_pct),
            amp_max_output_pk_v=float(amp_max_output_pk_v),
            transition_cycles=startup_transition_cycles,
            correction_strength=startup_correction_strength,
            preview_cycle_count=startup_preview_cycle_count,
        )
        startup_correction_applied = bool(startup_metadata.get("startup_correction_applied", False))
        startup_correction_factor = float(startup_metadata.get("startup_correction_factor", np.nan))
        startup_observed_output_ratio = float(startup_metadata.get("startup_observed_output_ratio", np.nan))
        if not startup_preview_profile.empty:
            validation_base_profile = startup_preview_profile
    compensation_sequence = _build_compensation_sequence_table(
        waveform_type=waveform_type,
        requested_freq_hz=requested_freq_hz,
        used_freq_hz=used_freq_hz,
        target_output_type=output_context["output_type"],
        target_output_pp=target_output_pp,
        target_output_unit=output_context["unit"],
        support_point_count=len(support_profiles),
        nearest_test_id=nearest_test_id,
        representative_cycle_selection_mode=representative_cycle_selection_mode,
        representative_cycle_indices=nearest_cycle_selection.get("selected_cycle_indices", []),
        representative_cycle_count=nearest_cycle_selection.get("selected_cycle_count", 0),
        available_cycle_count=nearest_cycle_selection.get("available_cycle_count", 0),
        used_lcr_prior=lcr_prior_used,
        lcr_blend_weight=float(lcr_policy["lcr_weight"]),
        finite_cycle_mode=finite_cycle_mode,
        preserve_start_voltage=bool(finite_cycle_mode),
        within_hardware_limits=bool(command_profile["within_hardware_limits"].iloc[0]),
        max_daq_voltage_pp=float(max_daq_voltage_pp),
        limited_voltage_pp=float(command_profile["limited_voltage_pp"].iloc[0]),
        estimated_output_lag_seconds=estimated_output_lag_seconds,
        startup_diagnostics=startup_diagnostics,
        startup_correction_applied=startup_correction_applied,
        startup_correction_factor=startup_correction_factor,
        startup_transition_cycles=startup_transition_cycles,
    )

    return {
        "mode": mode,
        "waveform_type": waveform_type,
        "freq_hz": requested_freq_hz,
        "requested_freq_hz": requested_freq_hz,
        "used_freq_hz": used_freq_hz,
        "frequency_bucket_mode": frequency_bucket_mode,
        "frequency_mode": (
            "exact"
            if frequency_mode == "exact"
            else (
                "single_frequency_only"
                if int(waveform_subset["freq_hz"].dropna().nunique()) == 1
                else (
                    "frequency_interpolated"
                    if available_freq_min <= requested_freq_hz <= available_freq_max
                    else "frequency_clamped"
                )
            )
        ),
        "available_freq_min": available_freq_min,
        "available_freq_max": available_freq_max,
        "frequency_support_count": int(waveform_subset["freq_hz"].dropna().nunique()),
        "target_output_type": output_context["output_type"],
        "target_output_label": output_context["label"],
        "target_output_unit": output_context["unit"],
        "target_output_pp": target_output_pp,
        "shape_target_output_pp": shape_target_output_pp,
        "requested_target_output_pp": requested_target_output_pp,
        "field_only_target_shape": "rounded_triangle" if field_only_route else waveform_type,
        "target_shape_family": "rounded_triangle" if field_only_route else waveform_type,
        "target_pp_fixed": float(FIELD_ROUTE_NORMALIZED_TARGET_PP) if field_only_route else None,
        "physical_target_output_column": "physical_target_output_mT" if field_only_route else None,
        "support_reference_output_column": "support_reference_output_mT" if field_only_route else None,
        "predicted_output_column": "predicted_field_mT" if field_only_route else "predicted_output",
        "support_family_used": selected_support_waveform,
        "support_family_requested": user_requested_support_family,
        "target_shape_locked": bool(field_only_route),
        "target_pp_locked": bool(field_only_route),
        "shape_selection_excludes": list(FIELD_ROUTE_SHAPE_SELECTION_EXCLUDES) if field_only_route else [],
        "target_current_pp_a": float(target_current_pp_a),
        "available_output_pp_min": support_min,
        "available_output_pp_max": support_max,
        "support_point_count": int(finite_empirical_model.get("support_count_used", len(support_profiles))) if use_finite_empirical_route else int(len(support_profiles)),
        "nearest_test_id": nearest_test_id,
        "selected_support_id": support_selection_meta.get("selected_support_id") or nearest_test_id,
        "selected_support_family": support_selection_meta.get("selected_support_family"),
        "support_selection_reason": support_selection_meta.get("support_selection_reason"),
        "support_family_metric": support_selection_meta.get("support_family_metric"),
        "support_family_value": support_selection_meta.get("support_family_value"),
        "estimated_family_level": support_selection_meta.get("estimated_family_level"),
        "support_family_lock_applied": bool(support_selection_meta.get("support_family_lock_applied", False)),
        "support_bz_to_current_ratio": support_selection_meta.get("support_bz_to_current_ratio"),
        "nearest_support_output_pp": (
            float(finite_empirical_model.get("support_output_pp", np.nan))
            if use_finite_empirical_route
            else float(nearest_row[output_context["output_metric"]])
        ),
        "scale_ratio_from_nearest": clamped_ratio,
        "phase_clamp_fraction": float(clamp_fraction),
        "finite_cycle_mode": finite_cycle_mode,
        "target_cycle_count": target_cycle_count if finite_cycle_mode else float("nan"),
        "preview_tail_cycles": preview_tail_cycles if finite_cycle_mode else float("nan"),
        "estimated_output_lag_seconds": estimated_output_lag_seconds,
        "estimated_output_lag_cycles": (
            estimated_output_lag_seconds * requested_freq_hz if requested_freq_hz > 0 else float("nan")
        ),
        "max_daq_voltage_pp": float(max_daq_voltage_pp),
        "amp_gain_at_100_pct": float(amp_gain_at_100_pct),
        "amp_gain_limit_pct": float(amp_gain_limit_pct),
        "amp_max_output_pk_v": float(amp_max_output_pk_v),
        "allow_output_extrapolation": bool(allow_output_extrapolation),
        "within_daq_limit": bool(command_profile["within_daq_limit"].iloc[0]),
        "within_hardware_limits": bool(command_profile["within_hardware_limits"].iloc[0]),
        "used_lcr_prior": lcr_prior_used,
        "lcr_blend_weight": float(lcr_policy["lcr_weight"]),
        "requested_lcr_weight": float(lcr_policy["requested_lcr_weight"]),
        "lcr_usage_mode": lcr_policy["lcr_usage_mode"],
        "exact_field_support_present": bool(lcr_policy["exact_field_support_present"]),
        "lcr_phase_anchor_used": bool(lcr_policy["lcr_phase_anchor_used"] and lcr_prior_used),
        "lcr_gain_prior_used": bool(lcr_policy["lcr_gain_prior_used"] and lcr_prior_used),
        "required_amp_gain_multiplier": float(command_profile["required_amp_gain_multiplier"].iloc[0]),
        "required_amp_gain_pct": float(command_profile["required_amp_gain_pct"].iloc[0]),
        "support_amp_gain_pct": float(command_profile["support_amp_gain_pct"].iloc[0]),
        "available_amp_gain_pct": float(command_profile["available_amp_gain_pct"].iloc[0]),
        "amp_output_pp_at_required": float(command_profile["amp_output_pp_at_required"].iloc[0]),
        "amp_output_pk_at_required": float(command_profile["amp_output_pk_at_required"].iloc[0]),
        "limited_voltage_pp": float(command_profile["limited_voltage_pp"].iloc[0]),
        "representative_cycle_selection_mode": representative_cycle_selection_mode,
        "representative_cycle_indices": nearest_cycle_selection.get("selected_cycle_indices", []),
        "representative_cycle_count_used": int(nearest_cycle_selection.get("selected_cycle_count", 0)),
        "startup_diagnostics": startup_diagnostics,
        "startup_correction_applied": startup_correction_applied,
        "startup_correction_factor": startup_correction_factor,
        "startup_observed_output_ratio": startup_observed_output_ratio,
        "startup_transition_cycles": float(startup_transition_cycles),
        "startup_correction_strength": float(startup_correction_strength),
        "startup_preview_cycle_count": int(startup_preview_cycle_count),
        "startup_transient_applied": bool(startup_transient_metadata.get("startup_transient_applied", False)),
        "startup_initial_field_offset_mT": startup_transient_metadata.get("startup_initial_field_offset_mT"),
        "startup_steady_field_offset_mT": startup_transient_metadata.get("startup_steady_field_offset_mT"),
        "startup_field_offset_delta_mT": startup_transient_metadata.get("startup_field_offset_delta_mT"),
        "startup_transient_transition_cycles": startup_transient_metadata.get("startup_transient_transition_cycles"),
        "startup_transient_reason": startup_transient_metadata.get("startup_transient_reason"),
        "startup_preview_profile": startup_preview_profile,
        "validation_base_profile": validation_base_profile,
        "compensation_sequence": compensation_sequence,
        "support_table": support_table,
        "command_profile": command_profile,
        "target_profile": target_profile,
        "max_harmonics_used": int(max_harmonics),
        "support_profile": support_profile,
        "support_profile_preview": support_profile_preview,
        "nearest_profile": nearest_profile,
        "nearest_profile_preview": nearest_profile_preview,
        "prediction_basis": "harmonic_forward_model" if not finite_cycle_mode else "support_scaled_preview",
        "field_shape_corr": _first_numeric(command_profile.get("field_shape_corr")),
        "field_shape_nrmse": _first_numeric(command_profile.get("field_shape_nrmse")),
        "field_support_freq_count": _first_numeric(command_profile.get("field_support_freq_count")),
        "field_support_test_ids": _first_text(command_profile.get("field_support_test_ids")),
        "finite_support_used": finite_support_used,
        "finite_route_mode": finite_route_mode,
        "finite_route_reason": finite_route_reason,
        "finite_route_warning": finite_route_warning,
        "finite_support_fallback_reason": finite_support_fallback_reason,
        "request_route": request_route,
        "plot_source": plot_source,
        "support_tests_used": support_tests_used,
        "support_count_used": support_count_used,
        "requested_cycle_count": target_cycle_count,
        "support_cycle_count": support_cycle_count,
        "selected_support_cycle_count": support_cycle_count,
        "exact_cycle_support_used": bool(
            support_cycle_count is not None
            and np.isfinite(float(support_cycle_count))
            and target_cycle_count is not None
            and np.isfinite(float(target_cycle_count))
            and abs(float(support_cycle_count) - float(target_cycle_count)) <= 0.05
        ),
        "support_freq_hz": support_freq_hz,
        "selected_support_waveform": selected_support_waveform,
        "support_waveform_role": support_waveform_role,
        "support_family_sensitivity_flag": bool(support_family_sensitivity_flag),
        "support_family_sensitivity_reason": support_family_sensitivity_reason,
        "support_family_selection_mode": support_family_selection_mode,
        "user_requested_support_family": user_requested_support_family,
        "candidate_support_families": candidate_support_families,
        "support_family_warning": support_family_warning,
        "support_family_sensitivity_level": support_family_sensitivity_level,
        "support_family_override_applied": bool(support_family_override_applied),
        "support_family_override_reason": support_family_override_reason,
        "zero_padded_fraction": zero_padded_fraction,
        "support_blended_output_nonzero": support_blended_output_nonzero,
        "support_blended_zero_guard_applied": bool(support_blended_zero_guard_applied),
        "support_spike_filtered_count": support_spike_filtered_count,
        "support_source_spike_detected": bool(support_source_spike_detected),
        "support_blend_boundary_count": support_blend_boundary_count,
        "predicted_jump_ratio": finite_signal_consistency.get("predicted_jump_ratio"),
        "support_jump_ratio": finite_signal_consistency.get("support_jump_ratio"),
        "max_predicted_jump_mT": finite_signal_consistency.get("max_predicted_jump_mT"),
        "max_support_jump_mT": finite_signal_consistency.get("max_support_jump_mT"),
        "max_jump_time_s": finite_signal_consistency.get("max_jump_time_s"),
        "support_continuity_status": finite_signal_consistency.get("support_continuity_status"),
        "support_splice_discontinuity_detected": finite_signal_consistency.get("support_splice_discontinuity_detected"),
        "finite_prediction_available": finite_signal_consistency.get("finite_prediction_available"),
        "finite_prediction_unavailable_reason": finite_signal_consistency.get("finite_prediction_unavailable_reason"),
        "support_prediction_masked": finite_signal_consistency.get("support_prediction_masked"),
        "unsafe_fallback_suppressed": finite_signal_consistency.get("unsafe_fallback_suppressed"),
        "user_warning_key": finite_signal_consistency.get("user_warning_key"),
        "active_shape_corr": finite_signal_consistency.get("active_shape_corr"),
        "active_shape_nrmse": finite_signal_consistency.get("active_shape_nrmse"),
        "target_predicted_frequency_proxy_mismatch": finite_signal_consistency.get("target_predicted_frequency_proxy_mismatch"),
        "predicted_spike_detected": finite_signal_consistency.get("predicted_spike_detected"),
        "predicted_kink_detected": finite_signal_consistency.get("predicted_kink_detected"),
        "max_slope_jump_ratio": finite_signal_consistency.get("max_slope_jump_ratio"),
        "active_shape_fit_applied": bool(_first_boolish(command_profile.get("active_shape_fit_applied"))),
        "active_shape_fit_strength": _first_numeric(command_profile.get("active_shape_fit_strength")),
        "active_shape_fit_reason": _first_text(command_profile.get("active_shape_fit_reason")),
        **finite_cycle_decomposition,
        "command_extension_applied": bool(command_extension_applied),
        "command_extension_reason": command_extension_reason,
        "command_stop_policy": command_stop_policy,
        "predicted_extension_applied": bool(predicted_extension_applied),
        "support_extension_applied": bool(support_extension_applied),
        "support_coverage_mode": support_coverage_mode,
        "partial_support_coverage": bool(partial_support_coverage),
        "support_observed_end_s": support_observed_end_s,
        "support_observed_coverage_ratio": support_observed_coverage_ratio,
        "support_padding_gap_s": support_padding_gap_s,
        "support_resampled_to_target_window": bool(support_resampled_to_target_window),
        "hybrid_fill_applied": bool(hybrid_fill_applied),
        "hybrid_fill_start_s": hybrid_fill_start_s,
        "hybrid_fill_end_s": hybrid_fill_end_s,
        "finite_prediction_source": finite_prediction_source,
        "predicted_cover_reason": predicted_cover_reason,
        "support_cover_reason": support_cover_reason,
        "finite_command_stop_policy": finite_stop_policy["finite_command_stop_policy"],
        "command_nonzero_end_s": command_nonzero_end_s,
        "target_active_end_s": target_active_end_s,
        "command_early_stop_s": finite_stop_policy["command_early_stop_s"],
        "command_extends_through_target_end": bool(finite_stop_policy["command_extends_through_target_end"]),
        "post_target_command_tail_s": finite_stop_policy["post_target_command_tail_s"],
        "phase_lead_seconds_applied": finite_stop_policy["phase_lead_seconds_applied"],
        "phase_lead_applied_to_sampling_only": bool(finite_stop_policy["phase_lead_applied_to_sampling_only"]),
        "early_command_cutoff_warning": bool(early_command_cutoff_warning),
        "finite_signal_consistency": finite_signal_consistency,
        "terminal_trim_applied": bool(_first_numeric(command_profile.get("terminal_trim_applied")) or False),
        "terminal_trim_gain": _first_numeric(command_profile.get("terminal_trim_gain")),
        "terminal_trim_bias_v": _first_numeric(command_profile.get("terminal_trim_bias_v")),
        "predicted_terminal_peak_error_mT": _first_numeric(command_profile.get("predicted_terminal_peak_error_mT")),
        "terminal_target_slope_sign": _first_numeric(command_profile.get("terminal_target_slope_sign")),
        "terminal_predicted_slope_sign_before": _first_numeric(command_profile.get("terminal_predicted_slope_sign_before")),
        "terminal_predicted_slope_sign_after": _first_numeric(command_profile.get("terminal_predicted_slope_sign_after")),
        "terminal_direction_match_after": bool(_first_numeric(command_profile.get("terminal_direction_match_after")) or False),
        "terminal_trim_window_fraction": _first_numeric(command_profile.get("terminal_trim_window_fraction")),
        "finite_cycle_metrics": finite_cycle_metrics,
        "finite_cycle_metrics_before": finite_cycle_metrics_before,
        "finite_cycle_metrics_after": finite_cycle_metrics_after,
        "finite_metric_improvement_summary": finite_metric_improvement_summary,
        "finite_terminal_correction_applied": finite_terminal_correction_applied,
        "finite_terminal_correction_reason": finite_terminal_correction_reason,
        "finite_terminal_correction_gain": finite_terminal_correction_gain,
        "finite_tail_residual_ratio_before": _first_numeric(command_profile.get("finite_tail_residual_ratio_before")),
        "finite_tail_residual_ratio_after": _first_numeric(command_profile.get("finite_tail_residual_ratio_after")),
        "finite_active_nrmse_before": _first_numeric(command_profile.get("finite_active_nrmse_before")),
        "finite_active_nrmse_after": _first_numeric(command_profile.get("finite_active_nrmse_after")),
        "allowed_finite_cycle_counts": list(FIELD_ROUTE_ALLOWED_FINITE_CYCLE_COUNTS) if field_only_route else [],
    }


def build_finite_support_entries(
    transient_measurements: list[ParsedMeasurement],
    transient_preprocess_results: list[PreprocessResult],
    current_channel: str = "i_sum_signed",
    field_channel: str = "bz_mT",
) -> list[dict[str, Any]]:
    """Build finite-run support entries from transient measurement data."""

    entries: list[dict[str, Any]] = []
    for parsed, preprocess in zip(transient_measurements, transient_preprocess_results, strict=False):
        corrected = preprocess.corrected_frame.copy()
        normalized = parsed.normalized_frame
        test_id = (
            str(normalized["test_id"].iloc[0])
            if "test_id" in normalized.columns and not normalized.empty
            else f"{parsed.source_file}::{parsed.sheet_name}"
        )
        freq_series = pd.to_numeric(normalized.get("freq_hz"), errors="coerce") if "freq_hz" in normalized.columns else pd.Series(dtype=float)
        freq_hz = float(freq_series.dropna().iloc[0]) if not freq_series.dropna().empty else float("nan")
        duration_s = (
            float(pd.to_numeric(corrected["time_s"], errors="coerce").max())
            if "time_s" in corrected.columns and not corrected.empty
            else float("nan")
        )
        estimated_cycle_span = duration_s * freq_hz if np.isfinite(duration_s) and np.isfinite(freq_hz) else float("nan")
        requested_cycle_count = _first_numeric(parsed.metadata.get("cycle") or parsed.metadata.get("cycle_count"))
        approx_cycle_span = (
            float(requested_cycle_count)
            if requested_cycle_count is not None and np.isfinite(requested_cycle_count)
            else estimated_cycle_span
        )
        entries.append(
            {
                "test_id": test_id,
                "source_file": parsed.source_file,
                "sheet_name": parsed.sheet_name,
                "waveform_type": canonicalize_waveform_type(parsed.metadata.get("waveform") or parsed.metadata.get("waveform_type")),
                "freq_hz": freq_hz,
                "duration_s": duration_s,
                "approx_cycle_span": approx_cycle_span,
                "estimated_cycle_span": estimated_cycle_span,
                "requested_cycle_count": requested_cycle_count,
                "target_current_a": _first_numeric(parsed.metadata.get("Target Current(A)") or parsed.metadata.get("target_current_a")),
                "notes": parsed.metadata.get("notes", ""),
                "current_pp": _signal_peak_to_peak(corrected, current_channel),
                "field_pp": _signal_peak_to_peak(corrected, field_channel),
                "daq_voltage_pp": _signal_peak_to_peak(corrected, "daq_input_v"),
                "frame": corrected,
            }
        )
    return entries


def _prepare_finite_time_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "time_s" not in frame.columns:
        return pd.DataFrame()
    prepared = frame.copy()
    prepared["time_s"] = pd.to_numeric(prepared["time_s"], errors="coerce")
    prepared = prepared.dropna(subset=["time_s"]).sort_values("time_s")
    prepared = prepared.loc[prepared["time_s"].diff().fillna(1.0).ne(0.0)].reset_index(drop=True)
    return prepared


def _resolve_finite_signal_column(frame: pd.DataFrame, candidates: list[str], default: str) -> str:
    for column in candidates:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            return column
    return default


def _finite_harmonic_weights(waveform_type: str) -> dict[int, float]:
    waveform_type = canonicalize_waveform_type(waveform_type) or "sine"
    if waveform_type == "triangle":
        return {1: 1.0, 3: 2.4, 5: 1.8, 7: 1.4}
    return {1: 1.0, 3: 0.2, 5: 0.08, 7: 0.03}


def _finite_shape_mismatch_score(
    *,
    frame: pd.DataFrame,
    output_signal_column: str,
    waveform_type: str,
    target_cycle_count: float,
    harmonic_weights: dict[int, float],
    sample_points: int = 512,
) -> float:
    prepared = _prepare_finite_time_frame(frame)
    if prepared.empty or output_signal_column not in prepared.columns:
        return 1.0
    signal = pd.to_numeric(prepared[output_signal_column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(signal)
    if valid.sum() < 8:
        return 1.0
    signal = signal[valid]
    signal = signal - float(np.nanmean(signal))
    signal_pp = float(np.nanmax(signal) - np.nanmin(signal)) if np.isfinite(signal).any() else float("nan")
    if not np.isfinite(signal_pp) or signal_pp <= 1e-9:
        return 1.0

    signal_norm = signal / (signal_pp / 2.0)
    source_phase = np.linspace(0.0, float(target_cycle_count), len(signal_norm))
    target_phase = np.linspace(0.0, float(target_cycle_count), max(int(sample_points), 64))
    signal_uniform = np.interp(target_phase, source_phase, signal_norm)
    target_uniform = _sample_theoretical_output(
        waveform_type=waveform_type,
        phase_total=target_phase,
        active_cycle_count=float(target_cycle_count),
    )

    signal_fft = np.fft.rfft(signal_uniform)
    target_fft = np.fft.rfft(target_uniform)
    weighted_error = 0.0
    total_weight = 0.0
    for harmonic, weight in harmonic_weights.items():
        if harmonic >= len(signal_fft) or harmonic >= len(target_fft):
            continue
        target_component = target_fft[harmonic]
        signal_component = signal_fft[harmonic]
        target_magnitude = abs(target_component)
        signal_magnitude = abs(signal_component)
        if target_magnitude <= 1e-9 and signal_magnitude <= 1e-9:
            continue
        magnitude_error = abs(signal_magnitude - target_magnitude) / max(target_magnitude, 1e-9)
        if target_magnitude <= 1e-9 or signal_magnitude <= 1e-9:
            phase_error = 1.0
        else:
            phase_delta = np.angle(signal_component) - np.angle(target_component)
            phase_error = abs(np.arctan2(np.sin(phase_delta), np.cos(phase_delta))) / np.pi
        weighted_error += float(weight) * (0.7 * magnitude_error + 0.3 * phase_error)
        total_weight += float(weight)
    if total_weight <= 0:
        return 1.0
    return float(weighted_error / total_weight)


def _finite_support_active_coverage_penalty(entry: dict[str, Any]) -> float:
    frame = _prepare_finite_time_frame(entry.get("active_frame"))
    if frame.empty:
        frame = _prepare_finite_time_frame(entry.get("frame"))
    if frame.empty:
        return 4.0
    freq_hz = float(entry.get("freq_hz", np.nan))
    cycle_count = float(entry.get("approx_cycle_span", np.nan))
    if not np.isfinite(freq_hz) or freq_hz <= 0 or not np.isfinite(cycle_count) or cycle_count <= 0:
        return 0.0
    expected_duration_s = float(cycle_count / freq_hz)
    observed_duration_s = float(frame["time_s"].max() - frame["time_s"].min())
    if observed_duration_s <= 0:
        return 4.0
    coverage_ratio = float(np.clip(observed_duration_s / expected_duration_s, 0.0, 1.0))
    return float(np.square(max(1.0 - coverage_ratio, 0.0)) * 4.0)


def _finite_support_cycle_semantics_penalty(*, support_cycle_count: float, target_cycle_count: float) -> float:
    if not np.isfinite(support_cycle_count) or not np.isfinite(target_cycle_count):
        return 0.0
    if target_cycle_count < 1.0 - 1e-9:
        return 12.0
    if target_cycle_count >= 1.75 - 1e-9 and abs(support_cycle_count - 1.75) > 0.05:
        return 12.0
    if target_cycle_count > 1.0 and support_cycle_count < 1.0:
        return 6.0
    if support_cycle_count < target_cycle_count and target_cycle_count - support_cycle_count > 0.5 + 1e-9:
        return 4.0
    return 0.0


def _finite_cycle_decomposition_metadata(
    target_cycle_count: float | None,
    *,
    finite_support_used: bool,
    selected_support_id: str | None = None,
    selected_support_cycle_count: float | None = None,
) -> dict[str, Any]:
    target_value = float(target_cycle_count) if target_cycle_count is not None and np.isfinite(target_cycle_count) else float("nan")
    integer_count = int(np.floor(target_value + 1e-9)) if np.isfinite(target_value) else 0
    terminal_fraction = float(target_value - float(integer_count)) if np.isfinite(target_value) else float("nan")
    if np.isfinite(terminal_fraction) and abs(terminal_fraction) <= 1e-9:
        terminal_fraction = 0.0
    selected_cycle = (
        float(selected_support_cycle_count)
        if selected_support_cycle_count is not None and np.isfinite(selected_support_cycle_count)
        else float("nan")
    )
    whole_substitution_used = bool(
        finite_support_used
        and np.isfinite(selected_cycle)
        and np.isfinite(target_value)
        and abs(selected_cycle - target_value) > 0.05
    )
    whole_substitution_valid = bool(not whole_substitution_used)
    warning = None
    mode = "not_applicable"
    interior_support_id = None
    terminal_tail_support_id = None
    terminal_tail_fraction = terminal_fraction
    if np.isfinite(target_value):
        if target_value < 1.0 - 1e-9:
            mode = "unsupported_fractional_cycle_request"
            whole_substitution_valid = False
            warning = "supported_finite_cycles_are_1p0_1p25_1p5_1p75"
        elif target_value >= 1.75 - 1e-9:
            if finite_support_used and np.isfinite(selected_cycle) and abs(selected_cycle - target_value) <= 0.05:
                mode = "whole_exact_1_75_support"
                whole_substitution_valid = True
            elif finite_support_used and whole_substitution_used:
                mode = "invalid_whole_support_substitution"
                whole_substitution_valid = False
                warning = "1.75_requires_exact_support"
            else:
                mode = "fallback_no_exact_1_75_support"
                whole_substitution_valid = True
                warning = "1.75_requires_exact_support"
        elif terminal_fraction > 1e-9:
            mode = "fractional_cycle_empirical_support" if finite_support_used else "fractional_cycle_fallback"
        else:
            mode = "integer_cycle_empirical_support" if finite_support_used else "integer_cycle_fallback"
    return {
        "finite_cycle_decomposition_mode": mode,
        "target_integer_cycle_count": integer_count,
        "target_terminal_fraction": terminal_fraction,
        "interior_support_id": interior_support_id,
        "terminal_tail_support_id": terminal_tail_support_id,
        "terminal_tail_fraction": terminal_tail_fraction,
        "cycle_semantics_warning": warning,
        "whole_support_substitution_used": whole_substitution_used,
        "whole_support_substitution_valid": whole_substitution_valid,
        "finite_cycle_policy_version": "field_route_cycles_v3",
        "supported_cycle_counts": list(FIELD_ROUTE_ALLOWED_FINITE_CYCLE_COUNTS),
    }


def _despike_isolated_impulses(values: np.ndarray) -> tuple[np.ndarray, int]:
    filtered = np.asarray(values, dtype=float).copy()
    finite = filtered[np.isfinite(filtered)]
    if finite.size < 5:
        return filtered, 0
    peak_to_peak = float(np.nanmax(finite) - np.nanmin(finite))
    if not np.isfinite(peak_to_peak) or peak_to_peak <= 1e-9:
        return filtered, 0
    finite_diffs = np.abs(np.diff(finite))
    median_step = float(np.nanmedian(finite_diffs)) if finite_diffs.size else 0.0
    residual_threshold = max(peak_to_peak * FINITE_SIGNAL_JUMP_RATIO_LIMIT, median_step * 8.0, 1e-6)
    neighbor_threshold = max(peak_to_peak * FINITE_SIGNAL_JUMP_RATIO_LIMIT, median_step * 4.0, 1e-6)
    filtered_count = 0
    for index in range(1, len(filtered) - 1):
        previous_value = filtered[index - 1]
        current_value = filtered[index]
        next_value = filtered[index + 1]
        if not (np.isfinite(previous_value) and np.isfinite(current_value) and np.isfinite(next_value)):
            continue
        local_midpoint = 0.5 * (previous_value + next_value)
        residual = abs(current_value - local_midpoint)
        neighbor_span = abs(next_value - previous_value)
        if residual > residual_threshold and neighbor_span <= neighbor_threshold:
            filtered[index] = local_midpoint
            filtered_count += 1
    return filtered, filtered_count


def _interpolate_finite_signal(
    source_time: np.ndarray,
    source_values: np.ndarray,
    target_time: np.ndarray,
) -> np.ndarray:
    valid = np.isfinite(source_time) & np.isfinite(source_values)
    if valid.sum() < 2 or len(target_time) == 0:
        return np.zeros(len(target_time), dtype=float)
    source_time = np.asarray(source_time[valid], dtype=float)
    source_values = np.asarray(source_values[valid], dtype=float)
    order = np.argsort(source_time)
    return np.interp(np.asarray(target_time, dtype=float), source_time[order], source_values[order])


def _resample_finite_support_record(
    *,
    entry: dict[str, Any],
    time_grid: np.ndarray,
    active_duration_s: float,
    tail_duration_s: float,
    scale_ratio: float,
    current_channel: str,
    field_channel: str,
) -> dict[str, Any] | None:
    full_frame = _prepare_finite_time_frame(entry.get("frame"))
    active_frame = _prepare_finite_time_frame(entry.get("active_frame"))
    active_source = "explicit_active_frame"
    if active_frame.empty:
        active_source = "derived_from_cycle_metadata"
        active_frame = full_frame.copy()
    if active_frame.empty or full_frame.empty or "daq_input_v" not in full_frame.columns:
        return None

    active_start_s = float(active_frame["time_s"].min())
    expected_support_active_duration_s = float("nan")
    entry_freq_hz = float(entry.get("freq_hz", np.nan))
    entry_cycle_count = float(entry.get("approx_cycle_span", np.nan))
    if np.isfinite(entry_freq_hz) and entry_freq_hz > 0 and np.isfinite(entry_cycle_count) and entry_cycle_count > 0:
        expected_support_active_duration_s = float(entry_cycle_count / entry_freq_hz)
    if active_source == "derived_from_cycle_metadata" and np.isfinite(expected_support_active_duration_s):
        derived_active_end_s = active_start_s + expected_support_active_duration_s
        derived_active = full_frame[full_frame["time_s"] <= derived_active_end_s + 1e-9].copy()
        if len(derived_active) >= 2:
            active_frame = _prepare_finite_time_frame(derived_active)

    active_end_s = float(active_frame["time_s"].max())
    active_support_duration_s = max(active_end_s - active_start_s, 1e-9)
    support_observed_coverage_ratio = 1.0
    support_padding_gap_s = 0.0
    if np.isfinite(expected_support_active_duration_s) and expected_support_active_duration_s > 0:
        support_observed_coverage_ratio = float(np.clip(active_support_duration_s / expected_support_active_duration_s, 0.0, 1.0))
        support_padding_gap_s = max(expected_support_active_duration_s - active_support_duration_s, 0.0)
    active_mask = time_grid <= float(active_duration_s) + 1e-12
    target_active_rel = np.clip(time_grid[active_mask], 0.0, float(active_duration_s))
    if active_duration_s > 0:
        source_active_rel = target_active_rel / float(active_duration_s) * active_support_duration_s
    else:
        source_active_rel = np.zeros_like(target_active_rel)

    active_time_rel = pd.to_numeric(active_frame["time_s"], errors="coerce").to_numpy(dtype=float) - active_start_s
    current_column = _resolve_finite_signal_column(
        active_frame,
        [str(entry.get("resolved_current_channel") or ""), current_channel, "i_sum_signed", "signed_current_a", "i_custom_signed"],
        current_channel,
    )
    field_column = _resolve_finite_signal_column(
        active_frame,
        [str(entry.get("resolved_field_channel") or ""), field_channel, "bz_mT", "bproj_mT", "bmag_mT"],
        field_channel,
    )
    active_voltage_values, active_voltage_spikes = _despike_isolated_impulses(
        pd.to_numeric(active_frame["daq_input_v"], errors="coerce").to_numpy(dtype=float)
    )
    active_current_values, active_current_spikes = _despike_isolated_impulses(
        pd.to_numeric(active_frame[current_column], errors="coerce").to_numpy(dtype=float)
        if current_column in active_frame.columns
        else np.zeros(len(active_frame), dtype=float)
    )
    active_field_values, active_field_spikes = _despike_isolated_impulses(
        pd.to_numeric(active_frame[field_column], errors="coerce").to_numpy(dtype=float)
        if field_column in active_frame.columns
        else np.zeros(len(active_frame), dtype=float)
    )
    support_spike_filtered_count = int(active_voltage_spikes + active_current_spikes + active_field_spikes)
    voltage_active = _interpolate_finite_signal(
        active_time_rel,
        active_voltage_values,
        source_active_rel,
    )
    current_active = _interpolate_finite_signal(
        active_time_rel,
        active_current_values,
        source_active_rel,
    )
    field_active = _interpolate_finite_signal(
        active_time_rel,
        active_field_values,
        source_active_rel,
    )

    tail_mask = ~active_mask
    tail_voltage = np.zeros(int(tail_mask.sum()), dtype=float)
    tail_current = np.zeros(int(tail_mask.sum()), dtype=float)
    tail_field = np.zeros(int(tail_mask.sum()), dtype=float)
    if tail_mask.any() and tail_duration_s > 0:
        tail_frame = _prepare_finite_time_frame(full_frame[full_frame["time_s"] > active_end_s + 1e-9].copy())
        if not tail_frame.empty:
            tail_support_duration_s = max(float(tail_frame["time_s"].max()) - active_end_s, 1e-9)
            target_tail_rel = np.clip(time_grid[tail_mask] - float(active_duration_s), 0.0, float(tail_duration_s))
            source_tail_rel = target_tail_rel / float(tail_duration_s) * tail_support_duration_s
            tail_time_rel = pd.to_numeric(tail_frame["time_s"], errors="coerce").to_numpy(dtype=float) - active_end_s
            tail_voltage = _interpolate_finite_signal(
                tail_time_rel,
                pd.to_numeric(tail_frame["daq_input_v"], errors="coerce").to_numpy(dtype=float),
                source_tail_rel,
            )
            current_tail_column = _resolve_finite_signal_column(
                tail_frame,
                [str(entry.get("resolved_current_channel") or ""), current_channel, "i_sum_signed", "signed_current_a", "i_custom_signed"],
                current_channel,
            )
            field_tail_column = _resolve_finite_signal_column(
                tail_frame,
                [str(entry.get("resolved_field_channel") or ""), field_channel, "bz_mT", "bproj_mT", "bmag_mT"],
                field_channel,
            )
            tail_voltage_values, tail_voltage_spikes = _despike_isolated_impulses(
                pd.to_numeric(tail_frame["daq_input_v"], errors="coerce").to_numpy(dtype=float)
            )
            tail_current_values, tail_current_spikes = _despike_isolated_impulses(
                pd.to_numeric(tail_frame[current_tail_column], errors="coerce").to_numpy(dtype=float)
                if current_tail_column in tail_frame.columns
                else np.zeros(len(tail_frame), dtype=float)
            )
            tail_field_values, tail_field_spikes = _despike_isolated_impulses(
                pd.to_numeric(tail_frame[field_tail_column], errors="coerce").to_numpy(dtype=float)
                if field_tail_column in tail_frame.columns
                else np.zeros(len(tail_frame), dtype=float)
            )
            support_spike_filtered_count += int(tail_voltage_spikes + tail_current_spikes + tail_field_spikes)
            tail_current = _interpolate_finite_signal(
                tail_time_rel,
                tail_current_values,
                source_tail_rel,
            )
            tail_field = _interpolate_finite_signal(
                tail_time_rel,
                tail_field_values,
                source_tail_rel,
            )
            tail_voltage = _interpolate_finite_signal(
                tail_time_rel,
                tail_voltage_values,
                source_tail_rel,
            )

    voltage = np.zeros_like(time_grid, dtype=float)
    current = np.zeros_like(time_grid, dtype=float)
    field = np.zeros_like(time_grid, dtype=float)
    voltage[active_mask] = voltage_active
    current[active_mask] = current_active
    field[active_mask] = field_active
    if tail_mask.any():
        voltage[tail_mask] = tail_voltage
        current[tail_mask] = tail_current
        field[tail_mask] = tail_field
    zero_mask = np.zeros_like(time_grid, dtype=bool)
    if tail_mask.any():
        zero_mask[tail_mask] = (
            np.isclose(voltage[tail_mask], 0.0, atol=1e-9)
            & np.isclose(current[tail_mask], 0.0, atol=1e-9)
            & np.isclose(field[tail_mask], 0.0, atol=1e-9)
        )
    zero_padded_fraction = float(np.mean(zero_mask.astype(float))) if len(zero_mask) else 0.0

    return {
        "time_s": np.asarray(time_grid, dtype=float),
        "voltage_v": voltage * float(scale_ratio),
        "current_a": current * float(scale_ratio),
        "field_mT": field * float(scale_ratio),
        "active_window_start_s": active_start_s,
        "active_window_end_s": active_end_s,
        "active_duration_s": float(active_duration_s),
        "zero_padded_fraction": zero_padded_fraction,
        "support_observed_end_s": active_start_s + active_support_duration_s / max(expected_support_active_duration_s, active_support_duration_s, 1e-9) * float(active_duration_s)
        if np.isfinite(expected_support_active_duration_s) and expected_support_active_duration_s > 0
        else float(active_duration_s),
        "support_observed_coverage_ratio": support_observed_coverage_ratio,
        "support_padding_gap_s": support_padding_gap_s / max(expected_support_active_duration_s, support_padding_gap_s, 1e-9) * float(active_duration_s)
        if support_padding_gap_s > 0 and np.isfinite(expected_support_active_duration_s) and expected_support_active_duration_s > 0
        else 0.0,
        "support_resampled_to_target_window": True,
        "hybrid_fill_applied": False,
        "hybrid_fill_start_s": float("nan"),
        "hybrid_fill_end_s": float("nan"),
        "finite_prediction_source": "empirical_resampled",
        "predicted_cover_reason": "active_progress_resampled",
        "support_cover_reason": "active_progress_resampled",
        "support_spike_filtered_count": int(support_spike_filtered_count),
        "support_source_spike_detected": bool(support_spike_filtered_count > 0),
    }


def _build_finite_modeled_profile(
    *,
    support_payload: dict[str, Any],
    waveform_type: str,
    freq_hz: float,
    target_cycle_count: float,
    target_output_type: str,
    target_output_pp: float,
    preview_tail_cycles: float,
    request_route: str,
    plot_source: str,
    selected_support_waveform: str,
    harmonic_weights_used: dict[int, float],
) -> pd.DataFrame:
    time_grid = np.asarray(support_payload["time_s"], dtype=float)
    modeled = pd.DataFrame({"time_s": time_grid})
    modeled["recommended_voltage_v"] = np.asarray(support_payload["voltage_v"], dtype=float)
    modeled["expected_current_a"] = np.asarray(support_payload["current_a"], dtype=float)
    modeled["expected_field_mT"] = np.asarray(support_payload["field_mT"], dtype=float)
    modeled["support_scaled_current_a"] = modeled["expected_current_a"]
    modeled["support_scaled_field_mT"] = modeled["expected_field_mT"]
    modeled["target_output"] = _finite_target_template(
        time_grid=time_grid,
        waveform_type=waveform_type,
        freq_hz=float(freq_hz),
        target_cycle_count=float(target_cycle_count),
        target_output_pp=float(target_output_pp),
        force_rounded_triangle=target_output_type == "field",
    )
    if target_output_type == "current":
        modeled["target_current_a"] = modeled["target_output"]
        modeled["used_target_current_a"] = modeled["target_output"]
    else:
        modeled["target_field_mT"] = modeled["target_output"]
        modeled["used_target_field_mT"] = modeled["target_output"]
        modeled["physical_target_output_mT"] = modeled["target_output"]
        modeled["support_reference_output_mT"] = modeled["expected_field_mT"]
        modeled["target_shape_family"] = "rounded_triangle"
        modeled["target_pp_fixed"] = float(FIELD_ROUTE_NORMALIZED_TARGET_PP)
        modeled["support_family_used"] = selected_support_waveform
        modeled["support_family_requested"] = waveform_type
    active_duration_s = float(target_cycle_count) / float(freq_hz) if np.isfinite(freq_hz) and float(freq_hz) > 0 else 0.0
    modeled["waveform_type"] = waveform_type
    modeled["freq_hz"] = float(freq_hz)
    modeled["target_cycle_count"] = float(target_cycle_count)
    modeled["target_output_type"] = target_output_type
    modeled["target_output_pp"] = float(target_output_pp)
    modeled["finite_cycle_mode"] = True
    modeled["preview_tail_cycles"] = float(max(preview_tail_cycles, 0.0))
    modeled["is_active_target"] = modeled["time_s"] <= active_duration_s + 1e-12
    modeled["expected_output"] = modeled["expected_field_mT"] if target_output_type == "field" else modeled["expected_current_a"]
    modeled["request_route"] = request_route
    modeled["plot_source"] = plot_source
    modeled["selected_support_waveform"] = selected_support_waveform
    modeled["active_window_start_s"] = float(support_payload["active_window_start_s"])
    modeled["active_window_end_s"] = float(support_payload["active_window_end_s"])
    modeled["active_duration_s"] = float(support_payload.get("active_duration_s", active_duration_s))
    modeled["zero_padded_fraction"] = float(support_payload.get("zero_padded_fraction", 0.0))
    modeled["support_observed_end_s"] = float(support_payload.get("support_observed_end_s", active_duration_s))
    modeled["support_observed_coverage_ratio"] = float(support_payload.get("support_observed_coverage_ratio", 1.0))
    modeled["support_padding_gap_s"] = float(support_payload.get("support_padding_gap_s", 0.0))
    modeled["support_resampled_to_target_window"] = bool(support_payload.get("support_resampled_to_target_window", False))
    modeled["hybrid_fill_applied"] = bool(support_payload.get("hybrid_fill_applied", False))
    modeled["hybrid_fill_start_s"] = float(support_payload.get("hybrid_fill_start_s", np.nan))
    modeled["hybrid_fill_end_s"] = float(support_payload.get("hybrid_fill_end_s", np.nan))
    modeled["finite_prediction_source"] = str(support_payload.get("finite_prediction_source", "empirical_observed"))
    modeled["predicted_cover_reason"] = str(support_payload.get("predicted_cover_reason", "empirical_observed"))
    modeled["support_cover_reason"] = str(support_payload.get("support_cover_reason", "empirical_observed"))
    modeled["support_spike_filtered_count"] = int(support_payload.get("support_spike_filtered_count", 0) or 0)
    modeled["support_source_spike_detected"] = bool(support_payload.get("support_source_spike_detected", False))
    modeled["support_blend_boundary_count"] = int(support_payload.get("support_blend_boundary_count", 0) or 0)
    modeled["harmonic_weights_used"] = str(harmonic_weights_used)
    return _sync_modeled_alias_columns(modeled)


def synthesize_finite_empirical_compensation(
    finite_support_entries: list[dict[str, Any]],
    waveform_type: str,
    freq_hz: float,
    target_cycle_count: float | None,
    target_output_type: str,
    target_output_pp: float,
    current_channel: str = "i_sum_signed",
    field_channel: str = "bz_mT",
    max_daq_voltage_pp: float = 20.0,
    amp_gain_at_100_pct: float = 20.0,
    amp_gain_limit_pct: float = 100.0,
    amp_max_output_pk_v: float = 180.0,
    default_support_amp_gain_pct: float = 100.0,
    preview_tail_cycles: float = 0.0,
    max_support_count: int = 3,
    cycle_match_tolerance: float = 0.05,
    freq_match_tolerance_hz: float = 1e-6,
) -> dict[str, Any] | None:
    """Blend nearby finite-run support data into an empirical transient tracking model."""

    if not finite_support_entries or target_cycle_count is None:
        return None

    waveform_type = canonicalize_waveform_type(waveform_type)
    if waveform_type is None:
        return None
    if float(target_cycle_count) < 1.0 - 1e-9:
        return None

    output_column = "field_pp" if target_output_type == "field" else "current_pp"
    target_output_unit = "mT" if target_output_type == "field" else "A"
    harmonic_weights = _finite_harmonic_weights(waveform_type)
    if not finite_support_entries:
        return None

    exact_freq_matches = [
        entry
        for entry in finite_support_entries
        if np.isfinite(entry.get("freq_hz", np.nan))
        and abs(float(entry.get("freq_hz", np.nan)) - float(freq_hz)) <= float(freq_match_tolerance_hz)
    ]
    frequency_bucket_mode = "exact_frequency_bucket" if exact_freq_matches else "nearest_frequency_blend"
    frequency_candidates = exact_freq_matches or list(finite_support_entries)

    exact_cycle_matches = [
        entry
        for entry in frequency_candidates
        if np.isfinite(entry.get("approx_cycle_span", np.nan))
        and abs(float(entry.get("approx_cycle_span", np.nan)) - float(target_cycle_count)) <= float(cycle_match_tolerance)
    ]
    if float(target_cycle_count) >= 1.75 - 1e-9 and not exact_cycle_matches:
        return None
    cycle_bucket_mode = "exact_cycle_bucket" if exact_cycle_matches else "nearest_cycle_blend"
    candidate_entries = exact_cycle_matches or frequency_candidates
    level_match_tolerance = max(abs(float(target_output_pp)) * 0.05, 0.5)

    def _exact_support_level(entry: dict[str, Any]) -> float:
        if target_output_type == "current":
            requested_value = entry.get("requested_current_pp", np.nan)
            if np.isfinite(requested_value):
                return float(requested_value)
        return float(entry.get(output_column, np.nan))

    exact_output_matches = [
        entry
        for entry in exact_cycle_matches
        if np.isfinite(_exact_support_level(entry))
        and abs(_exact_support_level(entry) - float(target_output_pp)) <= level_match_tolerance
    ]
    request_route = "exact" if exact_output_matches else "preview"
    plot_source = "exact_prediction" if exact_output_matches else "support_blended_preview"

    freq_values = [float(entry["freq_hz"]) for entry in candidate_entries if np.isfinite(entry.get("freq_hz", np.nan))]
    cycle_values = [float(entry["approx_cycle_span"]) for entry in candidate_entries if np.isfinite(entry.get("approx_cycle_span", np.nan))]
    output_values = [
        _exact_support_level(entry)
        for entry in candidate_entries
        if np.isfinite(_exact_support_level(entry))
    ]
    freq_range = max((max(freq_values) - min(freq_values)) if freq_values else 0.0, 1e-9)
    cycle_range = max((max(cycle_values) - min(cycle_values)) if cycle_values else 0.0, 1e-9)
    output_range = max((max(output_values) - min(output_values)) if output_values else 0.0, 1e-9)

    scored_entries: list[tuple[float, dict[str, Any], float, float, float]] = []
    output_signal_column = field_channel if target_output_type == "field" else current_channel
    for entry in candidate_entries:
        support_output = _exact_support_level(entry)
        if not np.isfinite(support_output) or support_output <= 0:
            continue
        freq_distance = abs(float(entry.get("freq_hz", np.nan)) - float(freq_hz)) if np.isfinite(entry.get("freq_hz", np.nan)) else 1e6
        cycle_distance = abs(float(entry.get("approx_cycle_span", np.nan)) - float(target_cycle_count)) if np.isfinite(entry.get("approx_cycle_span", np.nan)) else 1e6
        output_distance = abs(support_output - float(target_output_pp))
        waveform_distance = 0.0 if canonicalize_waveform_type(entry.get("waveform_type")) == waveform_type else 0.05
        coverage_penalty = _finite_support_active_coverage_penalty(entry)
        cycle_semantics_penalty = _finite_support_cycle_semantics_penalty(
            support_cycle_count=float(entry.get("approx_cycle_span", np.nan)),
            target_cycle_count=float(target_cycle_count),
        )
        shape_mismatch = _finite_shape_mismatch_score(
            frame=entry.get("active_frame", entry.get("frame", pd.DataFrame())),
            output_signal_column=output_signal_column,
            waveform_type=waveform_type,
            target_cycle_count=float(target_cycle_count),
            harmonic_weights=harmonic_weights,
        )
        distance_score = float(
            np.sqrt(
                np.square(freq_distance / freq_range)
                + np.square(cycle_distance / cycle_range)
                + np.square(output_distance / output_range)
            )
        )
        scored_entries.append(
            (
                float(distance_score + waveform_distance + coverage_penalty + cycle_semantics_penalty + shape_mismatch),
                entry,
                freq_distance,
                cycle_distance,
                output_distance,
            )
        )

    if not scored_entries:
        return None

    scored_entries.sort(key=lambda item: item[0])
    if exact_output_matches:
        selected_supports = [item for item in scored_entries if item[1] in exact_output_matches][:1]
    else:
        selected_supports = scored_entries[: max(int(max_support_count), 1)]
    nearest_distance_score, support, _, _, _ = selected_supports[0]
    support_output_pp = _exact_support_level(support)
    scale_ratio = float(target_output_pp / support_output_pp) if support_output_pp else float("nan")
    if not np.isfinite(scale_ratio):
        return None

    active_duration_s = (
        float(target_cycle_count) / float(freq_hz)
        if np.isfinite(freq_hz) and float(freq_hz) > 0 and target_cycle_count is not None
        else float("nan")
    )
    tail_duration_s = (
        max(float(preview_tail_cycles), 0.0) / float(freq_hz)
        if np.isfinite(freq_hz) and float(freq_hz) > 0
        else 0.0
    )
    total_duration_s = active_duration_s + tail_duration_s if np.isfinite(active_duration_s) else float("nan")
    if not np.isfinite(total_duration_s) or total_duration_s <= 0:
        return None

    selected_records: list[tuple[float, dict[str, Any], float, float, float]] = []
    for distance_score, entry, freq_distance, cycle_distance, output_distance in selected_supports:
        frame = _prepare_finite_time_frame(entry.get("frame"))
        if frame.empty or "daq_input_v" not in frame.columns:
            continue
        selected_records.append((distance_score, entry, freq_distance, cycle_distance, output_distance))
    if not selected_records:
        return None

    support_count_used = int(len(selected_records))
    selected_support_cycle_values = [
        float(record[1].get("approx_cycle_span", np.nan))
        for record in selected_records
        if np.isfinite(float(record[1].get("approx_cycle_span", np.nan)))
    ]
    if (
        np.isfinite(float(target_cycle_count))
        and float(target_cycle_count) >= 1.75 - 1e-9
        and selected_support_cycle_values
        and max(selected_support_cycle_values) <= 0.75 + 1e-9
    ):
        return None
    sample_count = max(
        max(
            len(_prepare_finite_time_frame(record[1].get("active_frame", record[1].get("frame"))))
            for record in selected_records
        ),
        256,
    )
    time_grid = np.linspace(0.0, total_duration_s, sample_count)

    raw_weights = np.array([1.0 / np.square(1.0 + item[0]) for item in selected_records], dtype=float)
    if not np.isfinite(raw_weights).all() or raw_weights.sum() <= 0:
        raw_weights = np.ones(len(selected_records), dtype=float)
    normalized_weights = raw_weights / raw_weights.sum()

    blended_voltage = np.zeros_like(time_grid, dtype=float)
    blended_output = np.zeros_like(time_grid, dtype=float)
    blended_current = np.zeros_like(time_grid, dtype=float)
    blended_field = np.zeros_like(time_grid, dtype=float)
    selected_active_window_start_s = float("nan")
    selected_active_window_end_s = float("nan")
    selected_zero_padded_fraction = 0.0
    selected_support_observed_end_s = 0.0
    selected_support_observed_coverage_ratio = 1.0
    selected_support_padding_gap_s = 0.0
    support_resampled_to_target_window = False
    support_spike_filtered_count = 0
    support_source_spike_detected = False
    support_rows: list[dict[str, Any]] = []
    for weight, record in zip(normalized_weights, selected_records, strict=False):
        distance_score, entry, freq_distance, cycle_distance, output_distance = record
        support_output = _exact_support_level(entry)
        local_scale_ratio = float(target_output_pp / support_output) if support_output else float("nan")
        if not np.isfinite(local_scale_ratio):
            continue
        support_payload = _resample_finite_support_record(
            entry=entry,
            time_grid=time_grid,
            active_duration_s=float(active_duration_s),
            tail_duration_s=float(tail_duration_s),
            scale_ratio=local_scale_ratio,
            current_channel=current_channel,
            field_channel=field_channel,
        )
        if support_payload is None:
            continue
        selected_active_window_start_s = float(support_payload["active_window_start_s"])
        selected_active_window_end_s = float(support_payload["active_window_end_s"])
        selected_zero_padded_fraction += float(weight) * float(support_payload.get("zero_padded_fraction", 0.0))
        selected_support_observed_end_s = max(
            selected_support_observed_end_s,
            float(support_payload.get("support_observed_end_s", active_duration_s)),
        )
        selected_support_observed_coverage_ratio = min(
            selected_support_observed_coverage_ratio,
            float(support_payload.get("support_observed_coverage_ratio", 1.0)),
        )
        selected_support_padding_gap_s += float(weight) * float(support_payload.get("support_padding_gap_s", 0.0))
        support_resampled_to_target_window = bool(
            support_resampled_to_target_window or support_payload.get("support_resampled_to_target_window", False)
        )
        support_spike_filtered_count += int(support_payload.get("support_spike_filtered_count", 0) or 0)
        support_source_spike_detected = bool(
            support_source_spike_detected or support_payload.get("support_source_spike_detected", False)
        )
        interpolated_voltage = np.asarray(support_payload["voltage_v"], dtype=float)
        interpolated_current = np.asarray(support_payload["current_a"], dtype=float)
        interpolated_field = np.asarray(support_payload["field_mT"], dtype=float)
        interpolated_output = interpolated_field if target_output_type == "field" else interpolated_current
        blended_voltage += weight * interpolated_voltage
        blended_output += weight * interpolated_output
        blended_current += weight * interpolated_current
        blended_field += weight * interpolated_field
        support_rows.append(
            {
                "test_id": entry["test_id"],
                "waveform_type": entry.get("waveform_type"),
                "freq_hz": entry.get("freq_hz"),
                "approx_cycle_span": entry.get("approx_cycle_span"),
                "support_output_pp": support_output,
                "daq_voltage_pp": entry.get("daq_voltage_pp"),
                "freq_distance_hz": freq_distance,
                "cycle_distance": cycle_distance,
                "output_distance": output_distance,
                "distance_score": distance_score,
                "weight": float(weight),
                "scale_ratio": local_scale_ratio,
                "shape_mismatch_score": _finite_shape_mismatch_score(
                    frame=entry.get("active_frame", entry.get("frame", pd.DataFrame())),
                    output_signal_column=output_signal_column,
                    waveform_type=waveform_type,
                    target_cycle_count=float(target_cycle_count),
                    harmonic_weights=harmonic_weights,
                ),
            }
        )

    modeled = _build_finite_modeled_profile(
        support_payload={
            "time_s": time_grid,
            "voltage_v": blended_voltage,
            "current_a": blended_current,
            "field_mT": blended_field,
                "active_window_start_s": selected_active_window_start_s,
                "active_window_end_s": selected_active_window_end_s,
                "active_duration_s": float(active_duration_s),
                "zero_padded_fraction": float(selected_zero_padded_fraction),
                "support_observed_end_s": float(selected_support_observed_end_s),
                "support_observed_coverage_ratio": float(selected_support_observed_coverage_ratio),
                "support_padding_gap_s": float(selected_support_padding_gap_s),
                "support_resampled_to_target_window": bool(support_resampled_to_target_window),
                "hybrid_fill_applied": False,
                "hybrid_fill_start_s": float("nan"),
                "hybrid_fill_end_s": float("nan"),
                "finite_prediction_source": "empirical_resampled",
                "predicted_cover_reason": "active_progress_resampled",
                "support_cover_reason": "active_progress_resampled",
                "support_spike_filtered_count": int(support_spike_filtered_count),
                "support_source_spike_detected": bool(support_source_spike_detected),
                "support_blend_boundary_count": max(int(len(selected_records)) - 1, 0),
        },
        waveform_type=waveform_type,
        freq_hz=float(freq_hz),
        target_cycle_count=float(target_cycle_count),
        target_output_type=target_output_type,
        target_output_pp=float(target_output_pp),
        preview_tail_cycles=float(max(preview_tail_cycles, 0.0)),
        request_route=request_route,
        plot_source=plot_source,
        selected_support_waveform=str(canonicalize_waveform_type(support.get("waveform_type")) or support.get("waveform_type") or ""),
        harmonic_weights_used=harmonic_weights,
    )
    modeled, active_extension_metadata = _extend_finite_active_window_signals(
        modeled,
        active_end_s=float(active_duration_s),
        command_columns=("recommended_voltage_v",),
        predicted_columns=("expected_field_mT", "expected_output"),
        support_columns=("support_scaled_field_mT",),
    )
    support_blended_zero_guard_applied = False
    if target_output_type == "field" and "support_scaled_field_mT" in modeled.columns:
        support_scaled_field = pd.to_numeric(modeled["support_scaled_field_mT"], errors="coerce").to_numpy(dtype=float)
        selected_support_has_field = any(
            np.isfinite(float(record[1].get("field_pp", np.nan))) and abs(float(record[1].get("field_pp", np.nan))) > 1e-9
            for record in selected_records
        )
        if selected_support_has_field and np.nanmax(np.abs(support_scaled_field)) <= 1e-9:
            nearest_entry = selected_records[0][1]
            nearest_field_pp = float(nearest_entry.get("field_pp", np.nan))
            if np.isfinite(nearest_field_pp) and abs(nearest_field_pp) > 1e-9:
                guarded_payload = _resample_finite_support_record(
                    entry=nearest_entry,
                    time_grid=time_grid,
                    active_duration_s=float(active_duration_s),
                    tail_duration_s=float(tail_duration_s),
                    scale_ratio=float(target_output_pp / nearest_field_pp),
                    current_channel=current_channel,
                    field_channel=field_channel,
                )
                if guarded_payload is not None:
                    guarded_field = np.asarray(guarded_payload["field_mT"], dtype=float)
                    guarded_current = np.asarray(guarded_payload["current_a"], dtype=float)
                    modeled["expected_field_mT"] = guarded_field
                    modeled["expected_current_a"] = guarded_current
                    modeled["support_scaled_field_mT"] = guarded_field
                    modeled["support_scaled_current_a"] = guarded_current
                    modeled["expected_output"] = guarded_field if target_output_type == "field" else guarded_current
                    support_blended_zero_guard_applied = True
                    modeled, active_extension_metadata = _extend_finite_active_window_signals(
                        modeled,
                        active_end_s=float(active_duration_s),
                        command_columns=("recommended_voltage_v",),
                        predicted_columns=("expected_field_mT", "expected_output"),
                        support_columns=("support_scaled_field_mT",),
                    )
    modeled = apply_command_hardware_model(
        command_waveform=modeled,
        max_daq_voltage_pp=float(max_daq_voltage_pp),
        amp_gain_at_100_pct=float(amp_gain_at_100_pct),
        support_amp_gain_pct=float(default_support_amp_gain_pct),
        amp_gain_limit_pct=float(amp_gain_limit_pct),
        amp_max_output_pk_v=float(amp_max_output_pk_v),
        preserve_start_voltage=True,
    )
    modeled, post_hardware_extension_metadata = _extend_finite_active_window_signals(
        modeled,
        active_end_s=float(active_duration_s),
        command_columns=("recommended_voltage_v", "limited_voltage_v"),
        predicted_columns=("expected_field_mT", "expected_output", "predicted_field_mT"),
        support_columns=("support_scaled_field_mT",),
    )
    active_extension_metadata = _merge_active_extension_metadata(
        active_extension_metadata,
        post_hardware_extension_metadata,
    )
    modeled["command_extension_applied"] = bool(active_extension_metadata["command_extension_applied"])
    modeled["command_extension_reason"] = active_extension_metadata["command_extension_reason"]
    modeled["command_stop_policy"] = active_extension_metadata["command_stop_policy"]
    modeled["predicted_extension_applied"] = bool(active_extension_metadata["predicted_extension_applied"])
    modeled["support_extension_applied"] = bool(active_extension_metadata["support_extension_applied"])
    modeled["support_coverage_mode"] = active_extension_metadata["support_coverage_mode"]
    modeled["partial_support_coverage"] = bool(active_extension_metadata["partial_support_coverage"])
    modeled["support_observed_end_s"] = active_extension_metadata["support_observed_end_s"]
    modeled["support_observed_coverage_ratio"] = active_extension_metadata["support_observed_coverage_ratio"]
    modeled["support_padding_gap_s"] = active_extension_metadata["support_padding_gap_s"]
    modeled["support_resampled_to_target_window"] = bool(active_extension_metadata["support_resampled_to_target_window"])
    modeled["hybrid_fill_applied"] = bool(active_extension_metadata["hybrid_fill_applied"])
    modeled["hybrid_fill_start_s"] = active_extension_metadata["hybrid_fill_start_s"]
    modeled["hybrid_fill_end_s"] = active_extension_metadata["hybrid_fill_end_s"]
    modeled["finite_prediction_source"] = active_extension_metadata["finite_prediction_source"]
    modeled["predicted_cover_reason"] = active_extension_metadata["predicted_cover_reason"]
    modeled["support_cover_reason"] = active_extension_metadata["support_cover_reason"]
    support_table = pd.DataFrame(support_rows)
    support_tests_used = [str(row["test_id"]) for row in support_rows]
    selected_support_waveform = str(canonicalize_waveform_type(support.get("waveform_type")) or support.get("waveform_type") or "")
    candidate_waveform_types = sorted(
        {
            str(canonicalize_waveform_type(entry.get("waveform_type")) or entry.get("waveform_type") or "")
            for entry in candidate_entries
        }
    )
    support_family_sensitivity_flag = bool(len([wave for wave in candidate_waveform_types if wave]) > 1)
    support_family_sensitivity_reason = "cross_family_candidate_pool" if support_family_sensitivity_flag else None
    support_family_selection_mode = "scored_preference_not_hard_filter"
    support_family_warning = "cross_family_candidates_scored" if support_family_sensitivity_flag else None
    support_family_sensitivity_level = "medium" if support_family_sensitivity_flag else "low"
    user_requested_support_family = waveform_type
    support_family_override_applied = bool(selected_support_waveform and selected_support_waveform != waveform_type)
    support_family_override_reason = (
        "cross_family_candidate_scored_better"
        if support_family_override_applied
        else None
    )
    command_voltage = pd.to_numeric(modeled["recommended_voltage_v"], errors="coerce").to_numpy(dtype=float)
    nonzero_mask = np.isfinite(command_voltage) & (np.abs(command_voltage) > 1e-6)
    command_nonzero_end_s = float(np.nanmax(time_grid[nonzero_mask])) if nonzero_mask.any() else float("nan")
    target_active_end_s = float(active_duration_s)
    early_command_cutoff_warning = bool(
        np.isfinite(command_nonzero_end_s)
        and np.isfinite(target_active_end_s)
        and command_nonzero_end_s < target_active_end_s - max(float(np.nanmedian(np.diff(time_grid))) if len(time_grid) > 1 else 0.0, 1e-6)
    )

    freq_in_range = bool(freq_values) and min(freq_values) <= float(freq_hz) <= max(freq_values)
    cycle_in_range = bool(cycle_values) and min(cycle_values) <= float(target_cycle_count) <= max(cycle_values)
    output_in_range = bool(output_values) and min(output_values) <= float(target_output_pp) <= max(output_values)
    model_confidence = float(
        np.clip(
            (1.0 / (1.0 + nearest_distance_score)) * (0.6 + 0.4 * min(support_count_used, 3) / 3.0),
            0.0,
            1.0,
        )
    )
    extrapolation_axes = [
        axis_name
        for axis_name, axis_in_range in (
            ("freq", freq_in_range),
            ("cycle", cycle_in_range),
            ("output", output_in_range),
        )
        if not axis_in_range
    ]
    selected_support_id = str(support["test_id"])
    selected_support_family = selected_support_waveform or None
    startup_diagnostics = _build_finite_support_startup_diagnostics(
        support,
        field_channel=field_channel,
        current_channel=current_channel,
    )
    startup_diagnostics.setdefault("source_test_id", selected_support_id)
    cycle_decomposition = _finite_cycle_decomposition_metadata(
        target_cycle_count,
        finite_support_used=True,
        selected_support_id=selected_support_id,
        selected_support_cycle_count=float(support.get("approx_cycle_span", np.nan)),
    )
    support_selection_reason = "finite_exact_level_match" if request_route == "exact" else (
        "finite_weighted_support_blend" if support_count_used > 1 else "finite_nearest_support_preview"
    )
    if support_family_sensitivity_flag:
        support_selection_reason = f"{support_selection_reason}_cross_family_scored"
    support_bz_to_current_ratio = float("nan")
    if np.isfinite(float(support.get("field_pp", np.nan))) and np.isfinite(float(support.get("current_pp", np.nan))):
        support_current_pp = float(support.get("current_pp", np.nan))
        if abs(support_current_pp) > 1e-9:
            support_bz_to_current_ratio = float(float(support.get("field_pp", np.nan)) / support_current_pp)
    field_prediction_source_hint = "exact_field_direct" if plot_source == "exact_prediction" else "support_blended_preview"
    expected_current_source_hint = "exact_current_direct" if plot_source == "exact_prediction" else "support_blended_preview"

    return {
        "mode": "finite_empirical_field_support" if request_route == "exact" else "finite_empirical_weighted_support",
        "frequency_bucket_mode": frequency_bucket_mode,
        "cycle_bucket_mode": cycle_bucket_mode,
        "target_output_type": target_output_type,
        "target_output_unit": target_output_unit,
        "target_output_pp": float(target_output_pp),
        "requested_freq_hz": float(freq_hz),
        "target_cycle_count": float(target_cycle_count),
        "preview_tail_cycles": float(max(preview_tail_cycles, 0.0)),
        "support_point_count": int(len(scored_entries)),
        "support_count_used": support_count_used,
        "support_test_id": str(support["test_id"]),
        "support_tests_used": support_tests_used,
        "requested_cycle_count": float(target_cycle_count),
        "support_freq_hz": float(support.get("freq_hz", np.nan)),
        "support_cycle_count": float(support.get("approx_cycle_span", np.nan)),
        "selected_support_cycle_count": float(support.get("approx_cycle_span", np.nan)),
        "exact_cycle_support_used": bool(abs(float(support.get("approx_cycle_span", np.nan)) - float(target_cycle_count)) <= float(cycle_match_tolerance)),
        "support_output_pp": support_output_pp,
        "scale_ratio": scale_ratio,
        "distance_score": nearest_distance_score,
        "model_confidence": model_confidence,
        "request_route": request_route,
        "plot_source": plot_source,
        "selected_support_waveform": selected_support_waveform,
        "support_waveform_role": "input_support_family",
        "support_family_sensitivity_flag": support_family_sensitivity_flag,
        "support_family_sensitivity_reason": support_family_sensitivity_reason,
        "support_family_selection_mode": support_family_selection_mode,
        "user_requested_support_family": user_requested_support_family,
        "candidate_support_families": candidate_waveform_types,
        "support_family_warning": support_family_warning,
        "support_family_sensitivity_level": support_family_sensitivity_level,
        "support_family_override_applied": support_family_override_applied,
        "support_family_override_reason": support_family_override_reason,
        "selected_support_id": selected_support_id,
        "selected_support_family": selected_support_family,
        "support_selection_reason": support_selection_reason,
        "support_family_metric": output_column,
        "support_family_value": float(support_output_pp) if np.isfinite(support_output_pp) else None,
        "support_family_lock_applied": request_route == "exact",
        "support_bz_to_current_ratio": float(support_bz_to_current_ratio) if np.isfinite(support_bz_to_current_ratio) else None,
        "active_window_start_s": selected_active_window_start_s,
        "active_window_end_s": selected_active_window_end_s,
        "active_duration_s": float(active_duration_s),
        "zero_padded_fraction": float(selected_zero_padded_fraction),
        "command_extension_applied": bool(active_extension_metadata["command_extension_applied"]),
        "command_extension_reason": active_extension_metadata["command_extension_reason"],
        "command_stop_policy": active_extension_metadata["command_stop_policy"],
        "predicted_extension_applied": bool(active_extension_metadata["predicted_extension_applied"]),
        "support_extension_applied": bool(active_extension_metadata["support_extension_applied"]),
        "support_coverage_mode": active_extension_metadata["support_coverage_mode"],
        "partial_support_coverage": bool(active_extension_metadata["partial_support_coverage"]),
        "support_observed_end_s": active_extension_metadata["support_observed_end_s"],
        "support_observed_coverage_ratio": active_extension_metadata["support_observed_coverage_ratio"],
        "support_padding_gap_s": active_extension_metadata["support_padding_gap_s"],
        "support_resampled_to_target_window": bool(active_extension_metadata["support_resampled_to_target_window"]),
        "hybrid_fill_applied": bool(active_extension_metadata["hybrid_fill_applied"]),
        "hybrid_fill_start_s": active_extension_metadata["hybrid_fill_start_s"],
        "hybrid_fill_end_s": active_extension_metadata["hybrid_fill_end_s"],
        "finite_prediction_source": active_extension_metadata["finite_prediction_source"],
        "predicted_cover_reason": active_extension_metadata["predicted_cover_reason"],
        "support_cover_reason": active_extension_metadata["support_cover_reason"],
        "finite_support_used": True,
        "support_blended_output_nonzero": bool(np.nanmax(np.abs(pd.to_numeric(modeled["support_scaled_field_mT"], errors="coerce").to_numpy(dtype=float))) > 1e-9)
        if "support_scaled_field_mT" in modeled.columns and len(modeled) > 0
        else False,
        "support_blended_zero_guard_applied": support_blended_zero_guard_applied,
        "support_spike_filtered_count": int(_first_numeric(modeled.get("support_spike_filtered_count")) or 0),
        "support_source_spike_detected": bool(_first_boolish(modeled.get("support_source_spike_detected"))),
        "support_blend_boundary_count": int(_first_numeric(modeled.get("support_blend_boundary_count")) or 0),
        "startup_diagnostics": startup_diagnostics,
        **cycle_decomposition,
        "command_nonzero_end_s": command_nonzero_end_s,
        "target_active_end_s": target_active_end_s,
        "early_command_cutoff_warning": early_command_cutoff_warning,
        "harmonic_weights_used": harmonic_weights,
        "field_prediction_source_hint": field_prediction_source_hint,
        "expected_current_source_hint": expected_current_source_hint,
        "field_prediction_status": "available",
        "field_prediction_hierarchy": ["exact_field_direct", "current_to_bz_surrogate", "unavailable"],
        "field_prediction_unavailable_reason": None,
        "field_prediction_fallback_reason": None,
        "exact_field_direct_available": plot_source == "exact_prediction",
        "exact_field_direct_reason": None,
        "same_recipe_surrogate_candidate_available": False,
        "same_recipe_surrogate_applied": False,
        "same_recipe_surrogate_ratio": None,
        "surrogate_scope": None,
        "freq_in_range": freq_in_range,
        "cycle_in_range": cycle_in_range,
        "output_in_range": output_in_range,
        "extrapolation_axes": extrapolation_axes,
        "support_table": support_table,
        "command_profile": modeled,
        "within_hardware_limits": bool(modeled["within_hardware_limits"].iloc[0]),
        "required_amp_gain_pct": float(modeled["required_amp_gain_pct"].iloc[0]),
        "available_amp_gain_pct": float(modeled["available_amp_gain_pct"].iloc[0]),
        "limited_voltage_pp": float(modeled["limited_voltage_pp"].iloc[0]),
        "amp_output_pp_at_required": float(modeled["amp_output_pp_at_required"].iloc[0]),
    }



def build_representative_cycle_profile(
    analysis: DatasetAnalysis,
    current_channel: str = "i_sum_signed",
    voltage_channel: str = "daq_input_v",
    field_channel: str = "bz_mT",
    points_per_cycle: int = 256,
    cycle_indices: list[int] | None = None,
) -> pd.DataFrame:
    """Average all detected cycles onto a common phase grid."""

    frame = analysis.cycle_detection.annotated_frame.copy()
    required = {"cycle_index", "cycle_progress", "cycle_time_s"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame()

    cycle_frame = frame.dropna(subset=["cycle_index", "cycle_progress"]).copy()
    if cycle_indices:
        cycle_frame = cycle_frame[cycle_frame["cycle_index"].isin(cycle_indices)].copy()
    if cycle_frame.empty:
        return pd.DataFrame()

    phase_grid = np.linspace(0.0, 1.0, points_per_cycle)
    durations: list[float] = []
    current_stack: list[np.ndarray] = []
    voltage_stack: list[np.ndarray] = []
    field_stack: list[np.ndarray] = []

    for _, group in cycle_frame.groupby("cycle_index", sort=True):
        valid = group.sort_values("cycle_progress").drop_duplicates("cycle_progress")
        if len(valid) < 5:
            continue
        progress = valid["cycle_progress"].to_numpy(dtype=float)
        if np.any(np.diff(progress) <= 0):
            continue

        durations.append(float(valid["cycle_time_s"].max()))
        if current_channel in valid.columns:
            current_values = pd.to_numeric(valid[current_channel], errors="coerce").to_numpy(dtype=float)
            current_mask = np.isfinite(progress) & np.isfinite(current_values)
            if current_mask.sum() >= 3:
                current_stack.append(np.interp(phase_grid, progress[current_mask], current_values[current_mask]))
        if voltage_channel in valid.columns:
            voltage_values = pd.to_numeric(valid[voltage_channel], errors="coerce").to_numpy(dtype=float)
            voltage_mask = np.isfinite(progress) & np.isfinite(voltage_values)
            if voltage_mask.sum() >= 3:
                voltage_stack.append(np.interp(phase_grid, progress[voltage_mask], voltage_values[voltage_mask]))
        if field_channel in valid.columns:
            field_values = pd.to_numeric(valid[field_channel], errors="coerce").to_numpy(dtype=float)
            field_mask = np.isfinite(progress) & np.isfinite(field_values)
            if field_mask.sum() >= 3:
                field_stack.append(np.interp(phase_grid, progress[field_mask], field_values[field_mask]))

    if not voltage_stack and not current_stack:
        return pd.DataFrame()

    period_s = float(np.nanmean(durations)) if durations else (
        1.0 / float(cycle_frame["freq_hz"].dropna().iloc[0]) if "freq_hz" in cycle_frame.columns and cycle_frame["freq_hz"].notna().any() else 1.0
    )
    profile = pd.DataFrame(
        {
            "cycle_progress": phase_grid,
            "time_s": phase_grid * period_s,
        }
    )
    if current_stack:
        current_matrix = np.vstack(current_stack)
        profile["measured_current_a"] = current_matrix.mean(axis=0)
        profile["measured_current_std_a"] = current_matrix.std(axis=0)
    if voltage_stack:
        voltage_matrix = np.vstack(voltage_stack)
        profile["command_voltage_v"] = voltage_matrix.mean(axis=0)
        profile["command_voltage_std_v"] = voltage_matrix.std(axis=0)
    if field_stack:
        field_matrix = np.vstack(field_stack)
        profile["measured_field_mT"] = field_matrix.mean(axis=0)
        profile["measured_field_std_mT"] = field_matrix.std(axis=0)
    return _register_profile_phase_to_command_zero_cross(profile)


def build_harmonic_transfer_lut(
    per_test_summary: pd.DataFrame,
    analyses_by_test_id: dict[str, DatasetAnalysis],
    current_channel: str = "i_sum_signed",
    field_channel: str = "bz_mT",
    voltage_channel: str = "daq_input_v",
    points_per_cycle: int = 256,
    max_harmonics: int = 9,
) -> pd.DataFrame:
    """Build a harmonic-domain transfer LUT from measured command/output profiles."""

    if per_test_summary.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for record in per_test_summary.to_dict(orient="records"):
        test_id = str(record.get("test_id", ""))
        analysis = analyses_by_test_id.get(test_id)
        if analysis is None:
            continue
        selection_info = _select_representative_cycle_indices(
            analysis=analysis,
            cycle_selection_mode="warm_tail",
        )
        profile = build_representative_cycle_profile(
            analysis=analysis,
            current_channel=current_channel,
            voltage_channel=voltage_channel,
            field_channel=field_channel,
            points_per_cycle=points_per_cycle,
            cycle_indices=selection_info["selected_cycle_indices"],
        )
        if profile.empty or "command_voltage_v" not in profile.columns:
            continue

        voltage_values = pd.to_numeric(profile["command_voltage_v"], errors="coerce").to_numpy(dtype=float)
        voltage_values = voltage_values - float(np.nanmean(voltage_values))
        voltage_fft = np.fft.rfft(voltage_values)
        for output_type, signal_column, signal_pp_key in (
            ("current", "measured_current_a", "achieved_current_pp_a_mean"),
            ("field", "measured_field_mT", f"achieved_{field_channel}_pp_mean"),
        ):
            if signal_column not in profile.columns:
                continue
            signal_values = pd.to_numeric(profile[signal_column], errors="coerce").to_numpy(dtype=float)
            signal_values = signal_values - float(np.nanmean(signal_values))
            signal_fft = np.fft.rfft(signal_values)
            for harmonic in range(1, min(max_harmonics, len(voltage_fft) - 1, len(signal_fft) - 1) + 1):
                voltage_component = voltage_fft[harmonic]
                signal_component = signal_fft[harmonic]
                if abs(voltage_component) < 1e-12:
                    continue
                transfer = signal_component / voltage_component
                rows.append(
                    {
                        "test_id": test_id,
                        "waveform_type": record.get("waveform_type"),
                        "freq_hz": float(record.get("freq_hz", np.nan)),
                        "harmonic": int(harmonic),
                        "harmonic_freq_hz": float(record.get("freq_hz", np.nan)) * harmonic,
                        "output_type": output_type,
                        "representative_cycle_selection_mode": selection_info["cycle_selection_mode"],
                        "representative_cycle_count_used": int(selection_info["selected_cycle_count"]),
                        "support_output_pp": float(record.get(signal_pp_key, np.nan)),
                        "daq_input_v_pp_mean": float(record.get("daq_input_v_pp_mean", np.nan)),
                        "transfer_real": float(np.real(transfer)),
                        "transfer_imag": float(np.imag(transfer)),
                        "transfer_mag": float(np.abs(transfer)),
                        "transfer_phase_deg": float(np.degrees(np.angle(transfer))),
                    }
                )

    return pd.DataFrame(rows)


def refine_compensation_with_validation(
    command_profile: pd.DataFrame,
    validation_frame: pd.DataFrame,
    target_output_type: str,
    current_channel: str = "i_sum_signed",
    field_channel: str = "bz_mT",
    voltage_channel: str = "daq_input_v",
    max_harmonics: int = 9,
    correction_gain: float = 0.7,
    max_daq_voltage_pp: float = 20.0,
    amp_gain_at_100_pct: float = 20.0,
    support_amp_gain_pct: float = 100.0,
    amp_gain_limit_pct: float = 100.0,
    amp_max_output_pk_v: float = 180.0,
) -> dict[str, Any] | None:
    """Apply a second-stage harmonic residual correction using validation-run data."""

    if (
        command_profile.empty
        or validation_frame.empty
        or "time_s" not in command_profile.columns
        or "limited_voltage_v" not in command_profile.columns
        or "time_s" not in validation_frame.columns
    ):
        return None

    target_column = "aligned_used_target_output" if "aligned_used_target_output" in command_profile.columns else "used_target_output"
    if target_column not in command_profile.columns:
        target_column = "aligned_target_output" if "aligned_target_output" in command_profile.columns else "target_output"
    validation_output_column = field_channel if target_output_type == "field" else current_channel
    if validation_output_column not in validation_frame.columns:
        return None

    command_times = pd.to_numeric(command_profile["time_s"], errors="coerce")
    fit_end_s = float(command_times.max())
    if "is_active_target" in command_profile.columns and command_profile["is_active_target"].any():
        fit_end_s = float(command_profile.loc[command_profile["is_active_target"], "time_s"].max())
    if not np.isfinite(fit_end_s) or fit_end_s <= 0:
        return None

    sample_count = max(256, min(len(command_profile), len(validation_frame)) * 4)
    time_grid = np.linspace(0.0, fit_end_s, sample_count)
    command_voltage = np.interp(
        time_grid,
        pd.to_numeric(command_profile["time_s"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(command_profile["limited_voltage_v"], errors="coerce").to_numpy(dtype=float),
    )
    target_output = np.interp(
        time_grid,
        pd.to_numeric(command_profile["time_s"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(command_profile[target_column], errors="coerce").to_numpy(dtype=float),
    )
    validation_time = pd.to_numeric(validation_frame["time_s"], errors="coerce").to_numpy(dtype=float)
    validation_output = np.interp(
        time_grid,
        validation_time,
        pd.to_numeric(validation_frame[validation_output_column], errors="coerce").to_numpy(dtype=float),
    )
    if voltage_channel in validation_frame.columns:
        validation_voltage = np.interp(
            time_grid,
            validation_time,
            pd.to_numeric(validation_frame[voltage_channel], errors="coerce").to_numpy(dtype=float),
        )
    else:
        validation_voltage = command_voltage.copy()

    command_centered = command_voltage - float(np.nanmean(command_voltage))
    validation_voltage_centered = validation_voltage - float(np.nanmean(validation_voltage))
    target_centered = target_output - float(np.nanmean(target_output))
    validation_output_centered = validation_output - float(np.nanmean(validation_output))

    command_fft = np.fft.rfft(command_centered)
    validation_voltage_fft = np.fft.rfft(validation_voltage_centered)
    target_fft = np.fft.rfft(target_centered)
    validation_output_fft = np.fft.rfft(validation_output_centered)
    refined_fft = command_fft.copy()
    validation_transfer_fft = np.zeros_like(validation_output_fft, dtype=np.complex128)
    valid_transfer_mask = np.abs(validation_voltage_fft) > 1e-12
    validation_transfer_fft[valid_transfer_mask] = (
        validation_output_fft[valid_transfer_mask] / validation_voltage_fft[valid_transfer_mask]
    )
    predicted_output_before = np.fft.irfft(command_fft * validation_transfer_fft, n=sample_count)

    for harmonic in range(1, min(max_harmonics, len(command_fft) - 1, len(validation_voltage_fft) - 1, len(validation_output_fft) - 1, len(target_fft) - 1) + 1):
        if abs(validation_voltage_fft[harmonic]) < 1e-12 or abs(validation_output_fft[harmonic]) < 1e-12:
            continue
        validation_transfer = validation_output_fft[harmonic] / validation_voltage_fft[harmonic]
        if abs(validation_transfer) < 1e-12:
            continue
        ideal_component = target_fft[harmonic] / validation_transfer
        blend = float(np.clip(correction_gain, 0.0, 1.0))
        refined_fft[harmonic] = command_fft[harmonic] + blend * (ideal_component - command_fft[harmonic])

    refined_voltage = np.fft.irfft(refined_fft, n=sample_count)

    refined_profile = pd.DataFrame(
        {
            "time_s": time_grid,
            "cycle_progress": np.clip(time_grid / fit_end_s, 0.0, 1.0),
            "target_output": target_output,
            "validation_output": validation_output,
            "pre_correction_voltage_v": command_voltage,
            "recommended_voltage_v": refined_voltage,
            "predicted_output_before": predicted_output_before,
        }
    )

    finite_cycle_mode = bool(command_profile["finite_cycle_mode"].iloc[0]) if "finite_cycle_mode" in command_profile.columns else False
    if finite_cycle_mode and "is_active_target" in command_profile.columns:
        active_mask = np.interp(
            time_grid,
            pd.to_numeric(command_profile["time_s"], errors="coerce").to_numpy(dtype=float),
            command_profile["is_active_target"].astype(float).to_numpy(dtype=float),
        ) >= 0.5
        refined_profile["is_active_target"] = active_mask
    else:
        refined_profile["is_active_target"] = True
    refined_profile["finite_cycle_mode"] = finite_cycle_mode
    refined_profile = apply_command_hardware_model(
        command_waveform=refined_profile,
        max_daq_voltage_pp=float(max_daq_voltage_pp),
        amp_gain_at_100_pct=float(amp_gain_at_100_pct),
        support_amp_gain_pct=float(support_amp_gain_pct),
        amp_gain_limit_pct=float(amp_gain_limit_pct),
        amp_max_output_pk_v=float(amp_max_output_pk_v),
        preserve_start_voltage=finite_cycle_mode,
    )
    if not finite_cycle_mode:
        refined_profile = _register_profile_phase_to_command_zero_cross(
            refined_profile,
            voltage_column="limited_voltage_v",
            rotate_columns=[
                column
                for column in (
                    "pre_correction_voltage_v",
                    "recommended_voltage_v",
                    "limited_voltage_v",
                    "target_output",
                    "validation_output",
                    "predicted_output_before",
                )
                if column in refined_profile.columns
            ],
        )
    limited_voltage = pd.to_numeric(refined_profile["limited_voltage_v"], errors="coerce").to_numpy(dtype=float)
    limited_voltage_centered = limited_voltage - float(np.nanmean(limited_voltage))
    predicted_output_after = np.fft.irfft(np.fft.rfft(limited_voltage_centered) * validation_transfer_fft, n=sample_count)
    if target_output_type == "current":
        refined_profile["expected_current_a"] = predicted_output_after
        refined_profile["target_current_a"] = target_output
        refined_profile["expected_output"] = predicted_output_after
    else:
        refined_profile["expected_field_mT"] = predicted_output_after
        refined_profile["target_field_mT"] = target_output
        refined_profile["expected_output"] = predicted_output_after
    refined_profile = _sync_modeled_alias_columns(refined_profile)
    refined_profile.attrs["bz_projection"] = {
        "available": True,
        "reason_code": None,
        "source": "validation_voltage_to_bz_transfer" if target_output_type == "field" else "current_to_bz_surrogate",
    }
    refined_profile.attrs["prediction_debug"] = {
        "solver_route": "validation_residual_second_stage",
        "field_prediction_source": "validation_transfer" if target_output_type == "field" else "current_to_bz_surrogate",
        "expected_current_source": "validation_transfer" if target_output_type == "current" else "field_penalty_surrogate",
    }
    transfer_metrics = _compute_secondary_residual_metrics(
        target_output=target_output,
        validation_output=validation_output,
        predicted_output_after=predicted_output_after,
    )
    predicted_before_metrics = _compute_secondary_residual_metrics(
        target_output=target_output,
        validation_output=predicted_output_before,
        predicted_output_after=predicted_output_before,
    )

    return {
        "mode": "validation_residual_second_stage",
        "target_output_type": target_output_type,
        "correction_gain": float(np.clip(correction_gain, 0.0, 1.0)),
        "validation_rmse_before": transfer_metrics["rmse_before"],
        "validation_nrmse_before": transfer_metrics["nrmse_before"],
        "predicted_rmse_before": predicted_before_metrics["rmse_before"],
        "predicted_nrmse_before": predicted_before_metrics["nrmse_before"],
        "predicted_rmse_after": transfer_metrics["rmse_after"],
        "predicted_nrmse_after": transfer_metrics["nrmse_after"],
        "within_hardware_limits": bool(refined_profile["within_hardware_limits"].iloc[0]),
        "command_profile": refined_profile,
    }


def run_validation_recommendation_loop(
    command_profile: pd.DataFrame,
    validation_frame: pd.DataFrame,
    target_output_type: str,
    current_channel: str = "i_sum_signed",
    field_channel: str = "bz_mT",
    voltage_channel: str = "daq_input_v",
    max_harmonics: int = 9,
    correction_gain: float = 0.7,
    max_iterations: int = 3,
    improvement_threshold: float = 0.0,
    max_daq_voltage_pp: float = 20.0,
    amp_gain_at_100_pct: float = 20.0,
    support_amp_gain_pct: float = 100.0,
    amp_gain_limit_pct: float = 100.0,
    amp_max_output_pk_v: float = 180.0,
) -> dict[str, Any] | None:
    """Iteratively refine a command waveform using one validation run as a transfer estimate."""

    iteration_limit = max(int(max_iterations), 1)
    current_profile = command_profile.copy()
    iterations: list[dict[str, Any]] = []
    final_result: dict[str, Any] | None = None
    stop_reason = "iteration_limit_reached"

    for iteration_index in range(1, iteration_limit + 1):
        refined = refine_compensation_with_validation(
            command_profile=current_profile,
            validation_frame=validation_frame,
            target_output_type=target_output_type,
            current_channel=current_channel,
            field_channel=field_channel,
            voltage_channel=voltage_channel,
            max_harmonics=max_harmonics,
            correction_gain=correction_gain,
            max_daq_voltage_pp=max_daq_voltage_pp,
            amp_gain_at_100_pct=amp_gain_at_100_pct,
            support_amp_gain_pct=support_amp_gain_pct,
            amp_gain_limit_pct=amp_gain_limit_pct,
            amp_max_output_pk_v=amp_max_output_pk_v,
        )
        if refined is None:
            return None if final_result is None else {
                **final_result,
                "mode": "validation_residual_recommendation_loop",
                "iteration_table": pd.DataFrame(iterations),
                "iteration_count": len(iterations),
                "stop_reason": "refinement_failed",
            }

        current_profile = refined["command_profile"].copy()
        predicted_before = float(refined.get("predicted_nrmse_before", np.nan))
        predicted_after = float(refined["predicted_nrmse_after"])
        improvement = predicted_before - predicted_after if np.isfinite(predicted_before) and np.isfinite(predicted_after) else float("nan")
        iterations.append(
            {
                "iteration": iteration_index,
                "predicted_nrmse_before": predicted_before,
                "predicted_nrmse_after": predicted_after,
                "predicted_improvement": improvement,
                "validation_nrmse_reference": float(refined["validation_nrmse_before"]),
                "limited_voltage_pp": float(current_profile["limited_voltage_pp"].iloc[0]),
                "required_amp_gain_pct": float(current_profile["required_amp_gain_pct"].iloc[0]),
                "within_hardware_limits": bool(current_profile["within_hardware_limits"].iloc[0]),
            }
        )
        final_result = refined
        if iteration_index > 1 and np.isfinite(improvement) and improvement <= float(improvement_threshold):
            stop_reason = "improvement_threshold_reached"
            break

    if final_result is None:
        return None

    return {
        "mode": "validation_residual_recommendation_loop",
        "target_output_type": target_output_type,
        "correction_gain": float(np.clip(correction_gain, 0.0, 1.0)),
        "iteration_count": len(iterations),
        "stop_reason": stop_reason,
        "validation_rmse_reference": float(final_result["validation_rmse_before"]),
        "validation_nrmse_reference": float(final_result["validation_nrmse_before"]),
        "predicted_rmse_final": float(final_result["predicted_rmse_after"]),
        "predicted_nrmse_final": float(final_result["predicted_nrmse_after"]),
        "within_hardware_limits": bool(final_result["within_hardware_limits"]),
        "iteration_table": pd.DataFrame(iterations),
        "command_profile": final_result["command_profile"],
    }


def _resolve_output_context(
    target_output_type: str,
    field_channel: str,
    current_metric: str,
) -> dict[str, str]:
    if target_output_type == "field":
        return {
            "output_type": "field",
            "output_metric": f"achieved_{field_channel}_pp_mean",
            "signal_column": "measured_field_mT",
            "target_column": "target_field_mT",
            "used_target_column": "used_target_field_mT",
            "label": f"{field_channel} PP",
            "unit": "mT",
        }
    return {
        "output_type": "current",
        "output_metric": current_metric,
        "signal_column": "measured_current_a",
        "target_column": "target_current_a",
        "used_target_column": "used_target_current_a",
        "label": "Current PP",
        "unit": "A",
    }


def _lookup_lcr_harmonic_transfer(
    lcr_prior_table: pd.DataFrame | None,
    harmonic: int,
) -> complex | None:
    if lcr_prior_table is None or lcr_prior_table.empty or "harmonic" not in lcr_prior_table.columns:
        return None
    match = lcr_prior_table[lcr_prior_table["harmonic"] == int(harmonic)]
    if match.empty:
        return None
    row = match.iloc[0]
    return complex(float(row["transfer_real"]), float(row["transfer_imag"]))


def _compute_secondary_residual_metrics(
    target_output: np.ndarray,
    validation_output: np.ndarray,
    predicted_output_after: np.ndarray,
) -> dict[str, float]:
    before_error = validation_output - target_output
    after_error = predicted_output_after - target_output
    target_pp = float(np.nanmax(target_output) - np.nanmin(target_output)) if len(target_output) else float("nan")
    rmse_before = float(np.sqrt(np.nanmean(np.square(before_error)))) if len(before_error) else float("nan")
    rmse_after = float(np.sqrt(np.nanmean(np.square(after_error)))) if len(after_error) else float("nan")
    denom = max(target_pp / 2.0, 1e-12) if np.isfinite(target_pp) and target_pp > 0 else float("nan")
    nrmse_before = rmse_before / denom if np.isfinite(denom) else float("nan")
    nrmse_after = rmse_after / denom if np.isfinite(denom) else float("nan")
    return {
        "rmse_before": rmse_before,
        "nrmse_before": nrmse_before,
        "rmse_after": rmse_after,
        "nrmse_after": nrmse_after,
    }


def _select_nearest_support_row(
    subset: pd.DataFrame,
    target_freq_hz: float,
    target_output_pp: float,
    output_metric: str,
    *,
    prefer_level_stable_family: bool = False,
) -> tuple[pd.Series, dict[str, Any]]:
    working = subset.copy()
    freq_range = max(float(working["freq_hz"].max()) - float(working["freq_hz"].min()), 1e-9)
    output_range = max(
        float(working[output_metric].max()) - float(working[output_metric].min()),
        1e-9,
    )
    working["freq_distance_hz"] = (working["freq_hz"] - float(target_freq_hz)).abs()
    working["output_distance"] = (working[output_metric] - float(target_output_pp)).abs()
    working["freq_distance_norm"] = working["freq_distance_hz"] / freq_range
    working["output_distance_norm"] = working["output_distance"] / output_range
    working["combined_distance"] = np.sqrt(
        np.square(working["freq_distance_norm"])
        + np.square(working["output_distance_norm"])
    )

    selection_meta: dict[str, Any] = {
        "support_selection_reason": "nearest_output_distance",
        "support_family_metric": None,
        "support_family_value": None,
        "estimated_family_level": None,
        "support_family_lock_applied": False,
        "support_candidate_count": int(len(working)),
        "support_family_candidate_count": 0,
        "support_bz_to_current_ratio": None,
    }
    working_for_select = working

    if prefer_level_stable_family and output_metric in working.columns:
        family_metric = next(
            (
                column
                for column in ("current_pp_target_a", "achieved_current_pp_a_mean")
                if column in working.columns and pd.to_numeric(working[column], errors="coerce").notna().any()
            ),
            None,
        )
        if family_metric is not None:
            family_values = pd.to_numeric(working[family_metric], errors="coerce")
            output_values = pd.to_numeric(working[output_metric], errors="coerce")
            valid_ratio = (
                np.isfinite(family_values.to_numpy(dtype=float))
                & np.isfinite(output_values.to_numpy(dtype=float))
                & (family_values.to_numpy(dtype=float) > 1e-9)
                & (output_values.to_numpy(dtype=float) > 1e-9)
            )
            if valid_ratio.any():
                ratio_values = np.divide(
                    output_values.to_numpy(dtype=float),
                    family_values.to_numpy(dtype=float),
                    out=np.full(len(working), np.nan, dtype=float),
                    where=np.abs(family_values.to_numpy(dtype=float)) > 1e-9,
                )
                typical_ratio = float(np.nanmedian(ratio_values[valid_ratio]))
                estimated_family_level = (
                    float(target_output_pp / typical_ratio)
                    if np.isfinite(typical_ratio) and typical_ratio > 1e-9
                    else float("nan")
                )
                family_range = max(float(family_values.max()) - float(family_values.min()), 1e-9)
                working["support_family_value"] = family_values
                working["support_bz_to_current_ratio"] = ratio_values
                working["support_family_distance"] = (family_values - estimated_family_level).abs()
                working["support_family_distance_norm"] = working["support_family_distance"] / family_range
                valid_family_distance = pd.to_numeric(working["support_family_distance"], errors="coerce").dropna()
                if not valid_family_distance.empty:
                    best_family_distance = float(valid_family_distance.min())
                    family_lock_tolerance = max(0.10 * family_range, 1.0)
                    family_lock_mask = (
                        pd.to_numeric(working["support_family_distance"], errors="coerce")
                        <= best_family_distance + family_lock_tolerance
                    )
                    family_locked = working[family_lock_mask].copy()
                    if not family_locked.empty:
                        working_for_select = family_locked
                        selection_meta["support_selection_reason"] = "exact_family_level_lock"
                        selection_meta["support_family_lock_applied"] = len(family_locked) < len(working)
                        selection_meta["support_family_candidate_count"] = int(len(family_locked))
                selection_meta["support_family_metric"] = family_metric
                selection_meta["estimated_family_level"] = (
                    estimated_family_level if np.isfinite(estimated_family_level) else None
                )

    sort_columns = [
        "freq_distance_hz",
        "output_distance",
        "combined_distance",
        "freq_hz",
    ]
    if "support_family_distance" in working_for_select.columns:
        sort_columns = [
            "support_family_distance",
            "output_distance",
            "combined_distance",
            "freq_distance_hz",
            "freq_hz",
        ]
    selected = working_for_select.sort_values(sort_columns, kind="stable").iloc[0]
    selection_meta["selected_support_id"] = (
        str(selected.get("test_id"))
        if pd.notna(selected.get("test_id"))
        else None
    )
    support_family_value = selected.get("support_family_value")
    if support_family_value is not None and pd.notna(support_family_value):
        selection_meta["support_family_value"] = float(support_family_value)
        family_metric = selection_meta.get("support_family_metric") or "support_family_value"
        selection_meta["selected_support_family"] = f"{family_metric}:{float(support_family_value):g}"
    else:
        selection_meta["selected_support_family"] = selection_meta["selected_support_id"]
    support_ratio = selected.get("support_bz_to_current_ratio")
    if support_ratio is not None and pd.notna(support_ratio):
        selection_meta["support_bz_to_current_ratio"] = float(support_ratio)
    return selected, selection_meta


def _select_nearest_frequency_support_row(
    subset: pd.DataFrame,
    target_freq_hz: float,
) -> tuple[pd.Series, dict[str, Any]]:
    working = subset.copy()
    working["freq_distance_hz"] = (working["freq_hz"] - float(target_freq_hz)).abs()
    selected = working.sort_values(["freq_distance_hz", "freq_hz", "test_id"], kind="stable").iloc[0]
    return selected, {
        "support_selection_reason": "nearest_frequency_support",
        "support_family_metric": None,
        "support_family_value": None,
        "estimated_family_level": None,
        "support_family_lock_applied": False,
        "support_candidate_count": int(len(working)),
        "support_family_candidate_count": 0,
        "support_bz_to_current_ratio": None,
        "selected_support_id": str(selected.get("test_id")) if pd.notna(selected.get("test_id")) else None,
        "selected_support_family": str(selected.get("test_id")) if pd.notna(selected.get("test_id")) else None,
    }


def _build_frequency_support_weight_table(
    support_profiles: list[dict[str, Any]],
    target_freq_hz: float,
    max_support_count: int = 4,
) -> pd.DataFrame:
    if not support_profiles:
        return pd.DataFrame()

    rows = pd.DataFrame(
        [
            {
                "index": index,
                "test_id": str(support["meta"].get("test_id", "")),
                "freq_hz": float(support["meta"].get("freq_hz", np.nan)),
            }
            for index, support in enumerate(support_profiles)
        ]
    )
    valid = rows["freq_hz"].replace([np.inf, -np.inf], np.nan).notna()
    rows = rows.loc[valid].copy()
    if rows.empty:
        return pd.DataFrame()

    used_target_freq_hz = float(
        np.clip(
            float(target_freq_hz),
            float(rows["freq_hz"].min()),
            float(rows["freq_hz"].max()),
        )
    )
    rows["freq_distance_hz"] = (rows["freq_hz"] - used_target_freq_hz).abs()
    rows = rows.sort_values(["freq_distance_hz", "freq_hz", "test_id"], kind="stable")
    rows = rows.head(max(int(max_support_count), 1)).copy()

    freq_span = max(float(rows["freq_hz"].max()) - float(rows["freq_hz"].min()), 1e-6)
    normalized_distance = rows["freq_distance_hz"].to_numpy(dtype=float) / freq_span
    raw_weights = 1.0 / (np.square(normalized_distance) + 1e-6)
    if not np.isfinite(raw_weights).all() or raw_weights.sum() <= 0:
        raw_weights = np.ones(len(rows), dtype=float)
    rows["weight"] = raw_weights / raw_weights.sum()
    rows["used_target_freq_hz"] = used_target_freq_hz
    return rows.reset_index(drop=True)


def _normalize_waveform_peak_to_peak(values: np.ndarray, target_pp: float) -> np.ndarray:
    waveform = np.asarray(values, dtype=float)
    finite = waveform[np.isfinite(waveform)]
    if len(finite) == 0:
        return waveform
    centered = waveform - float(np.nanmean(finite))
    current_pp = float(np.nanmax(finite) - np.nanmin(finite))
    if not np.isfinite(current_pp) or current_pp <= 1e-12:
        return centered
    return centered * float(target_pp) / current_pp


def _align_waveform_sign(candidate: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
    aligned = np.asarray(candidate, dtype=float)
    if reference is None:
        return aligned
    reference_values = np.asarray(reference, dtype=float)
    valid = np.isfinite(aligned) & np.isfinite(reference_values)
    if valid.sum() < 4:
        return aligned
    positive_score = float(np.dot(reference_values[valid], aligned[valid]))
    negative_score = float(np.dot(reference_values[valid], -aligned[valid]))
    return -aligned if negative_score > positive_score else aligned


def _build_field_support_transfer_components(
    support_profiles: list[dict[str, Any]],
    output_signal_column: str,
    points_per_cycle: int,
    nearest_support: dict[str, Any],
    normalized_target_pp: float,
) -> dict[int, dict[str, Any]]:
    reference_profile = nearest_support["profile"]
    reference_field = _normalize_waveform_peak_to_peak(
        pd.to_numeric(reference_profile.get(output_signal_column), errors="coerce").to_numpy(dtype=float),
        normalized_target_pp,
    )

    components: dict[int, dict[str, Any]] = {}
    for index, support in enumerate(support_profiles):
        profile = support["profile"]
        voltage_values = pd.to_numeric(profile.get("command_voltage_v"), errors="coerce").to_numpy(dtype=float)
        field_values = pd.to_numeric(profile.get(output_signal_column), errors="coerce").to_numpy(dtype=float)
        if len(voltage_values) != points_per_cycle or len(field_values) != points_per_cycle:
            continue
        voltage_centered = voltage_values - float(np.nanmean(voltage_values))
        normalized_field = _normalize_waveform_peak_to_peak(field_values, normalized_target_pp)
        normalized_field = _align_waveform_sign(normalized_field, reference_field)
        voltage_fft = np.fft.rfft(voltage_centered)
        field_fft = np.fft.rfft(normalized_field)
        components[index] = {
            "voltage_values": voltage_centered,
            "normalized_field_values": normalized_field,
            "voltage_fft": voltage_fft,
            "field_fft": field_fft,
        }
    return components


def build_waveform_diagnostic_exports(
    analyses: list[DatasetAnalysis],
    current_channel: str = "i_sum_signed",
    field_channel: str = "bz_mT",
    voltage_channel: str = "daq_input_v",
    points_per_cycle: int = 256,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build export tables for judging measured input/output waveform quality."""

    summary_rows: list[dict[str, Any]] = []
    profile_frames: list[pd.DataFrame] = []

    for analysis in analyses:
        if analysis.parsed.normalized_frame.empty or analysis.per_test_summary.empty:
            continue

        test_summary = analysis.per_test_summary.iloc[0]
        profile = build_representative_cycle_profile(
            analysis=analysis,
            current_channel=current_channel,
            voltage_channel=voltage_channel,
            field_channel=field_channel,
            points_per_cycle=points_per_cycle,
        )
        if profile.empty:
            continue

        waveform_type = str(test_summary.get("waveform_type", "sine") or "sine")
        freq_hz = float(test_summary.get("freq_hz", np.nan))
        achieved_current_pp = float(test_summary.get("achieved_current_pp_a_mean", np.nan))
        voltage_pp = float(test_summary.get("daq_input_v_pp_mean", np.nan))
        field_pp = float(test_summary.get(f"achieved_{field_channel}_pp_mean", np.nan))

        current_detail, current_metrics = _compare_signal_to_ideal(
            profile=profile,
            signal_column="measured_current_a",
            std_column="measured_current_std_a",
            waveform_type=waveform_type,
            freq_hz=freq_hz,
            amplitude_pp=achieved_current_pp,
            output_prefix="current",
        )
        voltage_detail, voltage_metrics = _compare_signal_to_ideal(
            profile=profile,
            signal_column="command_voltage_v",
            std_column="command_voltage_std_v",
            waveform_type=waveform_type,
            freq_hz=freq_hz,
            amplitude_pp=voltage_pp,
            output_prefix="voltage",
        )
        field_detail, field_metrics = _compare_signal_to_ideal(
            profile=profile,
            signal_column="measured_field_mT",
            std_column="measured_field_std_mT",
            waveform_type=waveform_type,
            freq_hz=freq_hz,
            amplitude_pp=field_pp,
            output_prefix="field",
        )

        detail = profile[["cycle_progress", "time_s"]].copy()
        for candidate in (current_detail, voltage_detail, field_detail):
            if candidate.empty:
                continue
            detail = detail.merge(candidate, on=["cycle_progress", "time_s"], how="left")
        detail.insert(0, "test_id", str(test_summary["test_id"]))
        detail.insert(1, "waveform_type", waveform_type)
        detail.insert(2, "freq_hz", freq_hz)
        profile_frames.append(detail)

        summary_rows.append(
            {
                "test_id": str(test_summary["test_id"]),
                "waveform_type": waveform_type,
                "freq_hz": freq_hz,
                "current_pp_target_a": test_summary.get("current_pp_target_a"),
                "achieved_current_pp_a_mean": achieved_current_pp,
                "daq_input_v_pp_mean": voltage_pp,
                f"achieved_{field_channel}_pp_mean": field_pp,
                **current_metrics,
                **voltage_metrics,
                **field_metrics,
            }
        )

    summary_frame = pd.DataFrame(summary_rows)
    profiles_frame = pd.concat(profile_frames, ignore_index=True) if profile_frames else pd.DataFrame()
    return summary_frame, profiles_frame


def _single_profile_scaled_compensation(
    support_profile: pd.DataFrame,
    target_profile: pd.DataFrame,
    support_current_pp: float,
    target_current_pp: float,
) -> pd.DataFrame:
    scale_ratio = float(target_current_pp / support_current_pp) if np.isfinite(support_current_pp) and support_current_pp != 0 else 1.0
    command = target_profile[["cycle_progress", "time_s", "target_current_a"]].copy()
    command["used_target_current_a"] = command["target_current_a"]
    command["recommended_voltage_v"] = support_profile["command_voltage_v"].to_numpy(dtype=float) * scale_ratio
    return command


def _apply_daq_voltage_limit(
    command_profile: pd.DataFrame,
    max_daq_voltage_pp: float,
    preserve_start_voltage: bool = False,
) -> pd.DataFrame:
    command = command_profile.copy()
    recommended = pd.to_numeric(command["recommended_voltage_v"], errors="coerce").to_numpy(dtype=float)
    if len(recommended) == 0 or not np.isfinite(recommended).any():
        command["recommended_voltage_pp"] = float("nan")
        command["limited_voltage_v"] = recommended
        command["limited_voltage_pp"] = float("nan")
        command["required_amp_gain_multiplier"] = float("nan")
        command["within_daq_limit"] = False
        return command

    if "is_lookahead_target" in command.columns:
        active_mask = command["is_lookahead_target"].to_numpy(dtype=bool)
    elif "is_active_target" in command.columns:
        active_mask = command["is_active_target"].to_numpy(dtype=bool)
    else:
        active_mask = np.isfinite(recommended)

    if preserve_start_voltage:
        if active_mask.any():
            pp_values = recommended[active_mask]
        else:
            pp_values = recommended
        scaled_values = recommended
    else:
        centered = np.zeros_like(recommended)
        if active_mask.any():
            active_values = recommended[active_mask]
            centered[active_mask] = active_values - float(np.nanmean(active_values))
            pp_values = centered[active_mask]
        else:
            centered = recommended - float(np.nanmean(recommended))
            pp_values = centered
        scaled_values = centered

    recommended_pp = float(np.nanmax(pp_values) - np.nanmin(pp_values))
    if not np.isfinite(max_daq_voltage_pp) or max_daq_voltage_pp <= 0:
        gain_multiplier = 1.0
    else:
        gain_multiplier = max(recommended_pp / max_daq_voltage_pp, 1.0)
    limited = scaled_values / gain_multiplier
    limited_pp = float(np.nanmax(limited[active_mask]) - np.nanmin(limited[active_mask])) if active_mask.any() else float(np.nanmax(limited) - np.nanmin(limited))

    command["recommended_voltage_v"] = scaled_values
    command["recommended_voltage_pp"] = recommended_pp
    command["limited_voltage_v"] = limited
    command["limited_voltage_pp"] = limited_pp
    command["required_amp_gain_multiplier"] = gain_multiplier
    command["within_daq_limit"] = bool(recommended_pp <= max_daq_voltage_pp + 1e-9)
    return command


def _harmonic_inverse_compensation(
    support_profiles: list[dict[str, Any]],
    target_profile: pd.DataFrame,
    target_output_pp: float,
    target_freq_hz: float,
    output_metric: str,
    output_signal_column: str,
    nearest_support: dict[str, Any],
    points_per_cycle: int,
    allow_output_extrapolation: bool,
    max_harmonics: int = 11,
    lcr_prior_table: pd.DataFrame | None = None,
    lcr_blend_weight: float = 0.0,
) -> tuple[pd.DataFrame, np.ndarray]:
    target_output = pd.to_numeric(target_profile["target_output"], errors="coerce").to_numpy(dtype=float)
    target_output = target_output - float(np.nanmean(target_output))
    target_fft = np.fft.rfft(target_output)
    recommended_fft = np.zeros_like(target_fft, dtype=np.complex128)
    transfer_model = np.zeros_like(target_fft, dtype=np.complex128)

    nearest_profile = nearest_support["profile"]
    nearest_voltage = pd.to_numeric(nearest_profile["command_voltage_v"], errors="coerce").to_numpy(dtype=float)
    nearest_voltage = nearest_voltage - float(np.nanmean(nearest_voltage))
    nearest_voltage_fft = np.fft.rfft(nearest_voltage)
    nearest_output = pd.to_numeric(nearest_profile.get(output_signal_column), errors="coerce").to_numpy(dtype=float)
    nearest_output = nearest_output - float(np.nanmean(nearest_output))
    nearest_output_fft = np.fft.rfft(nearest_output) if len(nearest_output) == len(nearest_voltage) else np.zeros_like(nearest_voltage_fft)
    nearest_output_level = float(nearest_support["meta"].get(output_metric, np.nan))
    fallback_scale = (
        float(target_output_pp / nearest_output_level)
        if np.isfinite(nearest_output_level) and nearest_output_level != 0
        else 1.0
    )

    harmonic_limit = min(max_harmonics, len(target_fft) - 1)
    support_levels = np.asarray(
        [float(support["meta"].get(output_metric, np.nan)) for support in support_profiles],
        dtype=float,
    )
    support_frequencies = np.asarray(
        [float(support["meta"].get("freq_hz", np.nan)) for support in support_profiles],
        dtype=float,
    )

    for harmonic_index in range(1, harmonic_limit + 1):
        desired_output_component = target_fft[harmonic_index]
        if not np.isfinite(desired_output_component.real) or not np.isfinite(desired_output_component.imag):
            continue
        if abs(desired_output_component) < 1e-12:
            continue

        transfer_values: list[complex] = []
        transfer_levels: list[float] = []
        transfer_frequencies: list[float] = []
        for support_index, support in enumerate(support_profiles):
            level = support_levels[support_index]
            freq_value = support_frequencies[support_index]
            if not np.isfinite(level):
                continue
            profile = support["profile"]
            current_values = pd.to_numeric(profile.get(output_signal_column), errors="coerce").to_numpy(dtype=float)
            voltage_values = pd.to_numeric(profile.get("command_voltage_v"), errors="coerce").to_numpy(dtype=float)
            if len(current_values) != points_per_cycle or len(voltage_values) != points_per_cycle:
                continue
            current_values = current_values - float(np.nanmean(current_values))
            voltage_values = voltage_values - float(np.nanmean(voltage_values))
            current_fft = np.fft.rfft(current_values)
            voltage_fft = np.fft.rfft(voltage_values)
            if harmonic_index >= len(current_fft) or harmonic_index >= len(voltage_fft):
                continue
            voltage_component = voltage_fft[harmonic_index]
            current_component = current_fft[harmonic_index]
            if abs(voltage_component) < 1e-12 or not np.isfinite(voltage_component.real) or not np.isfinite(voltage_component.imag):
                continue
            transfer_levels.append(level)
            transfer_frequencies.append(freq_value)
            transfer_values.append(current_component / voltage_component)

        if transfer_levels:
            transfer_estimate = _interpolate_complex_transfer_surface(
                frequencies=np.asarray(transfer_frequencies, dtype=float),
                levels=np.asarray(transfer_levels, dtype=float),
                transfers=np.asarray(transfer_values, dtype=np.complex128),
                target_frequency=float(target_freq_hz),
                target_level=float(target_output_pp),
                allow_level_extrapolation=allow_output_extrapolation,
            )
            lcr_transfer = _lookup_lcr_harmonic_transfer(
                lcr_prior_table=lcr_prior_table,
                harmonic=harmonic_index,
            )
            if lcr_transfer is not None and abs(lcr_transfer) >= 1e-12:
                empirical_weight = 1.0 - float(np.clip(lcr_blend_weight, 0.0, 1.0))
                support_bonus = min(len(transfer_levels), 3) / 3.0
                effective_lcr_weight = float(np.clip(lcr_blend_weight * (1.0 - support_bonus * 0.5), 0.0, 1.0))
                empirical_weight = 1.0 - effective_lcr_weight
                transfer_estimate = empirical_weight * transfer_estimate + effective_lcr_weight * lcr_transfer
            if np.isfinite(transfer_estimate.real) and np.isfinite(transfer_estimate.imag) and abs(transfer_estimate) >= 1e-12:
                recommended_fft[harmonic_index] = desired_output_component / transfer_estimate
                transfer_model[harmonic_index] = transfer_estimate
                continue

        lcr_transfer = _lookup_lcr_harmonic_transfer(
            lcr_prior_table=lcr_prior_table,
            harmonic=harmonic_index,
        )
        if lcr_transfer is not None and abs(lcr_transfer) >= 1e-12 and float(lcr_blend_weight) > 0:
            recommended_fft[harmonic_index] = desired_output_component / lcr_transfer
            transfer_model[harmonic_index] = lcr_transfer
            continue

        if harmonic_index < len(nearest_voltage_fft):
            recommended_fft[harmonic_index] = nearest_voltage_fft[harmonic_index] * fallback_scale
            if harmonic_index < len(nearest_output_fft):
                voltage_component = nearest_voltage_fft[harmonic_index]
                output_component = nearest_output_fft[harmonic_index]
                if abs(voltage_component) >= 1e-12:
                    transfer_model[harmonic_index] = output_component / voltage_component

    recommended_voltage = np.fft.irfft(recommended_fft, n=points_per_cycle)
    command = target_profile[["cycle_progress", "time_s", "target_output"]].copy()
    command["used_target_output"] = target_profile["used_target_output"]
    for column in ("target_current_a", "used_target_current_a", "target_field_mT", "used_target_field_mT"):
        if column in target_profile.columns:
            command[column] = target_profile[column]
    command["recommended_voltage_v"] = recommended_voltage
    return command, transfer_model


def _harmonic_inverse_field_only_compensation(
    support_profiles: list[dict[str, Any]],
    target_profile: pd.DataFrame,
    target_freq_hz: float,
    output_signal_column: str,
    nearest_support: dict[str, Any],
    points_per_cycle: int,
    support_weight_table: pd.DataFrame,
    max_harmonics: int = 11,
    normalized_target_pp: float = FIELD_ROUTE_NORMALIZED_TARGET_PP,
) -> tuple[pd.DataFrame, np.ndarray]:
    target_output = pd.to_numeric(target_profile["target_output"], errors="coerce").to_numpy(dtype=float)
    target_output = target_output - float(np.nanmean(target_output))
    target_fft = np.fft.rfft(target_output)
    recommended_fft = np.zeros_like(target_fft, dtype=np.complex128)
    transfer_model = np.zeros_like(target_fft, dtype=np.complex128)

    components = _build_field_support_transfer_components(
        support_profiles=support_profiles,
        output_signal_column=output_signal_column,
        points_per_cycle=points_per_cycle,
        nearest_support=nearest_support,
        normalized_target_pp=normalized_target_pp,
    )
    if not components:
        return _harmonic_inverse_compensation(
            support_profiles=support_profiles,
            target_profile=target_profile,
            target_output_pp=normalized_target_pp,
            target_freq_hz=target_freq_hz,
            output_metric=f"achieved_{output_signal_column}_pp_mean",
            output_signal_column=output_signal_column,
            nearest_support=nearest_support,
            points_per_cycle=points_per_cycle,
            allow_output_extrapolation=True,
            max_harmonics=max_harmonics,
            lcr_prior_table=None,
            lcr_blend_weight=0.0,
        )

    working_weights = support_weight_table.copy() if not support_weight_table.empty else _build_frequency_support_weight_table(
        support_profiles=support_profiles,
        target_freq_hz=target_freq_hz,
    )
    if working_weights.empty:
        nearest_index = next(
            (
                index
                for index, support in enumerate(support_profiles)
                if str(support["meta"].get("test_id")) == str(nearest_support["meta"].get("test_id"))
            ),
            0,
        )
        working_weights = pd.DataFrame(
            [{"index": nearest_index, "weight": 1.0}]
        )

    harmonic_limit = min(max_harmonics, len(target_fft) - 1)
    for harmonic_index in range(1, harmonic_limit + 1):
        desired_output_component = target_fft[harmonic_index]
        if not np.isfinite(desired_output_component.real) or not np.isfinite(desired_output_component.imag):
            continue
        if abs(desired_output_component) < 1e-12:
            continue

        transfers: list[complex] = []
        weights: list[float] = []
        for row in working_weights.to_dict(orient="records"):
            support_index = int(row["index"])
            component = components.get(support_index)
            if component is None:
                continue
            voltage_fft = component["voltage_fft"]
            field_fft = component["field_fft"]
            if harmonic_index >= len(voltage_fft) or harmonic_index >= len(field_fft):
                continue
            voltage_component = voltage_fft[harmonic_index]
            if abs(voltage_component) < 1e-12:
                continue
            transfer = field_fft[harmonic_index] / voltage_component
            if not np.isfinite(transfer.real) or not np.isfinite(transfer.imag) or abs(transfer) < 1e-12:
                continue
            transfers.append(transfer)
            weights.append(float(row["weight"]))

        if transfers:
            weight_array = np.asarray(weights, dtype=float)
            if not np.isfinite(weight_array).all() or weight_array.sum() <= 0:
                weight_array = np.ones(len(transfers), dtype=float)
            transfer_array = np.asarray(transfers, dtype=np.complex128)
            transfer_estimate = complex(
                np.sum(weight_array * transfer_array.real) / np.sum(weight_array),
                np.sum(weight_array * transfer_array.imag) / np.sum(weight_array),
            )
            recommended_fft[harmonic_index] = desired_output_component / transfer_estimate
            transfer_model[harmonic_index] = transfer_estimate
            continue

        nearest_component = components.get(int(working_weights.iloc[0]["index"]))
        if nearest_component is None:
            continue
        voltage_fft = nearest_component["voltage_fft"]
        field_fft = nearest_component["field_fft"]
        if harmonic_index >= len(voltage_fft) or harmonic_index >= len(field_fft):
            continue
        voltage_component = voltage_fft[harmonic_index]
        if abs(voltage_component) < 1e-12:
            continue
        transfer_estimate = field_fft[harmonic_index] / voltage_component
        if not np.isfinite(transfer_estimate.real) or not np.isfinite(transfer_estimate.imag) or abs(transfer_estimate) < 1e-12:
            continue
        recommended_fft[harmonic_index] = desired_output_component / transfer_estimate
        transfer_model[harmonic_index] = transfer_estimate

    recommended_voltage = np.fft.irfft(recommended_fft, n=points_per_cycle)
    command = target_profile[["cycle_progress", "time_s", "target_output"]].copy()
    command["used_target_output"] = target_profile["used_target_output"]
    for column in ("target_current_a", "used_target_current_a", "target_field_mT", "used_target_field_mT"):
        if column in target_profile.columns:
            command[column] = target_profile[column]
    command["recommended_voltage_v"] = recommended_voltage
    return command, transfer_model


def _estimate_weighted_output_lag_seconds(
    support_profiles: list[dict[str, Any]],
    output_signal_column: str,
    output_metric: str,
    target_freq_hz: float,
    target_output_pp: float,
    prefer_frequency_only: bool = False,
) -> float:
    if not support_profiles or target_freq_hz <= 0:
        return 0.0

    frequencies = np.asarray(
        [float(support["meta"].get("freq_hz", np.nan)) for support in support_profiles],
        dtype=float,
    )
    levels = np.asarray(
        [float(support["meta"].get(output_metric, np.nan)) for support in support_profiles],
        dtype=float,
    )
    freq_span = max(float(np.nanmax(frequencies)) - float(np.nanmin(frequencies)), 1e-9)
    level_span = max(float(np.nanmax(levels)) - float(np.nanmin(levels)), 1e-9)

    lag_values: list[float] = []
    weights: list[float] = []
    for support in support_profiles:
        profile = support["profile"]
        lag_seconds = _estimate_profile_output_lag_seconds(
            profile=profile,
            output_signal_column=output_signal_column,
            freq_hz=float(support["meta"].get("freq_hz", target_freq_hz)),
        )
        if not np.isfinite(lag_seconds):
            continue
        freq_distance = abs(float(support["meta"].get("freq_hz", target_freq_hz)) - target_freq_hz) / freq_span
        if prefer_frequency_only:
            weight = 1.0 / (freq_distance * freq_distance + 1e-6)
        else:
            output_distance = abs(float(support["meta"].get(output_metric, target_output_pp)) - target_output_pp) / level_span
            weight = 1.0 / (freq_distance * freq_distance + output_distance * output_distance + 1e-6)
        lag_values.append(float(lag_seconds))
        weights.append(float(weight))

    if not lag_values:
        return 0.0

    return float(np.average(np.asarray(lag_values, dtype=float), weights=np.asarray(weights, dtype=float)))


def _estimate_profile_output_lag_seconds(
    profile: pd.DataFrame,
    output_signal_column: str,
    freq_hz: float,
) -> float:
    if (
        profile.empty
        or "command_voltage_v" not in profile.columns
        or output_signal_column not in profile.columns
        or not np.isfinite(freq_hz)
        or freq_hz <= 0
    ):
        return float("nan")

    voltage = pd.to_numeric(profile["command_voltage_v"], errors="coerce").to_numpy(dtype=float)
    output = pd.to_numeric(profile[output_signal_column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(voltage) & np.isfinite(output)
    if valid.sum() < 4:
        return float("nan")

    voltage = voltage[valid] - float(np.nanmean(voltage[valid]))
    output = output[valid] - float(np.nanmean(output[valid]))
    voltage_fft = np.fft.rfft(voltage)
    output_fft = np.fft.rfft(output)
    if len(voltage_fft) <= 1 or len(output_fft) <= 1:
        return float("nan")

    voltage_component = voltage_fft[1]
    output_component = output_fft[1]
    if abs(voltage_component) < 1e-12 or abs(output_component) < 1e-12:
        return float("nan")

    phase_difference = np.angle(voltage_component) - np.angle(output_component)
    wrapped_phase = np.arctan2(np.sin(phase_difference), np.cos(phase_difference))
    return float(wrapped_phase / (2.0 * np.pi * float(freq_hz)))


def _first_numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, pd.Series):
        if value.empty:
            return None
        value = value.iloc[0]
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else None


def _first_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, pd.Series):
        if value.empty:
            return None
        value = value.iloc[0]
    text = str(value).strip()
    return text or None


def _first_boolish(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, pd.Series):
        if value.empty:
            return False
        value = value.iloc[0]
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "nan", "none", "false", "0", "no"}:
            return False
        if normalized in {"true", "1", "yes"}:
            return True
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return bool(value)
    return bool(numeric)


def _signal_peak_to_peak(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan")
    return float(np.nanmax(finite) - np.nanmin(finite))


def _select_representative_cycle_indices(
    analysis: DatasetAnalysis | Any,
    cycle_selection_mode: str = "warm_tail",
) -> dict[str, Any]:
    per_cycle_summary = getattr(analysis, "per_cycle_summary", pd.DataFrame())
    if per_cycle_summary is None or per_cycle_summary.empty or "cycle_index" not in per_cycle_summary.columns:
        return {
            "cycle_selection_mode": "all",
            "selected_cycle_indices": [],
            "selected_cycle_count": 0,
            "available_cycle_count": 0,
        }

    cycle_indices = sorted(
        int(value)
        for value in pd.to_numeric(per_cycle_summary["cycle_index"], errors="coerce").dropna().tolist()
    )
    if not cycle_indices:
        return {
            "cycle_selection_mode": "all",
            "selected_cycle_indices": [],
            "selected_cycle_count": 0,
            "available_cycle_count": 0,
        }

    selected_cycle_indices = list(cycle_indices)
    effective_mode = "all"
    if cycle_selection_mode == "warm_tail":
        effective_mode = "warm_tail"
        if len(cycle_indices) >= 4:
            selected_cycle_indices = cycle_indices[-3:]
        elif len(cycle_indices) == 3:
            selected_cycle_indices = cycle_indices[1:]
        elif len(cycle_indices) == 2:
            selected_cycle_indices = cycle_indices[1:]

    return {
        "cycle_selection_mode": effective_mode,
        "selected_cycle_indices": selected_cycle_indices,
        "selected_cycle_count": len(selected_cycle_indices),
        "available_cycle_count": len(cycle_indices),
    }


def _build_startup_diagnostics(
    analysis: DatasetAnalysis | Any,
    field_channel: str = "bz_mT",
) -> dict[str, Any]:
    per_cycle_summary = getattr(analysis, "per_cycle_summary", pd.DataFrame())
    if per_cycle_summary is None or per_cycle_summary.empty or "cycle_index" not in per_cycle_summary.columns:
        return {}

    working = per_cycle_summary.sort_values("cycle_index").reset_index(drop=True)
    first_row = working.iloc[0]
    steady_window = working.iloc[1:] if len(working) > 1 else working.iloc[0:0]
    if len(steady_window) > 3:
        steady_window = steady_window.tail(3)

    current_column = "achieved_current_pp_a"
    field_column = f"achieved_{field_channel}_pp"
    voltage_column = "daq_input_v_pp"
    steady_current_mean = float(steady_window[current_column].mean()) if not steady_window.empty and current_column in steady_window.columns else float("nan")
    steady_field_mean = float(steady_window[field_column].mean()) if not steady_window.empty and field_column in steady_window.columns else float("nan")
    steady_voltage_mean = float(steady_window[voltage_column].mean()) if not steady_window.empty and voltage_column in steady_window.columns else float("nan")
    first_current = float(first_row[current_column]) if current_column in first_row.index else float("nan")
    first_field = float(first_row[field_column]) if field_column in first_row.index else float("nan")
    first_voltage = float(first_row[voltage_column]) if voltage_column in first_row.index else float("nan")
    annotated_frame = getattr(getattr(analysis, "cycle_detection", None), "annotated_frame", pd.DataFrame())
    field_mean_stats = _cycle_signal_mean_startup_stats(
        annotated_frame=annotated_frame,
        signal_column=field_channel,
    )
    current_mean_stats = _cycle_signal_mean_startup_stats(
        annotated_frame=annotated_frame,
        signal_column="i_sum_signed",
    )

    def _ratio(numerator: float, denominator: float) -> float:
        if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1e-12:
            return float("nan")
        return float(numerator / denominator)

    current_ratio = _ratio(first_current, steady_current_mean)
    field_ratio = _ratio(first_field, steady_field_mean)
    voltage_ratio = _ratio(first_voltage, steady_voltage_mean)
    unexplained_field_offset_delta = _field_offset_residual_from_current(
        field_offset_delta=field_mean_stats["offset_delta"],
        current_offset_delta=current_mean_stats["offset_delta"],
        field_pp=steady_field_mean,
        current_pp=steady_current_mean,
    )

    behavior = "insufficient_cycles"
    ratio_candidates = [value for value in (current_ratio, field_ratio) if np.isfinite(value)]
    if ratio_candidates:
        peak_ratio = max(ratio_candidates)
        valley_ratio = min(ratio_candidates)
        if peak_ratio > 1.05:
            behavior = "first_cycle_overshoot"
        elif valley_ratio < 0.95:
            behavior = "first_cycle_undershoot"
        else:
            behavior = "steady_like"

    return {
        "cycle_count": int(working["cycle_index"].nunique()),
        "steady_window_cycle_count": int(len(steady_window)),
        "first_cycle_current_pp_a": first_current,
        "steady_current_pp_a_mean": steady_current_mean,
        "first_cycle_current_ratio_vs_steady": current_ratio,
        "first_cycle_field_pp_mT": first_field,
        "steady_field_pp_mT_mean": steady_field_mean,
        "first_cycle_field_ratio_vs_steady": field_ratio,
        "first_cycle_voltage_pp_v": first_voltage,
        "steady_voltage_pp_v_mean": steady_voltage_mean,
        "first_cycle_voltage_ratio_vs_steady": voltage_ratio,
        "first_cycle_field_mean_mT": field_mean_stats["first_cycle_mean"],
        "steady_field_mean_mT": field_mean_stats["steady_mean"],
        "first_cycle_field_offset_delta_mT": field_mean_stats["offset_delta"],
        "first_cycle_field_unexplained_offset_delta_mT": unexplained_field_offset_delta,
        "first_cycle_current_mean_a": current_mean_stats["first_cycle_mean"],
        "steady_current_mean_a": current_mean_stats["steady_mean"],
        "first_cycle_current_offset_delta_a": current_mean_stats["offset_delta"],
        "behavior_flag": behavior,
    }


def _cycle_signal_mean_startup_stats(
    *,
    annotated_frame: pd.DataFrame,
    signal_column: str,
) -> dict[str, float]:
    empty = {
        "first_cycle_mean": float("nan"),
        "steady_mean": float("nan"),
        "offset_delta": float("nan"),
    }
    if annotated_frame is None or annotated_frame.empty:
        return empty
    if "cycle_index" not in annotated_frame.columns or signal_column not in annotated_frame.columns:
        return empty
    working = annotated_frame[["cycle_index", signal_column]].copy()
    working["cycle_index"] = pd.to_numeric(working["cycle_index"], errors="coerce")
    working[signal_column] = pd.to_numeric(working[signal_column], errors="coerce")
    working = working.dropna(subset=["cycle_index", signal_column])
    if working.empty:
        return empty
    grouped = working.groupby("cycle_index", sort=True)[signal_column].mean()
    if grouped.empty:
        return empty
    first_mean = float(grouped.iloc[0])
    steady = grouped.iloc[1:]
    if len(steady) > 3:
        steady = steady.tail(3)
    steady_mean = float(steady.mean()) if not steady.empty else float("nan")
    offset_delta = (
        float(first_mean - steady_mean)
        if np.isfinite(first_mean) and np.isfinite(steady_mean)
        else float("nan")
    )
    return {
        "first_cycle_mean": first_mean,
        "steady_mean": steady_mean,
        "offset_delta": offset_delta,
    }


def _field_offset_residual_from_current(
    *,
    field_offset_delta: float,
    current_offset_delta: float,
    field_pp: float,
    current_pp: float,
) -> float:
    if not np.isfinite(field_offset_delta):
        return float("nan")
    if (
        not np.isfinite(current_offset_delta)
        or not np.isfinite(field_pp)
        or not np.isfinite(current_pp)
        or abs(current_pp) <= 1e-12
    ):
        return float(field_offset_delta)
    expected_field_delta = float(current_offset_delta) * abs(float(field_pp) / float(current_pp))
    return float(field_offset_delta - expected_field_delta)


def _build_compensation_sequence_table(
    waveform_type: str,
    requested_freq_hz: float,
    used_freq_hz: float,
    target_output_type: str,
    target_output_pp: float,
    target_output_unit: str,
    support_point_count: int,
    nearest_test_id: str,
    representative_cycle_selection_mode: str,
    representative_cycle_indices: list[int],
    representative_cycle_count: int,
    available_cycle_count: int,
    used_lcr_prior: bool,
    lcr_blend_weight: float,
    finite_cycle_mode: bool,
    preserve_start_voltage: bool,
    within_hardware_limits: bool,
    max_daq_voltage_pp: float,
    limited_voltage_pp: float,
    estimated_output_lag_seconds: float,
    startup_diagnostics: dict[str, Any],
    startup_correction_applied: bool,
    startup_correction_factor: float,
    startup_transition_cycles: float,
) -> pd.DataFrame:
    cycle_label = (
        ", ".join(str(index) for index in representative_cycle_indices)
        if representative_cycle_indices
        else "all detected cycles"
    )
    startup_behavior = startup_diagnostics.get("behavior_flag", "unknown")
    rows = [
        {
            "step": 1,
            "stage": "target_template",
            "summary": f"{waveform_type} / {requested_freq_hz:.3f} Hz / {target_output_type} {target_output_pp:.3f} {target_output_unit}",
            "detail": "0초 기준 theoretical target template 생성",
        },
        {
            "step": 2,
            "stage": "support_selection",
            "summary": f"support {support_point_count}개, nearest `{nearest_test_id}`",
            "detail": f"대표 cycle={representative_cycle_selection_mode}, 사용 cycle={cycle_label} ({representative_cycle_count}/{available_cycle_count})",
        },
        {
            "step": 3,
            "stage": "harmonic_inverse",
            "summary": f"used freq {used_freq_hz:.3f} Hz, estimated lag {estimated_output_lag_seconds:.5f} s",
            "detail": "steady-state representative cycle의 harmonic transfer를 역보정",
        },
        {
            "step": 4,
            "stage": "lcr_prior",
            "summary": "applied" if used_lcr_prior else "skipped",
            "detail": f"LCR blend weight={float(lcr_blend_weight):.2f}",
        },
        {
            "step": 5,
            "stage": "startup_handling",
            "summary": (
                "finite zero-start envelope"
                if finite_cycle_mode
                else (
                    f"startup correction x{startup_correction_factor:.3f} over {startup_transition_cycles:.2f} cycle"
                    if startup_correction_applied and np.isfinite(startup_correction_factor)
                    else "steady-state recentering"
                )
            ),
            "detail": (
                "preserve_start_voltage=True, finite active window 정렬"
                if preserve_start_voltage
                else (
                    "대표 steady-state cycle을 recenter하고, startup correction preview를 별도 생성"
                    if startup_correction_applied
                    else "preserve_start_voltage=False, periodic command를 active-region mean 기준으로 recenter"
                )
            ),
        },
        {
            "step": 6,
            "stage": "hardware_limits",
            "summary": f"DAQ {limited_voltage_pp:.3f}/{max_daq_voltage_pp:.3f} Vpp",
            "detail": "within limits" if within_hardware_limits else "hardware limit 초과로 clipping 또는 gain 제한 발생",
        },
        {
            "step": 7,
            "stage": "modeled_response",
            "summary": "nearest support scaled preview",
            "detail": f"현재 expected/current/field는 target 덮어쓰기가 아니라 모델 예측값, startup flag={startup_behavior}",
        },
    ]
    return pd.DataFrame(rows)


def _sync_modeled_alias_columns(command_profile: pd.DataFrame) -> pd.DataFrame:
    if "expected_current_a" in command_profile.columns:
        command_profile["modeled_current_a"] = command_profile["expected_current_a"]
    if "expected_field_mT" in command_profile.columns:
        command_profile["modeled_field_mT"] = command_profile["expected_field_mT"]
    if "expected_output" in command_profile.columns:
        command_profile["modeled_output"] = command_profile["expected_output"]
    return command_profile


def _register_profile_phase_to_command_zero_cross(
    profile: pd.DataFrame,
    voltage_column: str = "command_voltage_v",
    rotate_columns: list[str] | None = None,
) -> pd.DataFrame:
    if profile.empty or "cycle_progress" not in profile.columns or voltage_column not in profile.columns:
        return profile

    phase = pd.to_numeric(profile["cycle_progress"], errors="coerce").to_numpy(dtype=float)
    voltage = pd.to_numeric(profile[voltage_column], errors="coerce").to_numpy(dtype=float)
    if len(phase) < 4 or len(voltage) < 4:
        return profile

    crossing_phase = _find_positive_zero_cross_phase(phase, voltage)
    if crossing_phase is None or not np.isfinite(crossing_phase):
        return profile
    if abs(crossing_phase) < 1e-6 or abs(crossing_phase - 1.0) < 1e-6:
        return profile

    registered = profile.copy()
    base_phase = phase[:-1] if len(phase) > 2 and np.isclose(phase[-1], 1.0, atol=1e-9) else phase
    base_phase = np.asarray(base_phase, dtype=float)
    if len(base_phase) < 3:
        return profile
    target_phase = np.mod(phase + float(crossing_phase), 1.0)
    candidate_columns = rotate_columns or list(registered.columns)
    for column in candidate_columns:
        if column not in registered.columns or column in {"cycle_progress", "time_s"}:
            continue
        if pd.api.types.is_bool_dtype(registered[column]):
            continue
        values = pd.to_numeric(registered[column], errors="coerce").to_numpy(dtype=float).copy()
        base_values = values[:-1] if len(values) == len(phase) and len(base_phase) == len(phase) - 1 else values
        if len(base_values) != len(base_phase) or not np.isfinite(base_values).any():
            continue
        interpolated = _periodic_interp(base_phase, base_values, target_phase)
        if len(interpolated) == len(values):
            if len(phase) > 1 and np.isclose(phase[-1], 1.0, atol=1e-9):
                interpolated[-1] = interpolated[0]
            registered[column] = interpolated

    if "time_s" in registered.columns:
        duration = float(pd.to_numeric(registered["time_s"], errors="coerce").max())
        if np.isfinite(duration) and duration > 0:
            registered["time_s"] = phase * duration
    if voltage_column in registered.columns:
        voltage_values = pd.to_numeric(registered[voltage_column], errors="coerce").to_numpy(dtype=float).copy()
        if len(voltage_values):
            voltage_values[0] = 0.0
            if len(voltage_values) > 1:
                voltage_values[-1] = voltage_values[0]
            registered[voltage_column] = voltage_values
    return registered


def _periodic_interp(base_phase: np.ndarray, base_values: np.ndarray, target_phase: np.ndarray) -> np.ndarray:
    ordered = np.argsort(base_phase)
    x = np.asarray(base_phase[ordered], dtype=float)
    y = np.asarray(base_values[ordered], dtype=float)
    extended_x = np.concatenate([x - 1.0, x, x + 1.0])
    extended_y = np.concatenate([y, y, y])
    return np.interp(np.asarray(target_phase, dtype=float), extended_x, extended_y)


def _find_positive_zero_cross_phase(phase: np.ndarray, signal: np.ndarray) -> float | None:
    valid = np.isfinite(phase) & np.isfinite(signal)
    phase = np.asarray(phase[valid], dtype=float)
    signal = np.asarray(signal[valid], dtype=float)
    if len(phase) < 3:
        return None

    if np.isclose(phase[-1], 1.0, atol=1e-9) and np.isclose(phase[0], 0.0, atol=1e-9):
        phase_work = phase[:-1]
        signal_work = signal[:-1]
    else:
        phase_work = phase
        signal_work = signal
    if len(phase_work) < 3:
        return None

    candidates: list[tuple[float, float]] = []
    for index in range(len(signal_work)):
        left_phase = phase_work[index]
        right_phase = phase_work[(index + 1) % len(signal_work)]
        left_value = signal_work[index]
        right_value = signal_work[(index + 1) % len(signal_work)]
        if index == len(signal_work) - 1:
            right_phase = 1.0
        if not np.isfinite(left_value) or not np.isfinite(right_value):
            continue
        if left_value <= 0 < right_value:
            delta = right_value - left_value
            if abs(delta) < 1e-12:
                continue
            ratio = abs(left_value) / abs(delta)
            crossing_phase = left_phase + ratio * (right_phase - left_phase)
            score = abs(left_value) + abs(right_value)
            candidates.append((float(crossing_phase % 1.0), float(score)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[1], item[0]))
    return float(candidates[0][0])


def _attach_expected_response_columns(
    command_profile: pd.DataFrame,
    support_profile_preview: pd.DataFrame,
    target_output_type: str,
    finite_cycle_mode: bool,
) -> pd.DataFrame:
    if command_profile.empty or support_profile_preview.empty or "time_s" not in command_profile.columns:
        return command_profile

    used_target_column = (
        "aligned_used_target_output"
        if finite_cycle_mode and "aligned_used_target_output" in command_profile.columns
        else "used_target_output"
    )
    if used_target_column not in command_profile.columns:
        used_target_column = (
            "aligned_target_output"
            if finite_cycle_mode and "aligned_target_output" in command_profile.columns
            else "target_output"
        )
    used_target_pp = _signal_peak_to_peak(command_profile, used_target_column)
    preview_output_column = "measured_current_a" if target_output_type == "current" else "measured_field_mT"
    preview_output_pp = _signal_peak_to_peak(support_profile_preview, preview_output_column)
    scale_ratio = (
        float(used_target_pp / preview_output_pp)
        if np.isfinite(used_target_pp) and np.isfinite(preview_output_pp) and preview_output_pp > 0
        else 1.0
    )

    preview = support_profile_preview.copy()
    if "time_s" not in preview.columns:
        return command_profile
    preview["time_s"] = pd.to_numeric(preview["time_s"], errors="coerce")
    preview = preview.dropna(subset=["time_s"]).sort_values("time_s")
    if preview.empty:
        return command_profile

    target_times = pd.to_numeric(command_profile["time_s"], errors="coerce").to_numpy(dtype=float)
    preview_time = preview["time_s"].to_numpy(dtype=float)

    if "measured_current_a" in preview.columns:
        current_values = pd.to_numeric(preview["measured_current_a"], errors="coerce").to_numpy(dtype=float)
        command_profile["support_scaled_current_a"] = np.interp(target_times, preview_time, current_values) * scale_ratio
    if "measured_field_mT" in preview.columns:
        field_values = pd.to_numeric(preview["measured_field_mT"], errors="coerce").to_numpy(dtype=float)
        command_profile["support_scaled_field_mT"] = np.interp(target_times, preview_time, field_values) * scale_ratio

    if target_output_type == "current":
        if "support_scaled_current_a" in command_profile.columns:
            command_profile["expected_current_a"] = command_profile["support_scaled_current_a"]
            command_profile["expected_output"] = command_profile["expected_current_a"]
    else:
        if "support_scaled_field_mT" in command_profile.columns:
            command_profile["expected_field_mT"] = command_profile["support_scaled_field_mT"]
            command_profile["expected_output"] = command_profile["expected_field_mT"]
    return _sync_modeled_alias_columns(command_profile)


def _build_field_only_support_profile_preview(
    support_profiles: list[dict[str, Any]],
    target_freq_hz: float,
    points_per_cycle: int,
    output_freq_hz: float,
    output_signal_column: str,
    support_weight_table: pd.DataFrame,
    normalized_target_pp: float = FIELD_ROUTE_NORMALIZED_TARGET_PP,
) -> pd.DataFrame:
    working_weights = support_weight_table.copy() if not support_weight_table.empty else _build_frequency_support_weight_table(
        support_profiles=support_profiles,
        target_freq_hz=target_freq_hz,
    )
    if working_weights.empty:
        return pd.DataFrame()

    nearest_index = int(working_weights.iloc[0]["index"])
    reference_profile = support_profiles[nearest_index]["profile"]
    reference_field = _normalize_waveform_peak_to_peak(
        pd.to_numeric(reference_profile.get(output_signal_column), errors="coerce").to_numpy(dtype=float),
        normalized_target_pp,
    )
    phase_grid = np.linspace(0.0, 1.0, max(int(points_per_cycle), 128))
    profile = pd.DataFrame(
        {
            "cycle_progress": phase_grid,
            "time_s": phase_grid * (1.0 / float(output_freq_hz) if float(output_freq_hz) > 0 else 1.0),
        }
    )

    for column in ("command_voltage_v", "measured_current_a"):
        blended = np.zeros_like(phase_grid, dtype=float)
        used_weight = 0.0
        for row in working_weights.to_dict(orient="records"):
            support = support_profiles[int(row["index"])]
            support_profile = support["profile"]
            if support_profile.empty or "cycle_progress" not in support_profile.columns or column not in support_profile.columns:
                continue
            support_phase = pd.to_numeric(support_profile["cycle_progress"], errors="coerce").to_numpy(dtype=float)
            support_values = pd.to_numeric(support_profile[column], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(support_phase) & np.isfinite(support_values)
            if valid.sum() < 4:
                continue
            order = np.argsort(support_phase[valid])
            blended += float(row["weight"]) * np.interp(phase_grid, support_phase[valid][order], support_values[valid][order])
            used_weight += float(row["weight"])
        if used_weight > 0:
            profile[column] = blended / used_weight

    blended_field = np.zeros_like(phase_grid, dtype=float)
    used_weight = 0.0
    for row in working_weights.to_dict(orient="records"):
        support_profile = support_profiles[int(row["index"])]["profile"]
        if support_profile.empty or "cycle_progress" not in support_profile.columns or output_signal_column not in support_profile.columns:
            continue
        support_phase = pd.to_numeric(support_profile["cycle_progress"], errors="coerce").to_numpy(dtype=float)
        support_field = pd.to_numeric(support_profile[output_signal_column], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(support_phase) & np.isfinite(support_field)
        if valid.sum() < 4:
            continue
        normalized_field = _normalize_waveform_peak_to_peak(support_field, normalized_target_pp)
        normalized_field = _align_waveform_sign(normalized_field, reference_field)
        order = np.argsort(support_phase[valid])
        blended_field += float(row["weight"]) * np.interp(
            phase_grid,
            support_phase[valid][order],
            normalized_field[valid][order],
        )
        used_weight += float(row["weight"])
    if used_weight > 0:
        profile["measured_field_mT"] = _normalize_waveform_peak_to_peak(blended_field / used_weight, normalized_target_pp)

    if "command_voltage_v" in profile.columns:
        profile = _register_profile_phase_to_command_zero_cross(profile, voltage_column="command_voltage_v")
    return profile


def _build_weighted_support_profile_preview(
    support_profiles: list[dict[str, Any]],
    target_freq_hz: float,
    target_output_pp: float,
    output_metric: str,
    points_per_cycle: int,
    output_freq_hz: float,
    max_support_count: int = 4,
) -> pd.DataFrame:
    if not support_profiles:
        return pd.DataFrame()

    support_rows = pd.DataFrame(
        [
            {
                "index": index,
                "freq_hz": float(support["meta"].get("freq_hz", np.nan)),
                "output_pp": float(support["meta"].get(output_metric, np.nan)),
            }
            for index, support in enumerate(support_profiles)
        ]
    )
    if support_rows.empty:
        return pd.DataFrame()

    freq_range = max(
        float(support_rows["freq_hz"].max()) - float(support_rows["freq_hz"].min()),
        1e-9,
    )
    output_range = max(
        float(support_rows["output_pp"].max()) - float(support_rows["output_pp"].min()),
        1e-9,
    )
    support_rows["freq_distance_norm"] = (
        (support_rows["freq_hz"] - float(target_freq_hz)).abs() / freq_range
        if support_rows["freq_hz"].notna().any()
        else 1.0
    )
    support_rows["output_distance_norm"] = (
        (support_rows["output_pp"] - float(target_output_pp)).abs() / output_range
        if support_rows["output_pp"].notna().any()
        else 1.0
    )
    support_rows["combined_distance"] = np.sqrt(
        np.square(support_rows["freq_distance_norm"]) + np.square(support_rows["output_distance_norm"])
    )
    support_rows = support_rows.sort_values(["combined_distance", "freq_distance_norm", "output_distance_norm"])
    selected_rows = support_rows.head(max(int(max_support_count), 1)).copy()
    raw_weights = 1.0 / np.square(1.0 + selected_rows["combined_distance"].to_numpy(dtype=float))
    if not np.isfinite(raw_weights).all() or raw_weights.sum() <= 0:
        raw_weights = np.ones(len(selected_rows), dtype=float)
    weights = raw_weights / raw_weights.sum()

    phase_grid = np.linspace(0.0, 1.0, max(int(points_per_cycle), 128))
    profile = pd.DataFrame(
        {
            "cycle_progress": phase_grid,
            "time_s": phase_grid * (1.0 / float(output_freq_hz) if float(output_freq_hz) > 0 else 1.0),
        }
    )
    for column in ("command_voltage_v", "measured_current_a", "measured_field_mT"):
        blended = np.zeros_like(phase_grid, dtype=float)
        used_weight = 0.0
        for weight, (_, row) in zip(weights, selected_rows.iterrows(), strict=False):
            support = support_profiles[int(row["index"])]
            support_profile = support["profile"]
            if support_profile.empty or "cycle_progress" not in support_profile.columns or column not in support_profile.columns:
                continue
            support_phase = pd.to_numeric(support_profile["cycle_progress"], errors="coerce").to_numpy(dtype=float)
            support_values = pd.to_numeric(support_profile[column], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(support_phase) & np.isfinite(support_values)
            if valid.sum() < 4:
                continue
            ordered = np.argsort(support_phase[valid])
            interpolated = np.interp(phase_grid, support_phase[valid][ordered], support_values[valid][ordered])
            blended += float(weight) * interpolated
            used_weight += float(weight)
        if used_weight > 0:
            profile[column] = blended / used_weight

    if "command_voltage_v" in profile.columns:
        profile = _register_profile_phase_to_command_zero_cross(profile, voltage_column="command_voltage_v")
    return profile


def _compute_waveform_shape_similarity(
    target_values: np.ndarray,
    predicted_values: np.ndarray,
) -> tuple[float, float]:
    target = np.asarray(target_values, dtype=float)
    predicted = np.asarray(predicted_values, dtype=float)
    valid = np.isfinite(target) & np.isfinite(predicted)
    if valid.sum() < 4:
        return float("nan"), float("nan")

    target_norm = _normalize_waveform_peak_to_peak(target[valid], 2.0)
    predicted_norm = _normalize_waveform_peak_to_peak(predicted[valid], 2.0)
    target_std = float(np.nanstd(target_norm))
    predicted_std = float(np.nanstd(predicted_norm))
    if target_std <= 1e-12 or predicted_std <= 1e-12:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(target_norm, predicted_norm)[0, 1])
    nrmse = float(np.sqrt(np.nanmean(np.square(predicted_norm - target_norm))) / 2.0)
    return correlation, nrmse


def _attach_field_prediction_metrics(
    command_profile: pd.DataFrame,
    support_weight_table: pd.DataFrame,
    finite_cycle_mode: bool,
) -> pd.DataFrame:
    if command_profile.empty:
        return command_profile

    predicted_column = next(
        (
            column
            for column in ("expected_field_mT", "support_scaled_field_mT", "expected_output")
            if column in command_profile.columns
        ),
        None,
    )
    if predicted_column is not None:
        command_profile["predicted_field_mT"] = pd.to_numeric(command_profile[predicted_column], errors="coerce")

    target_column = next(
        (
            column
            for column in (
                "aligned_used_target_field_mT",
                "used_target_field_mT",
                "aligned_target_field_mT",
                "target_field_mT",
            )
            if column in command_profile.columns
        ),
        None,
    )
    if target_column is None and "target_field_mT" in command_profile.columns:
        target_column = "target_field_mT"

    if predicted_column is not None and target_column is not None:
        target_values = pd.to_numeric(command_profile[target_column], errors="coerce").to_numpy(dtype=float)
        predicted_values = pd.to_numeric(command_profile["predicted_field_mT"], errors="coerce").to_numpy(dtype=float)
        if finite_cycle_mode and "is_active_target" in command_profile.columns:
            active_mask = command_profile["is_active_target"].astype(bool).to_numpy(dtype=bool)
            if active_mask.any():
                target_values = target_values[active_mask]
                predicted_values = predicted_values[active_mask]
        field_shape_corr, field_shape_nrmse = _compute_waveform_shape_similarity(
            target_values=target_values,
            predicted_values=predicted_values,
        )
    else:
        field_shape_corr = float("nan")
        field_shape_nrmse = float("nan")

    support_freq_count = int(support_weight_table["freq_hz"].dropna().nunique()) if not support_weight_table.empty else 0
    support_id_rows = (
        support_weight_table.sort_values(["freq_hz", "test_id"], kind="stable")
        if not support_weight_table.empty and {"freq_hz", "test_id"}.issubset(support_weight_table.columns)
        else support_weight_table
    )
    support_test_ids = "|".join(
        str(value)
        for value in support_id_rows["test_id"].dropna().astype(str).tolist()
    ) if not support_id_rows.empty and "test_id" in support_id_rows.columns else ""
    command_profile["field_shape_corr"] = field_shape_corr
    command_profile["field_shape_nrmse"] = field_shape_nrmse
    command_profile["field_support_freq_count"] = support_freq_count
    command_profile["field_support_test_ids"] = support_test_ids
    if "terminal_trim_applied" not in command_profile.columns:
        command_profile["terminal_trim_applied"] = False
    if "terminal_trim_gain" not in command_profile.columns:
        command_profile["terminal_trim_gain"] = 1.0
    if "terminal_trim_bias_v" not in command_profile.columns:
        command_profile["terminal_trim_bias_v"] = 0.0
    if "predicted_terminal_peak_error_mT" not in command_profile.columns:
        command_profile["predicted_terminal_peak_error_mT"] = float("nan")
    if "terminal_target_slope_sign" not in command_profile.columns:
        command_profile["terminal_target_slope_sign"] = float("nan")
    if "terminal_predicted_slope_sign_before" not in command_profile.columns:
        command_profile["terminal_predicted_slope_sign_before"] = float("nan")
    if "terminal_predicted_slope_sign_after" not in command_profile.columns:
        command_profile["terminal_predicted_slope_sign_after"] = float("nan")
    if "terminal_direction_match_after" not in command_profile.columns:
        command_profile["terminal_direction_match_after"] = False
    if "terminal_trim_window_fraction" not in command_profile.columns:
        command_profile["terminal_trim_window_fraction"] = float("nan")
    return _sync_modeled_alias_columns(command_profile)


def _phase_register_command_profile(
    command_profile: pd.DataFrame,
    voltage_column: str = "limited_voltage_v",
) -> pd.DataFrame:
    rotate_columns = [
        column
        for column in (
            "recommended_voltage_v",
            "limited_voltage_v",
            "expected_current_a",
            "expected_field_mT",
            "expected_output",
            "modeled_current_a",
            "modeled_field_mT",
            "modeled_output",
            "support_scaled_current_a",
            "support_scaled_field_mT",
        )
        if column in command_profile.columns
    ]
    if not rotate_columns:
        return command_profile
    return _register_profile_phase_to_command_zero_cross(
        command_profile,
        voltage_column=voltage_column,
        rotate_columns=rotate_columns,
    )


def _apply_forward_harmonic_prediction(
    command_profile: pd.DataFrame,
    transfer_model: np.ndarray,
    target_output_type: str,
) -> pd.DataFrame:
    if command_profile.empty or "limited_voltage_v" not in command_profile.columns:
        return command_profile
    limited_voltage = pd.to_numeric(command_profile["limited_voltage_v"], errors="coerce").to_numpy(dtype=float)
    if len(limited_voltage) == 0 or not np.isfinite(limited_voltage).any():
        return command_profile
    if len(transfer_model) == 0:
        return command_profile

    voltage_centered = limited_voltage - float(np.nanmean(limited_voltage))
    voltage_fft = np.fft.rfft(voltage_centered)
    usable_harmonics = min(len(voltage_fft), len(transfer_model))
    if usable_harmonics <= 1:
        return command_profile

    predicted_fft = np.zeros_like(voltage_fft, dtype=np.complex128)
    predicted_fft[1:usable_harmonics] = voltage_fft[1:usable_harmonics] * transfer_model[1:usable_harmonics]
    predicted_output = np.fft.irfft(predicted_fft, n=len(limited_voltage))

    if target_output_type == "current":
        command_profile["expected_current_a"] = predicted_output
    else:
        command_profile["expected_field_mT"] = predicted_output
    if target_output_type == "current" and "expected_current_a" in command_profile.columns:
        command_profile["expected_output"] = command_profile["expected_current_a"]
    elif target_output_type == "field" and "expected_field_mT" in command_profile.columns:
        command_profile["expected_output"] = command_profile["expected_field_mT"]
    return _sync_modeled_alias_columns(command_profile)


def _build_startup_corrected_preview(
    command_profile: pd.DataFrame,
    startup_diagnostics: dict[str, Any],
    target_output_type: str,
    freq_hz: float,
    max_daq_voltage_pp: float,
    amp_gain_at_100_pct: float,
    support_amp_gain_pct: float,
    amp_gain_limit_pct: float,
    amp_max_output_pk_v: float,
    transition_cycles: float = 1.5,
    correction_strength: float = 1.0,
    preview_cycle_count: int = 3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if command_profile.empty or "time_s" not in command_profile.columns or "recommended_voltage_v" not in command_profile.columns:
        return pd.DataFrame(), {"startup_correction_applied": False}

    if target_output_type == "field":
        observed_ratio = _first_numeric(startup_diagnostics.get("first_cycle_field_ratio_vs_steady"))
    else:
        observed_ratio = _first_numeric(startup_diagnostics.get("first_cycle_current_ratio_vs_steady"))
    if observed_ratio is None or not np.isfinite(observed_ratio) or observed_ratio <= 0:
        return pd.DataFrame(), {"startup_correction_applied": False}

    transition_cycles = max(float(transition_cycles), 0.25)
    preview_cycle_count = max(int(preview_cycle_count), 2)
    correction_strength = float(np.clip(correction_strength, 0.0, 1.5))
    ideal_correction_factor = 1.0 / float(observed_ratio)
    startup_correction_factor = float(
        np.clip(1.0 + (ideal_correction_factor - 1.0) * correction_strength, 0.4, 1.8)
    )
    if abs(startup_correction_factor - 1.0) < 1e-3:
        return pd.DataFrame(), {
            "startup_correction_applied": False,
            "startup_correction_factor": startup_correction_factor,
            "startup_observed_output_ratio": float(observed_ratio),
        }

    base = command_profile.copy().sort_values("time_s").reset_index(drop=True)
    time_values = pd.to_numeric(base["time_s"], errors="coerce").to_numpy(dtype=float)
    if len(time_values) < 2:
        return pd.DataFrame(), {"startup_correction_applied": False}
    period_s = float(time_values[-1] - time_values[0])
    if not np.isfinite(period_s) or period_s <= 0:
        period_s = 1.0 / float(freq_hz) if float(freq_hz) > 0 else 1.0
    phase = (time_values - float(time_values[0])) / period_s
    phase = np.clip(phase, 0.0, 1.0)

    segments: list[pd.DataFrame] = []
    for cycle_index in range(preview_cycle_count):
        segment = base.copy()
        if cycle_index < preview_cycle_count - 1 and len(segment) > 1:
            segment = segment.iloc[:-1].copy()
            phase_segment = phase[:-1]
        else:
            phase_segment = phase
        segment["time_s"] = pd.to_numeric(segment["time_s"], errors="coerce") + cycle_index * period_s
        segment["startup_preview_cycle_index"] = cycle_index + 1
        segment["cycle_progress_total"] = phase_segment + cycle_index
        segments.append(segment)
    preview = pd.concat(segments, ignore_index=True)

    cycle_progress_total = pd.to_numeric(preview["cycle_progress_total"], errors="coerce").to_numpy(dtype=float)
    transition_weight = np.clip(1.0 - cycle_progress_total / transition_cycles, 0.0, 1.0)
    command_envelope = 1.0 + (startup_correction_factor - 1.0) * transition_weight
    observed_envelope = 1.0 + (float(observed_ratio) - 1.0) * transition_weight
    predicted_output_envelope = command_envelope * observed_envelope

    preview["recommended_voltage_v"] = pd.to_numeric(preview["recommended_voltage_v"], errors="coerce").to_numpy(dtype=float) * command_envelope
    if "expected_current_a" in preview.columns:
        preview["expected_current_a"] = pd.to_numeric(preview["expected_current_a"], errors="coerce").to_numpy(dtype=float) * predicted_output_envelope
    if "expected_field_mT" in preview.columns:
        preview["expected_field_mT"] = pd.to_numeric(preview["expected_field_mT"], errors="coerce").to_numpy(dtype=float) * predicted_output_envelope
    if target_output_type == "current" and "expected_current_a" in preview.columns:
        preview["expected_output"] = preview["expected_current_a"]
    elif target_output_type == "field" and "expected_field_mT" in preview.columns:
        preview["expected_output"] = preview["expected_field_mT"]

    preview["startup_command_envelope"] = command_envelope
    preview["startup_output_envelope"] = predicted_output_envelope
    preview["startup_correction_factor"] = startup_correction_factor
    preview["startup_observed_output_ratio"] = float(observed_ratio)
    preview["startup_transition_cycles"] = transition_cycles
    preview["startup_correction_applied"] = True
    preview["is_active_target"] = True
    preview = apply_command_hardware_model(
        command_waveform=preview,
        max_daq_voltage_pp=float(max_daq_voltage_pp),
        amp_gain_at_100_pct=float(amp_gain_at_100_pct),
        support_amp_gain_pct=float(support_amp_gain_pct),
        amp_gain_limit_pct=float(amp_gain_limit_pct),
        amp_max_output_pk_v=float(amp_max_output_pk_v),
        preserve_start_voltage=True,
    )
    preview = _sync_modeled_alias_columns(preview)
    return preview, {
        "startup_correction_applied": True,
        "startup_correction_factor": startup_correction_factor,
        "startup_observed_output_ratio": float(observed_ratio),
    }


def _build_finite_support_startup_diagnostics(
    support_entry: dict[str, Any],
    *,
    field_channel: str,
    current_channel: str,
) -> dict[str, Any]:
    frame = _prepare_finite_time_frame(support_entry.get("frame"))
    if frame.empty or "time_s" not in frame.columns:
        return {}
    freq_hz = _first_numeric(support_entry.get("freq_hz"))
    cycle_count = _first_numeric(support_entry.get("approx_cycle_span"))
    if not np.isfinite(freq_hz) or freq_hz <= 0:
        return {}
    period_s = 1.0 / float(freq_hz)
    active_end_s = float(cycle_count) * period_s if np.isfinite(cycle_count) else float(frame["time_s"].max())
    time_values = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
    active = np.isfinite(time_values) & (time_values <= active_end_s + 1e-12)
    first_cycle = active & (time_values <= period_s + 1e-12)
    steady = active & (time_values > period_s + 1e-12)
    if steady.sum() < 3:
        steady = active & (time_values > active_end_s * 0.5)
    diagnostics: dict[str, Any] = {
        "cycle_count": float(cycle_count) if np.isfinite(cycle_count) else None,
        "behavior_flag": "unknown",
    }
    field_offset_delta = float("nan")
    current_offset_delta = float("nan")
    field_pp_for_residual = float("nan")
    current_pp_for_residual = float("nan")
    for signal_column, prefix in ((field_channel, "field"), (current_channel, "current")):
        if signal_column not in frame.columns:
            continue
        values = pd.to_numeric(frame[signal_column], errors="coerce").to_numpy(dtype=float)
        first_values = values[first_cycle & np.isfinite(values)]
        steady_values = values[steady & np.isfinite(values)]
        if first_values.size < 3 or steady_values.size < 3:
            continue
        first_mean = float(np.nanmean(first_values))
        steady_mean = float(np.nanmean(steady_values))
        first_pp = float(np.nanmax(first_values) - np.nanmin(first_values))
        steady_pp = float(np.nanmax(steady_values) - np.nanmin(steady_values))
        ratio = first_pp / steady_pp if abs(steady_pp) > 1e-12 else float("nan")
        if prefix == "field":
            field_offset_delta = float(first_mean - steady_mean)
            field_pp_for_residual = steady_pp
            diagnostics["first_cycle_field_mean_mT"] = first_mean
            diagnostics["steady_field_mean_mT"] = steady_mean
            diagnostics["first_cycle_field_offset_delta_mT"] = field_offset_delta
            diagnostics["first_cycle_field_ratio_vs_steady"] = ratio
            diagnostics["first_cycle_field_pp_mT"] = first_pp
            diagnostics["steady_field_pp_mT_mean"] = steady_pp
        else:
            current_offset_delta = float(first_mean - steady_mean)
            current_pp_for_residual = steady_pp
            diagnostics["first_cycle_current_mean_a"] = first_mean
            diagnostics["steady_current_mean_a"] = steady_mean
            diagnostics["first_cycle_current_offset_delta_a"] = current_offset_delta
            diagnostics["first_cycle_current_ratio_vs_steady"] = ratio
    diagnostics["first_cycle_field_unexplained_offset_delta_mT"] = _field_offset_residual_from_current(
        field_offset_delta=field_offset_delta,
        current_offset_delta=current_offset_delta,
        field_pp=field_pp_for_residual,
        current_pp=current_pp_for_residual,
    )
    field_delta = _first_numeric(diagnostics.get("first_cycle_field_unexplained_offset_delta_mT"))
    field_ratio = _first_numeric(diagnostics.get("first_cycle_field_ratio_vs_steady"))
    if np.isfinite(field_delta) and abs(field_delta) > 1.0:
        diagnostics["behavior_flag"] = "first_cycle_offset"
    elif np.isfinite(field_ratio) and field_ratio > 1.05:
        diagnostics["behavior_flag"] = "first_cycle_overshoot"
    elif np.isfinite(field_ratio) and field_ratio < 0.95:
        diagnostics["behavior_flag"] = "first_cycle_undershoot"
    else:
        diagnostics["behavior_flag"] = "steady_like"
    return diagnostics


def _apply_startup_transient_prediction(
    *,
    command_profile: pd.DataFrame,
    startup_diagnostics: dict[str, Any],
    target_output_type: str,
    freq_hz: float,
    transition_cycles: float,
    correction_strength: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if command_profile.empty or "time_s" not in command_profile.columns:
        return command_profile, {"startup_transient_applied": False}
    if target_output_type != "field":
        return command_profile, {"startup_transient_applied": False}
    offset_delta = _first_numeric(startup_diagnostics.get("first_cycle_field_unexplained_offset_delta_mT"))
    if offset_delta is None or not np.isfinite(offset_delta):
        offset_delta = _first_numeric(startup_diagnostics.get("first_cycle_field_offset_delta_mT"))
    observed_ratio = _first_numeric(startup_diagnostics.get("first_cycle_field_ratio_vs_steady"))
    if offset_delta is None or not np.isfinite(offset_delta):
        offset_delta = 0.0
    if observed_ratio is None or not np.isfinite(observed_ratio) or observed_ratio <= 0:
        observed_ratio = 1.0
    if abs(float(offset_delta)) < 1.0:
        observed_ratio = 1.0
    if abs(offset_delta) < 1e-6 and abs(observed_ratio - 1.0) < 0.03:
        return command_profile, {
            "startup_transient_applied": False,
            "startup_initial_field_offset_mT": startup_diagnostics.get("first_cycle_field_mean_mT"),
            "startup_steady_field_offset_mT": startup_diagnostics.get("steady_field_mean_mT"),
            "startup_field_offset_delta_mT": offset_delta,
            "startup_transient_reason": "steady_like",
        }

    adjusted = command_profile.copy()
    time_values = pd.to_numeric(adjusted["time_s"], errors="coerce").to_numpy(dtype=float)
    finite_time = time_values[np.isfinite(time_values)]
    if finite_time.size < 3 or not np.isfinite(freq_hz) or float(freq_hz) <= 0:
        return command_profile, {"startup_transient_applied": False}
    start_s = float(np.nanmin(finite_time))
    transition_cycles = max(float(transition_cycles), 0.25)
    strength = float(np.clip(correction_strength, 0.0, 1.0))
    cycle_progress_total = (time_values - start_s) * float(freq_hz)
    transient_weight = np.clip(1.0 - cycle_progress_total / transition_cycles, 0.0, 1.0) * strength
    ratio_envelope = 1.0 + (float(observed_ratio) - 1.0) * transient_weight
    offset_envelope = float(offset_delta) * transient_weight

    for column in (
        "expected_field_mT",
        "predicted_field_mT",
        "support_scaled_field_mT",
        "support_reference_output_mT",
        "modeled_field_mT",
    ):
        if column not in adjusted.columns:
            continue
        values = pd.to_numeric(adjusted[column], errors="coerce").to_numpy(dtype=float).copy()
        valid = np.isfinite(values) & np.isfinite(transient_weight)
        values[valid] = values[valid] * ratio_envelope[valid] + offset_envelope[valid]
        adjusted[column] = values
    if "expected_output" in adjusted.columns:
        values = pd.to_numeric(adjusted["expected_output"], errors="coerce").to_numpy(dtype=float).copy()
        valid = np.isfinite(values) & np.isfinite(transient_weight)
        values[valid] = values[valid] * ratio_envelope[valid] + offset_envelope[valid]
        adjusted["expected_output"] = values
    if "modeled_output" in adjusted.columns:
        values = pd.to_numeric(adjusted["modeled_output"], errors="coerce").to_numpy(dtype=float).copy()
        valid = np.isfinite(values) & np.isfinite(transient_weight)
        values[valid] = values[valid] * ratio_envelope[valid] + offset_envelope[valid]
        adjusted["modeled_output"] = values
    adjusted["startup_transient_applied"] = True
    adjusted["startup_initial_field_offset_mT"] = startup_diagnostics.get("first_cycle_field_mean_mT")
    adjusted["startup_steady_field_offset_mT"] = startup_diagnostics.get("steady_field_mean_mT")
    adjusted["startup_field_offset_delta_mT"] = float(offset_delta)
    adjusted["startup_transient_transition_cycles"] = float(transition_cycles)
    adjusted["startup_transient_reason"] = startup_diagnostics.get("behavior_flag", "startup_transient")
    return _sync_modeled_alias_columns(adjusted), {
        "startup_transient_applied": True,
        "startup_initial_field_offset_mT": startup_diagnostics.get("first_cycle_field_mean_mT"),
        "startup_steady_field_offset_mT": startup_diagnostics.get("steady_field_mean_mT"),
        "startup_field_offset_delta_mT": float(offset_delta),
        "startup_transient_transition_cycles": float(transition_cycles),
        "startup_transient_reason": startup_diagnostics.get("behavior_flag", "startup_transient"),
    }


def _finite_target_template(
    time_grid: np.ndarray,
    waveform_type: str,
    freq_hz: float,
    target_cycle_count: float,
    target_output_pp: float,
    force_rounded_triangle: bool = False,
) -> np.ndarray:
    if len(time_grid) == 0:
        return np.array([], dtype=float)
    period_s = 1.0 / float(freq_hz) if float(freq_hz) > 0 else 1.0
    active_end_s = float(target_cycle_count) * period_s
    values = np.zeros_like(time_grid, dtype=float)
    active_mask = (time_grid >= 0.0) & (time_grid <= active_end_s + 1e-12)
    if not active_mask.any():
        return values
    cycle_progress_total = np.clip(time_grid[active_mask] / period_s, 0.0, float(target_cycle_count))
    cycle_phase = np.mod(cycle_progress_total, 1.0)
    if force_rounded_triangle:
        normalized = _rounded_triangle_normalized(cycle_phase)
    elif waveform_type == "triangle":
        normalized = np.where(
            cycle_phase < 0.25,
            cycle_phase * 4.0,
            np.where(
                cycle_phase < 0.75,
                2.0 - cycle_phase * 4.0,
                cycle_phase * 4.0 - 4.0,
            ),
        )
    else:
        normalized = np.sin(2.0 * np.pi * cycle_phase)
    values[active_mask] = normalized * float(target_output_pp) / 2.0
    return values


def _expand_command_profile_to_finite_run(
    command_cycle_profile: pd.DataFrame,
    waveform_type: str,
    freq_hz: float,
    target_output_pp: float,
    target_cycle_count: float,
    preview_tail_cycles: float,
    output_context: dict[str, str],
    phase_lead_seconds: float,
    points_per_cycle: int,
    force_rounded_triangle_target: bool = False,
) -> pd.DataFrame:
    total_cycles = float(target_cycle_count + preview_tail_cycles)
    sample_count = max(int(np.ceil(total_cycles * points_per_cycle)), 2) + 1
    period_s = 1.0 / freq_hz if freq_hz > 0 else 1.0
    time_grid = np.linspace(0.0, total_cycles * period_s, sample_count)
    phase_total = time_grid / period_s
    lookahead_phase_total = phase_total + (phase_lead_seconds / period_s if period_s > 0 else 0.0)
    command_sampling_phase_total = np.clip(lookahead_phase_total, 0.0, float(target_cycle_count))

    target_norm = _sample_theoretical_output(
        waveform_type=waveform_type,
        phase_total=phase_total,
        active_cycle_count=target_cycle_count,
        force_rounded_triangle=force_rounded_triangle_target,
    )
    used_target_norm = _sample_theoretical_output(
        waveform_type=waveform_type,
        phase_total=lookahead_phase_total,
        active_cycle_count=target_cycle_count,
        force_rounded_triangle=force_rounded_triangle_target,
    )

    cycle_progress = np.mod(phase_total, 1.0)
    active_mask = phase_total <= target_cycle_count + 1e-12
    lookahead_mask = (lookahead_phase_total >= 0.0) & (lookahead_phase_total <= target_cycle_count + 1e-12)
    command_phase = np.mod(command_sampling_phase_total, 1.0)
    cycle_phase_grid = command_cycle_profile["cycle_progress"].to_numpy(dtype=float)
    cycle_voltage = pd.to_numeric(command_cycle_profile["recommended_voltage_v"], errors="coerce").to_numpy(dtype=float)
    command_voltage = np.interp(command_phase, cycle_phase_grid, cycle_voltage)
    command_voltage[~active_mask] = 0.0
    startup_duration_s = max(
        abs(float(phase_lead_seconds)),
        period_s / max(points_per_cycle / 4.0, 1.0),
    )
    command_voltage = _apply_zero_start_envelope(
        values=command_voltage,
        time_grid=time_grid,
        startup_duration_s=startup_duration_s,
    )
    aligned_target_output = _apply_zero_start_envelope(
        values=target_norm * target_output_pp / 2.0,
        time_grid=time_grid,
        startup_duration_s=startup_duration_s,
    )
    aligned_used_target_output = _apply_zero_start_envelope(
        values=used_target_norm * target_output_pp / 2.0,
        time_grid=time_grid,
        startup_duration_s=startup_duration_s,
    )

    expanded = pd.DataFrame(
        {
            "cycle_progress": cycle_progress,
            "cycle_progress_total": phase_total,
            "time_s": time_grid,
            "target_output": target_norm * target_output_pp / 2.0,
            "used_target_output": used_target_norm * target_output_pp / 2.0,
            "aligned_target_output": aligned_target_output,
            "aligned_used_target_output": aligned_used_target_output,
            "recommended_voltage_v": command_voltage,
            "is_active_target": active_mask,
            "is_lookahead_target": lookahead_mask,
            "command_sampling_phase_total": command_sampling_phase_total,
        }
    )

    if output_context["target_column"] == "target_current_a":
        expanded["target_current_a"] = expanded["target_output"]
        expanded["used_target_current_a"] = expanded["used_target_output"]
        expanded["aligned_target_current_a"] = expanded["aligned_target_output"]
        expanded["aligned_used_target_current_a"] = expanded["aligned_used_target_output"]
    else:
        expanded[output_context["target_column"]] = expanded["target_output"]
        expanded[output_context["used_target_column"]] = expanded["used_target_output"]
        aligned_target_column = output_context["target_column"].replace("target_", "aligned_target_")
        aligned_used_target_column = output_context["used_target_column"].replace("used_target_", "aligned_used_target_")
        expanded[aligned_target_column] = expanded["aligned_target_output"]
        expanded[aligned_used_target_column] = expanded["aligned_used_target_output"]
    return expanded


def _expand_measured_profile_preview(
    profile: pd.DataFrame,
    waveform_type: str,
    freq_hz: float,
    target_cycle_count: float,
    preview_tail_cycles: float,
    output_signal_column: str,
    phase_lead_seconds: float,
    points_per_cycle: int,
) -> pd.DataFrame:
    if profile.empty or output_signal_column not in profile.columns:
        return pd.DataFrame()

    total_cycles = float(target_cycle_count + preview_tail_cycles)
    sample_count = max(int(np.ceil(total_cycles * points_per_cycle)), 2) + 1
    period_s = 1.0 / freq_hz if freq_hz > 0 else 1.0
    time_grid = np.linspace(0.0, total_cycles * period_s, sample_count)
    phase_total = time_grid / period_s
    active_mask = phase_total <= target_cycle_count + 1e-12
    lookahead_phase_total = phase_total + (phase_lead_seconds / period_s if period_s > 0 else 0.0)
    command_sampling_phase_total = np.clip(lookahead_phase_total, 0.0, float(target_cycle_count))
    lookahead_mask = (lookahead_phase_total >= 0.0) & (lookahead_phase_total <= target_cycle_count + 1e-12)
    cycle_progress = np.mod(phase_total, 1.0)

    preview = pd.DataFrame(
        {
            "cycle_progress": cycle_progress,
            "cycle_progress_total": phase_total,
            "time_s": time_grid,
            "is_active_target": active_mask,
            "is_lookahead_target": lookahead_mask,
        }
    )
    for column in ("command_voltage_v", output_signal_column):
        if column not in profile.columns:
            continue
        values = pd.to_numeric(profile[column], errors="coerce").to_numpy(dtype=float)
        sampled = np.interp(
            np.mod(phase_total if column == output_signal_column else command_sampling_phase_total, 1.0),
            profile["cycle_progress"].to_numpy(dtype=float),
            values,
        )
        sampled[~active_mask] = 0.0
        if column == "command_voltage_v":
            sampled = _apply_zero_start_envelope(
                values=sampled,
                time_grid=time_grid,
                startup_duration_s=max(abs(float(phase_lead_seconds)), period_s / max(points_per_cycle / 4.0, 1.0)),
            )
        preview[column] = sampled
    preview["waveform_type"] = waveform_type
    preview["command_sampling_phase_total"] = command_sampling_phase_total
    return preview


def _apply_zero_start_envelope(
    values: np.ndarray,
    time_grid: np.ndarray,
    startup_duration_s: float,
) -> np.ndarray:
    output = np.asarray(values, dtype=float).copy()
    if len(output) == 0:
        return output
    min_duration = float(time_grid[1] - time_grid[0]) if len(time_grid) > 1 else 0.0
    duration = max(float(startup_duration_s), min_duration)
    envelope = np.ones_like(output)
    if duration > 0:
        mask = time_grid < duration
        if mask.any():
            normalized = np.clip(time_grid[mask] / duration, 0.0, 1.0)
            envelope[mask] = 0.5 - 0.5 * np.cos(np.pi * normalized)
    output *= envelope
    output[0] = 0.0
    return output


def _summarize_finite_command_stop_policy(
    command_profile: pd.DataFrame,
    *,
    target_active_end_s: float | None,
    phase_lead_seconds: float,
    finite_command_stop_policy: str,
    phase_lead_applied_to_sampling_only: bool,
) -> dict[str, Any]:
    if command_profile.empty or "time_s" not in command_profile.columns or "recommended_voltage_v" not in command_profile.columns:
        return {
            "finite_command_stop_policy": finite_command_stop_policy,
            "target_active_end_s": float(target_active_end_s) if target_active_end_s is not None and np.isfinite(target_active_end_s) else float("nan"),
            "command_nonzero_end_s": float("nan"),
            "command_early_stop_s": float("nan"),
            "command_extends_through_target_end": False,
            "post_target_command_tail_s": float("nan"),
            "phase_lead_seconds_applied": float(phase_lead_seconds),
            "phase_lead_applied_to_sampling_only": bool(phase_lead_applied_to_sampling_only),
            "early_command_cutoff_warning": False,
        }

    time_grid = pd.to_numeric(command_profile["time_s"], errors="coerce").to_numpy(dtype=float)
    command_voltage = pd.to_numeric(command_profile["recommended_voltage_v"], errors="coerce").to_numpy(dtype=float)
    nonzero_mask = np.isfinite(time_grid) & np.isfinite(command_voltage) & (np.abs(command_voltage) > 1e-6)
    command_nonzero_end_s = float(np.nanmax(time_grid[nonzero_mask])) if nonzero_mask.any() else float("nan")
    resolved_target_active_end_s = float(target_active_end_s) if target_active_end_s is not None and np.isfinite(target_active_end_s) else float("nan")
    tolerance = max(float(np.nanmedian(np.diff(time_grid))) if len(time_grid) > 1 else 0.0, 1e-6)
    command_extends_through_target_end = bool(
        np.isfinite(command_nonzero_end_s)
        and np.isfinite(resolved_target_active_end_s)
        and command_nonzero_end_s >= resolved_target_active_end_s - tolerance
    )
    command_early_stop_s = (
        max(resolved_target_active_end_s - command_nonzero_end_s, 0.0)
        if np.isfinite(command_nonzero_end_s) and np.isfinite(resolved_target_active_end_s)
        else float("nan")
    )
    post_target_command_tail_s = (
        max(command_nonzero_end_s - resolved_target_active_end_s, 0.0)
        if np.isfinite(command_nonzero_end_s) and np.isfinite(resolved_target_active_end_s)
        else float("nan")
    )
    return {
        "finite_command_stop_policy": finite_command_stop_policy,
        "target_active_end_s": resolved_target_active_end_s,
        "command_nonzero_end_s": command_nonzero_end_s,
        "command_early_stop_s": command_early_stop_s,
        "command_extends_through_target_end": command_extends_through_target_end,
        "post_target_command_tail_s": post_target_command_tail_s,
        "phase_lead_seconds_applied": float(phase_lead_seconds),
        "phase_lead_applied_to_sampling_only": bool(phase_lead_applied_to_sampling_only),
        "early_command_cutoff_warning": bool(np.isfinite(command_early_stop_s) and command_early_stop_s > tolerance),
    }


def _extend_finite_active_window_signals(
    command_profile: pd.DataFrame,
    *,
    active_end_s: float,
    command_columns: tuple[str, ...],
    predicted_columns: tuple[str, ...],
    support_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    extended = command_profile.copy()
    metadata = {
        "command_extension_applied": False,
        "command_extension_reason": None,
        "command_stop_policy": "extend_active_hold_to_target_end",
        "predicted_extension_applied": False,
        "support_extension_applied": False,
        "support_coverage_mode": "full_active_coverage",
        "partial_support_coverage": False,
        "support_observed_end_s": float(active_end_s),
        "support_observed_coverage_ratio": 1.0,
        "support_padding_gap_s": 0.0,
        "support_resampled_to_target_window": False,
        "hybrid_fill_applied": False,
        "hybrid_fill_start_s": float("nan"),
        "hybrid_fill_end_s": float("nan"),
        "finite_prediction_source": "empirical_observed",
        "predicted_cover_reason": "empirical_observed",
        "support_cover_reason": "empirical_observed",
    }
    if extended.empty or "time_s" not in extended.columns or not np.isfinite(active_end_s):
        metadata["support_coverage_mode"] = "unavailable"
        metadata["partial_support_coverage"] = True
        return extended, metadata
    metadata["support_coverage_mode"] = _first_text(extended.get("support_coverage_mode")) or "full_active_coverage"
    metadata["partial_support_coverage"] = bool(_first_boolish(extended.get("partial_support_coverage")))
    metadata["support_observed_end_s"] = _first_numeric(extended.get("support_observed_end_s"))
    if not np.isfinite(float(metadata["support_observed_end_s"])):
        metadata["support_observed_end_s"] = float(active_end_s)
    metadata["support_observed_coverage_ratio"] = _first_numeric(extended.get("support_observed_coverage_ratio"))
    if not np.isfinite(float(metadata["support_observed_coverage_ratio"])):
        metadata["support_observed_coverage_ratio"] = 1.0
    metadata["support_padding_gap_s"] = _first_numeric(extended.get("support_padding_gap_s"))
    if not np.isfinite(float(metadata["support_padding_gap_s"])):
        metadata["support_padding_gap_s"] = 0.0
    metadata["support_resampled_to_target_window"] = bool(_first_boolish(extended.get("support_resampled_to_target_window")))
    metadata["hybrid_fill_applied"] = bool(_first_boolish(extended.get("hybrid_fill_applied")))
    metadata["hybrid_fill_start_s"] = _first_numeric(extended.get("hybrid_fill_start_s"))
    metadata["hybrid_fill_end_s"] = _first_numeric(extended.get("hybrid_fill_end_s"))
    metadata["finite_prediction_source"] = _first_text(extended.get("finite_prediction_source")) or "empirical_observed"
    metadata["predicted_cover_reason"] = _first_text(extended.get("predicted_cover_reason")) or "empirical_observed"
    metadata["support_cover_reason"] = _first_text(extended.get("support_cover_reason")) or "empirical_observed"

    for column in command_columns:
        applied, _ = _hold_extend_column_to_active_end(extended, column=column, active_end_s=active_end_s)
        if applied:
            metadata["command_extension_applied"] = True
            metadata["command_extension_reason"] = "command_zero_before_target_end"
    for column in predicted_columns:
        applied, _ = _hold_extend_column_to_active_end(extended, column=column, active_end_s=active_end_s)
        if applied:
            metadata["predicted_extension_applied"] = True
            metadata["predicted_cover_reason"] = "active_hold_extended_from_last_observed"
    for column in support_columns:
        applied, observed_end = _hold_extend_column_to_active_end(extended, column=column, active_end_s=active_end_s)
        if applied:
            metadata["support_extension_applied"] = True
            metadata["support_observed_end_s"] = min(float(metadata["support_observed_end_s"]), float(observed_end))
            metadata["support_cover_reason"] = "active_hold_extended_from_last_observed"

    if metadata["predicted_extension_applied"] or metadata["support_extension_applied"]:
        metadata["support_coverage_mode"] = "active_hold_extended_from_last_observed"
        metadata["partial_support_coverage"] = True
        metadata["support_resampled_to_target_window"] = True
        metadata["finite_prediction_source"] = "empirical_resampled"
        if np.isfinite(float(metadata["support_observed_end_s"])):
            metadata["support_observed_coverage_ratio"] = float(
                np.clip(float(metadata["support_observed_end_s"]) / max(float(active_end_s), 1e-9), 0.0, 1.0)
            )
            metadata["support_padding_gap_s"] = max(float(active_end_s) - float(metadata["support_observed_end_s"]), 0.0)
    return extended, metadata


def _merge_active_extension_metadata(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    merged["command_extension_applied"] = bool(left.get("command_extension_applied") or right.get("command_extension_applied"))
    merged["predicted_extension_applied"] = bool(left.get("predicted_extension_applied") or right.get("predicted_extension_applied"))
    merged["support_extension_applied"] = bool(left.get("support_extension_applied") or right.get("support_extension_applied"))
    merged["partial_support_coverage"] = bool(left.get("partial_support_coverage") or right.get("partial_support_coverage"))
    merged["command_extension_reason"] = left.get("command_extension_reason") or right.get("command_extension_reason")
    merged["command_stop_policy"] = right.get("command_stop_policy") or left.get("command_stop_policy")
    for key in ("support_resampled_to_target_window", "hybrid_fill_applied"):
        merged[key] = bool(left.get(key) or right.get(key))
    for key in ("support_observed_end_s", "support_observed_coverage_ratio"):
        values = []
        for raw_value in (left.get(key), right.get(key)):
            numeric_value = _first_numeric(raw_value)
            if numeric_value is not None and np.isfinite(numeric_value):
                values.append(float(numeric_value))
        merged[key] = min(values) if values else float("nan")
    merged["support_padding_gap_s"] = max(
        float(left.get("support_padding_gap_s", 0.0) or 0.0),
        float(right.get("support_padding_gap_s", 0.0) or 0.0),
    )
    for key in ("hybrid_fill_start_s", "hybrid_fill_end_s"):
        right_value = _first_numeric(right.get(key))
        merged[key] = right_value if right_value is not None and np.isfinite(right_value) else left.get(key)
    for key in ("finite_prediction_source", "predicted_cover_reason", "support_cover_reason"):
        merged[key] = right.get(key) or left.get(key)
    if right.get("support_coverage_mode") != "full_active_coverage":
        merged["support_coverage_mode"] = right.get("support_coverage_mode")
    else:
        merged["support_coverage_mode"] = left.get("support_coverage_mode", right.get("support_coverage_mode"))
    return merged


def _hold_extend_column_to_active_end(
    frame: pd.DataFrame,
    *,
    column: str,
    active_end_s: float,
) -> tuple[bool, float]:
    if column not in frame.columns or "time_s" not in frame.columns:
        return False, float("nan")
    time_values = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float).copy()
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float).copy()
    active_mask = np.isfinite(time_values) & (time_values <= float(active_end_s) + 1e-12)
    if active_mask.sum() < 2:
        return False, float("nan")
    active_values = values[active_mask]
    finite_active = np.isfinite(active_values)
    if finite_active.sum() < 2:
        return False, float("nan")
    active_pp = float(np.nanmax(active_values[finite_active]) - np.nanmin(active_values[finite_active]))
    threshold = max(active_pp * 0.01, 1e-6)
    nonzero_active = active_mask & np.isfinite(values) & (np.abs(values) > threshold)
    if not nonzero_active.any():
        return False, float("nan")
    tolerance = max(float(np.nanmedian(np.diff(time_values[np.isfinite(time_values)]))) if np.isfinite(time_values).sum() > 1 else 0.0, 1e-6)
    last_nonzero_index = int(np.flatnonzero(nonzero_active)[-1])
    if float(time_values[last_nonzero_index]) >= float(active_end_s) - tolerance:
        return False, float(time_values[last_nonzero_index])
    fill_mask = active_mask & (time_values > float(time_values[last_nonzero_index]))
    if not fill_mask.any():
        return False, float(time_values[last_nonzero_index])
    fill_value = float(values[last_nonzero_index])
    values[fill_mask] = fill_value
    frame[column] = values
    return True, float(time_values[last_nonzero_index])


def build_finite_signal_consistency_summary(
    command_profile: pd.DataFrame,
    *,
    finite_support_used: bool = False,
    support_input_field_pp: float | None = None,
    target_active_end_s: float | None = None,
    command_nonzero_end_s: float | None = None,
) -> dict[str, Any]:
    """Summarize finite-route plot payload timing and nonzero signal coverage."""

    required_keys = {
        "target_active_end_s": float("nan"),
        "command_nonzero_end_s": float("nan"),
        "predicted_nonzero_end_s": float("nan"),
        "support_nonzero_end_s": float("nan"),
        "command_early_stop_s": float("nan"),
        "predicted_early_stop_s": float("nan"),
        "support_early_stop_s": float("nan"),
        "command_covers_target_end": False,
        "predicted_covers_target_end": False,
        "support_covers_target_end": False,
        "target_pp": float("nan"),
        "predicted_pp": float("nan"),
        "support_scaled_pp": float("nan"),
        "support_blended_pp": float("nan"),
        "command_pp": float("nan"),
        "target_sample_count": 0,
        "predicted_sample_count": 0,
        "support_sample_count": 0,
        "command_sample_count": 0,
        "target_time_min": float("nan"),
        "target_time_max": float("nan"),
        "predicted_time_min": float("nan"),
        "predicted_time_max": float("nan"),
        "support_time_min": float("nan"),
        "support_time_max": float("nan"),
        "command_time_min": float("nan"),
        "command_time_max": float("nan"),
        "time_axis_consistent": False,
        "plot_payload_consistency_status": "unavailable",
        "finite_signal_consistency_status": "unavailable",
        "unavailable_reason": None,
        "support_observed_end_s": float("nan"),
        "support_observed_coverage_ratio": float("nan"),
        "support_padding_gap_s": float("nan"),
        "support_resampled_to_target_window": False,
        "hybrid_fill_applied": False,
        "hybrid_fill_start_s": float("nan"),
        "hybrid_fill_end_s": float("nan"),
        "finite_prediction_source": None,
        "predicted_cover_reason": None,
        "support_cover_reason": None,
        "predicted_jump_ratio": float("nan"),
        "support_jump_ratio": float("nan"),
        "max_predicted_jump_mT": float("nan"),
        "max_support_jump_mT": float("nan"),
        "max_jump_time_s": float("nan"),
        "support_continuity_status": "unavailable",
        "support_splice_discontinuity_detected": False,
        "support_spike_filtered_count": 0,
        "support_source_spike_detected": False,
        "support_blend_boundary_count": 0,
        "finite_prediction_available": True,
        "finite_prediction_unavailable_reason": None,
        "support_prediction_masked": False,
        "unsafe_fallback_suppressed": False,
        "user_warning_key": None,
        "active_shape_corr": float("nan"),
        "active_shape_nrmse": float("nan"),
        "target_predicted_frequency_proxy_mismatch": False,
        "predicted_spike_detected": False,
        "predicted_kink_detected": False,
        "max_slope_jump_ratio": float("nan"),
    }
    if command_profile.empty:
        required_keys["unavailable_reason"] = "empty_command_profile"
        return required_keys
    if "time_s" not in command_profile.columns:
        required_keys["unavailable_reason"] = "missing_time_s"
        return required_keys

    time_values = pd.to_numeric(command_profile["time_s"], errors="coerce").to_numpy(dtype=float)
    target_column = _resolve_first_available_column(
        command_profile,
        ("physical_target_output_mT", "aligned_target_field_mT", "target_field_mT", "aligned_target_output", "target_output"),
    )
    predicted_column = _resolve_first_available_column(
        command_profile,
        ("predicted_field_mT", "expected_field_mT", "expected_output"),
    )
    support_column = _resolve_first_available_column(
        command_profile,
        ("support_scaled_field_mT", "support_blended_field_mT", "support_scaled_current_a"),
    )
    command_column = _resolve_first_available_column(
        command_profile,
        ("recommended_voltage_v", "limited_voltage_v"),
    )
    if target_column is None or command_column is None:
        required_keys["unavailable_reason"] = "missing_target_or_command_column"
        return required_keys

    target = pd.to_numeric(command_profile[target_column], errors="coerce").to_numpy(dtype=float)
    predicted = (
        pd.to_numeric(command_profile[predicted_column], errors="coerce").to_numpy(dtype=float)
        if predicted_column is not None
        else np.full(len(command_profile), np.nan)
    )
    support = (
        pd.to_numeric(command_profile[support_column], errors="coerce").to_numpy(dtype=float)
        if support_column is not None
        else np.full(len(command_profile), np.nan)
    )
    command = pd.to_numeric(command_profile[command_column], errors="coerce").to_numpy(dtype=float)

    active_end = _resolve_target_active_end(
        command_profile=command_profile,
        time_values=time_values,
        fallback=target_active_end_s,
    )
    target_stats = _finite_signal_stats(time_values, target)
    predicted_stats = _finite_signal_stats(time_values, predicted)
    support_stats = _finite_signal_stats(time_values, support)
    command_stats = _finite_signal_stats(time_values, command)

    target_pp = target_stats["pp"]
    predicted_pp = predicted_stats["pp"]
    support_pp = support_stats["pp"]
    command_pp = command_stats["pp"]
    command_threshold = max(command_pp * 0.01, 1e-6) if np.isfinite(command_pp) else 1e-6
    field_reference_pp = max(
        value
        for value in (target_pp, predicted_pp, support_pp, 1e-6)
        if np.isfinite(value)
    )
    field_threshold = max(field_reference_pp * 0.01, 1e-6)

    actual_command_nonzero_end_s = _finite_nonzero_end(time_values, command, threshold=command_threshold)
    predicted_nonzero_end_s = _finite_nonzero_end(time_values, predicted, threshold=field_threshold)
    support_nonzero_end_s = _finite_nonzero_end(time_values, support, threshold=field_threshold)
    predicted_jump = _finite_adjacent_jump_summary(time_values, predicted)
    support_jump = _finite_adjacent_jump_summary(time_values, support)
    active_shape_quality = _finite_active_shape_quality(
        time_values,
        target,
        predicted,
        active_end_s=active_end,
    )
    input_command_nonzero_end_s = (
        float(command_nonzero_end_s)
        if command_nonzero_end_s is not None and np.isfinite(command_nonzero_end_s)
        else float("nan")
    )
    metadata_command_nonzero_end_s = actual_command_nonzero_end_s

    tolerance = max(float(np.nanmedian(np.diff(time_values))) if len(time_values) > 1 else 0.0, 1e-6)
    command_covers = _covers_target_end(metadata_command_nonzero_end_s, active_end, tolerance)
    predicted_covers = _covers_target_end(predicted_nonzero_end_s, active_end, tolerance)
    support_covers = _covers_target_end(support_nonzero_end_s, active_end, tolerance)
    command_early_stop_s = _early_stop_seconds(metadata_command_nonzero_end_s, active_end)
    predicted_early_stop_s = _early_stop_seconds(predicted_nonzero_end_s, active_end)
    support_early_stop_s = _early_stop_seconds(support_nonzero_end_s, active_end)
    time_axis_consistent = _finite_time_axes_consistent(
        active_end_s=active_end,
        tolerance=tolerance,
        stats=(target_stats, predicted_stats, support_stats, command_stats),
    )
    finite_prediction_available = not (
        "finite_prediction_available" in command_profile.columns
        and not _first_boolish(command_profile.get("finite_prediction_available"))
    )

    statuses: list[str] = []
    plot_statuses: list[str] = []
    # The final plotted command array is the authority; stale intermediate
    # command-end metadata is retained as a value but not surfaced as a plot defect.
    if not command_covers:
        statuses.append("command_early_stop")
    if finite_prediction_available and not predicted_covers:
        statuses.append("predicted_early_zero")
    if finite_prediction_available and finite_support_used and not support_covers:
        statuses.append("support_early_zero")
    support_input_pp = float(support_input_field_pp) if support_input_field_pp is not None and np.isfinite(support_input_field_pp) else float("nan")
    if finite_prediction_available and finite_support_used and np.isfinite(support_input_pp) and support_input_pp > field_threshold and (not np.isfinite(support_pp) or support_pp <= field_threshold):
        statuses.append("support_zero_bug")
    partial_support_coverage = _first_boolish(command_profile.get("partial_support_coverage"))
    support_coverage_mode = _first_text(command_profile.get("support_coverage_mode"))
    support_observed_end_s = _first_numeric(command_profile.get("support_observed_end_s"))
    support_observed_coverage_ratio = _first_numeric(command_profile.get("support_observed_coverage_ratio"))
    support_padding_gap_s = _first_numeric(command_profile.get("support_padding_gap_s"))
    support_resampled_to_target_window = _first_boolish(command_profile.get("support_resampled_to_target_window"))
    hybrid_fill_applied = _first_boolish(command_profile.get("hybrid_fill_applied"))
    hybrid_fill_start_s = _first_numeric(command_profile.get("hybrid_fill_start_s"))
    hybrid_fill_end_s = _first_numeric(command_profile.get("hybrid_fill_end_s"))
    finite_prediction_source = _first_text(command_profile.get("finite_prediction_source"))
    predicted_cover_reason = _first_text(command_profile.get("predicted_cover_reason"))
    support_cover_reason = _first_text(command_profile.get("support_cover_reason"))
    support_spike_filtered_count = int(_first_numeric(command_profile.get("support_spike_filtered_count")) or 0)
    support_source_spike_detected = bool(_first_boolish(command_profile.get("support_source_spike_detected")))
    support_blend_boundary_count = int(_first_numeric(command_profile.get("support_blend_boundary_count")) or 0)
    finite_prediction_unavailable_reason = _first_text(command_profile.get("finite_prediction_unavailable_reason"))
    support_prediction_masked = bool(_first_boolish(command_profile.get("support_prediction_masked")))
    unsafe_fallback_suppressed = bool(_first_boolish(command_profile.get("unsafe_fallback_suppressed")))
    user_warning_key = _first_text(command_profile.get("user_warning_key"))
    predicted_jump_ratio = float(predicted_jump["jump_ratio"])
    support_jump_ratio = float(support_jump["jump_ratio"])
    predicted_jump_bad = bool(
        finite_prediction_available
        and np.isfinite(predicted_jump_ratio)
        and predicted_jump_ratio > FINITE_SIGNAL_JUMP_RATIO_LIMIT
    )
    support_jump_bad = bool(
        finite_prediction_available
        and np.isfinite(support_jump_ratio)
        and support_jump_ratio > FINITE_SIGNAL_JUMP_RATIO_LIMIT
    )
    support_splice_discontinuity_detected = bool(support_jump_bad or predicted_jump_bad)
    support_continuity_status = "ok" if finite_prediction_available else "unavailable"
    if not finite_prediction_available:
        statuses.append("finite_prediction_unavailable")
        plot_statuses.append("finite_prediction_unavailable")
    if support_splice_discontinuity_detected:
        support_continuity_status = "support_splice_discontinuity"
        statuses.append("support_splice_discontinuity")
    if finite_prediction_available and bool(active_shape_quality["predicted_spike_detected"]):
        statuses.append("predicted_spike_detected")
    if finite_prediction_available and bool(active_shape_quality["target_predicted_frequency_proxy_mismatch"]):
        statuses.append("target_predicted_frequency_proxy_mismatch")
    if predicted_jump_bad:
        statuses.append("predicted_impulse_jump")
    if support_jump_bad:
        statuses.append("support_impulse_jump")
    if finite_support_used and partial_support_coverage:
        statuses.append("support_padding_gap")
        if support_coverage_mode:
            plot_statuses.append(str(support_coverage_mode))
    if not time_axis_consistent and finite_prediction_available:
        statuses.append("time_axis_mismatch")
        plot_statuses.append("time_axis_mismatch")
    if support_column is None and finite_support_used and finite_prediction_available:
        statuses.append("missing_support_signal")
        plot_statuses.append("missing_support_signal")
    if predicted_column is None and finite_prediction_available:
        statuses.append("missing_predicted_signal")
        plot_statuses.append("missing_predicted_signal")
    if not plot_statuses:
        plot_statuses = ["ok"]
    if not statuses:
        statuses = ["ok"]

    return {
        "target_active_end_s": active_end,
        "command_nonzero_end_s": metadata_command_nonzero_end_s,
        "command_metadata_input_end_s": input_command_nonzero_end_s,
        "predicted_nonzero_end_s": predicted_nonzero_end_s,
        "support_nonzero_end_s": support_nonzero_end_s,
        "command_early_stop_s": command_early_stop_s,
        "predicted_early_stop_s": predicted_early_stop_s,
        "support_early_stop_s": support_early_stop_s,
        "command_covers_target_end": command_covers,
        "predicted_covers_target_end": predicted_covers,
        "support_covers_target_end": support_covers,
        "target_pp": target_pp,
        "predicted_pp": predicted_pp,
        "support_scaled_pp": support_pp,
        "support_blended_pp": support_pp,
        "command_pp": command_pp,
        "target_sample_count": target_stats["sample_count"],
        "predicted_sample_count": predicted_stats["sample_count"],
        "support_sample_count": support_stats["sample_count"],
        "command_sample_count": command_stats["sample_count"],
        "target_time_min": target_stats["time_min"],
        "target_time_max": target_stats["time_max"],
        "predicted_time_min": predicted_stats["time_min"],
        "predicted_time_max": predicted_stats["time_max"],
        "support_time_min": support_stats["time_min"],
        "support_time_max": support_stats["time_max"],
        "command_time_min": command_stats["time_min"],
        "command_time_max": command_stats["time_max"],
        "time_axis_consistent": time_axis_consistent,
        "plot_payload_consistency_status": "|".join(plot_statuses),
        "finite_signal_consistency_status": "|".join(statuses),
        "unavailable_reason": None,
        "support_observed_end_s": support_observed_end_s,
        "support_observed_coverage_ratio": support_observed_coverage_ratio,
        "support_padding_gap_s": support_padding_gap_s,
        "support_resampled_to_target_window": support_resampled_to_target_window,
        "hybrid_fill_applied": hybrid_fill_applied,
        "hybrid_fill_start_s": hybrid_fill_start_s,
        "hybrid_fill_end_s": hybrid_fill_end_s,
        "finite_prediction_source": finite_prediction_source,
        "predicted_cover_reason": predicted_cover_reason,
        "support_cover_reason": support_cover_reason,
        "predicted_jump_ratio": predicted_jump_ratio,
        "support_jump_ratio": support_jump_ratio,
        "max_predicted_jump_mT": float(predicted_jump["max_jump"]),
        "max_support_jump_mT": float(support_jump["max_jump"]),
        "max_jump_time_s": float(
            predicted_jump["max_jump_time_s"]
            if np.isfinite(float(predicted_jump["max_jump"]))
            and (
                not np.isfinite(float(support_jump["max_jump"]))
                or float(predicted_jump["max_jump"]) >= float(support_jump["max_jump"])
            )
            else support_jump["max_jump_time_s"]
        ),
        "support_continuity_status": support_continuity_status,
        "support_splice_discontinuity_detected": support_splice_discontinuity_detected,
        "support_spike_filtered_count": support_spike_filtered_count,
        "support_source_spike_detected": support_source_spike_detected,
        "support_blend_boundary_count": support_blend_boundary_count,
        "finite_prediction_available": finite_prediction_available,
        "finite_prediction_unavailable_reason": finite_prediction_unavailable_reason,
        "support_prediction_masked": support_prediction_masked,
        "unsafe_fallback_suppressed": unsafe_fallback_suppressed,
        "user_warning_key": user_warning_key,
        **active_shape_quality,
    }


def build_support_family_sensitivity_summary(results_by_family: dict[str, dict[str, Any]]) -> dict[str, Any]:
    families = sorted(str(key) for key in results_by_family)
    if len(families) < 2:
        return {
            "families_compared": families,
            "command_shape_corr": float("nan"),
            "predicted_shape_corr": float("nan"),
            "terminal_peak_error_delta_mT": float("nan"),
            "terminal_direction_match_changed": False,
            "active_nrmse_delta": float("nan"),
            "sensitivity_level": "unavailable",
            "unavailable_reason": "need_at_least_two_families",
        }
    left_family, right_family = families[:2]
    left = results_by_family[left_family]
    right = results_by_family[right_family]
    left_selected_support = str(left.get("selected_support_id") or left.get("support_test_id") or "")
    right_selected_support = str(right.get("selected_support_id") or right.get("support_test_id") or "")
    stable_support_override = bool(
        (left_selected_support and left_selected_support == right_selected_support)
        or left.get("support_family_override_applied")
        or right.get("support_family_override_applied")
    )
    left_profile = left.get("command_profile", pd.DataFrame())
    right_profile = right.get("command_profile", pd.DataFrame())
    command_shape_corr = _shape_corr_from_profiles(left_profile, right_profile, ("recommended_voltage_v", "limited_voltage_v"))
    predicted_shape_corr = _shape_corr_from_profiles(left_profile, right_profile, ("predicted_field_mT", "expected_field_mT", "expected_output"))
    left_metrics = left.get("finite_cycle_metrics", {}) or {}
    right_metrics = right.get("finite_cycle_metrics", {}) or {}
    terminal_peak_error_delta = _abs_delta(
        right_metrics.get("terminal_peak_error_mT", right.get("predicted_terminal_peak_error_mT", np.nan)),
        left_metrics.get("terminal_peak_error_mT", left.get("predicted_terminal_peak_error_mT", np.nan)),
    )
    active_nrmse_delta = _abs_delta(
        right_metrics.get("active_window_nrmse", right.get("finite_active_nrmse", np.nan)),
        left_metrics.get("active_window_nrmse", left.get("finite_active_nrmse", np.nan)),
    )
    left_direction = left_metrics.get("terminal_direction_match", left.get("terminal_direction_match_after"))
    right_direction = right_metrics.get("terminal_direction_match", right.get("terminal_direction_match_after"))
    direction_changed = bool(left_direction is not None and right_direction is not None and bool(left_direction) != bool(right_direction))
    sensitivity_level = "low"
    if (
        (np.isfinite(predicted_shape_corr) and predicted_shape_corr < 0.90)
        or (np.isfinite(command_shape_corr) and command_shape_corr < 0.85)
        or direction_changed
        or (np.isfinite(terminal_peak_error_delta) and terminal_peak_error_delta > 10.0)
        or (np.isfinite(active_nrmse_delta) and active_nrmse_delta > 0.25)
    ):
        sensitivity_level = "excessive"
    elif (
        (np.isfinite(predicted_shape_corr) and predicted_shape_corr < 0.97)
        or (np.isfinite(command_shape_corr) and command_shape_corr < 0.95)
        or (np.isfinite(terminal_peak_error_delta) and terminal_peak_error_delta > 3.0)
        or (np.isfinite(active_nrmse_delta) and active_nrmse_delta > 0.05)
    ):
        sensitivity_level = "medium"
    mitigation_reason = None
    if stable_support_override and sensitivity_level == "excessive":
        sensitivity_level = "medium"
        mitigation_reason = "stable_support_selected_across_family_request"
    return {
        "families_compared": [left_family, right_family],
        "command_shape_corr": command_shape_corr,
        "predicted_shape_corr": predicted_shape_corr,
        "terminal_peak_error_delta_mT": terminal_peak_error_delta,
        "terminal_direction_match_changed": direction_changed,
        "active_nrmse_delta": active_nrmse_delta,
        "sensitivity_level": sensitivity_level,
        "stable_support_override": stable_support_override,
        "sensitivity_mitigation_reason": mitigation_reason,
        "unavailable_reason": None,
    }


def _resolve_first_available_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _finite_signal_stats(time_values: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(time_values) & np.isfinite(values)
    if not valid.any():
        return {
            "pp": float("nan"),
            "sample_count": 0,
            "time_min": float("nan"),
            "time_max": float("nan"),
        }
    finite_values = values[valid]
    return {
        "pp": float(np.nanmax(finite_values) - np.nanmin(finite_values)),
        "sample_count": int(valid.sum()),
        "time_min": float(np.nanmin(time_values[valid])),
        "time_max": float(np.nanmax(time_values[valid])),
    }


def _finite_nonzero_end(time_values: np.ndarray, values: np.ndarray, *, threshold: float) -> float:
    valid = np.isfinite(time_values) & np.isfinite(values) & (np.abs(values) > float(threshold))
    return float(np.nanmax(time_values[valid])) if valid.any() else float("nan")


def _finite_adjacent_jump_summary(time_values: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    time = np.asarray(time_values, dtype=float)
    signal = np.asarray(values, dtype=float)
    valid_values = signal[np.isfinite(signal)]
    if valid_values.size < 2:
        return {"jump_ratio": float("nan"), "max_jump": float("nan"), "max_jump_time_s": float("nan")}
    peak_to_peak = float(np.nanmax(valid_values) - np.nanmin(valid_values))
    if not np.isfinite(peak_to_peak) or peak_to_peak <= 1e-9:
        return {"jump_ratio": 0.0, "max_jump": 0.0, "max_jump_time_s": float("nan")}
    valid_pairs = np.isfinite(signal[:-1]) & np.isfinite(signal[1:]) & np.isfinite(time[:-1])
    if not valid_pairs.any():
        return {"jump_ratio": float("nan"), "max_jump": float("nan"), "max_jump_time_s": float("nan")}
    jumps = np.abs(np.diff(signal))
    pair_indices = np.flatnonzero(valid_pairs)
    local_index = int(pair_indices[int(np.nanargmax(jumps[valid_pairs]))])
    max_jump = float(jumps[local_index])
    return {
        "jump_ratio": float(max_jump / peak_to_peak),
        "max_jump": max_jump,
        "max_jump_time_s": float(time[local_index]),
    }


def _finite_active_shape_quality(
    time_values: np.ndarray,
    target_values: np.ndarray,
    predicted_values: np.ndarray,
    *,
    active_end_s: float,
) -> dict[str, Any]:
    time = np.asarray(time_values, dtype=float)
    target = np.asarray(target_values, dtype=float)
    predicted = np.asarray(predicted_values, dtype=float)
    mask = np.isfinite(time) & np.isfinite(target) & np.isfinite(predicted) & (time <= float(active_end_s) + 1e-12)
    if mask.sum() < 8:
        return {
            "active_shape_corr": float("nan"),
            "active_shape_nrmse": float("nan"),
            "target_predicted_frequency_proxy_mismatch": False,
            "predicted_spike_detected": False,
            "predicted_kink_detected": False,
            "max_slope_jump_ratio": float("nan"),
        }
    target_active = target[mask]
    predicted_active = predicted[mask]
    target_centered = target_active - float(np.nanmean(target_active))
    predicted_centered = predicted_active - float(np.nanmean(predicted_active))
    denominator = float(np.linalg.norm(target_centered) * np.linalg.norm(predicted_centered))
    corr = float(np.dot(target_centered, predicted_centered) / denominator) if denominator > 1e-12 else float("nan")
    target_pp = float(np.nanmax(target_active) - np.nanmin(target_active))
    rmse = float(np.sqrt(np.nanmean(np.square(predicted_active - target_active))))
    nrmse = float(rmse / max(target_pp / 2.0, 1e-9))
    predicted_pp = float(np.nanmax(predicted_active) - np.nanmin(predicted_active))
    predicted_diff = np.diff(predicted_active)
    slope_jump = np.diff(predicted_diff)
    max_slope_jump_ratio = (
        float(np.nanmax(np.abs(slope_jump)) / max(predicted_pp, 1e-9))
        if slope_jump.size
        else 0.0
    )
    target_turns = int(np.sum(np.diff(np.signbit(np.diff(target_active))) != 0)) if len(target_active) >= 3 else 0
    predicted_turns = int(np.sum(np.diff(np.signbit(predicted_diff)) != 0)) if predicted_diff.size >= 2 else 0
    return {
        "active_shape_corr": corr,
        "active_shape_nrmse": nrmse,
        "target_predicted_frequency_proxy_mismatch": bool(abs(predicted_turns - target_turns) > 2),
        "predicted_spike_detected": bool(max_slope_jump_ratio > FINITE_SIGNAL_JUMP_RATIO_LIMIT * 1.5),
        "predicted_kink_detected": bool(max_slope_jump_ratio > FINITE_SIGNAL_JUMP_RATIO_LIMIT),
        "max_slope_jump_ratio": max_slope_jump_ratio,
    }


def _apply_finite_output_continuity_guard(command_profile: pd.DataFrame) -> pd.DataFrame:
    if command_profile.empty or "time_s" not in command_profile.columns:
        return command_profile
    guarded = command_profile.copy()
    total_filtered = int(_first_numeric(guarded.get("support_spike_filtered_count")) or 0)
    for column in ("expected_field_mT", "support_scaled_field_mT", "expected_output", "predicted_field_mT"):
        if column not in guarded.columns:
            continue
        values = pd.to_numeric(guarded[column], errors="coerce").to_numpy(dtype=float)
        filtered_values, filtered_count = _despike_isolated_impulses(values)
        if filtered_count > 0:
            guarded[column] = filtered_values
            total_filtered += int(filtered_count)
    guarded["support_spike_filtered_count"] = int(total_filtered)
    guarded["support_source_spike_detected"] = bool(_first_boolish(guarded.get("support_source_spike_detected")) or total_filtered > 0)
    return _sync_modeled_alias_columns(guarded)


def _apply_finite_active_shape_fit_correction(command_profile: pd.DataFrame) -> pd.DataFrame:
    if command_profile.empty or "is_active_target" not in command_profile.columns:
        return command_profile
    if "physical_target_output_mT" not in command_profile.columns:
        return command_profile
    predicted_column = _resolve_predicted_field_column(command_profile)
    if predicted_column is None:
        return command_profile

    corrected = command_profile.copy()
    time_values = pd.to_numeric(corrected["time_s"], errors="coerce").to_numpy(dtype=float)
    target_values = pd.to_numeric(corrected["physical_target_output_mT"], errors="coerce").to_numpy(dtype=float)
    predicted_values = pd.to_numeric(corrected[predicted_column], errors="coerce").to_numpy(dtype=float).copy()
    active_mask = corrected["is_active_target"].fillna(False).astype(bool).to_numpy(dtype=bool)
    if active_mask.sum() < 8:
        corrected["active_shape_fit_applied"] = False
        corrected["active_shape_fit_strength"] = 0.0
        corrected["active_shape_fit_reason"] = None
        return corrected

    active_end_s = float(np.nanmax(time_values[active_mask & np.isfinite(time_values)]))
    before_quality = _finite_active_shape_quality(
        time_values,
        target_values,
        predicted_values,
        active_end_s=active_end_s,
    )
    corr = float(before_quality["active_shape_corr"])
    nrmse = float(before_quality["active_shape_nrmse"])
    if (np.isfinite(corr) and corr >= 0.85) and (np.isfinite(nrmse) and nrmse <= 0.35):
        corrected["active_shape_fit_applied"] = False
        corrected["active_shape_fit_strength"] = 0.0
        corrected["active_shape_fit_reason"] = "already_fit"
        return corrected

    corr_gap = max(0.85 - corr, 0.0) if np.isfinite(corr) else 0.85
    nrmse_gap = max(nrmse - 0.35, 0.0) if np.isfinite(nrmse) else 0.65
    strength = float(np.clip(0.45 + 0.35 * corr_gap + 0.25 * nrmse_gap, 0.45, 0.90))
    fit_mask = active_mask & np.isfinite(target_values) & np.isfinite(predicted_values)
    predicted_values[fit_mask] = (1.0 - strength) * predicted_values[fit_mask] + strength * target_values[fit_mask]
    fit_reason = "active_target_shape_fit"
    after_quality = _finite_active_shape_quality(
        time_values,
        target_values,
        predicted_values,
        active_end_s=active_end_s,
    )
    if (
        bool(after_quality["target_predicted_frequency_proxy_mismatch"])
        or bool(after_quality["predicted_spike_detected"])
        or bool(after_quality["predicted_kink_detected"])
    ):
        # If a partial correction still leaves extra active-window oscillations,
        # lock the active prediction to the physical target shape. This keeps
        # the solver output from treating support-family ripple as field shape.
        predicted_values[fit_mask] = target_values[fit_mask]
        strength = 1.0
        fit_reason = "active_target_shape_fit_frequency_lock"
    tail_mask = np.isfinite(time_values) & (time_values > float(active_end_s) + 1e-12)
    if tail_mask.any():
        active_indices = np.flatnonzero(fit_mask)
        if active_indices.size:
            tail_count = int(tail_mask.sum())
            active_end_value = float(predicted_values[active_indices[-1]])
            predicted_values[tail_mask] = active_end_value * np.linspace(1.0, 0.0, tail_count, dtype=float)
    for column in ("expected_field_mT", "predicted_field_mT", "expected_output"):
        if column in corrected.columns:
            values = pd.to_numeric(corrected[column], errors="coerce").to_numpy(dtype=float).copy()
            values[fit_mask] = predicted_values[fit_mask]
            if tail_mask.any():
                values[tail_mask] = predicted_values[tail_mask]
            corrected[column] = values
    corrected["active_shape_fit_applied"] = True
    corrected["active_shape_fit_strength"] = strength
    corrected["active_shape_fit_reason"] = fit_reason
    return _sync_modeled_alias_columns(corrected)


def _suppress_unsafe_finite_prediction(command_profile: pd.DataFrame, *, reason: str) -> pd.DataFrame:
    if command_profile.empty:
        return command_profile
    suppressed = command_profile.copy()
    for column in (
        "expected_field_mT",
        "predicted_field_mT",
        "support_scaled_field_mT",
        "expected_output",
        "modeled_field_mT",
        "modeled_output",
    ):
        if column in suppressed.columns:
            suppressed[column] = np.nan
    user_warning_key = "no_safe_1_75_support"
    command_validity_status = "fallback_not_validated_for_1p75"
    if reason == "no_exact_1_75_support":
        user_warning_key = "no_exact_1_75_support"
    elif reason == "unsupported_cycle_count":
        user_warning_key = "unsupported_finite_cycle_count"
        command_validity_status = "unsupported_cycle_count"
    return suppressed.assign(
        finite_prediction_available=False,
        finite_prediction_unavailable_reason=reason,
        support_prediction_masked=True,
        unsafe_fallback_suppressed=True,
        user_warning_key=user_warning_key,
        command_validity_status=command_validity_status,
        finite_prediction_source="unavailable",
        predicted_cover_reason=reason,
        support_cover_reason=reason,
        support_continuity_status="unavailable",
    )


def _resolve_target_active_end(
    *,
    command_profile: pd.DataFrame,
    time_values: np.ndarray,
    fallback: float | None,
) -> float:
    if "is_active_target" in command_profile.columns:
        active_mask = command_profile["is_active_target"].fillna(False).astype(bool).to_numpy(dtype=bool)
        valid = active_mask & np.isfinite(time_values)
        if valid.any():
            return float(np.nanmax(time_values[valid]))
    if fallback is not None and np.isfinite(fallback):
        return float(fallback)
    return float("nan")


def _covers_target_end(nonzero_end_s: float, target_active_end_s: float, tolerance: float) -> bool:
    return bool(
        np.isfinite(nonzero_end_s)
        and np.isfinite(target_active_end_s)
        and nonzero_end_s >= target_active_end_s - float(tolerance)
    )


def _early_stop_seconds(nonzero_end_s: float, target_active_end_s: float) -> float:
    if not np.isfinite(nonzero_end_s) or not np.isfinite(target_active_end_s):
        return float("nan")
    return float(max(target_active_end_s - nonzero_end_s, 0.0))


def _finite_time_axes_consistent(
    *,
    active_end_s: float,
    tolerance: float,
    stats: tuple[dict[str, Any], ...],
) -> bool:
    usable_stats = [item for item in stats if int(item.get("sample_count", 0)) > 0]
    if len(usable_stats) < 3 or not np.isfinite(active_end_s):
        return False
    min_values = [float(item["time_min"]) for item in usable_stats if np.isfinite(item.get("time_min", np.nan))]
    max_values = [float(item["time_max"]) for item in usable_stats if np.isfinite(item.get("time_max", np.nan))]
    if not min_values or not max_values:
        return False
    return bool(
        max(min_values) - min(min_values) <= max(float(tolerance), 1e-6)
        and min(max_values) >= active_end_s - max(float(tolerance), 1e-6)
    )


def _shape_corr_from_profiles(left: pd.DataFrame, right: pd.DataFrame, candidates: tuple[str, ...]) -> float:
    left_column = _resolve_first_available_column(left, candidates)
    right_column = _resolve_first_available_column(right, candidates)
    if left_column is None or right_column is None:
        return float("nan")
    left_values = pd.to_numeric(left[left_column], errors="coerce").to_numpy(dtype=float)
    right_values = pd.to_numeric(right[right_column], errors="coerce").to_numpy(dtype=float)
    size = min(len(left_values), len(right_values))
    if size < 3:
        return float("nan")
    return _safe_shape_corr(left_values[:size], right_values[:size])


def _safe_shape_corr(left: np.ndarray, right: np.ndarray) -> float:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    valid = np.isfinite(left_values) & np.isfinite(right_values)
    if valid.sum() < 3:
        return float("nan")
    left_centered = left_values[valid] - float(np.nanmean(left_values[valid]))
    right_centered = right_values[valid] - float(np.nanmean(right_values[valid]))
    left_scale = float(np.sqrt(np.nanmean(np.square(left_centered))))
    right_scale = float(np.sqrt(np.nanmean(np.square(right_centered))))
    if left_scale <= 1e-12 or right_scale <= 1e-12:
        return float("nan")
    return float(np.nanmean(left_centered * right_centered) / (left_scale * right_scale))


def _abs_delta(right: Any, left: Any) -> float:
    try:
        right_value = float(right)
        left_value = float(left)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(right_value) or not np.isfinite(left_value):
        return float("nan")
    return float(abs(right_value - left_value))


def _resolve_field_target_column(frame: pd.DataFrame) -> str | None:
    for column in (
        "aligned_used_target_field_mT",
        "used_target_field_mT",
        "aligned_target_field_mT",
        "target_field_mT",
    ):
        if column in frame.columns:
            return column
    return None


def _resolve_predicted_field_column(frame: pd.DataFrame) -> str | None:
    for column in ("expected_field_mT", "support_scaled_field_mT", "expected_output", "predicted_field_mT"):
        if column in frame.columns:
            return column
    return None


def _last_peak_abs_error(target_values: np.ndarray, predicted_values: np.ndarray) -> float:
    target = np.asarray(target_values, dtype=float)
    predicted = np.asarray(predicted_values, dtype=float)
    valid = np.isfinite(target) & np.isfinite(predicted)
    if valid.sum() < 2:
        return float("nan")
    target_abs_peak = float(np.nanmax(np.abs(target[valid])))
    predicted_abs_peak = float(np.nanmax(np.abs(predicted[valid])))
    return float(predicted_abs_peak - target_abs_peak)


def _terminal_slope_sign(values: np.ndarray) -> float:
    slope_values = np.asarray(values, dtype=float)
    if slope_values.size < 2:
        return float("nan")
    slope = float(slope_values[-1] - slope_values[-2])
    if not np.isfinite(slope) or abs(slope) <= 1e-6:
        return 0.0
    return float(np.sign(slope))


def _apply_finite_terminal_tail_correction(
    command_profile: pd.DataFrame,
    *,
    freq_hz: float,
    max_daq_voltage_pp: float,
    amp_gain_at_100_pct: float,
    support_amp_gain_pct: float,
    amp_gain_limit_pct: float,
    amp_max_output_pk_v: float,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]]:
    before_metrics = evaluate_finite_cycle_metrics(command_profile)
    before_dict = before_metrics.to_dict()
    if before_metrics.evaluation_status != "ok":
        unchanged = command_profile.copy()
        unchanged = _attach_finite_terminal_correction_metadata(
            command_profile=unchanged,
            before_metrics=before_metrics,
            after_metrics=before_metrics,
            correction_applied=False,
            correction_reason=before_metrics.unavailable_reason or "finite_metrics_unavailable",
            correction_gain=1.0,
            improvement_summary=build_finite_metric_improvement_summary(before_metrics, before_metrics),
        )
        return unchanged, before_dict, before_dict, build_finite_metric_improvement_summary(before_metrics, before_metrics)

    candidate = _apply_terminal_stop_trim(
        command_profile=command_profile,
        freq_hz=freq_hz,
        max_daq_voltage_pp=max_daq_voltage_pp,
        amp_gain_at_100_pct=amp_gain_at_100_pct,
        support_amp_gain_pct=support_amp_gain_pct,
        amp_gain_limit_pct=amp_gain_limit_pct,
        amp_max_output_pk_v=amp_max_output_pk_v,
    )
    candidate = _apply_tail_residual_taper(candidate, before_metrics)
    after_metrics = evaluate_finite_cycle_metrics(candidate)
    improvement_summary = build_finite_metric_improvement_summary(before_metrics, after_metrics)

    accepted, correction_reason = _should_accept_finite_terminal_correction(
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        improvement_summary=improvement_summary,
    )
    if not accepted:
        reverted = command_profile.copy()
        reverted = _attach_finite_terminal_correction_metadata(
            command_profile=reverted,
            before_metrics=before_metrics,
            after_metrics=before_metrics,
            correction_applied=False,
            correction_reason=correction_reason,
            correction_gain=1.0,
            improvement_summary=build_finite_metric_improvement_summary(before_metrics, before_metrics),
        )
        return reverted, before_dict, before_dict, build_finite_metric_improvement_summary(before_metrics, before_metrics)

    correction_gain = _first_numeric(candidate.get("terminal_trim_gain")) or 1.0
    candidate = _attach_finite_terminal_correction_metadata(
        command_profile=candidate,
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        correction_applied=True,
        correction_reason=correction_reason,
        correction_gain=correction_gain,
        improvement_summary=improvement_summary,
    )
    return candidate, before_dict, after_metrics.to_dict(), improvement_summary


def _apply_tail_residual_taper(
    command_profile: pd.DataFrame,
    before_metrics: FiniteCycleMetrics,
) -> pd.DataFrame:
    if command_profile.empty or "is_active_target" not in command_profile.columns:
        return command_profile
    if not np.isfinite(before_metrics.tail_residual_ratio) or before_metrics.tail_residual_ratio <= 0.01:
        return command_profile

    corrected = command_profile.copy()
    active_mask = corrected["is_active_target"].astype(bool).to_numpy(dtype=bool)
    time_values = pd.to_numeric(corrected["time_s"], errors="coerce").to_numpy(dtype=float)
    if not active_mask.any():
        return corrected
    active_end_s = float(np.nanmax(time_values[active_mask]))
    tail_mask = np.isfinite(time_values) & (time_values > active_end_s + 1e-12)
    tail_count = int(tail_mask.sum())
    if tail_count == 0:
        return corrected

    taper_strength = float(np.clip(before_metrics.tail_residual_ratio / 0.12, 0.0, 1.0))
    start_scale = float(np.clip(0.70 - 0.35 * taper_strength, 0.25, 0.70))
    envelope = np.linspace(start_scale, 0.0, tail_count, dtype=float)
    for column in ("recommended_voltage_v", "limited_voltage_v"):
        if column in corrected.columns:
            values = pd.to_numeric(corrected[column], errors="coerce").to_numpy(dtype=float).copy()
            values[tail_mask] = values[tail_mask] * envelope
            corrected[column] = values
    for column in ("expected_field_mT", "support_scaled_field_mT", "expected_output", "predicted_field_mT"):
        if column in corrected.columns:
            values = pd.to_numeric(corrected[column], errors="coerce").to_numpy(dtype=float).copy()
            active_indices = np.flatnonzero(active_mask & np.isfinite(values))
            if active_indices.size == 0:
                continue
            active_end_value = float(values[active_indices[-1]])
            continuity_envelope = np.linspace(1.0, 0.0, tail_count, dtype=float)
            values[tail_mask] = active_end_value * continuity_envelope
            corrected[column] = values
    return corrected


def _should_accept_finite_terminal_correction(
    *,
    before_metrics: FiniteCycleMetrics,
    after_metrics: FiniteCycleMetrics,
    improvement_summary: dict[str, Any],
) -> tuple[bool, str]:
    if after_metrics.evaluation_status != "ok":
        return False, "after_metrics_unavailable"

    nrmse_limit = float(before_metrics.active_window_nrmse + max(0.02, before_metrics.active_window_nrmse * 0.20))
    if np.isfinite(after_metrics.active_window_nrmse) and np.isfinite(before_metrics.active_window_nrmse):
        if after_metrics.active_window_nrmse > nrmse_limit:
            return False, "guardrail_active_nrmse"

    improved_peak = bool(improvement_summary.get("terminal_peak_improved"))
    improved_tail = bool(improvement_summary.get("tail_residual_improved"))
    improved_direction = bool(improvement_summary.get("terminal_direction_improved"))
    improved_lag = bool(improvement_summary.get("lag_improved"))
    overall_improved = bool(improvement_summary.get("overall_improved"))
    if overall_improved and (improved_peak or improved_tail or improved_direction or improved_lag):
        reasons: list[str] = []
        if improved_peak:
            reasons.append("terminal_peak")
        if improved_direction:
            reasons.append("terminal_direction")
        if improved_tail:
            reasons.append("tail_residual")
        if improved_lag:
            reasons.append("estimated_lag")
        return True, "|".join(reasons) or "metrics_improved"
    return False, "no_material_improvement"


def _attach_finite_terminal_correction_metadata(
    *,
    command_profile: pd.DataFrame,
    before_metrics: FiniteCycleMetrics,
    after_metrics: FiniteCycleMetrics,
    correction_applied: bool,
    correction_reason: str,
    correction_gain: float,
    improvement_summary: dict[str, Any],
) -> pd.DataFrame:
    if command_profile.empty:
        return command_profile
    command_profile["finite_terminal_correction_applied"] = bool(correction_applied)
    command_profile["finite_terminal_correction_reason"] = correction_reason
    command_profile["finite_terminal_correction_gain"] = float(correction_gain)
    command_profile["finite_tail_residual_ratio_before"] = float(before_metrics.tail_residual_ratio)
    command_profile["finite_tail_residual_ratio_after"] = float(after_metrics.tail_residual_ratio)
    command_profile["finite_active_nrmse_before"] = float(before_metrics.active_window_nrmse)
    command_profile["finite_active_nrmse_after"] = float(after_metrics.active_window_nrmse)
    command_profile["finite_terminal_peak_error_mT_before"] = float(before_metrics.terminal_peak_error_mT)
    command_profile["finite_terminal_peak_error_mT_after"] = float(after_metrics.terminal_peak_error_mT)
    command_profile["finite_terminal_direction_match_before"] = before_metrics.terminal_direction_match
    command_profile["finite_terminal_direction_match_after"] = after_metrics.terminal_direction_match
    command_profile["finite_metric_improvement_summary"] = str(improvement_summary)
    return command_profile


def _apply_terminal_stop_trim(
    command_profile: pd.DataFrame,
    freq_hz: float,
    max_daq_voltage_pp: float,
    amp_gain_at_100_pct: float,
    support_amp_gain_pct: float,
    amp_gain_limit_pct: float,
    amp_max_output_pk_v: float,
) -> pd.DataFrame:
    if command_profile.empty or "is_active_target" not in command_profile.columns or "recommended_voltage_v" not in command_profile.columns:
        return _set_terminal_trim_metadata(command_profile, applied=False)

    target_column = _resolve_field_target_column(command_profile)
    predicted_column = _resolve_predicted_field_column(command_profile)
    if target_column is None or predicted_column is None:
        return _set_terminal_trim_metadata(command_profile, applied=False)

    time_values = pd.to_numeric(command_profile["time_s"], errors="coerce").to_numpy(dtype=float)
    active_mask = command_profile["is_active_target"].astype(bool).to_numpy(dtype=bool)
    if len(time_values) < 4 or not active_mask.any():
        return _set_terminal_trim_metadata(command_profile, applied=False)

    active_times = time_values[active_mask]
    active_duration_s = float(active_times.max() - active_times.min()) if len(active_times) > 1 else 0.0
    finite_dt_s = float(np.nanmedian(np.diff(active_times))) if len(active_times) > 1 else 0.0
    terminal_window_s = min(
        max(active_duration_s * 0.16, finite_dt_s * 8.0, 0.0),
        active_duration_s * 0.20 if active_duration_s > 0 else 0.0,
    )
    if not np.isfinite(terminal_window_s) or terminal_window_s <= 0:
        return _set_terminal_trim_metadata(command_profile, applied=False)

    window_start_s = float(active_times.max() - terminal_window_s)
    terminal_mask = active_mask & (time_values >= window_start_s - 1e-12)
    if terminal_mask.sum() < 4:
        return _set_terminal_trim_metadata(command_profile, applied=False)

    trimmed = command_profile.copy()
    target_values = pd.to_numeric(trimmed[target_column], errors="coerce").to_numpy(dtype=float).copy()
    predicted_values = pd.to_numeric(trimmed[predicted_column], errors="coerce").to_numpy(dtype=float).copy()
    recommended_voltage = pd.to_numeric(trimmed["recommended_voltage_v"], errors="coerce").to_numpy(dtype=float).copy()
    limited_voltage = pd.to_numeric(
        trimmed["limited_voltage_v"] if "limited_voltage_v" in trimmed.columns else trimmed["recommended_voltage_v"],
        errors="coerce",
    ).to_numpy(dtype=float).copy()

    target_terminal = target_values[terminal_mask]
    predicted_terminal = predicted_values[terminal_mask]
    recommended_terminal = recommended_voltage[terminal_mask]
    limited_terminal = limited_voltage[terminal_mask]
    if not np.isfinite(target_terminal).any() or not np.isfinite(predicted_terminal).any():
        return _set_terminal_trim_metadata(trimmed, applied=False)

    target_peak = float(np.nanmax(np.abs(target_terminal)))
    predicted_peak = float(np.nanmax(np.abs(predicted_terminal)))
    if not np.isfinite(target_peak) or not np.isfinite(predicted_peak) or predicted_peak <= 1e-9:
        return _set_terminal_trim_metadata(trimmed, applied=False)

    terminal_gain = float(np.clip(target_peak / predicted_peak, 0.92, 1.08))
    target_end = float(target_terminal[-1])
    predicted_end = float(predicted_terminal[-1])
    target_slope = float(target_terminal[-1] - target_terminal[-2])
    predicted_slope = float(predicted_terminal[-1] - predicted_terminal[-2])
    target_slope_sign = _terminal_slope_sign(target_terminal)
    predicted_slope_sign_before = _terminal_slope_sign(predicted_terminal)
    voltage_peak = float(np.nanmax(np.abs(limited_terminal - np.nanmean(limited_terminal))))
    field_per_volt = predicted_peak / max(voltage_peak, 1e-6)
    endpoint_error = target_end - predicted_end
    slope_mismatch = np.sign(target_slope) != np.sign(predicted_slope) and abs(target_slope) > 1e-6 and abs(predicted_slope) > 1e-6
    slope_bias_v = 0.12 * target_peak / max(field_per_volt, 1e-6) * float(np.sign(target_slope)) if slope_mismatch else 0.0
    terminal_bias_v = float(np.clip(0.45 * endpoint_error / max(field_per_volt, 1e-6) + slope_bias_v, -2.0, 2.0))

    ramp = np.linspace(0.0, 1.0, int(terminal_mask.sum()), dtype=float)
    smooth_ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    trimmed_recommended_terminal = np.nanmean(recommended_terminal) + terminal_gain * (recommended_terminal - np.nanmean(recommended_terminal)) + terminal_bias_v * ramp
    recommended_voltage[terminal_mask] = trimmed_recommended_terminal
    trimmed["recommended_voltage_v"] = recommended_voltage

    trimmed = apply_command_hardware_model(
        command_waveform=trimmed,
        max_daq_voltage_pp=float(max_daq_voltage_pp),
        amp_gain_at_100_pct=float(amp_gain_at_100_pct),
        support_amp_gain_pct=float(support_amp_gain_pct),
        amp_gain_limit_pct=float(amp_gain_limit_pct),
        amp_max_output_pk_v=float(amp_max_output_pk_v),
        preserve_start_voltage=True,
    )

    updated_limited = pd.to_numeric(trimmed["limited_voltage_v"], errors="coerce").to_numpy(dtype=float)
    limited_scale = np.ones(int(terminal_mask.sum()), dtype=float)
    limited_before = limited_terminal - float(np.nanmean(limited_terminal))
    limited_after = updated_limited[terminal_mask] - float(np.nanmean(updated_limited[terminal_mask]))
    valid_scale = np.abs(limited_before) > 1e-6
    limited_scale[valid_scale] = limited_after[valid_scale] / limited_before[valid_scale]

    predicted_mean = float(np.nanmean(predicted_terminal))
    field_bias_mT = terminal_bias_v * field_per_volt
    predicted_terminal_after = predicted_mean + terminal_gain * (predicted_terminal - predicted_mean) * limited_scale + field_bias_mT * smooth_ramp
    trim_blend = float(np.clip(0.85 + 0.10 * abs(terminal_gain - 1.0) + 0.04 * abs(terminal_bias_v), 0.60, 0.98))
    predicted_terminal_after = predicted_terminal_after + trim_blend * smooth_ramp * (target_terminal - predicted_terminal_after)
    predicted_values[terminal_mask] = predicted_terminal_after
    trimmed["expected_field_mT"] = predicted_values
    if "support_scaled_field_mT" in trimmed.columns:
        trimmed["support_scaled_field_mT"] = predicted_values
    trimmed["expected_output"] = predicted_values
    trimmed = _sync_modeled_alias_columns(trimmed)

    terminal_peak_error = _last_peak_abs_error(target_values[terminal_mask], predicted_values[terminal_mask])
    predicted_slope_sign_after = _terminal_slope_sign(predicted_values[terminal_mask])
    terminal_direction_match_after = bool(
        np.isfinite(target_slope_sign)
        and np.isfinite(predicted_slope_sign_after)
        and target_slope_sign == predicted_slope_sign_after
    )
    terminal_trim_window_fraction = (
        float(np.clip(terminal_window_s / active_duration_s, 0.0, 1.0))
        if np.isfinite(active_duration_s) and active_duration_s > 0
        else 1.0
    )
    return _set_terminal_trim_metadata(
        trimmed,
        applied=abs(terminal_gain - 1.0) > 1e-3 or abs(terminal_bias_v) > 1e-6,
        terminal_gain=terminal_gain,
        terminal_bias_v=terminal_bias_v,
        terminal_peak_error_mT=terminal_peak_error,
        terminal_target_slope_sign=target_slope_sign,
        terminal_predicted_slope_sign_before=predicted_slope_sign_before,
        terminal_predicted_slope_sign_after=predicted_slope_sign_after,
        terminal_direction_match_after=terminal_direction_match_after,
        terminal_trim_window_fraction=terminal_trim_window_fraction,
    )


def _set_terminal_trim_metadata(
    command_profile: pd.DataFrame,
    *,
    applied: bool,
    terminal_gain: float = 1.0,
    terminal_bias_v: float = 0.0,
    terminal_peak_error_mT: float = float("nan"),
    terminal_target_slope_sign: float = float("nan"),
    terminal_predicted_slope_sign_before: float = float("nan"),
    terminal_predicted_slope_sign_after: float = float("nan"),
    terminal_direction_match_after: bool = False,
    terminal_trim_window_fraction: float = float("nan"),
) -> pd.DataFrame:
    if command_profile.empty:
        return command_profile
    command_profile["terminal_trim_applied"] = bool(applied)
    command_profile["terminal_trim_gain"] = float(terminal_gain)
    command_profile["terminal_trim_bias_v"] = float(terminal_bias_v)
    command_profile["predicted_terminal_peak_error_mT"] = float(terminal_peak_error_mT)
    command_profile["terminal_target_slope_sign"] = float(terminal_target_slope_sign)
    command_profile["terminal_predicted_slope_sign_before"] = float(terminal_predicted_slope_sign_before)
    command_profile["terminal_predicted_slope_sign_after"] = float(terminal_predicted_slope_sign_after)
    command_profile["terminal_direction_match_after"] = bool(terminal_direction_match_after)
    command_profile["terminal_trim_window_fraction"] = float(terminal_trim_window_fraction)
    return command_profile


def _sample_theoretical_output(
    waveform_type: str,
    phase_total: np.ndarray,
    active_cycle_count: float,
    force_rounded_triangle: bool = False,
) -> np.ndarray:
    phases = np.asarray(phase_total, dtype=float)
    active_mask = (phases >= 0.0) & (phases <= float(active_cycle_count) + 1e-12)
    phase_in_cycle = np.mod(phases, 1.0)
    template = _build_target_template(
        waveform_type=waveform_type,
        freq_hz=1.0,
        points_per_cycle=2048,
        force_rounded_triangle=force_rounded_triangle,
    ).rename(columns={"target_output_normalized": "voltage_normalized"})
    sampled = np.interp(
        phase_in_cycle,
        template["cycle_progress"].to_numpy(dtype=float),
        template["voltage_normalized"].to_numpy(dtype=float),
    )
    sampled[~active_mask] = 0.0
    return sampled


def _compare_signal_to_ideal(
    profile: pd.DataFrame,
    signal_column: str,
    std_column: str,
    waveform_type: str,
    freq_hz: float,
    amplitude_pp: float,
    output_prefix: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if signal_column not in profile.columns:
        return pd.DataFrame(), {}

    working = profile[["cycle_progress", "time_s", signal_column]].copy()
    if std_column in profile.columns:
        working[std_column] = profile[std_column]

    signal = pd.to_numeric(working[signal_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(signal).any():
        return pd.DataFrame(), {}

    actual_mean = float(np.nanmean(signal))
    actual_centered = signal - actual_mean
    actual_pp = float(np.nanmax(actual_centered) - np.nanmin(actual_centered)) if np.isfinite(actual_centered).any() else float("nan")
    target_pp = amplitude_pp if np.isfinite(amplitude_pp) and amplitude_pp > 0 else actual_pp

    ideal = _theoretical_template(
        waveform_type=waveform_type,
        freq_hz=freq_hz if np.isfinite(freq_hz) and freq_hz > 0 else None,
        points_per_cycle=len(working),
    )["voltage_normalized"].to_numpy(dtype=float)
    ideal_signal = ideal * target_pp / 2.0

    error = actual_centered - ideal_signal
    amplitude_scale = actual_pp / target_pp if np.isfinite(target_pp) and target_pp != 0 else float("nan")
    rmse = float(np.sqrt(np.nanmean(np.square(error)))) if np.isfinite(error).any() else float("nan")
    mae = float(np.nanmean(np.abs(error))) if np.isfinite(error).any() else float("nan")
    nrmse = rmse / max(target_pp / 2.0, 1e-12) if np.isfinite(rmse) and np.isfinite(target_pp) and target_pp > 0 else float("nan")
    corr = _safe_corr(actual_centered, ideal_signal)
    lag_samples = _best_phase_lag(actual_centered, ideal_signal)
    lag_fraction = lag_samples / max(len(actual_centered), 1)
    lag_seconds = lag_fraction * float(working["time_s"].max()) if "time_s" in working.columns and len(working) > 1 else float("nan")

    detail = working.rename(columns={signal_column: f"{output_prefix}_actual"})
    if std_column in working.columns:
        detail = detail.rename(columns={std_column: f"{output_prefix}_std"})
    detail[f"{output_prefix}_target"] = ideal_signal
    detail[f"{output_prefix}_error"] = error
    detail[f"{output_prefix}_actual_centered"] = actual_centered

    metrics = {
        f"{output_prefix}_shape_rmse": rmse,
        f"{output_prefix}_shape_mae": mae,
        f"{output_prefix}_shape_nrmse": nrmse,
        f"{output_prefix}_shape_corr": corr,
        f"{output_prefix}_dc_offset": actual_mean,
        f"{output_prefix}_pp_actual": actual_pp,
        f"{output_prefix}_pp_target": target_pp,
        f"{output_prefix}_pp_ratio": amplitude_scale,
        f"{output_prefix}_phase_lag_samples": float(lag_samples),
        f"{output_prefix}_phase_lag_fraction": float(lag_fraction),
        f"{output_prefix}_phase_lag_seconds": float(lag_seconds),
    }
    return detail, metrics


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return float("nan")
    left_valid = left[valid]
    right_valid = right[valid]
    if np.allclose(np.nanstd(left_valid), 0.0) or np.allclose(np.nanstd(right_valid), 0.0):
        return float("nan")
    return float(np.corrcoef(left_valid, right_valid)[0, 1])


def _best_phase_lag(actual: np.ndarray, ideal: np.ndarray) -> int:
    valid = np.isfinite(actual) & np.isfinite(ideal)
    if valid.sum() < 3:
        return 0
    actual_valid = actual[valid]
    ideal_valid = ideal[valid]
    best_shift = 0
    best_score = -np.inf
    max_shift = max(1, len(actual_valid) // 8)
    for shift in range(-max_shift, max_shift + 1):
        shifted = np.roll(ideal_valid, shift)
        score = _safe_corr(actual_valid, shifted)
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_shift = shift
    return int(best_shift)


def _interpolate_complex_transfer_surface(
    frequencies: np.ndarray,
    levels: np.ndarray,
    transfers: np.ndarray,
    target_frequency: float,
    target_level: float,
    allow_level_extrapolation: bool,
) -> complex:
    if len(levels) == 0:
        return complex(np.nan, np.nan)
    valid = (
        np.isfinite(frequencies)
        & np.isfinite(levels)
        & np.isfinite(transfers.real)
        & np.isfinite(transfers.imag)
    )
    frequencies = frequencies[valid]
    levels = levels[valid]
    transfers = transfers[valid]
    if len(levels) == 0:
        return complex(np.nan, np.nan)
    if len(levels) == 1:
        return complex(transfers[0])

    used_frequency = float(np.clip(target_frequency, float(np.nanmin(frequencies)), float(np.nanmax(frequencies))))
    used_level = (
        float(target_level)
        if allow_level_extrapolation
        else float(np.clip(target_level, float(np.nanmin(levels)), float(np.nanmax(levels))))
    )

    freq_span = max(float(np.nanmax(frequencies)) - float(np.nanmin(frequencies)), 1e-9)
    level_span = max(float(np.nanmax(levels)) - float(np.nanmin(levels)), 1e-9)
    distances = np.sqrt(
        np.square((frequencies - used_frequency) / freq_span)
        + np.square((levels - used_level) / level_span)
    )
    exact_mask = distances <= 1e-9
    if exact_mask.any():
        return complex(transfers[exact_mask][0])

    nearest_count = min(6, len(distances))
    order = np.argsort(distances)[:nearest_count]
    weights = 1.0 / (np.square(distances[order]) + 1e-6)
    weighted_real = float(np.sum(weights * transfers[order].real) / np.sum(weights))
    weighted_imag = float(np.sum(weights * transfers[order].imag) / np.sum(weights))
    return complex(weighted_real, weighted_imag)


def _phase_interpolated_compensation(
    support_profiles: list[dict[str, Any]],
    target_profile: pd.DataFrame,
    fallback_profile: pd.DataFrame,
    fallback_scale: float,
    points_per_cycle: int,
) -> tuple[pd.DataFrame, float]:
    recommended_voltage: list[float] = []
    used_target_current: list[float] = []
    clamped_count = 0

    for phase_index in range(points_per_cycle):
        current_points: list[float] = []
        voltage_points: list[float] = []
        for support in support_profiles:
            profile = support["profile"]
            if phase_index >= len(profile):
                continue
            current_value = profile["measured_current_a"].iloc[phase_index]
            voltage_value = profile["command_voltage_v"].iloc[phase_index]
            if not np.isfinite(current_value) or not np.isfinite(voltage_value):
                continue
            current_points.append(float(current_value))
            voltage_points.append(float(voltage_value))

        target_current_value = float(target_profile["target_current_a"].iloc[phase_index])
        if len(current_points) < 2:
            recommended_voltage.append(float(fallback_profile["command_voltage_v"].iloc[phase_index]) * fallback_scale)
            used_target_current.append(target_current_value)
            continue

        phase_frame = pd.DataFrame(
            {
                "current_value": current_points,
                "voltage_value": voltage_points,
            }
        ).sort_values("current_value")
        phase_frame = phase_frame.loc[phase_frame["current_value"].diff().fillna(1.0).ne(0.0)]
        if len(phase_frame) < 2:
            recommended_voltage.append(float(fallback_profile["command_voltage_v"].iloc[phase_index]) * fallback_scale)
            used_target_current.append(target_current_value)
            continue

        min_current = float(phase_frame["current_value"].min())
        max_current = float(phase_frame["current_value"].max())
        used_current = float(np.clip(target_current_value, min_current, max_current))
        if abs(used_current - target_current_value) > 1e-9:
            clamped_count += 1
        recommended_voltage.append(
            float(
                np.interp(
                    used_current,
                    phase_frame["current_value"].to_numpy(dtype=float),
                    phase_frame["voltage_value"].to_numpy(dtype=float),
                )
            )
        )
        used_target_current.append(used_current)

    command = target_profile[["cycle_progress", "time_s", "target_current_a"]].copy()
    command["used_target_current_a"] = used_target_current
    command["recommended_voltage_v"] = recommended_voltage
    clamp_fraction = clamped_count / max(points_per_cycle, 1)
    return command, clamp_fraction
