from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .hardware import apply_command_hardware_model
from .lcr import build_lcr_harmonic_prior, build_lcr_impedance_table
from .lut import _theoretical_template
from .models import DatasetAnalysis, ParsedMeasurement, PreprocessResult
from .recommendation_lcr_runtime import resolve_lcr_runtime_policy
from .utils import canonicalize_waveform_type


def _default_harmonic_count(waveform_type: str, points_per_cycle: int) -> int:
    max_available = max(int(points_per_cycle) // 2 - 1, 1)
    waveform_type = canonicalize_waveform_type(waveform_type) or "sine"
    if waveform_type == "triangle":
        return max(3, min(max_available, 31))
    return max(1, min(max_available, 11))


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

    output_context = _resolve_output_context(
        target_output_type=target_output_type,
        field_channel=field_channel,
        current_metric=current_metric,
    )
    target_output_pp = float(target_output_pp if target_output_pp is not None else target_current_pp_a)
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
    preview_tail_cycles = max(float(preview_tail_cycles), 0.0)

    target_profile = _theoretical_template(
        waveform_type=waveform_type,
        freq_hz=requested_freq_hz,
        points_per_cycle=points_per_cycle,
    ).rename(columns={"voltage_normalized": "target_output_normalized"})
    target_profile["target_output"] = target_profile["target_output_normalized"] * target_output_pp / 2.0
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
    support_amp_gain_pct = float(nearest_row.get("amp_gain_setting_mean", np.nan))
    if not np.isfinite(support_amp_gain_pct) or support_amp_gain_pct <= 0:
        support_amp_gain_pct = float(default_support_amp_gain_pct)
    output_scale = 1.0
    if target_output_type == "field":
        nearest_current_pp = float(nearest_row.get(current_metric, np.nan))
        nearest_output_pp = float(nearest_row.get(output_context["output_metric"], np.nan))
        if np.isfinite(nearest_current_pp) and nearest_current_pp > 0 and np.isfinite(nearest_output_pp):
            output_scale = nearest_output_pp / nearest_current_pp
    lcr_policy = resolve_lcr_runtime_policy(
        requested_lcr_weight=lcr_blend_weight,
        lcr_prior_available=bool(lcr_measurements is not None and not lcr_measurements.empty),
        exact_field_support_present=bool(output_context["output_type"] == "field" and not exact_freq_subset.empty),
        support_point_count=len(support_profiles),
        waveform_type=waveform_type,
        official_band_applied=bool(requested_freq_hz <= 5.0),
    )
    lcr_prior_table = pd.DataFrame()
    if lcr_measurements is not None and not lcr_measurements.empty and float(lcr_policy["lcr_weight"]) > 0.0:
        lcr_prior_table = build_lcr_harmonic_prior(
            lcr_impedance_table=build_lcr_impedance_table(lcr_measurements),
            base_freq_hz=used_freq_hz,
            harmonics=range(1, max_harmonics + 1),
            daq_to_amp_gain=float(amp_gain_at_100_pct) * float(support_amp_gain_pct) / 100.0,
            output_scale=output_scale,
        )
    lcr_prior_used = bool(not lcr_prior_table.empty and float(lcr_policy["lcr_weight"]) > 0.0)

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

    estimated_output_lag_seconds = _estimate_weighted_output_lag_seconds(
        support_profiles=support_profiles,
        output_signal_column=output_context["signal_column"],
        output_metric=output_context["output_metric"],
        target_freq_hz=used_freq_hz,
        target_output_pp=target_output_pp,
    )
    period_s = 1.0 / requested_freq_hz if requested_freq_hz > 0 else 1.0
    max_reasonable_lag = 0.45 * period_s
    estimated_output_lag_seconds = float(
        np.clip(estimated_output_lag_seconds, -max_reasonable_lag, max_reasonable_lag)
    )

    if finite_cycle_mode and target_cycle_count is not None:
        command_profile = _expand_command_profile_to_finite_run(
            command_cycle_profile=command_profile,
            waveform_type=waveform_type,
            freq_hz=requested_freq_hz,
            target_output_pp=target_output_pp,
            target_cycle_count=target_cycle_count,
            preview_tail_cycles=preview_tail_cycles,
            output_context=output_context,
            phase_lead_seconds=estimated_output_lag_seconds,
            points_per_cycle=points_per_cycle,
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

    command_profile["target_output_pp"] = target_output_pp
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
    if "freq_distance_hz" in support_table.columns and "output_distance" in support_table.columns:
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

    nearest_profile = nearest_support["profile"].copy()
    nearest_profile["nearest_test_id"] = nearest_test_id
    nearest_profile["nearest_current_pp_a"] = float(nearest_row.get(current_metric, np.nan))
    nearest_profile["nearest_output_pp"] = float(nearest_row.get(output_context["output_metric"], np.nan))
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
    nearest_cycle_selection = nearest_support.get("cycle_selection", {})
    startup_diagnostics = dict(nearest_support.get("startup_diagnostics", {}))
    startup_diagnostics.setdefault("source_test_id", nearest_test_id)
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
        "target_current_pp_a": float(target_current_pp_a),
        "available_output_pp_min": support_min,
        "available_output_pp_max": support_max,
        "support_point_count": int(len(support_profiles)),
        "nearest_test_id": nearest_test_id,
        "selected_support_id": support_selection_meta.get("selected_support_id") or nearest_test_id,
        "selected_support_family": support_selection_meta.get("selected_support_family"),
        "support_selection_reason": support_selection_meta.get("support_selection_reason"),
        "support_family_metric": support_selection_meta.get("support_family_metric"),
        "support_family_value": support_selection_meta.get("support_family_value"),
        "estimated_family_level": support_selection_meta.get("estimated_family_level"),
        "support_family_lock_applied": bool(support_selection_meta.get("support_family_lock_applied", False)),
        "support_bz_to_current_ratio": support_selection_meta.get("support_bz_to_current_ratio"),
        "nearest_support_output_pp": float(nearest_row[output_context["output_metric"]]),
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
        requested_cycle_count = _first_numeric(parsed.metadata.get("cycle"))
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
                "waveform_type": canonicalize_waveform_type(parsed.metadata.get("waveform")),
                "freq_hz": freq_hz,
                "duration_s": duration_s,
                "approx_cycle_span": approx_cycle_span,
                "estimated_cycle_span": estimated_cycle_span,
                "requested_cycle_count": requested_cycle_count,
                "target_current_a": _first_numeric(parsed.metadata.get("Target Current(A)")),
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
    if active_frame.empty:
        active_frame = full_frame.copy()
    if active_frame.empty or full_frame.empty or "daq_input_v" not in full_frame.columns:
        return None

    active_start_s = float(active_frame["time_s"].min())
    active_end_s = float(active_frame["time_s"].max())
    active_support_duration_s = max(active_end_s - active_start_s, 1e-9)
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
    voltage_active = _interpolate_finite_signal(
        active_time_rel,
        pd.to_numeric(active_frame["daq_input_v"], errors="coerce").to_numpy(dtype=float),
        source_active_rel,
    )
    current_active = _interpolate_finite_signal(
        active_time_rel,
        pd.to_numeric(active_frame[current_column], errors="coerce").to_numpy(dtype=float)
        if current_column in active_frame.columns
        else np.zeros(len(active_frame), dtype=float),
        source_active_rel,
    )
    field_active = _interpolate_finite_signal(
        active_time_rel,
        pd.to_numeric(active_frame[field_column], errors="coerce").to_numpy(dtype=float)
        if field_column in active_frame.columns
        else np.zeros(len(active_frame), dtype=float),
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
            tail_current = _interpolate_finite_signal(
                tail_time_rel,
                pd.to_numeric(tail_frame[current_tail_column], errors="coerce").to_numpy(dtype=float)
                if current_tail_column in tail_frame.columns
                else np.zeros(len(tail_frame), dtype=float),
                source_tail_rel,
            )
            tail_field = _interpolate_finite_signal(
                tail_time_rel,
                pd.to_numeric(tail_frame[field_tail_column], errors="coerce").to_numpy(dtype=float)
                if field_tail_column in tail_frame.columns
                else np.zeros(len(tail_frame), dtype=float),
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
    if plot_source == "support_blended_preview":
        modeled["support_scaled_current_a"] = modeled["expected_current_a"]
        modeled["support_scaled_field_mT"] = modeled["expected_field_mT"]
    modeled["target_output"] = _finite_target_template(
        time_grid=time_grid,
        waveform_type=waveform_type,
        freq_hz=float(freq_hz),
        target_cycle_count=float(target_cycle_count),
        target_output_pp=float(target_output_pp),
    )
    if target_output_type == "current":
        modeled["target_current_a"] = modeled["target_output"]
        modeled["used_target_current_a"] = modeled["target_output"]
    else:
        modeled["target_field_mT"] = modeled["target_output"]
        modeled["used_target_field_mT"] = modeled["target_output"]
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

    output_column = "field_pp" if target_output_type == "field" else "current_pp"
    target_output_unit = "mT" if target_output_type == "field" else "A"
    harmonic_weights = _finite_harmonic_weights(waveform_type)
    waveform_matches = [
        entry
        for entry in finite_support_entries
        if canonicalize_waveform_type(entry.get("waveform_type")) == waveform_type
    ]
    if not waveform_matches:
        return None

    exact_freq_matches = [
        entry
        for entry in waveform_matches
        if np.isfinite(entry.get("freq_hz", np.nan))
        and abs(float(entry.get("freq_hz", np.nan)) - float(freq_hz)) <= float(freq_match_tolerance_hz)
    ]
    frequency_bucket_mode = "exact_frequency_bucket" if exact_freq_matches else "nearest_frequency_blend"
    frequency_candidates = exact_freq_matches or waveform_matches

    exact_cycle_matches = [
        entry
        for entry in frequency_candidates
        if np.isfinite(entry.get("approx_cycle_span", np.nan))
        and abs(float(entry.get("approx_cycle_span", np.nan)) - float(target_cycle_count)) <= float(cycle_match_tolerance)
    ]
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
        waveform_distance = (
            0.0
            if canonicalize_waveform_type(entry.get("waveform_type")) == waveform_type
            else 5.0
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
                float(distance_score + waveform_distance + shape_mismatch),
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
    modeled = apply_command_hardware_model(
        command_waveform=modeled,
        max_daq_voltage_pp=float(max_daq_voltage_pp),
        amp_gain_at_100_pct=float(amp_gain_at_100_pct),
        support_amp_gain_pct=float(default_support_amp_gain_pct),
        amp_gain_limit_pct=float(amp_gain_limit_pct),
        amp_max_output_pk_v=float(amp_max_output_pk_v),
        preserve_start_voltage=True,
    )
    support_table = pd.DataFrame(support_rows)
    support_tests_used = [str(row["test_id"]) for row in support_rows]

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
    selected_support_family = f"{output_column}:{support_output_pp:g}" if np.isfinite(support_output_pp) else None
    support_selection_reason = "finite_exact_level_match" if request_route == "exact" else (
        "finite_weighted_support_blend" if support_count_used > 1 else "finite_nearest_support_preview"
    )
    support_bz_to_current_ratio = float("nan")
    if np.isfinite(float(support.get("field_pp", np.nan))) and np.isfinite(float(support.get("current_pp", np.nan))):
        support_current_pp = float(support.get("current_pp", np.nan))
        if abs(support_current_pp) > 1e-9:
            support_bz_to_current_ratio = float(float(support.get("field_pp", np.nan)) / support_current_pp)
    field_prediction_source_hint = "exact_field_direct" if plot_source == "exact_prediction" else "support_blended_preview"
    expected_current_source_hint = "exact_current_direct" if plot_source == "exact_prediction" else "support_blended_preview"

    return {
        "mode": "finite_exact_direct" if request_route == "exact" else ("finite_empirical_weighted_support" if support_count_used > 1 else "finite_empirical_nearest_support"),
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
        "support_freq_hz": float(support.get("freq_hz", np.nan)),
        "support_cycle_count": float(support.get("approx_cycle_span", np.nan)),
        "support_output_pp": support_output_pp,
        "scale_ratio": scale_ratio,
        "distance_score": nearest_distance_score,
        "model_confidence": model_confidence,
        "request_route": request_route,
        "plot_source": plot_source,
        "selected_support_waveform": str(canonicalize_waveform_type(support.get("waveform_type")) or support.get("waveform_type") or ""),
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


def _estimate_weighted_output_lag_seconds(
    support_profiles: list[dict[str, Any]],
    output_signal_column: str,
    output_metric: str,
    target_freq_hz: float,
    target_output_pp: float,
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
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else None


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

    def _ratio(numerator: float, denominator: float) -> float:
        if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1e-12:
            return float("nan")
        return float(numerator / denominator)

    current_ratio = _ratio(first_current, steady_current_mean)
    field_ratio = _ratio(first_field, steady_field_mean)
    voltage_ratio = _ratio(first_voltage, steady_voltage_mean)

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
        "behavior_flag": behavior,
    }


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
        values = pd.to_numeric(registered[column], errors="coerce").to_numpy(dtype=float)
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
        voltage_values = pd.to_numeric(registered[voltage_column], errors="coerce").to_numpy(dtype=float)
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


def _finite_target_template(
    time_grid: np.ndarray,
    waveform_type: str,
    freq_hz: float,
    target_cycle_count: float,
    target_output_pp: float,
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
    if waveform_type == "triangle":
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
) -> pd.DataFrame:
    total_cycles = float(target_cycle_count + preview_tail_cycles)
    sample_count = max(int(np.ceil(total_cycles * points_per_cycle)), 2) + 1
    period_s = 1.0 / freq_hz if freq_hz > 0 else 1.0
    time_grid = np.linspace(0.0, total_cycles * period_s, sample_count)
    phase_total = time_grid / period_s
    lookahead_phase_total = phase_total + (phase_lead_seconds / period_s if period_s > 0 else 0.0)

    target_norm = _sample_theoretical_output(
        waveform_type=waveform_type,
        phase_total=phase_total,
        active_cycle_count=target_cycle_count,
    )
    used_target_norm = _sample_theoretical_output(
        waveform_type=waveform_type,
        phase_total=lookahead_phase_total,
        active_cycle_count=target_cycle_count,
    )

    cycle_progress = np.mod(phase_total, 1.0)
    active_mask = phase_total <= target_cycle_count + 1e-12
    lookahead_mask = (lookahead_phase_total >= 0.0) & (lookahead_phase_total <= target_cycle_count + 1e-12)
    command_phase = np.mod(lookahead_phase_total, 1.0)
    cycle_phase_grid = command_cycle_profile["cycle_progress"].to_numpy(dtype=float)
    cycle_voltage = pd.to_numeric(command_cycle_profile["recommended_voltage_v"], errors="coerce").to_numpy(dtype=float)
    command_voltage = np.interp(command_phase, cycle_phase_grid, cycle_voltage)
    command_voltage[~lookahead_mask] = 0.0
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
            np.mod(phase_total if column == output_signal_column else lookahead_phase_total, 1.0),
            profile["cycle_progress"].to_numpy(dtype=float),
            values,
        )
        sampled[~(active_mask if column == output_signal_column else lookahead_mask)] = 0.0
        if column == "command_voltage_v":
            sampled = _apply_zero_start_envelope(
                values=sampled,
                time_grid=time_grid,
                startup_duration_s=max(abs(float(phase_lead_seconds)), period_s / max(points_per_cycle / 4.0, 1.0)),
            )
        preview[column] = sampled
    preview["waveform_type"] = waveform_type
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


def _sample_theoretical_output(
    waveform_type: str,
    phase_total: np.ndarray,
    active_cycle_count: float,
) -> np.ndarray:
    phases = np.asarray(phase_total, dtype=float)
    active_mask = (phases >= 0.0) & (phases <= float(active_cycle_count) + 1e-12)
    phase_in_cycle = np.mod(phases, 1.0)
    template = _theoretical_template(
        waveform_type=waveform_type,
        freq_hz=1.0,
        points_per_cycle=2048,
    )
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
