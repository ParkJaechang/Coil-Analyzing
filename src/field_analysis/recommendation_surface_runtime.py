from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .compensation import (
    _build_weighted_support_profile_preview,
    _estimate_weighted_output_lag_seconds,
    _phase_register_command_profile,
    _resolve_output_context,
    _select_nearest_support_row,
    _sync_modeled_alias_columns,
)
from .hardware import apply_command_hardware_model
from .lcr import build_lcr_harmonic_prior, build_lcr_impedance_table
from .lut import _theoretical_template
from .plant_model.base import ModelContext
from .plant_model.harmonic_surface import (
    HarmonicSurfaceModel,
    build_harmonic_observation_frame,
    harmonic_cap,
)
from .recommendation_constants import OFFICIAL_OPERATION_MAX_FREQ_HZ
from .recommendation_lcr_runtime import resolve_lcr_runtime_policy
from .recommendation_models import LegacyRecommendationContext, RecommendationOptions, TargetRequest
from .recommendation_route_selection import _classify_support_state, _summarize_frequency_geometry
from .recommendation_shape_metrics import _build_surface_confidence_summary
from .recommendation_surface_support import (
    _attach_support_scaled_preview,
    _build_surface_support_profiles,
    _build_surface_support_table,
    _estimate_surface_sample_rate_hz,
)
from .utils import canonicalize_waveform_type
from .validation import ValidationReport


def _build_lcr_current_prior_lookup(
    *,
    lcr_measurements: pd.DataFrame | None,
    requested_freq_hz: float,
    max_harmonics: int,
) -> tuple[dict[int, dict[str, Any]], pd.DataFrame]:
    if lcr_measurements is None or lcr_measurements.empty:
        return {}, pd.DataFrame()
    impedance_table = build_lcr_impedance_table(lcr_measurements)
    if impedance_table.empty:
        return {}, pd.DataFrame()
    prior_table = build_lcr_harmonic_prior(
        lcr_impedance_table=impedance_table,
        base_freq_hz=float(requested_freq_hz),
        harmonics=range(1, int(max_harmonics) + 1),
    )
    if prior_table.empty:
        return {}, pd.DataFrame()
    lookup = {
        int(row["harmonic"]): dict(row)
        for row in prior_table.to_dict(orient="records")
        if pd.notna(row.get("harmonic"))
    }
    return lookup, prior_table


def _signal_peak_to_peak_array(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.nanmax(finite) - np.nanmin(finite))


def _resolve_exact_field_prediction_hierarchy(
    *,
    command_profile: pd.DataFrame,
    exact_frequency_match: bool,
    target_output_type: str,
    support_selection_meta: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    hierarchy = ["exact_field_direct", "current_to_bz_surrogate", "unavailable"]
    if str(target_output_type) != "field":
        return command_profile, {
            "field_prediction_hierarchy": hierarchy,
            "field_prediction_status": "available",
            "field_prediction_source_hint": None,
            "field_prediction_unavailable_reason": None,
            "field_prediction_fallback_reason": None,
            "exact_field_direct_available": None,
            "exact_field_direct_reason": None,
            "same_recipe_surrogate_candidate_available": None,
            "same_recipe_surrogate_applied": None,
            "same_recipe_surrogate_ratio": None,
            "surrogate_scope": None,
        }

    updated = command_profile.copy()
    direct_field = (
        pd.to_numeric(updated["expected_field_mT"], errors="coerce").to_numpy(dtype=float)
        if "expected_field_mT" in updated.columns
        else np.array([], dtype=float)
    )
    if direct_field.size == len(updated):
        updated["exact_field_direct_mT"] = direct_field
    target_field = (
        pd.to_numeric(updated["target_field_mT"], errors="coerce").to_numpy(dtype=float)
        if "target_field_mT" in updated.columns
        else np.array([], dtype=float)
    )
    expected_current = (
        pd.to_numeric(updated["expected_current_a"], errors="coerce").to_numpy(dtype=float)
        if "expected_current_a" in updated.columns
        else np.array([], dtype=float)
    )
    direct_field_pp = _signal_peak_to_peak_array(direct_field)
    target_field_pp = _signal_peak_to_peak_array(target_field)
    direct_amplitude_ratio = (
        float(direct_field_pp / target_field_pp)
        if np.isfinite(direct_field_pp) and np.isfinite(target_field_pp) and float(target_field_pp) > 1e-9
        else float("nan")
    )
    direct_reason: str | None = None
    if exact_frequency_match:
        if direct_field.size == 0 or not np.isfinite(direct_field).any():
            direct_reason = "missing_expected_field_prediction"
        elif not np.isfinite(direct_field_pp) or direct_field_pp <= 1e-6:
            direct_reason = "expected_field_near_zero"
        elif np.isfinite(direct_amplitude_ratio) and direct_amplitude_ratio < 0.02:
            direct_reason = "expected_field_collapse"

    support_ratio = support_selection_meta.get("support_bz_to_current_ratio")
    surrogate_candidate_available = bool(
        exact_frequency_match
        and expected_current.size == len(updated)
        and np.isfinite(expected_current).any()
        and support_ratio is not None
        and np.isfinite(float(support_ratio))
        and float(support_ratio) > 1e-9
    )
    metadata: dict[str, Any] = {
        "field_prediction_hierarchy": hierarchy,
        "field_prediction_status": "available",
        "field_prediction_source_hint": "exact_field_direct" if exact_frequency_match else None,
        "field_prediction_unavailable_reason": None,
        "field_prediction_fallback_reason": None,
        "exact_field_direct_available": direct_reason is None if exact_frequency_match else None,
        "exact_field_direct_reason": direct_reason,
        "same_recipe_surrogate_candidate_available": surrogate_candidate_available,
        "same_recipe_surrogate_applied": False,
        "same_recipe_surrogate_ratio": float(support_ratio) if surrogate_candidate_available else None,
        "surrogate_scope": None,
    }
    if not exact_frequency_match:
        return updated, metadata
    if direct_reason is None:
        metadata["field_prediction_source_hint"] = "exact_field_direct"
        return updated, metadata
    if surrogate_candidate_available:
        surrogate = expected_current * float(support_ratio)
        surrogate_pp = _signal_peak_to_peak_array(surrogate)
        if np.isfinite(surrogate_pp) and surrogate_pp > 1e-6:
            updated["expected_field_mT"] = surrogate
            updated["expected_output"] = surrogate
            updated["same_recipe_surrogate_field_mT"] = surrogate
            metadata["field_prediction_source_hint"] = "current_to_bz_surrogate"
            metadata["field_prediction_status"] = "available"
            metadata["field_prediction_fallback_reason"] = direct_reason
            metadata["same_recipe_surrogate_applied"] = True
            metadata["surrogate_scope"] = "same_recipe_validated_exact_support"
            return updated, metadata
    updated["expected_field_mT"] = np.full(len(updated), np.nan, dtype=float)
    updated["expected_output"] = np.full(len(updated), np.nan, dtype=float)
    metadata["field_prediction_source_hint"] = "exact_field_direct"
    metadata["field_prediction_status"] = "unavailable"
    metadata["field_prediction_unavailable_reason"] = direct_reason or "field_prediction_unavailable"
    return updated, metadata


def _recommend_continuous_harmonic_surface(
    *,
    continuous_runs: list[Any],
    target: TargetRequest,
    options: RecommendationOptions,
    legacy_context: LegacyRecommendationContext,
    validation_report: ValidationReport,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Use the PR4 harmonic-surface model for continuous steady-state recommendations."""

    if target.regime != "continuous":
        return None, {"reason": "not_continuous"}
    if bool(target.context.get("finite_cycle_mode", False)):
        return None, {"reason": "finite_cycle_mode_uses_legacy"}
    if str(target.context.get("request_kind", "waveform_compensation")) != "waveform_compensation":
        return None, {"reason": "unsupported_request_kind"}
    if target.freq_hz is None or not np.isfinite(target.freq_hz) or target.freq_hz <= 0:
        return None, {"reason": "missing_frequency"}
    if target.target_level_value is None or not np.isfinite(target.target_level_value):
        return None, {"reason": "missing_target_level"}
    if legacy_context.per_test_summary.empty or not legacy_context.analysis_lookup:
        return None, {"reason": "missing_legacy_support_context"}

    waveform_type = canonicalize_waveform_type(target.command_waveform or target.target_waveform)
    if waveform_type is None:
        return None, {"reason": "unsupported_waveform"}

    requested_frequency_mode = str(target.context.get("frequency_mode", options.frequency_mode))
    interpolation_requested = requested_frequency_mode != "exact"
    exact_frequency_match = bool(validation_report.exact_freq_match)

    output_context = _resolve_output_context(
        target_output_type=target.target_type,
        field_channel=options.field_channel,
        current_metric="achieved_current_pp_a_mean",
    )
    target_output_type = output_context["output_type"]
    requested_freq_hz = float(target.freq_hz)
    target_output_pp = float(target.target_level_value)
    points_per_cycle = 256
    official_band_applied = bool(
        np.isfinite(requested_freq_hz) and requested_freq_hz <= float(OFFICIAL_OPERATION_MAX_FREQ_HZ)
    )

    waveform_subset = legacy_context.per_test_summary[
        legacy_context.per_test_summary["waveform_type"].map(canonicalize_waveform_type) == waveform_type
    ].copy()
    waveform_subset = waveform_subset.dropna(subset=["freq_hz", output_context["output_metric"]]).copy()
    if official_band_applied:
        waveform_subset = waveform_subset[
            pd.to_numeric(waveform_subset["freq_hz"], errors="coerce") <= float(OFFICIAL_OPERATION_MAX_FREQ_HZ)
        ].copy()
    if waveform_subset.empty:
        return None, {"reason": "no_waveform_subset"}
    support_state = _classify_support_state(
        waveform_subset["freq_hz"].tolist(),
        requested_freq_hz,
        exact_frequency_match=exact_frequency_match,
    )
    frequency_geometry = _summarize_frequency_geometry(
        waveform_subset["freq_hz"].tolist(),
        requested_freq_hz,
        exact_frequency_match=exact_frequency_match,
    )
    if not exact_frequency_match and not interpolation_requested:
        return None, {
            "reason": "exact_frequency_support_required_for_exact_request",
            "support_state": support_state,
        }

    exact_subset = waveform_subset[
        np.isclose(waveform_subset["freq_hz"], requested_freq_hz, atol=1e-6, equal_nan=False)
    ].copy()
    support_subset = exact_subset.copy() if not exact_subset.empty else waveform_subset.copy()
    if support_subset.empty:
        return None, {"reason": "no_surface_support_subset", "support_state": support_state}
    frequency_bucket_mode = "exact_frequency_bucket" if exact_frequency_match else "interpolated_frequency_preview"

    support_profiles = _build_surface_support_profiles(
        subset=support_subset,
        analysis_lookup=legacy_context.analysis_lookup,
        current_channel=options.current_channel,
        field_channel=options.field_channel,
        points_per_cycle=points_per_cycle,
    )
    if not support_profiles:
        return None, {"reason": "no_support_profiles"}

    nearest_row, support_selection_meta = _select_nearest_support_row(
        subset=support_subset,
        target_freq_hz=requested_freq_hz,
        target_output_pp=target_output_pp,
        output_metric=output_context["output_metric"],
        prefer_level_stable_family=bool(exact_frequency_match and target_output_type == "field"),
    )
    nearest_test_id = str(nearest_row["test_id"])
    nearest_support = next(
        (item for item in support_profiles if str(item["meta"]["test_id"]) == nearest_test_id),
        support_profiles[0],
    )

    support_amp_gain_pct = float(nearest_row.get("amp_gain_setting_mean", np.nan))
    if not np.isfinite(support_amp_gain_pct) or support_amp_gain_pct <= 0:
        support_amp_gain_pct = float(options.default_support_amp_gain_pct)

    support_sample_rate_hz = _estimate_surface_sample_rate_hz(
        continuous_runs=continuous_runs,
        waveform_type=waveform_type,
        freq_hz=requested_freq_hz,
        fallback_hz=float(points_per_cycle) * requested_freq_hz,
    )
    user_harmonic_cap = 31 if waveform_type == "triangle" else 11
    max_harmonics = harmonic_cap(
        sample_rate_hz=support_sample_rate_hz,
        fundamental_freq_hz=requested_freq_hz,
        user_cap=user_harmonic_cap,
    )
    lcr_prior_lookup: dict[int, dict[str, Any]] = {}
    lcr_prior_table = pd.DataFrame()
    lcr_impedance_table = pd.DataFrame()
    lcr_policy = resolve_lcr_runtime_policy(
        requested_lcr_weight=options.lcr_blend_weight,
        lcr_prior_available=bool(options.lcr_measurements is not None and not options.lcr_measurements.empty),
        exact_field_support_present=bool(target_output_type == "field" and exact_frequency_match),
        support_point_count=len(support_profiles),
        waveform_type=waveform_type,
        official_band_applied=official_band_applied,
    )
    use_lcr_prior = bool(lcr_policy["lcr_weight"] > 0.0)
    if use_lcr_prior:
        lcr_impedance_table = build_lcr_impedance_table(options.lcr_measurements)
        lcr_prior_lookup, lcr_prior_table = _build_lcr_current_prior_lookup(
            lcr_measurements=options.lcr_measurements,
            requested_freq_hz=requested_freq_hz,
            max_harmonics=max_harmonics,
        )
        use_lcr_prior = bool(lcr_prior_lookup) and not lcr_impedance_table.empty
        if not use_lcr_prior:
            lcr_policy = resolve_lcr_runtime_policy(
                requested_lcr_weight=0.0,
                lcr_prior_available=False,
                exact_field_support_present=bool(target_output_type == "field" and exact_frequency_match),
                support_point_count=len(support_profiles),
                waveform_type=waveform_type,
                official_band_applied=official_band_applied,
            )
    lcr_prior_used = bool(use_lcr_prior and lcr_prior_lookup and not lcr_impedance_table.empty)

    observation_frames: list[pd.DataFrame] = []
    for support in support_profiles:
        profile = support["profile"]
        if profile.empty or "command_voltage_v" not in profile.columns:
            continue
        time_s = pd.to_numeric(profile["time_s"], errors="coerce").to_numpy(dtype=float)
        command_v = pd.to_numeric(profile["command_voltage_v"], errors="coerce").to_numpy(dtype=float)
        current_pp = float(support["meta"].get("achieved_current_pp_a_mean", np.nan))
        field_pp = float(support["meta"].get(output_context["output_metric"], np.nan))
        if "measured_current_a" in profile.columns:
            current_signal = pd.to_numeric(profile["measured_current_a"], errors="coerce").to_numpy(dtype=float)
            observation_frames.append(
                build_harmonic_observation_frame(
                    run_id=str(support["meta"]["test_id"]),
                    waveform_type=waveform_type,
                    freq_hz=float(support["meta"]["freq_hz"]),
                    target_level_value=current_pp,
                    sample_rate_hz=support_sample_rate_hz,
                    reference_axis="current",
                    output_type="current",
                    time_s=time_s,
                    input_v=command_v,
                    output_signal=current_signal,
                    max_harmonics=max_harmonics,
                )
            )
        if "measured_field_mT" in profile.columns:
            field_signal = pd.to_numeric(profile["measured_field_mT"], errors="coerce").to_numpy(dtype=float)
            observation_frames.append(
                build_harmonic_observation_frame(
                    run_id=str(support["meta"]["test_id"]),
                    waveform_type=waveform_type,
                    freq_hz=float(support["meta"]["freq_hz"]),
                    target_level_value=field_pp,
                    sample_rate_hz=support_sample_rate_hz,
                    reference_axis=options.field_channel,
                    output_type="field",
                    time_s=time_s,
                    input_v=command_v,
                    output_signal=field_signal,
                    max_harmonics=max_harmonics,
                )
            )

    observation_frame = pd.concat(observation_frames, ignore_index=True) if observation_frames else pd.DataFrame()
    if observation_frame.empty:
        return None, {"reason": "no_harmonic_observations"}

    model = HarmonicSurfaceModel()
    transfer_frame = model.fit(observation_frame)
    if transfer_frame.empty:
        return None, {"reason": "surface_fit_empty"}

    model_context = ModelContext(
        waveform_type=waveform_type,
        freq_hz=requested_freq_hz,
        target_level_value=target_output_pp,
        target_level_kind=target.target_level_kind,
        commanded_cycles=None,
        metadata={
            "allow_output_extrapolation": options.allow_output_extrapolation,
            "field_channel": options.field_channel,
            "max_harmonics": max_harmonics,
            "exact_frequency_match": exact_frequency_match,
            "interpolation_requested": interpolation_requested,
            "lcr_blend_weight": float(lcr_policy["lcr_weight"]) if use_lcr_prior else 0.0,
            "lcr_current_prior_lookup": lcr_prior_lookup if use_lcr_prior else {},
            "lcr_impedance_table": lcr_impedance_table if use_lcr_prior else pd.DataFrame(),
        },
    )
    surface_support = model.support_summary(
        context=model_context,
        output_type=target_output_type,
        reference_axis="current" if target_output_type == "current" else options.field_channel,
    )
    if surface_support.get("harmonic_count", 0) <= 0:
        return None, {
            "reason": "surface_support_has_no_harmonics",
            "surface_support": surface_support,
        }

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

    inverse_result = model.invert_target(
        target_output=pd.to_numeric(target_profile["target_output"], errors="coerce").to_numpy(dtype=float),
        output_type=target_output_type,
        context=model_context,
        max_harmonics=max_harmonics,
        reference_axis="current" if target_output_type == "current" else options.field_channel,
    )
    if inverse_result is None:
        return None, {
            "reason": "surface_inverse_failed",
            "surface_support": surface_support,
        }
    recommended_voltage, inverse_debug_frame, inverse_meta = inverse_result

    command_profile = target_profile[["cycle_progress", "time_s", "target_output"]].copy()
    command_profile["used_target_output"] = target_profile["used_target_output"]
    for column in ("target_current_a", "used_target_current_a", "target_field_mT", "used_target_field_mT"):
        if column in target_profile.columns:
            command_profile[column] = target_profile[column]
    command_profile["recommended_voltage_v"] = recommended_voltage
    command_profile = apply_command_hardware_model(
        command_waveform=command_profile,
        max_daq_voltage_pp=float(options.max_daq_voltage_pp),
        amp_gain_at_100_pct=float(options.amp_gain_at_100_pct),
        support_amp_gain_pct=support_amp_gain_pct,
        amp_gain_limit_pct=float(options.amp_gain_limit_pct),
        amp_max_output_pk_v=float(options.amp_max_output_pk_v),
        preserve_start_voltage=False,
    )

    prediction = model.predict(
        pd.to_numeric(command_profile["limited_voltage_v"], errors="coerce").to_numpy(dtype=float),
        model_context,
    )
    if prediction.predicted_current_a is not None:
        command_profile["expected_current_a"] = prediction.predicted_current_a
    if prediction.predicted_field_mT is not None:
        command_profile["expected_field_mT"] = prediction.predicted_field_mT
    if target_output_type == "current" and "expected_current_a" in command_profile.columns:
        command_profile["expected_output"] = command_profile["expected_current_a"]
    elif target_output_type == "field" and "expected_field_mT" in command_profile.columns:
        command_profile["expected_output"] = command_profile["expected_field_mT"]
    command_profile, field_prediction_meta = _resolve_exact_field_prediction_hierarchy(
        command_profile=command_profile,
        exact_frequency_match=exact_frequency_match,
        target_output_type=target_output_type,
        support_selection_meta=support_selection_meta,
    )

    request_route = "exact" if exact_frequency_match else "preview"
    plot_source = "exact_prediction" if exact_frequency_match else "support_blended_preview"
    selected_support_waveform = str(
        canonicalize_waveform_type(nearest_support["meta"].get("waveform_type")) or nearest_support["meta"].get("waveform_type") or waveform_type
    )
    support_profile_preview = _build_weighted_support_profile_preview(
        support_profiles=support_profiles,
        target_freq_hz=requested_freq_hz,
        target_output_pp=target_output_pp,
        output_metric=output_context["output_metric"],
        points_per_cycle=points_per_cycle,
        output_freq_hz=requested_freq_hz,
    )
    if plot_source == "support_blended_preview":
        _attach_support_scaled_preview(
            command_profile=command_profile,
            support_profile_preview=support_profile_preview,
            target_output_type=target_output_type,
        )
    command_profile = _sync_modeled_alias_columns(command_profile)
    command_profile = _phase_register_command_profile(
        command_profile,
        voltage_column="limited_voltage_v",
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
    command_profile["freq_hz"] = requested_freq_hz
    command_profile["finite_cycle_mode"] = False
    command_profile["target_cycle_count"] = np.nan
    command_profile["preview_tail_cycles"] = np.nan
    command_profile["request_route"] = request_route
    command_profile["plot_source"] = plot_source
    command_profile["selected_support_waveform"] = selected_support_waveform
    command_profile["harmonic_weights_used"] = str({1: 1.0} if waveform_type == "sine" else {1: 1.0, 3: 2.4, 5: 1.8, 7: 1.4})

    estimated_output_lag_seconds = _estimate_weighted_output_lag_seconds(
        support_profiles=support_profiles,
        output_signal_column=output_context["signal_column"],
        output_metric=output_context["output_metric"],
        target_freq_hz=requested_freq_hz,
        target_output_pp=target_output_pp,
    )
    command_profile["estimated_output_lag_seconds"] = estimated_output_lag_seconds
    command_profile["estimated_output_lag_cycles"] = estimated_output_lag_seconds * requested_freq_hz

    available_output_pp_min = float(support_subset[output_context["output_metric"]].min())
    available_output_pp_max = float(support_subset[output_context["output_metric"]].max())
    support_table = _build_surface_support_table(
        subset=support_subset,
        target_output_type=target_output_type,
        output_metric=output_context["output_metric"],
        target_output_pp=target_output_pp,
        target_freq_hz=requested_freq_hz,
    )
    confidence_summary = _build_surface_confidence_summary(
        exact_frequency_match=exact_frequency_match,
        interpolation_requested=interpolation_requested,
        support_run_count=len(support_profiles),
        harmonic_cap_value=max_harmonics,
        harmonics_used=int(inverse_meta.get("usable_harmonic_count", 0)),
        phase_clamp_fraction=float(inverse_meta.get("phase_clamp_fraction", 0.0)),
        validation_report=validation_report,
        command_profile=command_profile,
        frequency_geometry=frequency_geometry,
        inverse_debug_frame=inverse_debug_frame,
    )

    payload = {
        "mode": "harmonic_surface_inverse_exact" if exact_frequency_match else "harmonic_surface_inverse_interpolated_preview",
        "frequency_mode": "exact" if exact_frequency_match else "frequency_interpolated",
        "frequency_bucket_mode": frequency_bucket_mode,
        "target_output_type": target_output_type,
        "target_output_label": output_context["label"],
        "target_output_unit": output_context["unit"],
        "target_output_pp": target_output_pp,
        "requested_freq_hz": requested_freq_hz,
        "used_freq_hz": requested_freq_hz,
        "official_support_band_applied": official_band_applied,
        "available_freq_min": float(waveform_subset["freq_hz"].min()),
        "available_freq_max": float(waveform_subset["freq_hz"].max()),
        "frequency_support_count": int(waveform_subset["freq_hz"].nunique()),
        "available_output_pp_min": available_output_pp_min,
        "available_output_pp_max": available_output_pp_max,
        "allow_output_extrapolation": bool(options.allow_output_extrapolation),
        "phase_clamp_fraction": float(inverse_meta.get("phase_clamp_fraction", 0.0)),
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
        "nearest_profile": nearest_support["profile"],
        "nearest_profile_preview": nearest_support["profile"],
        "support_profile_preview": support_profile_preview if not support_profile_preview.empty else nearest_support["profile"],
        "support_table": support_table,
        "command_profile": command_profile,
        "lookup_table": transfer_frame,
        "within_hardware_limits": bool(command_profile["within_hardware_limits"].iloc[0]),
        "support_amp_gain_pct": support_amp_gain_pct,
        "required_amp_gain_pct": float(command_profile["required_amp_gain_pct"].iloc[0]),
        "available_amp_gain_pct": float(command_profile["available_amp_gain_pct"].iloc[0]),
        "limited_voltage_pp": float(command_profile["limited_voltage_pp"].iloc[0]),
        "amp_output_pp_at_required": float(command_profile["amp_output_pp_at_required"].iloc[0]),
        "max_daq_voltage_pp": float(options.max_daq_voltage_pp),
        "estimated_output_lag_seconds": float(estimated_output_lag_seconds),
        "estimated_output_lag_cycles": float(estimated_output_lag_seconds * requested_freq_hz),
        "finite_cycle_mode": False,
        "target_cycle_count": np.nan,
        "preview_tail_cycles": np.nan,
        "request_route": request_route,
        "plot_source": plot_source,
        "selected_support_waveform": selected_support_waveform,
        "harmonic_weights_used": {1: 1.0} if waveform_type == "sine" else {1: 1.0, 3: 2.4, 5: 1.8, 7: 1.4},
        "max_harmonics_used": int(inverse_meta.get("max_harmonics_used", max_harmonics)),
        **field_prediction_meta,
        "used_lcr_prior": lcr_prior_used,
        "lcr_blend_weight": float(lcr_policy["lcr_weight"]) if use_lcr_prior else 0.0,
        "requested_lcr_weight": float(lcr_policy["requested_lcr_weight"]),
        "lcr_usage_mode": lcr_policy["lcr_usage_mode"],
        "exact_field_support_present": bool(lcr_policy["exact_field_support_present"]),
        "lcr_phase_anchor_used": bool(lcr_policy["lcr_phase_anchor_used"] and lcr_prior_used),
        "lcr_gain_prior_used": bool(lcr_policy["lcr_gain_prior_used"] and lcr_prior_used),
        "lcr_prior_harmonic_count": int(len(lcr_prior_lookup)),
        "scale_ratio_from_nearest": np.nan,
        "validation_base_profile": command_profile,
        "startup_diagnostics": nearest_support.get("startup_diagnostics", {}),
        "startup_preview_profile": pd.DataFrame(),
        "model_debug_frame": inverse_debug_frame,
        "prediction_basis": "harmonic_surface_model",
        "engine_summary": {
            "selected_engine": "harmonic_surface",
            "rollout_mode": "auto_recommendation" if exact_frequency_match else "preview_only",
            "exact_frequency_match": exact_frequency_match,
            "support_state": support_state,
        },
        "support_summary": surface_support,
        "confidence_summary": confidence_summary,
    }
    return payload, {
        "reason": "ok",
        "support_state": support_state,
        "surface_support": surface_support,
        "frequency_geometry": frequency_geometry,
        "inverse_meta": inverse_meta,
        "support_sample_rate_hz": support_sample_rate_hz,
        "confidence_summary": confidence_summary,
        "used_lcr_prior": lcr_prior_used,
        "lcr_prior_harmonic_count": int(len(lcr_prior_lookup)),
        "lcr_prior_table": lcr_prior_table,
        "lcr_usage_mode": lcr_policy["lcr_usage_mode"],
        "lcr_weight": float(lcr_policy["lcr_weight"]),
        "requested_lcr_weight": float(lcr_policy["requested_lcr_weight"]),
        "exact_field_support_present": bool(lcr_policy["exact_field_support_present"]),
        "lcr_phase_anchor_used": bool(lcr_policy["lcr_phase_anchor_used"] and lcr_prior_used),
        "lcr_gain_prior_used": bool(lcr_policy["lcr_gain_prior_used"] and lcr_prior_used),
    }
