from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .canonical_runs import CanonicalRun
from .compensation import build_representative_cycle_profile, _select_representative_cycle_indices
from .recommendation_shape_metrics import _signal_peak_to_peak
from .utils import canonicalize_waveform_type


def _build_surface_support_profiles(
    *,
    subset: pd.DataFrame,
    analysis_lookup: dict[str, Any],
    current_channel: str,
    field_channel: str,
    points_per_cycle: int,
) -> list[dict[str, Any]]:
    support_profiles: list[dict[str, Any]] = []
    for row in subset.to_dict(orient="records"):
        analysis = analysis_lookup.get(str(row["test_id"]))
        if analysis is None:
            continue
        selection_info = _select_representative_cycle_indices(
            analysis=analysis,
            cycle_selection_mode="warm_tail",
        )
        profile = build_representative_cycle_profile(
            analysis=analysis,
            current_channel=current_channel,
            voltage_channel="daq_input_v",
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
                "startup_diagnostics": {},
            }
        )
    return support_profiles


def _estimate_surface_sample_rate_hz(
    *,
    continuous_runs: list[CanonicalRun],
    waveform_type: str,
    freq_hz: float,
    fallback_hz: float,
) -> float:
    sample_rates = [
        float(run.sample_rate_hz)
        for run in continuous_runs
        if run.sample_rate_hz is not None
        and np.isfinite(run.sample_rate_hz)
        and run.command_waveform is not None
        and (canonicalize_waveform_type(run.command_waveform) or run.command_waveform) == waveform_type
        and run.freq_hz is not None
        and np.isclose(float(run.freq_hz), float(freq_hz), atol=1e-6, equal_nan=False)
    ]
    if sample_rates:
        return float(np.median(sample_rates))
    return float(fallback_hz)


def _build_surface_support_table(
    *,
    subset: pd.DataFrame,
    target_output_type: str,
    output_metric: str,
    target_output_pp: float,
    target_freq_hz: float,
) -> pd.DataFrame:
    columns = [
        column
        for column in dict.fromkeys(
            (
                "test_id",
                "waveform_type",
                "freq_hz",
                "current_pp_target_a",
                output_metric,
                "achieved_current_pp_a_mean",
                "daq_input_v_pp_mean",
                "amp_gain_setting_mean",
                "achieved_bz_mT_pp_mean",
                "achieved_bmag_mT_pp_mean",
            )
        )
        if column in subset.columns
    ]
    table = subset[columns].copy().reset_index(drop=True)
    if "freq_hz" in table.columns:
        table["freq_distance_hz"] = (table["freq_hz"] - float(target_freq_hz)).abs()
    else:
        table["freq_distance_hz"] = np.nan
    if output_metric in table.columns:
        table["output_distance"] = (table[output_metric] - float(target_output_pp)).abs()
    else:
        table["output_distance"] = np.nan
    if "freq_distance_hz" in table.columns and "output_distance" in table.columns:
        table = table.sort_values(["freq_distance_hz", "output_distance", "test_id"]).reset_index(drop=True)
    return table


def _attach_support_scaled_preview(
    *,
    command_profile: pd.DataFrame,
    support_profile_preview: pd.DataFrame,
    target_output_type: str,
) -> None:
    if command_profile.empty or support_profile_preview.empty or "time_s" not in command_profile.columns:
        return
    if "time_s" not in support_profile_preview.columns:
        return

    used_target_pp = _signal_peak_to_peak(
        command_profile,
        "used_target_output" if "used_target_output" in command_profile.columns else "target_output",
    )
    preview_output_column = "measured_current_a" if target_output_type == "current" else "measured_field_mT"
    preview_output_pp = _signal_peak_to_peak(support_profile_preview, preview_output_column)
    scale_ratio = (
        float(used_target_pp / preview_output_pp)
        if np.isfinite(used_target_pp) and np.isfinite(preview_output_pp) and preview_output_pp > 0
        else 1.0
    )

    target_time = pd.to_numeric(command_profile["time_s"], errors="coerce").to_numpy(dtype=float)
    preview_time = pd.to_numeric(support_profile_preview["time_s"], errors="coerce").to_numpy(dtype=float)
    if "measured_current_a" in support_profile_preview.columns:
        current_values = pd.to_numeric(support_profile_preview["measured_current_a"], errors="coerce").to_numpy(dtype=float)
        command_profile["support_scaled_current_a"] = np.interp(target_time, preview_time, current_values) * scale_ratio
    if "measured_field_mT" in support_profile_preview.columns:
        field_values = pd.to_numeric(support_profile_preview["measured_field_mT"], errors="coerce").to_numpy(dtype=float)
        command_profile["support_scaled_field_mT"] = np.interp(target_time, preview_time, field_values) * scale_ratio
