from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .hardware import apply_command_hardware_model
from .models import DatasetAnalysis
from .utils import canonicalize_waveform_type, field_axis_display_name


TARGET_LABELS = {
    "achieved_current_pp_a_mean": "Achieved Current PP (A)",
    "achieved_bz_mT_pp_mean": f"Achieved {field_axis_display_name('bz_mT')} PP (mT)",
    "achieved_bmag_mT_pp_mean": "Achieved |B| PP (mT)",
    "achieved_bproj_mT_pp_mean": "Achieved Bproj PP (mT)",
}
CURRENT_DEBUG_TARGET_METRICS = {"achieved_current_pp_a_mean"}


def target_metric_label(metric: str) -> str:
    if metric in TARGET_LABELS:
        return TARGET_LABELS[metric]
    if metric.startswith("achieved_") and metric.endswith("_pp_mean"):
        axis_name = metric.removeprefix("achieved_").removesuffix("_pp_mean")
        return f"Achieved {field_axis_display_name(axis_name)} PP (mT)"
    return metric


def target_metric_unit(metric: str) -> str:
    return "A" if metric == "achieved_current_pp_a_mean" else "mT"


def prioritize_lut_target_metrics(
    metric_options: list[str],
    main_field_axis: str,
    include_current_debug: bool = False,
) -> list[str]:
    preferred_order = [
        f"achieved_{main_field_axis}_pp_mean",
        "achieved_bz_mT_pp_mean",
        "achieved_bmag_mT_pp_mean",
        "achieved_bproj_mT_pp_mean",
    ]
    ordered = [metric for metric in dict.fromkeys(preferred_order) if metric in metric_options]
    ordered.extend(
        metric
        for metric in metric_options
        if metric not in ordered and metric not in CURRENT_DEBUG_TARGET_METRICS
    )
    if include_current_debug or not ordered:
        ordered.extend(
            metric
            for metric in metric_options
            if metric in CURRENT_DEBUG_TARGET_METRICS and metric not in ordered
        )
    return ordered


def build_lut_recommendation_display_context(
    *,
    target_metric: str,
    used_target_value: float,
    estimated_current_pp: float,
    estimated_bz_pp: float,
    estimated_bmag_pp: float,
    finite_cycle_mode: bool,
) -> dict[str, Any]:
    target_label = target_metric_label(target_metric)
    target_unit = target_metric_unit(target_metric)
    scope = "finite_cycle" if finite_cycle_mode else "continuous"
    scope_label = "finite-cycle" if finite_cycle_mode else "continuous"

    primary_output_label = target_label
    primary_output_value = float(used_target_value)
    primary_output_unit = target_unit

    if target_metric == "achieved_current_pp_a_mean":
        if np.isfinite(estimated_bz_pp):
            primary_output_label = target_metric_label("achieved_bz_mT_pp_mean")
            primary_output_value = float(estimated_bz_pp)
            primary_output_unit = "mT"
        elif np.isfinite(estimated_bmag_pp):
            primary_output_label = target_metric_label("achieved_bmag_mT_pp_mean")
            primary_output_value = float(estimated_bmag_pp)
            primary_output_unit = "mT"
        else:
            primary_output_value = float(estimated_current_pp)

    return {
        "target_output_label": target_label,
        "target_output_unit": target_unit,
        "target_output_pp": float(used_target_value),
        "primary_output_label": primary_output_label,
        "primary_output_unit": primary_output_unit,
        "primary_output_pp": float(primary_output_value),
        "recommendation_scope": scope,
        "recommendation_scope_label": scope_label,
    }


def recommend_voltage_waveform(
    per_test_summary: pd.DataFrame,
    analyses_by_test_id: dict[str, DatasetAnalysis],
    waveform_type: str,
    freq_hz: float,
    target_metric: str,
    target_value: float,
    voltage_metric: str = "daq_input_v_pp_mean",
    voltage_channel: str = "daq_input_v",
    frequency_mode: str = "interpolate",
    finite_cycle_mode: bool = False,
    target_cycle_count: float | None = None,
    preview_tail_cycles: float = 0.25,
    max_daq_voltage_pp: float = 20.0,
    amp_gain_at_100_pct: float = 20.0,
    amp_gain_limit_pct: float = 100.0,
    amp_max_output_pk_v: float = 180.0,
    default_support_amp_gain_pct: float = 100.0,
    allow_target_extrapolation: bool = True,
) -> dict[str, Any] | None:
    """Recommend a continuous or finite-cycle voltage waveform from measured LUT data."""

    if (
        per_test_summary.empty
        or target_metric not in per_test_summary.columns
        or voltage_metric not in per_test_summary.columns
    ):
        return None

    waveform_type = canonicalize_waveform_type(waveform_type)
    if waveform_type is None:
        return None

    requested_freq_hz = float(freq_hz)
    waveform_subset = per_test_summary[
        per_test_summary["waveform_type"].map(canonicalize_waveform_type) == waveform_type
    ].copy()
    waveform_subset = waveform_subset.dropna(subset=[target_metric, voltage_metric, "freq_hz"])
    if waveform_subset.empty:
        return None

    if frequency_mode == "exact":
        subset = waveform_subset[
            np.isclose(waveform_subset["freq_hz"], requested_freq_hz, atol=1e-6, equal_nan=False)
        ].copy()
    else:
        subset = waveform_subset.copy()
    if subset.empty:
        return None

    frequency_support_table = _build_frequency_support_table(
        subset=subset,
        target_metric=target_metric,
        target_value=float(target_value),
        voltage_metric=voltage_metric,
        allow_target_extrapolation=allow_target_extrapolation,
    )
    if frequency_support_table.empty:
        return None

    available_freq_values = frequency_support_table["freq_hz"].to_numpy(dtype=float)
    available_freq_min = float(np.nanmin(available_freq_values))
    available_freq_max = float(np.nanmax(available_freq_values))
    distinct_frequency_count = int(pd.Series(available_freq_values).nunique())
    used_freq_hz = (
        requested_freq_hz
        if frequency_mode == "exact"
        else float(np.clip(requested_freq_hz, available_freq_min, available_freq_max))
    )

    if frequency_mode == "exact":
        selected_frequency_row = frequency_support_table.iloc[0]
        estimated_voltage_pp = float(selected_frequency_row["estimated_voltage_pp"])
        estimated_current_pp = float(selected_frequency_row["estimated_current_pp"])
        estimated_bz_pp = float(selected_frequency_row["estimated_bz_pp"])
        estimated_bmag_pp = float(selected_frequency_row["estimated_bmag_pp"])
        clamped_target = float(selected_frequency_row["used_target_value"])
        recommendation_mode = str(selected_frequency_row["local_mode"])
        frequency_mode_used = "exact"
    else:
        frequency_support_table = frequency_support_table.sort_values("freq_hz").reset_index(drop=True)
        selected_frequency_row = _select_nearest_frequency_row(
            frequency_support_table=frequency_support_table,
            used_freq_hz=used_freq_hz,
        )
        estimated_voltage_pp = _interpolate_frequency_metric(
            frequency_support_table,
            value_column="estimated_voltage_pp",
            used_freq_hz=used_freq_hz,
        )
        estimated_current_pp = _interpolate_frequency_metric(
            frequency_support_table,
            value_column="estimated_current_pp",
            used_freq_hz=used_freq_hz,
        )
        estimated_bz_pp = _interpolate_frequency_metric(
            frequency_support_table,
            value_column="estimated_bz_pp",
            used_freq_hz=used_freq_hz,
        )
        estimated_bmag_pp = _interpolate_frequency_metric(
            frequency_support_table,
            value_column="estimated_bmag_pp",
            used_freq_hz=used_freq_hz,
        )
        clamped_target = _interpolate_frequency_metric(
            frequency_support_table,
            value_column="used_target_value",
            used_freq_hz=used_freq_hz,
        )
        if distinct_frequency_count == 1:
            frequency_mode_used = "single_frequency_only"
        else:
            frequency_mode_used = (
                "frequency_interpolated"
                if available_freq_min <= requested_freq_hz <= available_freq_max
                else "frequency_clamped"
            )

        if str(selected_frequency_row["local_mode"]) == "single_point_only":
            recommendation_mode = "single_point_only"
        elif str(selected_frequency_row["local_mode"]) == "extrapolated":
            recommendation_mode = "frequency_extrapolated" if distinct_frequency_count > 1 else "extrapolated"
        elif distinct_frequency_count == 1:
            recommendation_mode = str(selected_frequency_row["local_mode"])
        else:
            recommendation_mode = frequency_mode_used

    template_waveform = _build_frequency_aware_template(
        analyses_by_test_id=analyses_by_test_id,
        frequency_support_table=frequency_support_table,
        requested_freq_hz=requested_freq_hz,
        used_freq_hz=used_freq_hz,
        voltage_channel=voltage_channel,
        waveform_type=waveform_type,
    )
    if finite_cycle_mode and target_cycle_count is not None:
        command_waveform = _expand_template_waveform(
            template_waveform=template_waveform,
            freq_hz=requested_freq_hz,
            estimated_voltage_pp=estimated_voltage_pp,
            target_cycle_count=max(float(target_cycle_count), 0.25),
            preview_tail_cycles=max(float(preview_tail_cycles), 0.0),
        )
        preserve_start_voltage = True
    else:
        command_waveform = template_waveform.copy()
        command_waveform["recommended_voltage_v"] = (
            command_waveform["voltage_normalized"] * estimated_voltage_pp / 2.0
        )
        preserve_start_voltage = False
    template_amp_gain_pct = float(selected_frequency_row.get("template_amp_gain_setting", np.nan))
    if not np.isfinite(template_amp_gain_pct) or template_amp_gain_pct <= 0:
        template_amp_gain_pct = float(default_support_amp_gain_pct)
    command_waveform = apply_command_hardware_model(
        command_waveform=command_waveform,
        max_daq_voltage_pp=float(max_daq_voltage_pp),
        amp_gain_at_100_pct=float(amp_gain_at_100_pct),
        support_amp_gain_pct=template_amp_gain_pct,
        amp_gain_limit_pct=float(amp_gain_limit_pct),
        amp_max_output_pk_v=float(amp_max_output_pk_v),
        preserve_start_voltage=preserve_start_voltage,
    )
    command_waveform["waveform_type"] = waveform_type
    command_waveform["freq_hz"] = requested_freq_hz
    command_waveform["target_metric"] = target_metric
    command_waveform["target_value"] = target_value
    command_waveform["finite_cycle_mode"] = finite_cycle_mode
    command_waveform["target_cycle_count"] = (
        max(float(target_cycle_count), 0.25)
        if finite_cycle_mode and target_cycle_count is not None
        else np.nan
    )
    command_waveform["preview_tail_cycles"] = (
        max(float(preview_tail_cycles), 0.0)
        if finite_cycle_mode and target_cycle_count is not None
        else np.nan
    )

    support_point_count = int(len(subset))
    distinct_target_count = int(pd.Series(subset[target_metric]).dropna().nunique())
    selected_frequency_subset = subset[
        np.isclose(subset["freq_hz"], float(selected_frequency_row["freq_hz"]), atol=1e-6, equal_nan=False)
    ].copy()
    selected_target_values = selected_frequency_subset[target_metric].to_numpy(dtype=float)
    min_target = float(np.nanmin(selected_target_values))
    max_target = float(np.nanmax(selected_target_values))
    in_range = min_target <= target_value <= max_target

    neighbor_points = _build_neighbor_points(
        subset=subset,
        target_metric=target_metric,
        clamped_target=clamped_target,
        requested_freq_hz=requested_freq_hz,
        used_freq_hz=used_freq_hz,
        frequency_mode=frequency_mode_used,
    )
    if support_point_count == 1:
        recommendation_mode = "single_point_only"

    display_context = build_lut_recommendation_display_context(
        target_metric=target_metric,
        used_target_value=clamped_target,
        estimated_current_pp=estimated_current_pp,
        estimated_bz_pp=estimated_bz_pp,
        estimated_bmag_pp=estimated_bmag_pp,
        finite_cycle_mode=finite_cycle_mode,
    )

    return {
        "waveform_type": waveform_type,
        "freq_hz": requested_freq_hz,
        "target_metric": target_metric,
        "target_label": target_metric_label(target_metric),
        "requested_target_value": float(target_value),
        "used_target_value": clamped_target,
        "in_range": in_range,
        "recommendation_mode": recommendation_mode,
        "requested_freq_hz": requested_freq_hz,
        "used_freq_hz": used_freq_hz,
        "frequency_mode": frequency_mode_used,
        "available_freq_min": available_freq_min,
        "available_freq_max": available_freq_max,
        "frequency_in_range": available_freq_min <= requested_freq_hz <= available_freq_max,
        "frequency_support_count": distinct_frequency_count,
        "support_point_count": support_point_count,
        "distinct_target_count": distinct_target_count,
        "available_target_min": min_target,
        "available_target_max": max_target,
        "estimated_voltage_pp": estimated_voltage_pp,
        "estimated_voltage_peak": estimated_voltage_pp / 2.0,
        "finite_cycle_mode": finite_cycle_mode,
        "target_cycle_count": (
            max(float(target_cycle_count), 0.25)
            if finite_cycle_mode and target_cycle_count is not None
            else float("nan")
        ),
        "preview_tail_cycles": (
            max(float(preview_tail_cycles), 0.0)
            if finite_cycle_mode and target_cycle_count is not None
            else float("nan")
        ),
        "max_daq_voltage_pp": float(max_daq_voltage_pp),
        "amp_gain_at_100_pct": float(amp_gain_at_100_pct),
        "amp_gain_limit_pct": float(amp_gain_limit_pct),
        "amp_max_output_pk_v": float(amp_max_output_pk_v),
        "allow_target_extrapolation": bool(allow_target_extrapolation),
        "within_daq_limit": bool(command_waveform["within_daq_limit"].iloc[0]),
        "within_hardware_limits": bool(command_waveform["within_hardware_limits"].iloc[0]),
        "required_amp_gain_multiplier": float(command_waveform["required_amp_gain_multiplier"].iloc[0]),
        "required_amp_gain_pct": float(command_waveform["required_amp_gain_pct"].iloc[0]),
        "support_amp_gain_pct": float(command_waveform["support_amp_gain_pct"].iloc[0]),
        "available_amp_gain_pct": float(command_waveform["available_amp_gain_pct"].iloc[0]),
        "amp_output_pp_at_required": float(command_waveform["amp_output_pp_at_required"].iloc[0]),
        "amp_output_pk_at_required": float(command_waveform["amp_output_pk_at_required"].iloc[0]),
        "limited_voltage_pp": float(command_waveform["limited_voltage_pp"].iloc[0]),
        "estimated_current_pp": estimated_current_pp,
        "estimated_bz_pp": estimated_bz_pp,
        "estimated_bmag_pp": estimated_bmag_pp,
        "frequency_support_table": frequency_support_table,
        "lookup_table": subset,
        "neighbor_points": neighbor_points,
        "template_test_id": str(selected_frequency_row["template_test_id"]),
        "template_waveform": template_waveform,
        "command_waveform": command_waveform,
        **display_context,
    }


def build_voltage_template(
    analysis: DatasetAnalysis,
    voltage_channel: str = "daq_input_v",
    fallback_waveform: str | None = None,
    fallback_freq_hz: float | None = None,
    points_per_cycle: int = 300,
) -> pd.DataFrame:
    """Build a representative one-cycle normalized voltage waveform."""

    frame = analysis.cycle_detection.annotated_frame.copy()
    if (
        voltage_channel not in frame.columns
        or "cycle_index" not in frame.columns
        or frame["cycle_index"].dropna().empty
    ):
        return _theoretical_template(
            waveform_type=fallback_waveform or "sine",
            freq_hz=fallback_freq_hz,
            points_per_cycle=points_per_cycle,
        )

    grid = np.linspace(0.0, 1.0, points_per_cycle)
    waveforms: list[np.ndarray] = []
    durations: list[float] = []

    for _, cycle_frame in frame.dropna(subset=["cycle_index"]).groupby("cycle_index", sort=True):
        valid = cycle_frame[["cycle_progress", voltage_channel, "cycle_time_s"]].dropna().copy()
        valid = valid.sort_values("cycle_progress").drop_duplicates("cycle_progress")
        if len(valid) < 5:
            continue
        progress = valid["cycle_progress"].to_numpy(dtype=float)
        voltage = valid[voltage_channel].to_numpy(dtype=float)
        centered = voltage - float(np.nanmean(voltage))
        amplitude = float(np.nanmax(np.abs(centered)))
        if not np.isfinite(amplitude) or amplitude <= 0:
            continue
        waveforms.append(np.interp(grid, progress, centered / amplitude))
        durations.append(float(valid["cycle_time_s"].max()))

    if not waveforms:
        return _theoretical_template(
            waveform_type=fallback_waveform or "sine",
            freq_hz=fallback_freq_hz,
            points_per_cycle=points_per_cycle,
        )

    normalized = np.vstack(waveforms).mean(axis=0)
    waveform_guess = fallback_waveform
    if waveform_guess is None and "waveform_type" in frame.columns and frame["waveform_type"].notna().any():
        waveform_guess = str(frame["waveform_type"].dropna().iloc[0])
    normalized = _align_normalized_waveform_to_theoretical_phase(
        normalized=normalized,
        waveform_type=waveform_guess or "sine",
    )

    freq_hz = fallback_freq_hz or float(
        pd.to_numeric(frame["freq_hz"], errors="coerce").dropna().iloc[0]
    )
    period_s = float(np.mean(durations)) if durations else (1.0 / freq_hz if freq_hz else 1.0)
    if not np.isfinite(period_s) or period_s <= 0:
        period_s = 1.0 / freq_hz if freq_hz else 1.0

    return pd.DataFrame(
        {
            "cycle_progress": grid,
            "time_s": grid * period_s,
            "voltage_normalized": normalized,
        }
    )


def _interpolate_metric(
    subset: pd.DataFrame,
    metric_column: str,
    clamped_target: float,
    target_metric: str,
    allow_target_extrapolation: bool = False,
) -> float:
    if metric_column not in subset.columns:
        return float("nan")
    metric_subset = subset.dropna(subset=[metric_column, target_metric]).sort_values(target_metric)
    if metric_subset.empty:
        return float("nan")
    x_values = metric_subset[target_metric].to_numpy(dtype=float)
    y_values = metric_subset[metric_column].to_numpy(dtype=float)
    if allow_target_extrapolation:
        return _interpolate_or_extrapolate_metric(x_values, y_values, clamped_target)
    return float(np.interp(clamped_target, x_values, y_values))


def _interpolate_or_extrapolate_metric(
    x_values: np.ndarray,
    y_values: np.ndarray,
    target_value: float,
) -> float:
    if len(x_values) == 0:
        return float("nan")
    if len(x_values) == 1:
        return float(y_values[0])
    if target_value < float(x_values[0]):
        x0, x1 = float(x_values[0]), float(x_values[1])
        y0, y1 = float(y_values[0]), float(y_values[1])
        slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0.0
        return float(y0 + slope * (target_value - x0))
    if target_value > float(x_values[-1]):
        x0, x1 = float(x_values[-2]), float(x_values[-1])
        y0, y1 = float(y_values[-2]), float(y_values[-1])
        slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0.0
        return float(y1 + slope * (target_value - x1))
    return float(np.interp(target_value, x_values, y_values))


def _build_frequency_support_table(
    subset: pd.DataFrame,
    target_metric: str,
    target_value: float,
    voltage_metric: str,
    allow_target_extrapolation: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for freq_hz, group in subset.groupby("freq_hz", sort=True):
        working = group.dropna(subset=[target_metric, voltage_metric]).sort_values(target_metric).copy()
        working = working.loc[working[target_metric].diff().fillna(1).ne(0)].copy()
        if working.empty:
            continue

        target_values = working[target_metric].to_numpy(dtype=float)
        voltage_values = working[voltage_metric].to_numpy(dtype=float)
        available_target_min = float(np.nanmin(target_values))
        available_target_max = float(np.nanmax(target_values))
        used_target_value = (
            float(target_value)
            if allow_target_extrapolation
            else float(np.clip(target_value, available_target_min, available_target_max))
        )
        support_point_count = int(len(working))
        distinct_target_count = int(pd.Series(target_values).nunique())

        if distinct_target_count < 2:
            nearest_row = working.iloc[0]
            estimated_voltage_pp = float(nearest_row[voltage_metric])
            estimated_current_pp = float(nearest_row.get("achieved_current_pp_a_mean", np.nan))
            estimated_bz_pp = float(nearest_row.get("achieved_bz_mT_pp_mean", np.nan))
            estimated_bmag_pp = float(nearest_row.get("achieved_bmag_mT_pp_mean", np.nan))
            local_mode = "single_point_only"
        else:
            estimated_voltage_pp = _interpolate_or_extrapolate_metric(target_values, voltage_values, used_target_value)
            estimated_current_pp = _interpolate_metric(
                working,
                "achieved_current_pp_a_mean",
                used_target_value,
                target_metric,
                allow_target_extrapolation=allow_target_extrapolation,
            )
            estimated_bz_pp = _interpolate_metric(
                working,
                "achieved_bz_mT_pp_mean",
                used_target_value,
                target_metric,
                allow_target_extrapolation=allow_target_extrapolation,
            )
            estimated_bmag_pp = _interpolate_metric(
                working,
                "achieved_bmag_mT_pp_mean",
                used_target_value,
                target_metric,
                allow_target_extrapolation=allow_target_extrapolation,
            )
            nearest_row = working.iloc[(working[target_metric] - used_target_value).abs().argmin()]
            if available_target_min <= target_value <= available_target_max:
                local_mode = "interpolated"
            else:
                local_mode = "extrapolated" if allow_target_extrapolation else "clamped"

        rows.append(
            {
                "freq_hz": float(freq_hz),
                "used_target_value": used_target_value,
                "available_target_min": available_target_min,
                "available_target_max": available_target_max,
                "support_point_count": support_point_count,
                "distinct_target_count": distinct_target_count,
                "estimated_voltage_pp": estimated_voltage_pp,
                "estimated_current_pp": estimated_current_pp,
                "estimated_bz_pp": estimated_bz_pp,
                "estimated_bmag_pp": estimated_bmag_pp,
                "local_mode": local_mode,
                "template_test_id": str(nearest_row["test_id"]),
                "template_amp_gain_setting": float(nearest_row.get("amp_gain_setting_mean", np.nan)),
            }
        )

    return pd.DataFrame(rows)


def _interpolate_frequency_metric(
    frequency_support_table: pd.DataFrame,
    value_column: str,
    used_freq_hz: float,
) -> float:
    if value_column not in frequency_support_table.columns:
        return float("nan")
    working = frequency_support_table.dropna(subset=["freq_hz", value_column]).sort_values("freq_hz")
    if working.empty:
        return float("nan")
    if len(working) == 1:
        return float(working[value_column].iloc[0])
    return float(
        np.interp(
            float(used_freq_hz),
            working["freq_hz"].to_numpy(dtype=float),
            working[value_column].to_numpy(dtype=float),
        )
    )


def _select_nearest_frequency_row(
    frequency_support_table: pd.DataFrame,
    used_freq_hz: float,
) -> pd.Series:
    working = frequency_support_table.copy()
    working["freq_distance_hz"] = (working["freq_hz"] - float(used_freq_hz)).abs()
    return working.sort_values(["freq_distance_hz", "support_point_count"], ascending=[True, False]).iloc[0]


def _build_frequency_aware_template(
    analyses_by_test_id: dict[str, DatasetAnalysis],
    frequency_support_table: pd.DataFrame,
    requested_freq_hz: float,
    used_freq_hz: float,
    voltage_channel: str,
    waveform_type: str,
) -> pd.DataFrame:
    if frequency_support_table.empty:
        return _theoretical_template(
            waveform_type=waveform_type,
            freq_hz=requested_freq_hz,
            points_per_cycle=300,
        )

    working = frequency_support_table.sort_values("freq_hz").reset_index(drop=True)
    lower = working.loc[working["freq_hz"] <= used_freq_hz]
    upper = working.loc[working["freq_hz"] >= used_freq_hz]
    lower_row = lower.iloc[-1] if not lower.empty else working.iloc[0]
    upper_row = upper.iloc[0] if not upper.empty else working.iloc[-1]

    lower_template = _load_template_for_test(
        analyses_by_test_id=analyses_by_test_id,
        test_id=str(lower_row["template_test_id"]),
        voltage_channel=voltage_channel,
        waveform_type=waveform_type,
        freq_hz=float(lower_row["freq_hz"]),
    )
    if str(lower_row["template_test_id"]) == str(upper_row["template_test_id"]):
        template = lower_template.copy()
    else:
        upper_template = _load_template_for_test(
            analyses_by_test_id=analyses_by_test_id,
            test_id=str(upper_row["template_test_id"]),
            voltage_channel=voltage_channel,
            waveform_type=waveform_type,
            freq_hz=float(upper_row["freq_hz"]),
        )
        template = _blend_templates(
            lower_template=lower_template,
            upper_template=upper_template,
            lower_freq_hz=float(lower_row["freq_hz"]),
            upper_freq_hz=float(upper_row["freq_hz"]),
            used_freq_hz=float(used_freq_hz),
            output_freq_hz=float(requested_freq_hz),
        )

    template["time_s"] = template["cycle_progress"] * (
        1.0 / requested_freq_hz if requested_freq_hz > 0 else 1.0
    )
    return template


def _load_template_for_test(
    analyses_by_test_id: dict[str, DatasetAnalysis],
    test_id: str,
    voltage_channel: str,
    waveform_type: str,
    freq_hz: float,
) -> pd.DataFrame:
    analysis = analyses_by_test_id.get(test_id)
    if analysis is None:
        return _theoretical_template(
            waveform_type=waveform_type,
            freq_hz=freq_hz,
            points_per_cycle=300,
        )
    return build_voltage_template(
        analysis=analysis,
        voltage_channel=voltage_channel,
        fallback_waveform=waveform_type,
        fallback_freq_hz=freq_hz,
    )


def _blend_templates(
    lower_template: pd.DataFrame,
    upper_template: pd.DataFrame,
    lower_freq_hz: float,
    upper_freq_hz: float,
    used_freq_hz: float,
    output_freq_hz: float,
) -> pd.DataFrame:
    if abs(upper_freq_hz - lower_freq_hz) <= 1e-12:
        template = lower_template.copy()
        template["time_s"] = template["cycle_progress"] * (1.0 / output_freq_hz if output_freq_hz > 0 else 1.0)
        return template

    grid = np.linspace(0.0, 1.0, max(len(lower_template), len(upper_template), 300))
    lower_values = np.interp(
        grid,
        lower_template["cycle_progress"].to_numpy(dtype=float),
        lower_template["voltage_normalized"].to_numpy(dtype=float),
    )
    upper_values = np.interp(
        grid,
        upper_template["cycle_progress"].to_numpy(dtype=float),
        upper_template["voltage_normalized"].to_numpy(dtype=float),
    )
    weight = (float(used_freq_hz) - float(lower_freq_hz)) / (float(upper_freq_hz) - float(lower_freq_hz))
    weight = float(np.clip(weight, 0.0, 1.0))
    blended = (1.0 - weight) * lower_values + weight * upper_values
    reference_waveform = "triangle" if _looks_triangle_template(lower_template, upper_template) else "sine"
    blended = _align_normalized_waveform_to_theoretical_phase(
        normalized=blended,
        waveform_type=reference_waveform,
    )
    return pd.DataFrame(
        {
            "cycle_progress": grid,
            "time_s": grid * (1.0 / output_freq_hz if output_freq_hz > 0 else 1.0),
            "voltage_normalized": blended,
        }
    )


def _expand_template_waveform(
    template_waveform: pd.DataFrame,
    freq_hz: float,
    estimated_voltage_pp: float,
    target_cycle_count: float,
    preview_tail_cycles: float,
) -> pd.DataFrame:
    total_cycles = float(target_cycle_count + preview_tail_cycles)
    sample_count = max(int(np.ceil(total_cycles * max(len(template_waveform), 300))), 2) + 1
    period_s = 1.0 / freq_hz if freq_hz > 0 else 1.0
    time_grid = np.linspace(0.0, total_cycles * period_s, sample_count)
    phase_total = time_grid / period_s
    active_mask = phase_total <= target_cycle_count + 1e-12
    cycle_progress = np.mod(phase_total, 1.0)
    normalized = np.interp(
        cycle_progress,
        template_waveform["cycle_progress"].to_numpy(dtype=float),
        template_waveform["voltage_normalized"].to_numpy(dtype=float),
    )
    normalized[~active_mask] = 0.0
    startup_duration_s = max(period_s / max(len(template_waveform) / 4.0, 1.0), 0.0)
    normalized = _apply_zero_start_envelope(
        values=normalized,
        time_grid=time_grid,
        startup_duration_s=startup_duration_s,
    )
    return pd.DataFrame(
        {
            "cycle_progress": cycle_progress,
            "cycle_progress_total": phase_total,
            "time_s": time_grid,
            "voltage_normalized": normalized,
            "recommended_voltage_v": normalized * estimated_voltage_pp / 2.0,
            "is_active_target": active_mask,
        }
    )


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


def _build_neighbor_points(
    subset: pd.DataFrame,
    target_metric: str,
    clamped_target: float,
    requested_freq_hz: float | None = None,
    used_freq_hz: float | None = None,
    frequency_mode: str = "exact",
) -> pd.DataFrame:
    working = subset.copy()
    working["distance_to_target"] = (working[target_metric] - clamped_target).abs()
    sort_columns = ["distance_to_target"]
    if requested_freq_hz is not None and "freq_hz" in working.columns and frequency_mode != "exact":
        effective_freq = float(used_freq_hz if used_freq_hz is not None else requested_freq_hz)
        working["distance_to_freq_hz"] = (working["freq_hz"] - effective_freq).abs()
        target_range = max(
            float(working[target_metric].max()) - float(working[target_metric].min()),
            1e-9,
        )
        freq_range = max(float(working["freq_hz"].max()) - float(working["freq_hz"].min()), 1e-9)
        working["combined_distance"] = np.sqrt(
            np.square(working["distance_to_target"] / target_range)
            + np.square(working["distance_to_freq_hz"] / freq_range)
        )
        sort_columns = ["combined_distance", "distance_to_target", "distance_to_freq_hz"]
    columns = [
        column
        for column in dict.fromkeys(
            (
                "test_id",
                "waveform_type",
                "freq_hz",
                "current_pp_target_a",
                target_metric,
                "daq_input_v_pp_mean",
                "amp_gain_setting_mean",
                "achieved_current_pp_a_mean",
                "achieved_bz_mT_pp_mean",
                "achieved_bmag_mT_pp_mean",
                "distance_to_target",
                "distance_to_freq_hz",
                "combined_distance",
            )
        )
        if column in working.columns
    ]
    return working.sort_values(sort_columns).head(6)[columns].reset_index(drop=True)


def _theoretical_template(
    waveform_type: str,
    freq_hz: float | None,
    points_per_cycle: int,
) -> pd.DataFrame:
    grid = np.linspace(0.0, 1.0, points_per_cycle)
    waveform_type = waveform_type.lower()
    if waveform_type == "triangle":
        normalized = np.piecewise(
            grid,
            [
                grid < 0.25,
                (grid >= 0.25) & (grid < 0.5),
                (grid >= 0.5) & (grid < 0.75),
                grid >= 0.75,
            ],
            [
                lambda value: 4.0 * value,
                lambda value: 2.0 - 4.0 * value,
                lambda value: -4.0 * (value - 0.5),
                lambda value: -4.0 + 4.0 * value,
            ],
        )
    else:
        normalized = np.sin(2.0 * np.pi * grid)
    period_s = 1.0 / freq_hz if freq_hz and freq_hz > 0 else 1.0
    return pd.DataFrame(
        {
            "cycle_progress": grid,
            "time_s": grid * period_s,
            "voltage_normalized": normalized,
        }
    )


def _align_normalized_waveform_to_theoretical_phase(
    normalized: np.ndarray,
    waveform_type: str,
) -> np.ndarray:
    values = np.asarray(normalized, dtype=float)
    if len(values) < 4 or not np.isfinite(values).any():
        return values

    centered = values - float(np.nanmean(values))
    ideal = _theoretical_template(
        waveform_type=waveform_type,
        freq_hz=1.0,
        points_per_cycle=len(centered),
    )["voltage_normalized"].to_numpy(dtype=float)
    centered_fft = np.fft.rfft(centered)
    ideal_fft = np.fft.rfft(ideal)
    aligned_fft = np.zeros_like(centered_fft, dtype=np.complex128)

    for harmonic_index in range(1, len(centered_fft)):
        magnitude = abs(centered_fft[harmonic_index])
        if magnitude < 1e-12:
            continue
        reference_component = ideal_fft[harmonic_index] if harmonic_index < len(ideal_fft) else 0.0j
        phase = np.angle(reference_component) if abs(reference_component) >= 1e-12 else np.angle(centered_fft[harmonic_index])
        aligned_fft[harmonic_index] = magnitude * np.exp(1j * phase)

    aligned = np.fft.irfft(aligned_fft, n=len(centered))
    max_abs = float(np.nanmax(np.abs(aligned)))
    if max_abs > 0:
        aligned = aligned / max_abs
    aligned[0] = 0.0
    if len(aligned) > 1:
        aligned[-1] = aligned[0]
    return aligned


def _looks_triangle_template(
    lower_template: pd.DataFrame,
    upper_template: pd.DataFrame,
) -> bool:
    for template in (lower_template, upper_template):
        if template.empty or "voltage_normalized" not in template.columns:
            continue
        values = pd.to_numeric(template["voltage_normalized"], errors="coerce").to_numpy(dtype=float)
        if len(values) < 8 or not np.isfinite(values).any():
            continue
        fft_values = np.fft.rfft(values - float(np.nanmean(values)))
        even_energy = float(np.nansum(np.square(np.abs(fft_values[2::2]))))
        odd_energy = float(np.nansum(np.square(np.abs(fft_values[1::2]))))
        if odd_energy > 0 and even_energy / odd_energy < 0.08:
            return True
    return False


def _apply_daq_voltage_limit(
    command_waveform: pd.DataFrame,
    max_daq_voltage_pp: float,
    preserve_start_voltage: bool = False,
) -> pd.DataFrame:
    command = command_waveform.copy()
    recommended = pd.to_numeric(command["recommended_voltage_v"], errors="coerce").to_numpy(dtype=float)
    if "is_active_target" in command.columns:
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
    recommended_pp = float(np.nanmax(pp_values) - np.nanmin(pp_values)) if len(pp_values) else float("nan")
    gain_multiplier = (
        max(recommended_pp / max_daq_voltage_pp, 1.0)
        if np.isfinite(recommended_pp) and np.isfinite(max_daq_voltage_pp) and max_daq_voltage_pp > 0
        else 1.0
    )
    limited = scaled_values / gain_multiplier
    limited_pp = (
        float(np.nanmax(limited[active_mask]) - np.nanmin(limited[active_mask]))
        if active_mask.any()
        else float(np.nanmax(limited) - np.nanmin(limited))
    )

    command["recommended_voltage_v"] = scaled_values
    command["recommended_voltage_pp"] = recommended_pp
    command["limited_voltage_v"] = limited
    command["limited_voltage_pp"] = limited_pp
    command["required_amp_gain_multiplier"] = gain_multiplier
    command["within_daq_limit"] = bool(recommended_pp <= max_daq_voltage_pp + 1e-9) if np.isfinite(recommended_pp) else False
    return command
