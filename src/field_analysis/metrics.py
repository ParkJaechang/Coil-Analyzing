from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .utils import canonicalize_waveform_type, column_stats, flatten_messages


FIELD_AXES = ("bx_mT", "by_mT", "bz_mT", "bmag_mT", "bproj_mT")


def compute_cycle_and_test_metrics(
    annotated_frame: pd.DataFrame,
    current_channel: str = "i_sum",
    main_field_axis: str = "bz_mT",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute cycle-level and test-level engineering metrics."""

    if "cycle_index" not in annotated_frame.columns:
        return pd.DataFrame(), pd.DataFrame()

    working = annotated_frame.dropna(subset=["cycle_index"]).copy()
    if working.empty:
        return pd.DataFrame(), pd.DataFrame()

    cycle_rows: list[dict[str, Any]] = []
    for cycle_index, cycle_frame in working.groupby("cycle_index", sort=True):
        cycle_rows.append(
            _compute_single_cycle_metrics(
                cycle_frame=cycle_frame.copy(),
                current_channel=current_channel,
                main_field_axis=main_field_axis,
            )
        )

    per_cycle = pd.DataFrame(cycle_rows).sort_values("cycle_index").reset_index(drop=True)
    if per_cycle.empty:
        return per_cycle, pd.DataFrame()

    per_cycle = _append_drift_columns(per_cycle, main_field_axis=main_field_axis)
    per_test = _build_test_summary(
        annotated_frame=annotated_frame,
        per_cycle=per_cycle,
        current_channel=current_channel,
        main_field_axis=main_field_axis,
    )
    return per_cycle, per_test


def build_calculation_details(
    annotated_frame: pd.DataFrame,
    per_cycle_summary: pd.DataFrame,
    cycle_index: int,
    current_channel: str = "i_sum",
    main_field_axis: str = "bz_mT",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return formula-level details plus an intermediate sample table."""

    cycle_frame = annotated_frame[annotated_frame["cycle_index"] == cycle_index].copy()
    if cycle_frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    metric_row = per_cycle_summary[per_cycle_summary["cycle_index"] == cycle_index]
    if metric_row.empty:
        return pd.DataFrame(), pd.DataFrame()
    metric_row = metric_row.iloc[0]

    current_max = float(pd.to_numeric(cycle_frame[current_channel], errors="coerce").max())
    current_min = float(pd.to_numeric(cycle_frame[current_channel], errors="coerce").min())
    field_max = float(pd.to_numeric(cycle_frame[main_field_axis], errors="coerce").max())
    field_min = float(pd.to_numeric(cycle_frame[main_field_axis], errors="coerce").min())
    peak_row = cycle_frame.loc[pd.to_numeric(cycle_frame["bmag_mT"], errors="coerce").idxmax()]

    details = pd.DataFrame(
        [
            {
                "metric": "Bmag",
                "formula": "sqrt(Bx^2 + By^2 + Bz^2)",
                "substitution": (
                    f"sqrt({peak_row['bx_mT']:.4f}^2 + {peak_row['by_mT']:.4f}^2 + "
                    f"{peak_row['bz_mT']:.4f}^2)"
                ),
                "result": f"{peak_row['bmag_mT']:.4f} mT",
            },
            {
                "metric": "Selected Current PP",
                "formula": "max(I_selected) - min(I_selected)",
                "substitution": f"{current_max:.4f} - ({current_min:.4f})",
                "result": f"{metric_row['achieved_current_pp_a']:.4f} A",
            },
            {
                "metric": f"{main_field_axis} PP",
                "formula": f"max({main_field_axis}) - min({main_field_axis})",
                "substitution": f"{field_max:.4f} - ({field_min:.4f})",
                "result": f"{metric_row[f'achieved_{main_field_axis}_pp']:.4f} mT",
            },
            {
                "metric": "Field Gain",
                "formula": f"{main_field_axis}_pp / current_pp",
                "substitution": (
                    f"{metric_row[f'achieved_{main_field_axis}_pp']:.4f} / "
                    f"{metric_row['achieved_current_pp_a']:.4f}"
                ),
                "result": f"{metric_row[f'{main_field_axis}_gain_mT_per_a']:.4f} mT/A",
            },
            {
                "metric": "Cycle Drift",
                "formula": "(cycle_n_metric - cycle_1_metric) / cycle_1_metric",
                "substitution": f"cycle {cycle_index} vs cycle 1 on {main_field_axis}_pp",
                "result": f"{metric_row[f'{main_field_axis}_pp_drift_ratio']:.4f}",
            },
        ]
    )

    intermediate_columns = [
        column
        for column in (
            "time_s",
            current_channel,
            "coil1_current_a",
            "coil2_current_a",
            "bx_mT",
            "by_mT",
            "bz_mT",
            "bmag_mT",
            "bproj_mT",
            "branch_direction",
        )
        if column in cycle_frame.columns
    ]
    intermediate_table = cycle_frame[intermediate_columns].head(25).reset_index(drop=True)
    return details, intermediate_table


def build_coverage_matrix(per_test_summary: pd.DataFrame) -> pd.DataFrame:
    """Build a 56-condition style coverage matrix by waveform/frequency/current."""

    if per_test_summary.empty:
        return pd.DataFrame()

    coverage = per_test_summary.copy()
    coverage["waveform_type"] = coverage["waveform_type"].map(canonicalize_waveform_type).fillna(
        coverage["waveform_type"]
    )
    coverage["coverage"] = 1
    pivot = coverage.pivot_table(
        index=["waveform_type", "freq_hz"],
        columns="current_pp_target_a",
        values="coverage",
        aggfunc="sum",
        fill_value=0,
    )
    return pivot.sort_index(axis=0).sort_index(axis=1)


def apply_reference_normalization(
    per_test_summary: pd.DataFrame,
    reference_test_id: str | None,
    field_axis: str = "bz_mT",
) -> pd.DataFrame:
    """Normalize current and field metrics to a user-selected reference test."""

    summary = per_test_summary.copy()
    if summary.empty or not reference_test_id:
        summary["reference_current_ratio"] = np.nan
        summary["reference_field_ratio"] = np.nan
        return summary

    reference_row = summary[summary["test_id"] == reference_test_id]
    if reference_row.empty:
        summary["reference_current_ratio"] = np.nan
        summary["reference_field_ratio"] = np.nan
        return summary

    reference_row = reference_row.iloc[0]
    current_ref = float(reference_row["achieved_current_pp_a_mean"])
    field_ref = float(reference_row[f"achieved_{field_axis}_pp_mean"])
    summary["reference_current_ratio"] = summary["achieved_current_pp_a_mean"] / current_ref
    summary["reference_field_ratio"] = summary[f"achieved_{field_axis}_pp_mean"] / field_ref
    return summary


def estimate_drive_for_target_field(
    per_test_summary: pd.DataFrame,
    waveform_type: str,
    freq_hz: float,
    target_field_pp: float,
    field_axis: str = "bz_mT",
) -> dict[str, float] | None:
    """Interpolate required current/voltage to reach a target field."""

    if per_test_summary.empty:
        return None

    waveform_type = canonicalize_waveform_type(waveform_type)
    if waveform_type is None:
        return None

    subset = per_test_summary[
        (per_test_summary["waveform_type"].map(canonicalize_waveform_type) == waveform_type)
        & (np.isclose(per_test_summary["freq_hz"], freq_hz, equal_nan=False))
    ].copy()
    field_metric = f"achieved_{field_axis}_pp_mean"
    if subset.empty or subset[field_metric].nunique() < 2:
        return None

    subset = subset.sort_values(field_metric)
    field_values = subset[field_metric].to_numpy(dtype=float)
    current_values = subset["current_pp_target_a"].to_numpy(dtype=float)
    voltage_values = subset["daq_input_v_pp_mean"].to_numpy(dtype=float)

    if target_field_pp < np.nanmin(field_values) or target_field_pp > np.nanmax(field_values):
        return None

    return {
        "estimated_current_pp_a": float(np.interp(target_field_pp, field_values, current_values)),
        "estimated_input_voltage_pp_v": float(np.interp(target_field_pp, field_values, voltage_values)),
    }


def _compute_single_cycle_metrics(
    cycle_frame: pd.DataFrame,
    current_channel: str,
    main_field_axis: str,
) -> dict[str, Any]:
    cycle_frame = cycle_frame.sort_values("time_s").reset_index(drop=True)
    first_row = cycle_frame.iloc[0]

    duration = float(cycle_frame["time_s"].max() - cycle_frame["time_s"].min())
    detected_frequency = 1.0 / duration if duration > 0 else np.nan
    current_stats = column_stats(cycle_frame[current_channel])
    daq_stats = column_stats(cycle_frame["daq_input_v"])
    temperature_stats = column_stats(cycle_frame["temperature_c"])

    row: dict[str, Any] = {
        "source_file": first_row["source_file"],
        "sheet_name": first_row["sheet_name"],
        "test_id": first_row["test_id"],
        "waveform_type": first_row["waveform_type"],
        "freq_hz": first_row["freq_hz"],
        "current_pp_target_a": first_row["current_pp_target_a"],
        "current_pk_target_a": first_row["current_pk_target_a"],
        "cycle_total_expected": first_row["cycle_total_expected"],
        "cycle_index": int(first_row["cycle_index"]),
        "cycle_duration_s": duration,
        "detected_frequency_hz": detected_frequency,
        "selected_current_channel": current_channel,
        "main_field_axis": main_field_axis,
        "selected_current_mean_a": current_stats["mean"],
        "selected_current_std_a": current_stats["std"],
        "selected_current_rms_a": current_stats["rms"],
        "selected_current_peak_a": current_stats["peak"],
        "selected_current_valley_a": current_stats["valley"],
        "achieved_current_pp_a": current_stats["peak_to_peak"],
        "achieved_current_peak_a": max(abs(current_stats["peak"]), abs(current_stats["valley"])),
        "temperature_mean_c": temperature_stats["mean"],
        "temperature_rise_in_cycle_c": float(cycle_frame["temperature_c"].iloc[-1] - cycle_frame["temperature_c"].iloc[0]),
        "daq_input_v_pp": daq_stats["peak_to_peak"],
        "daq_input_v_rms": daq_stats["rms"],
        "amp_gain_setting_mean": float(pd.to_numeric(cycle_frame.get("amp_gain_setting"), errors="coerce").mean())
        if "amp_gain_setting" in cycle_frame.columns
        else np.nan,
        "parse_warnings": first_row.get("parse_warnings", ""),
    }

    if row["daq_input_v_pp"] and np.isfinite(row["daq_input_v_pp"]) and row["daq_input_v_pp"] != 0:
        row["input_current_gain_a_per_v"] = row["achieved_current_pp_a"] / row["daq_input_v_pp"]
    else:
        row["input_current_gain_a_per_v"] = np.nan

    for axis in FIELD_AXES:
        if axis not in cycle_frame.columns:
            continue
        stats = column_stats(cycle_frame[axis])
        row[f"{axis}_mean"] = stats["mean"]
        row[f"{axis}_std"] = stats["std"]
        row[f"{axis}_rms"] = stats["rms"]
        row[f"achieved_{axis}_peak"] = max(abs(stats["peak"]), abs(stats["valley"]))
        row[f"achieved_{axis}_pp"] = stats["peak_to_peak"]
        if row["achieved_current_pp_a"] and np.isfinite(row["achieved_current_pp_a"]) and row["achieved_current_pp_a"] != 0:
            row[f"{axis}_gain_mT_per_a"] = stats["peak_to_peak"] / row["achieved_current_pp_a"]
        else:
            row[f"{axis}_gain_mT_per_a"] = np.nan

    row["loop_area_main"] = compute_loop_area(
        x=cycle_frame[current_channel],
        y=cycle_frame[main_field_axis],
    )
    row["coercive_like_current_a"] = interpolate_x_at_y_zero(
        x=cycle_frame[current_channel],
        y=cycle_frame[main_field_axis],
    )
    row["zero_crossing_offset_mT"] = interpolate_y_at_x_zero(
        x=cycle_frame[current_channel],
        y=cycle_frame[main_field_axis],
    )

    rising_frame = cycle_frame[cycle_frame["branch_direction"] == "rising"]
    falling_frame = cycle_frame[cycle_frame["branch_direction"] == "falling"]
    rising_pp = column_stats(rising_frame[main_field_axis])["peak_to_peak"] if not rising_frame.empty else np.nan
    falling_pp = column_stats(falling_frame[main_field_axis])["peak_to_peak"] if not falling_frame.empty else np.nan
    row["rising_branch_pp_mT"] = rising_pp
    row["falling_branch_pp_mT"] = falling_pp
    if np.isfinite(rising_pp) and np.isfinite(falling_pp):
        denominator = max(abs(rising_pp), abs(falling_pp), 1e-12)
        row["branch_asymmetry_ratio"] = (rising_pp - falling_pp) / denominator
    else:
        row["branch_asymmetry_ratio"] = np.nan

    return row


def _append_drift_columns(per_cycle: pd.DataFrame, main_field_axis: str) -> pd.DataFrame:
    summary = per_cycle.copy()
    if summary.empty:
        return summary

    first = summary.iloc[0]
    drift_targets = {
        "current_pp_drift_ratio": "achieved_current_pp_a",
        f"{main_field_axis}_pp_drift_ratio": f"achieved_{main_field_axis}_pp",
        "temperature_drift_ratio": "temperature_mean_c",
        "gain_drift_ratio": f"{main_field_axis}_gain_mT_per_a",
    }
    for drift_column, source_column in drift_targets.items():
        baseline = float(first[source_column]) if np.isfinite(first[source_column]) else np.nan
        if not np.isfinite(baseline) or baseline == 0:
            summary[drift_column] = np.nan
            continue
        summary[drift_column] = (summary[source_column] - baseline) / baseline
    return summary


def _build_test_summary(
    annotated_frame: pd.DataFrame,
    per_cycle: pd.DataFrame,
    current_channel: str,
    main_field_axis: str,
) -> pd.DataFrame:
    first_row = annotated_frame.iloc[0]
    cold = per_cycle.iloc[0]
    warm = per_cycle.tail(min(3, len(per_cycle)))
    warm_field_gain = float(warm[f"{main_field_axis}_gain_mT_per_a"].mean())
    cold_field_gain = float(cold[f"{main_field_axis}_gain_mT_per_a"])
    first_field_pp = float(cold[f"achieved_{main_field_axis}_pp"])
    last_field_pp = float(per_cycle.iloc[-1][f"achieved_{main_field_axis}_pp"])
    first_current_pp = float(cold["achieved_current_pp_a"])
    last_current_pp = float(per_cycle.iloc[-1]["achieved_current_pp_a"])

    row: dict[str, Any] = {
        "source_file": first_row["source_file"],
        "sheet_name": first_row["sheet_name"],
        "test_id": first_row["test_id"],
        "waveform_type": first_row["waveform_type"],
        "freq_hz": first_row["freq_hz"],
        "current_pp_target_a": first_row["current_pp_target_a"],
        "current_pk_target_a": first_row["current_pk_target_a"],
        "cycle_total_expected": first_row["cycle_total_expected"],
        "cycle_detected_count": int(per_cycle["cycle_index"].nunique()),
        "selected_current_channel": current_channel,
        "main_field_axis": main_field_axis,
        "detected_frequency_hz_mean": float(per_cycle["detected_frequency_hz"].mean()),
        "achieved_current_pp_a_mean": float(per_cycle["achieved_current_pp_a"].mean()),
        "achieved_current_peak_a_mean": float(per_cycle["achieved_current_peak_a"].mean()),
        "daq_input_v_pp_mean": float(per_cycle["daq_input_v_pp"].mean()),
        "amp_gain_setting_mean": float(per_cycle["amp_gain_setting_mean"].mean())
        if "amp_gain_setting_mean" in per_cycle.columns
        else np.nan,
        "temperature_mean_c_mean": float(per_cycle["temperature_mean_c"].mean()),
        "temperature_rise_total_c": float(
            annotated_frame["temperature_c"].iloc[-1] - annotated_frame["temperature_c"].iloc[0]
        ),
        "current_retention": float(per_cycle["achieved_current_pp_a"].mean() / first_row["current_pp_target_a"])
        if pd.notna(first_row["current_pp_target_a"]) and first_row["current_pp_target_a"] not in (0, np.nan)
        else np.nan,
        "first_vs_last_current_change_ratio": (last_current_pp - first_current_pp) / first_current_pp
        if first_current_pp
        else np.nan,
        "first_vs_last_field_change_ratio": (last_field_pp - first_field_pp) / first_field_pp
        if first_field_pp
        else np.nan,
        "warm_gain_mean": warm_field_gain,
        "cold_gain_mean": cold_field_gain,
        "thermal_drift_ratio": warm_field_gain / cold_field_gain if cold_field_gain else np.nan,
        "warning_flags": "",
    }

    for axis in FIELD_AXES:
        if f"achieved_{axis}_pp" not in per_cycle.columns:
            continue
        row[f"achieved_{axis}_pp_mean"] = float(per_cycle[f"achieved_{axis}_pp"].mean())
        row[f"achieved_{axis}_peak_mean"] = float(per_cycle[f"achieved_{axis}_peak"].mean())
        row[f"{axis}_gain_mT_per_a_mean"] = float(per_cycle[f"{axis}_gain_mT_per_a"].mean())

    warning_flags: list[str] = []
    if row["cycle_detected_count"] != int(first_row["cycle_total_expected"]):
        warning_flags.append("cycle_count_mismatch")
    if abs(row["first_vs_last_field_change_ratio"]) > 0.1:
        warning_flags.append("excessive_field_drift")
    if abs(row["first_vs_last_current_change_ratio"]) > 0.1:
        warning_flags.append("excessive_current_drift")

    row["warning_flags"] = flatten_messages(warning_flags)
    return pd.DataFrame([row])


def compute_loop_area(x: pd.Series, y: pd.Series) -> float:
    """Approximate loop area via line integral."""

    x_numeric = pd.to_numeric(x, errors="coerce")
    y_numeric = pd.to_numeric(y, errors="coerce")
    valid = x_numeric.notna() & y_numeric.notna()
    if valid.sum() < 3:
        return float("nan")
    return float(abs(np.trapezoid(y_numeric.loc[valid], x_numeric.loc[valid])))


def interpolate_x_at_y_zero(x: pd.Series, y: pd.Series) -> float:
    """Estimate x where y crosses zero."""

    return _interpolate_axis_at_zero(x=x, y=y, solve_for="x")


def interpolate_y_at_x_zero(x: pd.Series, y: pd.Series) -> float:
    """Estimate y where x crosses zero."""

    return _interpolate_axis_at_zero(x=x, y=y, solve_for="y")


def _interpolate_axis_at_zero(x: pd.Series, y: pd.Series, solve_for: str) -> float:
    x_numeric = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_numeric = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    values: list[float] = []

    for index in range(len(x_numeric) - 1):
        x0, x1 = x_numeric[index], x_numeric[index + 1]
        y0, y1 = y_numeric[index], y_numeric[index + 1]
        if not np.isfinite(x0) or not np.isfinite(x1) or not np.isfinite(y0) or not np.isfinite(y1):
            continue
        if solve_for == "x" and y0 == 0:
            values.append(x0)
            continue
        if solve_for == "y" and x0 == 0:
            values.append(y0)
            continue
        if solve_for == "x" and y0 * y1 < 0:
            ratio = abs(y0) / (abs(y0) + abs(y1))
            values.append(float(x0 + (x1 - x0) * ratio))
        if solve_for == "y" and x0 * x1 < 0:
            ratio = abs(x0) / (abs(x0) + abs(x1))
            values.append(float(y0 + (y1 - y0) * ratio))

    if not values:
        return float("nan")
    return float(np.mean(values))
