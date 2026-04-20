from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utils import canonicalize_waveform_type


PLOT_TEMPLATE = "plotly_white"


def plot_waveforms(
    frame: pd.DataFrame,
    channels: list[str],
    title: str,
) -> go.Figure:
    """Plot selected time-series channels."""

    figure = go.Figure()
    for channel in channels:
        if channel not in frame.columns:
            continue
        figure.add_trace(
            go.Scatter(
                x=frame["time_s"],
                y=frame[channel],
                mode="lines",
                name=channel,
                hovertemplate="time=%{x:.4f}s<br>value=%{y:.4f}<extra>" + channel + "</extra>",
            )
        )
    figure.update_layout(
        template=PLOT_TEMPLATE,
        height=460,
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Signal",
        legend_title="Channel",
    )
    return figure


def plot_cycle_detection_overlay(
    frame: pd.DataFrame,
    reference_channel: str,
    boundaries: list,
) -> go.Figure:
    """Overlay detected cycle boundaries on the reference signal."""

    figure = plot_waveforms(frame, [reference_channel], title="Cycle Detection Overlay")
    for boundary in boundaries:
        figure.add_vline(
            x=boundary.start_s,
            line_dash="dash",
            line_color="firebrick",
            opacity=0.7,
        )
    if boundaries:
        figure.add_vline(
            x=boundaries[-1].end_s,
            line_dash="dash",
            line_color="firebrick",
            opacity=0.7,
        )
    return figure


def plot_cycle_overlay(
    frame: pd.DataFrame,
    channel: str,
    x_mode: str = "cycle_progress",
    show_mean_band: bool = True,
) -> go.Figure:
    """Overlay all cycles and optionally add mean/std band."""

    figure = go.Figure()
    cycle_frame = frame.dropna(subset=["cycle_index"]).copy()
    if cycle_frame.empty or channel not in cycle_frame.columns:
        figure.update_layout(template=PLOT_TEMPLATE, title="Cycle Overlay")
        return figure

    x_column = "cycle_progress" if x_mode == "cycle_progress" else "cycle_time_s"
    for cycle_index, group in cycle_frame.groupby("cycle_index", sort=True):
        figure.add_trace(
            go.Scatter(
                x=group[x_column],
                y=group[channel],
                mode="lines",
                name=f"Cycle {int(cycle_index)}",
                opacity=0.42,
            )
        )

    if show_mean_band:
        x_grid, mean_values, std_values = _interpolated_cycle_band(cycle_frame, channel, x_column)
        if x_grid is not None:
            figure.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=mean_values + std_values,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=mean_values - std_values,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(31, 119, 180, 0.15)",
                    name="Mean ± 1σ",
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=mean_values,
                    mode="lines",
                    line=dict(color="black", width=3),
                    name="Mean Cycle",
                )
            )

    figure.update_layout(
        template=PLOT_TEMPLATE,
        height=500,
        title=f"Cycle Overlay: {channel}",
        xaxis_title="Cycle Progress" if x_mode == "cycle_progress" else "Cycle Time (s)",
        yaxis_title=channel,
    )
    return figure


def plot_loop(
    frame: pd.DataFrame,
    current_channel: str,
    field_channel: str,
    color_by: str = "branch_direction",
) -> go.Figure:
    """Plot a B vs I loop with color grouping."""

    plot_frame = frame.dropna(subset=[current_channel, field_channel]).copy()
    if plot_frame.empty:
        return go.Figure(layout=dict(template=PLOT_TEMPLATE, title="Loop Analysis"))

    if color_by not in plot_frame.columns:
        color_by = "cycle_index" if "cycle_index" in plot_frame.columns else None

    figure = px.line(
        plot_frame,
        x=current_channel,
        y=field_channel,
        color=color_by,
        line_group="cycle_index" if "cycle_index" in plot_frame.columns else None,
        hover_data=["time_s", "cycle_index", "branch_direction"],
        template=PLOT_TEMPLATE,
        title=f"Loop Analysis: {field_channel} vs {current_channel}",
    )
    figure.update_layout(height=520)
    return figure


def plot_frequency_comparison(
    per_test_summary: pd.DataFrame,
    metric: str,
) -> go.Figure:
    """Compare one metric across frequency, current, and waveform."""

    if per_test_summary.empty or metric not in per_test_summary.columns:
        return go.Figure(layout=dict(template=PLOT_TEMPLATE, title="Frequency Comparison"))

    figure = px.line(
        per_test_summary.sort_values(["waveform_type", "current_pp_target_a", "freq_hz"]),
        x="freq_hz",
        y=metric,
        color="waveform_type",
        line_dash="current_pp_target_a",
        markers=True,
        hover_data=["test_id", "current_pp_target_a"],
        template=PLOT_TEMPLATE,
        title=f"Frequency / Amplitude Comparison: {metric}",
    )
    figure.update_layout(height=500, xaxis_title="Frequency (Hz)", yaxis_title=metric)
    return figure


def plot_shape_overlay(
    overlay_frame: pd.DataFrame,
    y_column: str,
    title: str,
    x_column: str = "cycle_progress",
) -> go.Figure:
    """Overlay normalized representative cycles for multiple tests."""

    required = {x_column, y_column, "legend_label", "test_id"}
    if overlay_frame.empty or not required.issubset(overlay_frame.columns):
        return go.Figure(layout=dict(template=PLOT_TEMPLATE, title=title))

    plot_frame = overlay_frame.sort_values(["current_pp_target_a", "test_id", x_column]).copy()
    hover_columns = [
        column
        for column in (
            "test_id",
            "current_pp_target_a",
            "achieved_current_pp_a_mean",
            "phase_lag_deg",
            "shape_nrmse_aligned",
            "is_reference",
        )
        if column in plot_frame.columns
    ]
    figure = px.line(
        plot_frame,
        x=x_column,
        y=y_column,
        color="legend_label",
        line_group="test_id",
        hover_data=hover_columns,
        template=PLOT_TEMPLATE,
        title=title,
    )
    figure.update_layout(
        height=460,
        xaxis_title="Cycle Progress" if x_column == "cycle_progress" else "Time (s)",
        yaxis_title="Normalized Signal",
        legend_title="Test",
    )
    figure.add_hline(y=0.0, line_dash="dot", line_color="gray", opacity=0.6)
    return figure


def plot_shape_metric_trend(
    summary_frame: pd.DataFrame,
    metric: str,
    x_metric: str = "current_pp_target_a",
    title: str | None = None,
) -> go.Figure:
    """Plot one shape/phase metric against current level."""

    required = {metric, x_metric}
    if summary_frame.empty or not required.issubset(summary_frame.columns):
        return go.Figure(layout=dict(template=PLOT_TEMPLATE, title=title or "Shape Metric Trend"))

    plot_frame = summary_frame.sort_values(x_metric).copy()
    hover_columns = [
        column
        for column in ("test_id", "achieved_current_pp_a_mean", "shape_corr_aligned", "shape_nrmse_aligned")
        if column in plot_frame.columns
    ]
    figure = px.line(
        plot_frame,
        x=x_metric,
        y=metric,
        markers=True,
        hover_data=hover_columns,
        template=PLOT_TEMPLATE,
        title=title or f"{metric} vs {x_metric}",
    )
    figure.update_layout(height=380, xaxis_title=x_metric, yaxis_title=metric)
    return figure


def plot_metric_heatmap(
    per_test_summary: pd.DataFrame,
    metric: str,
    waveform_type: str | None = None,
) -> go.Figure:
    """Build a frequency-current heatmap for a selected metric."""

    if per_test_summary.empty or metric not in per_test_summary.columns:
        return go.Figure(layout=dict(template=PLOT_TEMPLATE, title="Heatmap"))

    plot_frame = per_test_summary.copy()
    if waveform_type and waveform_type != "all":
        requested_waveform = canonicalize_waveform_type(waveform_type)
        if requested_waveform is None:
            return go.Figure(layout=dict(template=PLOT_TEMPLATE, title="Heatmap"))
        plot_frame = plot_frame[plot_frame["waveform_type"].map(canonicalize_waveform_type) == requested_waveform]

    if plot_frame.empty:
        return go.Figure(layout=dict(template=PLOT_TEMPLATE, title="Heatmap"))

    pivot = plot_frame.pivot_table(
        index="current_pp_target_a",
        columns="freq_hz",
        values=metric,
        aggfunc="mean",
    ).sort_index(axis=0).sort_index(axis=1)

    figure = go.Figure(
        data=go.Heatmap(
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            z=pivot.to_numpy(dtype=float),
            colorscale="Viridis",
            colorbar_title=metric,
        )
    )
    figure.update_layout(
        template=PLOT_TEMPLATE,
        height=460,
        title=f"Heatmap: {metric}",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Current Target PP (A)",
    )
    return figure


def plot_drift(
    per_cycle_summary: pd.DataFrame,
    metric: str,
) -> go.Figure:
    """Plot cycle-to-cycle drift for one metric."""

    if per_cycle_summary.empty or metric not in per_cycle_summary.columns:
        return go.Figure(layout=dict(template=PLOT_TEMPLATE, title="Drift"))

    figure = px.line(
        per_cycle_summary,
        x="cycle_index",
        y=metric,
        markers=True,
        color="test_id",
        template=PLOT_TEMPLATE,
        title=f"Cycle Drift: {metric}",
    )
    figure.update_layout(height=460, xaxis_title="Cycle Index", yaxis_title=metric)
    return figure


def plot_temperature_vs_drift(
    per_cycle_summary: pd.DataFrame,
    drift_metric: str,
) -> go.Figure:
    """Plot temperature versus drift-like metric."""

    if per_cycle_summary.empty or drift_metric not in per_cycle_summary.columns:
        return go.Figure(layout=dict(template=PLOT_TEMPLATE, title="Thermal / Drift"))

    figure = px.scatter(
        per_cycle_summary,
        x="temperature_mean_c",
        y=drift_metric,
        color="test_id",
        hover_data=["cycle_index"],
        template=PLOT_TEMPLATE,
        title=f"Thermal / Drift: {drift_metric}",
    )
    figure.update_layout(height=460, xaxis_title="Temperature Mean (C)", yaxis_title=drift_metric)
    return figure


def plot_operating_map(
    per_test_summary: pd.DataFrame,
    field_axis: str = "bz_mT",
) -> go.Figure:
    """Plot achieved field operating points across frequency/current."""

    metric = f"achieved_{field_axis}_pp_mean"
    if per_test_summary.empty or metric not in per_test_summary.columns:
        return go.Figure(layout=dict(template=PLOT_TEMPLATE, title="Operating Map"))

    figure = px.scatter(
        per_test_summary,
        x="freq_hz",
        y="current_pp_target_a",
        size=metric,
        color="waveform_type",
        hover_data=["test_id", metric, "current_retention", "temperature_rise_total_c"],
        template=PLOT_TEMPLATE,
        title=f"Operating Map: {field_axis}",
    )
    figure.update_layout(
        height=500,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Current Target PP (A)",
    )
    return figure


def plot_lut_lookup_curve(
    lookup_table: pd.DataFrame,
    target_metric: str,
    voltage_metric: str = "daq_input_v_pp_mean",
) -> go.Figure:
    """Plot measured LUT points for target metric versus voltage."""

    if lookup_table.empty or target_metric not in lookup_table.columns or voltage_metric not in lookup_table.columns:
        return go.Figure(layout=dict(template=PLOT_TEMPLATE, title="LUT Lookup"))

    figure = px.line(
        lookup_table.sort_values(target_metric),
        x=target_metric,
        y=voltage_metric,
        markers=True,
        hover_data=["test_id", "current_pp_target_a", "achieved_bz_mT_pp_mean", "achieved_current_pp_a_mean"],
        template=PLOT_TEMPLATE,
        title=f"LUT Lookup: {target_metric} -> {voltage_metric}",
    )
    figure.update_layout(height=420, xaxis_title=target_metric, yaxis_title=voltage_metric)
    return figure


def plot_frequency_support_curve(
    frequency_support_table: pd.DataFrame,
    voltage_metric: str = "estimated_voltage_pp",
    requested_freq_hz: float | None = None,
    used_freq_hz: float | None = None,
) -> go.Figure:
    """Plot the frequency-trend support points used for LUT interpolation."""

    if (
        frequency_support_table.empty
        or "freq_hz" not in frequency_support_table.columns
        or voltage_metric not in frequency_support_table.columns
    ):
        return go.Figure(layout=dict(template=PLOT_TEMPLATE, title="Frequency Trend"))

    hover_columns = [
        column
        for column in (
            "template_test_id",
            "support_point_count",
            "available_target_min",
            "available_target_max",
            "local_mode",
        )
        if column in frequency_support_table.columns
    ]
    figure = px.line(
        frequency_support_table.sort_values("freq_hz"),
        x="freq_hz",
        y=voltage_metric,
        markers=True,
        hover_data=hover_columns,
        template=PLOT_TEMPLATE,
        title="Frequency Trend Support",
    )
    if requested_freq_hz is not None:
        figure.add_vline(
            x=float(requested_freq_hz),
            line_dash="dash",
            line_color="firebrick",
            annotation_text="requested",
            annotation_position="top right",
        )
    if used_freq_hz is not None and (
        requested_freq_hz is None or abs(float(used_freq_hz) - float(requested_freq_hz)) > 1e-9
    ):
        figure.add_vline(
            x=float(used_freq_hz),
            line_dash="dot",
            line_color="darkorange",
            annotation_text="used",
            annotation_position="top left",
        )
    figure.update_layout(
        height=420,
        xaxis_title="Frequency (Hz)",
        yaxis_title=voltage_metric,
    )
    return figure


def plot_command_waveform(
    command_waveform: pd.DataFrame,
    value_column: str = "recommended_voltage_v",
) -> go.Figure:
    """Plot the recommended one-cycle command waveform."""

    if command_waveform.empty or value_column not in command_waveform.columns:
        return go.Figure(layout=dict(template=PLOT_TEMPLATE, title="Command Waveform"))

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=command_waveform["time_s"],
            y=command_waveform[value_column],
            mode="lines",
            name=value_column,
        )
    )
    figure.update_layout(
        template=PLOT_TEMPLATE,
        height=420,
        title="Recommended Command Waveform",
        xaxis_title="Time (s)",
        yaxis_title="Voltage (V)",
    )
    if "is_active_target" in command_waveform.columns and command_waveform["is_active_target"].any():
        active_end_s = float(command_waveform.loc[command_waveform["is_active_target"], "time_s"].max())
        figure.add_vline(
            x=active_end_s,
            line_dash="dash",
            line_color="firebrick",
            annotation_text="target end",
            annotation_position="top right",
        )
    return figure


def plot_command_response_overview(
    frame: pd.DataFrame,
    title: str = "Recommended Command and Expected Response",
    voltage_column: str = "limited_voltage_v",
    current_column: str = "expected_current_a",
    field_column: str = "expected_field_mT",
) -> go.Figure:
    """Show recommended command voltage with expected current and field response."""

    if frame.empty or "time_s" not in frame.columns:
        return go.Figure(layout=dict(template=PLOT_TEMPLATE, title=title))

    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Command Voltage", "Expected Current", "Expected Field"),
    )
    for row_index, column, label, yaxis_title in (
        (1, voltage_column, "Voltage", "Voltage (V)"),
        (2, current_column, "Current", "Current (A)"),
        (3, field_column, "Field", "Field (mT)"),
    ):
        if column not in frame.columns:
            continue
        figure.add_trace(
            go.Scatter(
                x=frame["time_s"],
                y=frame[column],
                mode="lines",
                name=label,
            ),
            row=row_index,
            col=1,
        )
        figure.update_yaxes(title_text=yaxis_title, row=row_index, col=1)

    if "target_output" in frame.columns:
        figure.add_trace(
            go.Scatter(
                x=frame["time_s"],
                y=frame["target_output"],
                mode="lines",
                name="Target Output",
                line=dict(dash="dash"),
            ),
            row=3 if field_column in frame.columns else 2,
            col=1,
        )

    if "is_active_target" in frame.columns and frame["is_active_target"].any():
        active_end_s = float(frame.loc[frame["is_active_target"], "time_s"].max())
        for row_index in (1, 2, 3):
            figure.add_vline(
                x=active_end_s,
                line_dash="dash",
                line_color="firebrick",
                row=row_index,
                col=1,
            )

    figure.update_layout(
        template=PLOT_TEMPLATE,
        height=820,
        title=title,
        xaxis3_title="Time (s)",
        legend_title="Signal",
    )
    return figure


def plot_formula_comparison(
    reconstruction_frame: pd.DataFrame,
) -> go.Figure:
    """Compare the original recommended command waveform against its Fourier reconstruction."""

    if reconstruction_frame.empty:
        return go.Figure(layout=dict(template=PLOT_TEMPLATE, title="Formula Comparison"))

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=reconstruction_frame["time_s"],
            y=reconstruction_frame["original_voltage_v"],
            mode="lines",
            name="Original Command",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=reconstruction_frame["time_s"],
            y=reconstruction_frame["formula_voltage_v"],
            mode="lines",
            name="Fourier Formula",
            line=dict(dash="dash"),
        )
    )
    figure.update_layout(
        template=PLOT_TEMPLATE,
        height=420,
        title="Fourier Formula Comparison",
        xaxis_title="Time (s)",
        yaxis_title="Voltage (V)",
    )
    return figure


def plot_current_compensation_waveforms(
    command_profile: pd.DataFrame,
    nearest_profile: pd.DataFrame | None = None,
) -> go.Figure:
    """Compare target current waveform against nearest measured current waveform."""

    figure = go.Figure()
    if not command_profile.empty and "target_current_a" in command_profile.columns:
        figure.add_trace(
            go.Scatter(
                x=command_profile["time_s"],
                y=command_profile["target_current_a"],
                mode="lines",
                name="Target Current",
            )
        )
    if not command_profile.empty and "used_target_current_a" in command_profile.columns:
        figure.add_trace(
            go.Scatter(
                x=command_profile["time_s"],
                y=command_profile["used_target_current_a"],
                mode="lines",
                name="Used Target Current",
                line=dict(dash="dash"),
            )
        )
    if nearest_profile is not None and not nearest_profile.empty and "measured_current_a" in nearest_profile.columns:
        figure.add_trace(
            go.Scatter(
                x=nearest_profile["time_s"],
                y=nearest_profile["measured_current_a"],
                mode="lines",
                name="Support-Blended Current",
                line=dict(color="#1f77b4", dash="dot"),
            )
        )
    figure.update_layout(
        template=PLOT_TEMPLATE,
        height=420,
        title="Current Waveform Compensation",
        xaxis_title="Time (s)",
        yaxis_title="Current (A)",
    )
    return figure


def plot_output_compensation_waveforms(
    command_profile: pd.DataFrame,
    nearest_profile: pd.DataFrame | None,
    nearest_column: str,
    title: str,
    yaxis_title: str,
    predicted_label: str = "Predicted Output",
    reference_label: str = "Support-Blended Output",
) -> go.Figure:
    """Compare generic target output waveform against nearest measured output waveform."""

    figure = go.Figure()
    target_column = "aligned_target_output" if "aligned_target_output" in command_profile.columns else "target_output"
    used_target_column = (
        "aligned_used_target_output"
        if "aligned_used_target_output" in command_profile.columns
        else "used_target_output"
    )
    if not command_profile.empty and target_column in command_profile.columns:
        figure.add_trace(
            go.Scatter(
                x=command_profile["time_s"],
                y=command_profile[target_column],
                mode="lines",
                name="Target Output",
            )
        )
    if not command_profile.empty and used_target_column in command_profile.columns:
        target_values = pd.to_numeric(command_profile[target_column], errors="coerce").to_numpy(dtype=float)
        used_values = pd.to_numeric(command_profile[used_target_column], errors="coerce").to_numpy(dtype=float)
        if not np.allclose(target_values, used_values, equal_nan=True):
            figure.add_trace(
                go.Scatter(
                    x=command_profile["time_s"],
                    y=command_profile[used_target_column],
                    mode="lines",
                    name="Lag-Compensated Target",
                    line=dict(dash="dash"),
                )
            )
    predicted_column = None
    if "expected_output" in command_profile.columns:
        predicted_column = "expected_output"
    elif nearest_column == "measured_current_a" and "expected_current_a" in command_profile.columns:
        predicted_column = "expected_current_a"
    elif nearest_column == "measured_field_mT" and "expected_field_mT" in command_profile.columns:
        predicted_column = "expected_field_mT"
    elif "modeled_output" in command_profile.columns:
        predicted_column = "modeled_output"
    if predicted_column is not None and predicted_column in command_profile.columns:
        figure.add_trace(
            go.Scatter(
                x=command_profile["time_s"],
                y=command_profile[predicted_column],
                mode="lines",
                name=predicted_label,
                line=dict(color="#00cc96"),
            )
        )
    if nearest_profile is not None and not nearest_profile.empty and nearest_column in nearest_profile.columns:
        figure.add_trace(
            go.Scatter(
                x=nearest_profile["time_s"],
                y=nearest_profile[nearest_column],
                mode="lines",
                name=reference_label,
                line=dict(color="#1f77b4", dash="dot"),
            )
        )
    figure.update_layout(
        template=PLOT_TEMPLATE,
        height=420,
        title=title,
        xaxis_title="Time (s)",
        yaxis_title=yaxis_title,
    )
    if "is_active_target" in command_profile.columns and command_profile["is_active_target"].any():
        active_end_s = float(command_profile.loc[command_profile["is_active_target"], "time_s"].max())
        figure.add_vline(
            x=active_end_s,
            line_dash="dash",
            line_color="firebrick",
            annotation_text="target end",
            annotation_position="top right",
        )
    return figure


def plot_coverage_matrix(
    coverage: pd.DataFrame,
    *,
    level_unit: str = "A",
    title: str = "Test Coverage Matrix",
    xaxis_title: str | None = None,
) -> go.Figure:
    """Visualize which waveform/frequency/current conditions are present."""

    if coverage.empty:
        return go.Figure(layout=dict(template=PLOT_TEMPLATE, title=title))

    def _format_index_label(value: object) -> str:
        if isinstance(value, tuple):
            parts: list[str] = []
            for index, part in enumerate(value):
                if isinstance(part, (int, float, np.integer, np.floating)) and np.isfinite(float(part)):
                    if index == 1:
                        parts.append(f"{float(part):g}Hz")
                    else:
                        parts.append(f"{float(part):g}")
                else:
                    parts.append(str(part))
            return " | ".join(parts)
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
            return f"{float(value):g}"
        return str(value)

    def _format_level_label(value: object) -> str:
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
            return f"{float(value):g}{level_unit}"
        return str(value)

    y_labels = [_format_index_label(index_value) for index_value in coverage.index.tolist()]
    figure = go.Figure(
        data=go.Heatmap(
            x=[_format_level_label(value) for value in coverage.columns.tolist()],
            y=y_labels,
            z=coverage.to_numpy(dtype=float),
            colorscale=[[0.0, "#f3f4f6"], [1.0, "#0f766e"]],
            showscale=False,
        )
    )
    figure.update_layout(
        template=PLOT_TEMPLATE,
        height=max(320, 40 * len(y_labels)),
        title=title,
        xaxis_title=xaxis_title or ("Current Target PP" if level_unit == "A" else "Target Level"),
        yaxis_title="Waveform / Frequency",
    )
    return figure


def _interpolated_cycle_band(
    cycle_frame: pd.DataFrame,
    channel: str,
    x_column: str,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    grouped = list(cycle_frame.groupby("cycle_index", sort=True))
    if len(grouped) < 2:
        return None, None, None

    x_grid = np.linspace(0.0, 1.0, 250) if x_column == "cycle_progress" else np.linspace(
        0.0,
        float(cycle_frame[x_column].max()),
        250,
    )
    stacked: list[np.ndarray] = []
    for _, group in grouped:
        valid = group[[x_column, channel]].dropna()
        if len(valid) < 3:
            continue
        x_values = valid[x_column].to_numpy(dtype=float)
        y_values = valid[channel].to_numpy(dtype=float)
        if np.any(np.diff(x_values) <= 0):
            continue
        stacked.append(np.interp(x_grid, x_values, y_values))

    if len(stacked) < 2:
        return None, None, None

    matrix = np.vstack(stacked)
    return x_grid, matrix.mean(axis=0), matrix.std(axis=0)
