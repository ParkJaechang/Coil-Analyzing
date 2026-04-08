"""Plotly figure builders for the Streamlit UI and export."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go


def waveform_figure(
    df: pd.DataFrame,
    fitted_signals: dict[str, list[float]] | None = None,
    title: str = "Raw Waveforms",
) -> go.Figure:
    fig = go.Figure()
    for column in [column for column in df.columns if column != "time_s"]:
        fig.add_trace(go.Scatter(x=df["time_s"], y=df[column], mode="lines", name=column))
        if fitted_signals and column in fitted_signals:
            fig.add_trace(
                go.Scatter(
                    x=df["time_s"],
                    y=fitted_signals[column],
                    mode="lines",
                    name=f"{column} fundamental fit",
                    line={"dash": "dash"},
                )
            )
    fig.update_layout(title=title, xaxis_title="Time [s]", yaxis_title="Amplitude")
    return fig


def loop_figure(x: pd.Series, y: pd.Series, title: str, x_title: str, y_title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=title))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
    return fig


def status_heatmap(status_df: pd.DataFrame) -> go.Figure:
    mapping = {"not tested": 0, "data loaded": 1, "analyzed": 2, "flagged": 3}
    z_values = [[mapping.get(value, 0)] for value in status_df["status"]]
    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=["status"],
            y=status_df["frequency_hz"].astype(str),
            colorscale=[
                [0.0, "#e5e7eb"],
                [0.33, "#60a5fa"],
                [0.66, "#34d399"],
                [1.0, "#f87171"],
            ],
            showscale=False,
        )
    )
    fig.update_layout(title="Requested / Loaded / Analyzed / Flagged Status")
    return fig


def frequency_summary_figure(summary_df: pd.DataFrame, y_column: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=summary_df["frequency_hz"],
            y=summary_df[y_column],
            mode="lines+markers",
            name=y_column,
        )
    )
    fig.update_layout(title=title, xaxis_title="Frequency [Hz]", yaxis_title=y_column)
    return fig


def reference_comparison_figure(
    measured_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    measured_col: str,
    reference_col: str,
    title: str,
) -> go.Figure:
    fig = go.Figure()
    if not measured_df.empty:
        fig.add_trace(
            go.Scatter(
                x=measured_df["frequency_hz"],
                y=measured_df[measured_col],
                mode="lines+markers",
                name=f"Large-signal measured {measured_col}",
            )
        )
    if not reference_df.empty:
        fig.add_trace(
            go.Scatter(
                x=reference_df["frequency_hz"],
                y=reference_df[reference_col],
                mode="lines+markers",
                name=f"Small-signal reference {reference_col}",
            )
        )
    fig.update_layout(title=title, xaxis_title="Frequency [Hz]")
    return fig


def phasor_summary_figure(electrical_metrics: dict[str, Any]) -> go.Figure:
    fig = go.Figure()
    if not electrical_metrics:
        return fig
    fig.add_trace(
        go.Bar(
            x=["Voltage", "Current"],
            y=[electrical_metrics["phase_V_deg"], electrical_metrics["phase_I_deg"]],
            name="Phase [deg]",
        )
    )
    fig.update_layout(title="V-I Phasor Phase Summary", yaxis_title="Phase [deg]")
    return fig
