"""Small plot helpers for finite second-modeling UI."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_labeled_frame(frame: pd.DataFrame, columns: list[str], *, title: str, yaxis_title: str) -> go.Figure:
    figure = go.Figure()
    for column in columns:
        if column not in frame.columns:
            continue
        figure.add_trace(
            go.Scatter(
                x=frame["time_s"],
                y=frame[column],
                mode="lines",
                name=column,
                hovertemplate="시간=%{x:.4f}s<br>값=%{y:.4f}<extra>" + column + "</extra>",
            )
        )
    figure.update_layout(
        template="plotly_white",
        height=360,
        title=title,
        xaxis_title="시간 (s)",
        yaxis_title=yaxis_title,
        legend_title="항목",
    )
    return figure


def render_correction_discontinuity_diagnostics(st: Any, metadata: dict[str, object], *, title: str) -> None:
    diagnostic_keys = [
        "correction_discontinuity_detected",
        "correction_discontinuity_time_s",
        "correction_discontinuity_source",
        "max_abs_delta_step_v",
        "max_abs_second_voltage_step_v",
        "discontinuity_threshold_v",
        "active_end_kink_detected",
        "active_end_kink_time_s",
        "active_end_kink_source",
        "active_end_delta_step_v",
        "active_end_second_voltage_step_v",
        "polarity_guard_mode",
    ]
    with st.expander(title, expanded=False):
        st.dataframe(
            pd.DataFrame([(key, metadata.get(key)) for key in diagnostic_keys], columns=["항목", "값"]),
            use_container_width=True,
            hide_index=True,
        )
        if metadata.get("correction_discontinuity_detected"):
            st.warning("2차 보정 전압에 큰 불연속이 감지되었습니다. 상세 진단에서 원인을 확인하십시오.")
        if metadata.get("active_end_kink_detected"):
            st.warning(f"2차 command active 끝부분에 큰 꺾임이 감지되었습니다. 원인: {metadata.get('active_end_kink_source', 'unknown')}")


def add_peak_alignment_markers(
    figure: go.Figure,
    frame: pd.DataFrame,
    metadata: dict[str, object],
    *,
    target_label: str,
    smoothed_label: str,
    aligned_label: str,
    target_peak_label: str,
    measured_peak_label: str,
    aligned_peak_label: str,
) -> go.Figure:
    marker_specs = [
        ("target_first_peak_time_s", target_label, target_peak_label),
        ("measured_first_peak_time_s", smoothed_label, measured_peak_label),
        ("target_first_peak_time_s", aligned_label, aligned_peak_label),
    ]
    x_values = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
    for time_key, column, label in marker_specs:
        if column not in frame.columns:
            continue
        peak_time = metadata.get(time_key)
        try:
            peak_time_value = float(peak_time)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(peak_time_value):
            continue
        y_value = np.interp(
            peak_time_value,
            x_values,
            pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float),
            left=np.nan,
            right=np.nan,
        )
        if not np.isfinite(y_value):
            continue
        figure.add_trace(
            go.Scatter(
                x=[peak_time_value],
                y=[y_value],
                mode="markers",
                name=label,
                marker={"size": 10, "symbol": "circle"},
                hovertemplate="시간=%{x:.4f}s<br>값=%{y:.4f}<extra>" + label + "</extra>",
            )
        )
    return figure
