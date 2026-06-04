"""Small plot helpers for finite/continuous modeling review UI."""

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


def add_max_error_marker(
    figure: go.Figure,
    frame: pd.DataFrame,
    *,
    residual_label: str,
    target_label: str | None = None,
    marker_label: str = "최대 오차 지점",
    mask_column: str | None = None,
) -> go.Figure:
    if "time_s" not in frame.columns or residual_label not in frame.columns:
        return figure
    time_s = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
    residual = pd.to_numeric(frame[residual_label], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(time_s) & np.isfinite(residual)
    if mask_column and mask_column in frame.columns:
        mask_values = pd.Series(frame[mask_column]).astype(bool).to_numpy(dtype=bool)
        if mask_values.size == valid.size:
            valid &= mask_values
    if not np.any(valid):
        return figure
    valid_indices = np.flatnonzero(valid)
    max_index = int(valid_indices[int(np.nanargmax(np.abs(residual[valid])))] )
    max_time = float(time_s[max_index])
    max_residual = float(residual[max_index])
    target_peak = np.nan
    if target_label and target_label in frame.columns:
        target = pd.to_numeric(frame[target_label], errors="coerce").to_numpy(dtype=float)
        finite_target = np.abs(target[np.isfinite(target)])
        if finite_target.size:
            target_peak = float(np.nanmax(finite_target))
    ratio_text = ""
    if np.isfinite(target_peak) and target_peak > 1e-12:
        ratio_text = f"<br>오차율={abs(max_residual) / target_peak * 100.0:.2f}%"
    figure.add_vline(x=max_time, line_dash="dot", line_color="#ff9800", annotation_text=marker_label)
    figure.add_trace(
        go.Scatter(
            x=[max_time],
            y=[max_residual],
            mode="markers",
            name=marker_label,
            marker={"size": 13, "symbol": "x", "color": "#ff9800"},
            hovertemplate=(
                "시간=%{x:.4f}s<br>"
                "residual=%{y:.4f} mT"
                + ratio_text
                + "<extra>"
                + marker_label
                + "</extra>"
            ),
        )
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
            st.warning("2차 보정 전압에 불연속이 감지되었습니다. 상세 진단에서 원인을 확인하십시오.")
        if metadata.get("active_end_kink_detected"):
            st.warning(
                f"2차 command active 끝부분에 꺾임이 감지되었습니다. 원인: {metadata.get('active_end_kink_source', 'unknown')}"
            )


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
