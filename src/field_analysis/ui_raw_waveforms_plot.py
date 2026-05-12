from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .plotting import plot_waveforms
from .ui_raw_waveforms_quality import (
    add_finite_visual_markers,
    build_finite_marker_times,
    preferred_marker_channel,
    render_anomaly_helper,
    render_channel_timebase_summary,
    render_finite_marker_summary,
)


METADATA_HIDDEN_COLUMNS = {"source_file", "sheet_name", "test_id", "notes", "parse_warnings"}


def render_waveform_normalization_summary(metadata: dict[str, Any]) -> None:
    if not metadata:
        return
    st.markdown("#### 정규화 요약")
    st.caption("Raw data는 보존하고 normalized data는 review/modeling용입니다.")
    if "finite_normalization_mode" in metadata:
        st.caption("finite: active segment 기준 정규화입니다. pre/post rest는 scale 계산에서 제외합니다.")
    if "waveform_normalization_window" in metadata:
        st.caption("continuous: startup / steady-state window 구분을 유지합니다.")
    keys = [
        "waveform_normalization_enabled",
        "waveform_normalization_mode",
        "waveform_normalization_window",
        "waveform_normalization_status",
        "waveform_normalization_source_peak_mT",
        "waveform_normalization_scale_factor",
        "startup_window_start_s",
        "startup_window_end_s",
        "steady_state_start_s",
        "steady_state_end_s",
        "finite_normalization_enabled",
        "finite_normalization_mode",
        "finite_normalization_status",
        "finite_normalization_source_peak_mT",
        "finite_normalization_scale_factor",
        "finite_active_window_start_s",
        "finite_active_window_end_s",
        "raw_field_peak_mT",
        "normalized_field_peak_mT",
        "finite_positive_peak_normalized_mT",
        "finite_negative_peak_normalized_mT",
        "source_pre_baseline_excluded_from_reference",
        "source_tail_excluded_from_reference",
    ]
    rows = [{"field": _normalization_label(key), "value": metadata.get(key), "metadata_key": key} for key in keys if key in metadata]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_finite_symmetric_peak_review(display_frame: pd.DataFrame, metadata: dict[str, Any] | None) -> None:
    if metadata is None:
        return
    st.markdown("#### Finite 대칭 peak 검토")
    st.caption("절대 gain 평가가 아니라 개형/대칭성 검토용입니다.")
    st.caption("지원 cycles: 1.0 / 1.5")
    st.caption("미지원 cycles: 1.25 / 1.75")
    st.caption("1.25 / 1.75는 phase delay 때문에 peak amplitude correction 주력 경로에서 제외합니다.")
    st.caption("1.0 / 1.5 finite-cycle에 대해 양/음 peak symmetry를 검토합니다.")
    status = str(metadata.get("finite_symmetric_peak_status", "unavailable"))
    if status == "unsupported_cycle":
        st.warning("unsupported_cycle: phase-delay peak correction disabled")
    elif status != "ok":
        st.info(f"Finite 대칭 peak 검토 사용 불가: {status}")
    rows = [
        {"field": "finite_symmetric_peak_modeling_enabled", "value": metadata.get("finite_symmetric_peak_modeling_enabled")},
        {"field": "finite_symmetric_peak_cycle_supported", "value": metadata.get("finite_symmetric_peak_cycle_supported")},
        {"field": "finite_symmetric_peak_status", "value": status},
        {"field": "positive_peak_mT", "value": metadata.get("positive_peak_mT")},
        {"field": "negative_peak_mT", "value": metadata.get("negative_peak_mT")},
        {"field": "peak_symmetry_error_mT", "value": metadata.get("peak_symmetry_error_mT")},
        {"field": "peak_symmetry_ratio", "value": metadata.get("peak_symmetry_ratio")},
        {"field": "command_voltage_peak_v", "value": metadata.get("command_voltage_peak_v")},
        {"field": "command_voltage_limit_status", "value": metadata.get("command_voltage_limit_status")},
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        "그래프: 정규화 목표 자기장, 정규화 field/predicted field, 양/음 lobe marker, "
        "baseline command, symmetric peak command candidate, command delta, residual / lobe error."
    )
    _render_symmetric_peak_graphs(display_frame)


def render_raw_waveform_plot(
    record: Any,
    dataset_mode: str,
    display_frame: pd.DataFrame,
) -> None:
    default_channels = [
        "daq_input_v",
        "coil1_current_a",
        "coil2_current_a",
        "temperature_c",
        "bx_mT",
        "by_mT",
        "bz_mT",
        "bmag_mT",
        "normalized_continuous_field_mT",
        "normalized_finite_field_mT",
        "symmetric_peak_predicted_field_mT",
        "symmetric_peak_recommended_voltage_v",
        "symmetric_peak_command_delta_v",
    ]
    plottable_columns = [
        column
        for column in display_frame.columns
        if column not in METADATA_HIDDEN_COLUMNS and pd.api.types.is_numeric_dtype(display_frame[column])
    ]
    selected_channels = st.multiselect(
        "검토할 신호",
        options=plottable_columns,
        default=[channel for channel in default_channels if channel in plottable_columns],
        key="raw_channels_audit",
    )
    st.caption(
        f"Plot view: {dataset_mode} | source={record.source_file_label or 'unknown'} | "
        f"signals={', '.join(selected_channels) if selected_channels else 'none selected'}"
    )
    st.caption("raw_* 및 normalized_* field column이 함께 있으면 raw/정규화 field trace를 같이 표시합니다.")
    if not selected_channels:
        st.warning("plot할 numeric signal을 하나 이상 선택하십시오.")
        return
    marker_channel = preferred_marker_channel(display_frame, selected_channels)
    marker_times = build_finite_marker_times(display_frame, marker_channel)
    figure = plot_waveforms(display_frame, selected_channels, title=f"{record.label} / {dataset_mode}")
    if record.source_type == "finite-cycle":
        add_finite_visual_markers(figure, marker_times)
        render_finite_marker_summary(marker_times)
    st.plotly_chart(figure, use_container_width=True)
    render_channel_timebase_summary(display_frame, selected_channels)
    render_anomaly_helper(
        display_frame,
        selected_channels,
        source_type=record.source_type,
        duration_s=record.duration_s,
        freq_hz=record.freq_hz,
        cycle_count=record.cycle_count,
    )
    st.dataframe(display_frame.head(200), use_container_width=True)


def _render_symmetric_peak_graphs(frame: pd.DataFrame) -> None:
    if "time_s" not in frame.columns:
        return
    field_columns = [
        ("normalized_finite_field_mT", "normalized field/predicted field"),
        ("symmetric_peak_predicted_field_mT", "symmetric peak predicted field"),
    ]
    command_columns = [
        ("baseline_recommended_voltage_v", "baseline command"),
        ("symmetric_peak_recommended_voltage_v", "symmetric peak command candidate"),
        ("symmetric_peak_command_delta_v", "command delta"),
    ]
    residual_columns = [
        ("positive_lobe_mask", "positive lobe marker"),
        ("negative_lobe_mask", "negative lobe marker"),
    ]
    _line_chart(frame, field_columns, "정규화 목표 자기장 / field 검토", "mT")
    _line_chart(frame, command_columns, "대칭 peak command 검토", "V")
    _line_chart(frame, residual_columns, "양/음 lobe marker", "mask")


def _line_chart(frame: pd.DataFrame, columns: list[tuple[str, str]], title: str, yaxis_title: str) -> None:
    figure = go.Figure()
    for column, label in columns:
        if column not in frame.columns:
            continue
        figure.add_trace(go.Scatter(x=frame["time_s"], y=frame[column], mode="lines", name=label))
    if not figure.data:
        return
    figure.update_layout(template="plotly_white", height=300, title=title, xaxis_title="시간 (s)", yaxis_title=yaxis_title)
    st.plotly_chart(figure, use_container_width=True)


def _normalization_label(key: str) -> str:
    labels = {
        "waveform_normalization_enabled": "정규화 사용",
        "finite_normalization_enabled": "정규화 사용",
        "waveform_normalization_mode": "정규화 방식",
        "finite_normalization_mode": "정규화 방식",
        "waveform_normalization_source_peak_mT": "source peak",
        "finite_normalization_source_peak_mT": "source peak",
        "waveform_normalization_scale_factor": "scale factor",
        "finite_normalization_scale_factor": "scale factor",
        "finite_active_window_start_s": "active window start/end",
        "finite_active_window_end_s": "active window start/end",
        "raw_field_peak_mT": "raw peak",
        "normalized_field_peak_mT": "normalized peak",
    }
    return labels.get(key, key)
