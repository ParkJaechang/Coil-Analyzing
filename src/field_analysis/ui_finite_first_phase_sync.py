from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .utils import first_number


def render_finite_first_phase_sync_review(command_profile: pd.DataFrame, metadata: dict[str, object]) -> None:
    if not isinstance(command_profile, pd.DataFrame) or command_profile.empty:
        return
    st.markdown("#### Finite 1차 phase sync 확인")
    st.caption("전압 피크와 실측 자기장 피크를 맞춘 뒤, ±50mT 정규화 기준으로 residual을 계산합니다.")
    summary = {
        "source file": metadata.get("finite_first_measured_source_file"),
        "source label": metadata.get("finite_first_measured_source_label"),
        "measured field column": metadata.get("finite_first_measured_source_column"),
        "actual measured source": metadata.get("finite_first_measured_source_is_actual_measured"),
        "source data origin": metadata.get("finite_first_source_data_origin"),
        "source_input_waveform_family": command_profile.get("waveform_type", pd.Series(["triangle"])).iloc[0],
        "target_field_shape": "fixed_rounded_triangle",
        "finite_first_modeling_mode": metadata.get("finite_first_modeling_mode", "phase_synced"),
        "phase_delay_s": metadata.get("phase_delay_s"),
        "phase_delay_cycles": metadata.get("phase_delay_cycles"),
        "실측 field peak (mT)": metadata.get("measured_abs_peak_effective_mT"),
        "±50mT 정규화 scale": metadata.get("measured_field_scale_to_50mT"),
        "정규화 후 peak (mT)": metadata.get("measured_aligned_normalized_peak_mT"),
        "correction_gain": metadata.get("correction_gain_used"),
        "voltage_headroom_v": metadata.get("voltage_headroom_v"),
        "clipping_fraction": metadata.get("clipping_fraction"),
        "active residual finite ratio": metadata.get("active_residual_finite_ratio"),
        "phase support status": metadata.get("phase_support_status"),
        "required source end": metadata.get("required_phase_aligned_source_end_s"),
        "actual source end": metadata.get("actual_source_time_end_s"),
        "active_end_kink_detected": metadata.get("active_end_kink_detected"),
        "target ripple check": metadata.get("target_template_ripple_check_passed"),
        "target linear deviation mT": metadata.get("target_linear_segment_deviation_max_mT"),
    }
    st.dataframe(pd.DataFrame([summary]), use_container_width=True, hide_index=True)
    if "measured_field_aligned_mT" not in command_profile.columns:
        st.caption("기존 delay 포함 방식, review only: phase sync trace는 생성하지 않습니다.")
        return
    st.plotly_chart(_finite_first_phase_sync_plot(command_profile, metadata), use_container_width=True)
    st.plotly_chart(_finite_first_residual_plot(command_profile), use_container_width=True)
    st.caption("다운로드 voltage_v source: limited_voltage_v")
    st.plotly_chart(_finite_first_command_plot(command_profile), use_container_width=True)
    with st.expander("1차 command diagnostic traces", expanded=False):
        st.plotly_chart(_finite_first_command_plot(command_profile, diagnostics=True), use_container_width=True)


def _finite_first_phase_sync_plot(command_profile: pd.DataFrame, metadata: dict[str, object]) -> go.Figure:
    fig = go.Figure()
    measured_column = str(metadata.get("finite_first_measured_source_column") or "actual source")
    _add_profile_trace(fig, command_profile, "finite_first_base_voltage_v", "source/base voltage, scaled")
    _add_profile_trace(fig, command_profile, "measured_field_smoothed_mT", f"measured field smoothed, actual source: {measured_column}")
    _add_profile_trace(fig, command_profile, "measured_field_aligned_mT", f"measured field aligned, actual source: {measured_column}")
    for key, label in (
        ("voltage_first_peak_time_s", "voltage first peak"),
        ("measured_first_peak_time_s", "measured first peak before alignment"),
    ):
        value = first_number(metadata.get(key))
        if value is not None:
            fig.add_vline(x=float(value), line_dash="dash", annotation_text=label)
    shifted = first_number(metadata.get("voltage_first_peak_time_s"))
    if shifted is not None:
        fig.add_vline(x=float(shifted), line_dash="dot", annotation_text="measured first peak after alignment")
    fig.update_layout(template="plotly_white", height=320, title="Finite 1차 phase sync 확인", xaxis_title="time_s")
    return fig


def _finite_first_residual_plot(command_profile: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    _add_profile_trace(fig, command_profile, "physical_target_output_mT", "target field")
    _add_profile_trace(fig, command_profile, "measured_field_aligned_mT", "phase-aligned measured field")
    _add_profile_trace(fig, command_profile, "residual_for_modeling_mT", "residual")
    fig.update_layout(template="plotly_white", height=320, title="목표 자기장 vs phase-aligned 실측 자기장", xaxis_title="time_s")
    return fig


def _finite_first_command_plot(command_profile: pd.DataFrame, *, diagnostics: bool = False) -> go.Figure:
    fig = go.Figure()
    if diagnostics:
        _add_profile_trace(fig, command_profile, "finite_first_base_voltage_v", "source/base voltage")
        _add_profile_trace(fig, command_profile, "correction_delta_v", "correction_delta_v")
        _add_profile_trace(fig, command_profile, "limited_voltage_v", "limited_voltage_v")
        title = "Finite 1차 command diagnostics"
    else:
        _add_profile_trace(fig, command_profile, "limited_voltage_v", "1차 모델링 command")
        title = "1차 모델링 command"
    fig.update_layout(template="plotly_white", height=320, title=title, xaxis_title="time_s")
    return fig


def _add_profile_trace(fig: go.Figure, frame: pd.DataFrame, column: str, label: str) -> None:
    if "time_s" in frame.columns and column in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=pd.to_numeric(frame["time_s"], errors="coerce"),
                y=pd.to_numeric(frame[column], errors="coerce"),
                mode="lines",
                name=label,
            )
        )
