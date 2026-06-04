from __future__ import annotations

from uuid import uuid4
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .continuous_steady_state_runtime import run_continuous_first_modeling
from .ui_continuous_final_lut_export import (
    normalize_continuous_result_contract,
    render_continuous_final_voltage_lut_export_section,
)
from .ui_modeling_error_summary import render_error_ratio_metrics
from .ui_second_modeling_plots import add_max_error_marker
from .voltage_policy import COMMAND_VOLTAGE_LIMIT_V


def render_continuous_first_modeling_controls(*, waveform_type: str | None, freq_hz: float | None) -> None:
    target_config = st.session_state.get("quick_lut_target_config") or {}
    target_peak_field_mT = float(target_config.get("user_target_peak_field_mT") or 50.0)
    st.caption(
        f"Continuous도 finite와 동일하게 목표 피크 ±{target_peak_field_mT:g} mT 기준으로 field와 입력 전압 scale을 계산합니다."
    )
    if st.button("Continuous 1차 모델링 실행", key="continuous_first_modeling_button"):
        case = st.session_state.get("continuous_steady_state_extraction_result")
        if not isinstance(case, dict) or st.session_state.get("continuous_steady_state_dirty"):
            st.warning("먼저 현재 설정으로 Steady-state 1cycle 추출을 실행하십시오.")
            return
        result = run_continuous_first_modeling(
            extraction_result=case,
            waveform_type=waveform_type,
            freq_hz=freq_hz,
            target_peak_field_mT=target_peak_field_mT,
        )
        if result.get("status") != "ok":
            st.warning(f"Continuous 1차 모델링 command 생성에 실패했습니다: {result.get('error_reason')}")
            return
        command = result.get("command_profile")
        if not isinstance(command, pd.DataFrame) or command.empty:
            st.warning("Continuous 1차 모델링 command 생성에 실패했습니다: command_profile_empty")
            return
        normalized_result = normalize_continuous_result_contract(
            result["first_model_result"],
            "first",
        )
        st.session_state["quick_lut_first_model_result_continuous"] = normalized_result
        st.session_state["quick_lut_first_model_result_continuous_metadata"] = (
            normalized_result.get("metadata") if isinstance(normalized_result, dict) else result["first_model_metadata"]
        )
        st.session_state["continuous_first_modeling_run_id"] = uuid4().hex
        st.success("Continuous 1차 모델링 command 생성 완료")

    first = st.session_state.get("quick_lut_first_model_result_continuous")
    command = first.get("command_profile") if isinstance(first, dict) else None
    metadata = dict(first.get("metadata") or {}) if isinstance(first, dict) else {}
    if not isinstance(command, pd.DataFrame) or command.empty:
        st.caption("Continuous 1차 모델링 실행 후 command plot과 export source가 표시됩니다.")
        return
    st.markdown("#### Phase alignment 확인")
    st.plotly_chart(_phase_alignment_plot(command), use_container_width=True)
    render_error_ratio_metrics(st, metadata, title="Continuous 1차 보정 오차율 요약")
    st.markdown("#### 목표 자기장 vs phase-aligned 실측 자기장")
    st.plotly_chart(_target_residual_plot(command), use_container_width=True)
    st.markdown("#### Continuous 1차 modeling command")
    st.plotly_chart(_command_plot(command, metadata), use_container_width=True)
    if metadata.get("continuous_voltage_clip_status") in {"warning", "severe"}:
        st.warning(str(metadata.get("continuous_clipping_warning")))
    st.caption("이 결과는 continuous steady-state 1cycle 반복 출력용 LUT입니다.")
    st.caption("초반 startup transient는 모델링에 사용하지 않았습니다.")
    st.caption("자기장 첫 피크를 전압 피크에 맞춘 뒤 residual을 계산했습니다.")
    st.markdown("#### 1cycle 반복 출력용 voltage LUT")
    st.dataframe(pd.DataFrame([_summary(metadata, command)]), use_container_width=True, hide_index=True)
    render_continuous_final_voltage_lut_export_section(
        waveform_type=waveform_type,
        freq_hz=freq_hz,
        key_namespace="quick_lut_first_modeling",
    )


def _phase_alignment_plot(command: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    _add_trace(fig, command, "limited_voltage_v", "voltage, scaled")
    _add_trace(fig, command, "measured_field_smoothed_mT", "measured field smoothed")
    _add_trace(fig, command, "measured_field_aligned_mT", "measured field aligned")
    fig.update_layout(template="plotly_white", height=320, title="Phase alignment 확인", xaxis_title="time_s")
    return fig


def _target_residual_plot(command: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    _add_trace(fig, command, "target_field_1cycle_mT", "목표 자기장")
    _add_trace(fig, command, "measured_field_aligned_mT", "phase-aligned measured field")
    _add_trace(fig, command, "residual_for_modeling_mT", "residual")
    marker_frame = command.rename(
        columns={
            "target_field_1cycle_mT": "target field",
            "residual_for_modeling_mT": "residual",
        }
    )
    add_max_error_marker(
        fig,
        marker_frame,
        residual_label="residual",
        target_label="target field",
        marker_label="최대 오차 지점",
        mask_column="modeling_error_evaluation_mask",
    )
    fig.update_layout(template="plotly_white", height=320, title="목표 자기장 vs phase-aligned 실측 자기장")
    return fig


def _command_plot(command: pd.DataFrame, metadata: dict[str, Any]) -> go.Figure:
    fig = go.Figure()
    _add_trace(fig, command, "source_voltage_v", "원본 입력 전압")
    _add_trace(fig, command, "base_voltage_v", "모델링 입력 전압")
    _add_trace(fig, command, "correction_delta_v", "correction_delta_v")
    _add_trace(fig, command, "limited_voltage_v", "limited_voltage_v")
    limit_v = float(metadata.get("continuous_final_voltage_limit_v") or COMMAND_VOLTAGE_LIMIT_V)
    base_peak_v = float(metadata.get("continuous_base_voltage_peak_v") or 0.0)
    fig.add_hline(y=limit_v, line_dash="dash", line_color="red", annotation_text=f"+{limit_v:g}V limit")
    fig.add_hline(y=-limit_v, line_dash="dash", line_color="red", annotation_text=f"-{limit_v:g}V limit")
    if base_peak_v > 0.0:
        fig.add_hline(y=base_peak_v, line_dash="dot", line_color="gray", annotation_text=f"modeling input peak {base_peak_v:g}V")
        fig.add_hline(y=-base_peak_v, line_dash="dot", line_color="gray")
    fig.update_layout(template="plotly_white", height=320, title="Continuous 1차 modeling command")
    return fig


def _add_trace(fig: go.Figure, command: pd.DataFrame, column: str, label: str) -> None:
    if column in command.columns:
        fig.add_trace(go.Scatter(x=command["time_s"], y=command[column], mode="lines", name=label))


def _summary(metadata: dict[str, Any], command: pd.DataFrame) -> dict[str, Any]:
    return {
        "continuous_first_modeling_status": metadata.get("continuous_first_modeling_status") or "ok",
        "continuous_phase_delay_s": metadata.get("continuous_phase_delay_s"),
        "continuous_loop_output": metadata.get("continuous_loop_output", True),
        "loop_endpoint_policy": metadata.get("loop_endpoint_policy") or "period_exclusive",
        "command_profile_rows": len(command),
        "continuous_base_voltage_peak_v": metadata.get("continuous_base_voltage_peak_v"),
        "source_voltage_base_normalized_peak_v": metadata.get("source_voltage_base_normalized_peak_v"),
        "field_per_volt_mT_per_v": metadata.get("field_per_volt_mT_per_v"),
        "correction_delta_mode": metadata.get("correction_delta_mode"),
        "continuous_clipping_fraction": metadata.get("continuous_clipping_fraction", 0.0),
        "continuous_voltage_clip_status": metadata.get("continuous_voltage_clip_status") or "ok",
    }
