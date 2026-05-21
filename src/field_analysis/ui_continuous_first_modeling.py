from __future__ import annotations

from uuid import uuid4
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .continuous_steady_state_runtime import run_continuous_first_modeling


def render_continuous_first_modeling_controls(*, waveform_type: str | None, freq_hz: float | None) -> None:
    if st.button("Continuous 1차 모델링 실행", key="continuous_first_modeling_button"):
        case = st.session_state.get("continuous_steady_state_extraction_result")
        if not isinstance(case, dict) or st.session_state.get("continuous_steady_state_dirty"):
            st.warning("먼저 현재 설정으로 Steady-state 1cycle 추출을 실행하십시오.")
            return
        result = run_continuous_first_modeling(
            extraction_result=case,
            waveform_type=waveform_type,
            freq_hz=freq_hz,
        )
        if result.get("status") != "ok":
            st.warning(f"Continuous 1차 모델링 command 생성에 실패했습니다: {result.get('error_reason')}")
            return
        command = result.get("command_profile")
        if not isinstance(command, pd.DataFrame) or command.empty:
            st.warning("Continuous 1차 모델링 command 생성에 실패했습니다: command_profile_empty")
            return
        st.session_state["quick_lut_first_model_result_continuous"] = result["first_model_result"]
        st.session_state["quick_lut_first_model_result_continuous_metadata"] = result["first_model_metadata"]
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
    st.markdown("#### 목표 자기장 vs phase-aligned 실측 자기장")
    st.plotly_chart(_target_residual_plot(command), use_container_width=True)
    st.markdown("#### Continuous 1차 modeling command")
    st.plotly_chart(_command_plot(command), use_container_width=True)
    st.caption("이 결과는 continuous steady-state 1cycle 반복 출력용 LUT입니다.")
    st.caption("초반 startup transient는 모델링에 사용하지 않았습니다.")
    st.caption("자기장 첫 피크를 전압 피크에 맞춘 뒤 residual을 계산했습니다.")
    st.markdown("#### 1cycle 반복 출력용 voltage LUT")
    st.dataframe(pd.DataFrame([_summary(metadata, command)]), use_container_width=True, hide_index=True)


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
    fig.update_layout(template="plotly_white", height=320, title="목표 자기장 vs phase-aligned 실측 자기장")
    return fig


def _command_plot(command: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    _add_trace(fig, command, "base_voltage_v", "source/base voltage")
    _add_trace(fig, command, "correction_delta_v", "correction_delta_v")
    _add_trace(fig, command, "limited_voltage_v", "limited_voltage_v")
    fig.update_layout(template="plotly_white", height=320, title="Continuous 1차 modeling command")
    return fig


def _add_trace(fig: go.Figure, command: pd.DataFrame, column: str, label: str) -> None:
    if column in command.columns:
        fig.add_trace(go.Scatter(x=command["time_s"], y=command[column], mode="lines", name=label))


def _summary(metadata: dict[str, Any], command: pd.DataFrame) -> dict[str, Any]:
    return {
        "continuous_first_modeling_status": metadata.get("continuous_first_modeling_status"),
        "continuous_phase_delay_s": metadata.get("continuous_phase_delay_s"),
        "continuous_loop_output": metadata.get("continuous_loop_output"),
        "loop_endpoint_policy": metadata.get("loop_endpoint_policy"),
        "command_profile_rows": len(command),
    }
