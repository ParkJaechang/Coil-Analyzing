"""UI helpers for user-triggered finite second modeled LUT generation."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import streamlit as st

from .finite_second_modeling import generate_second_modeled_voltage_lut
from .plotting import plot_waveforms
from .ui_voltage_lut_review import render_final_voltage_lut_export_panel


def render_second_modeling_controls(
    *,
    command_profile: pd.DataFrame,
    feedback_selection: dict[str, object] | None,
    freq_hz: float,
    cycle_count: float,
) -> None:
    st.markdown("#### 2차 모델링")
    st.caption("사용자가 버튼을 눌렀을 때만 생성합니다. 업로드나 옵션 변경만으로 2차 보정을 자동 생성하지 않습니다.")
    st.caption("Raw peak 값은 참고용입니다. 최종 적합성은 사용자가 그래프를 보고 판단합니다. 자동 pass/fail 판정은 하지 않습니다.")
    supported = np.isfinite(cycle_count) and any(
        abs(float(cycle_count) - supported_cycle) <= 1e-9 for supported_cycle in (1.0, 1.5)
    )
    if not supported:
        st.info("2차 모델링 사용 불가: Production finite 보정은 1.0 / 1.5 cycle만 지원합니다. 1.25 / 1.75 / 2.0 cycle은 검토용입니다.")
        return
    if not feedback_selection or not feedback_selection.get("csv_bytes"):
        st.info("2차 모델링 전압 LUT를 만들려면 먼저 실구동 결과 CSV를 업로드/선택하고 검토하십시오.")
        return
    gain = float(
        st.number_input(
            "2차 보정 gain",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            key="second_modeling_correction_gain",
        )
    )
    if not st.button("2차 모델링 전압 LUT 생성", key="generate_second_modeled_voltage_lut"):
        cached = st.session_state.get("quick_lut_second_model_result")
        if isinstance(cached, dict) and isinstance(cached.get("command_profile"), pd.DataFrame):
            _render_second_modeling_result(cached["command_profile"], dict(cached.get("metadata") or {}))
        else:
            st.info("2차 모델링 결과가 아직 없습니다. 버튼을 눌러 생성하십시오.")
        return
    suffix = "_" + Path(str(feedback_selection.get("filename") or "actual_drive_result.csv")).name
    with NamedTemporaryFile(prefix="quick_lut_second_model_", suffix=suffix, delete=False) as handle:
        temp_path = Path(handle.name)
        handle.write(bytes(feedback_selection["csv_bytes"]))
    try:
        second_profile, metadata = generate_second_modeled_voltage_lut(
            command_profile,
            temp_path,
            freq_hz=freq_hz,
            cycle_count=cycle_count,
            correction_gain=gain,
        )
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
    second_profile["second_modeling_available"] = bool(metadata.get("second_modeling_available", False))
    second_profile["second_modeling_status"] = str(metadata.get("second_modeling_status", "unavailable"))
    st.session_state["quick_lut_second_model_result"] = {
        "command_profile": second_profile,
        "metadata": metadata,
    }
    st.session_state["quick_lut_final_export_source"] = "second_model" if metadata.get("second_modeling_status") == "ok" else "first_model"
    _render_second_modeling_result(second_profile, metadata)


def _render_second_modeling_result(command_profile: pd.DataFrame, metadata: dict[str, object]) -> None:
    with st.expander("상세 진단", expanded=False):
        st.dataframe(pd.DataFrame([metadata]), use_container_width=True)
    if metadata.get("second_modeling_status") != "ok":
        st.info(f"2차 모델링 사용 불가: {metadata.get('second_modeling_status', 'unknown')}")
        return
    st.success("2차 모델링 완료")
    st.markdown("##### 2차 모델링 결과 검토")
    st.plotly_chart(
        plot_waveforms(
            command_profile,
            ["physical_target_output_mT", "measured_field_normalized_mT", "first_model_residual_mT"],
            title="목표 자기장 vs 실측 자기장 / 오차",
        ).update_layout(xaxis_title="시간 (s)", yaxis_title="자기장 / 오차 (mT)"),
        use_container_width=True,
    )
    st.markdown("##### 1차 전압 vs 2차 보정 전압")
    st.plotly_chart(
        plot_waveforms(
            command_profile,
            ["first_modeled_voltage_v", "actual_drive_voltage_normalized_v", "second_correction_delta_v", "second_limited_voltage_v"],
            title="1차 전압 vs 2차 보정 전압",
        ).update_layout(xaxis_title="시간 (s)", yaxis_title="전압 (V)"),
        use_container_width=True,
    )
    with st.expander("Raw 데이터 상세 보기", expanded=False):
        st.plotly_chart(
            plot_waveforms(
                command_profile,
                [
                    "measured_field_raw_mT",
                    "measured_field_effective_mT",
                    "measured_field_normalized_mT",
                    "actual_drive_voltage_v",
                    "actual_drive_voltage_normalized_v",
                ],
                title="Raw 실구동 데이터",
            ).update_layout(xaxis_title="시간 (s)", yaxis_title="측정값"),
            use_container_width=True,
        )
    render_final_voltage_lut_export_panel(
        command_profile=command_profile,
        finite_cycle_mode=True,
        waveform_type=None,
        freq_hz=None,
        cycle_count=1.0,
    )
