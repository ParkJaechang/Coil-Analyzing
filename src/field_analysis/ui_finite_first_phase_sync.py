from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .utils import first_number


def render_finite_first_phase_sync_review(command_profile: pd.DataFrame, metadata: dict[str, object]) -> None:
    if not isinstance(command_profile, pd.DataFrame) or command_profile.empty:
        return
    st.markdown("#### Finite 1차 phase sync 확인")
    st.caption("전압 피크와 실측 자기장 피크를 맞춘 뒤, 실측 field mT 값을 그대로 사용해 residual을 계산합니다.")
    summary = {
        "source file": metadata.get("finite_first_measured_source_file"),
        "source label": metadata.get("finite_first_measured_source_label"),
        "measured field column": metadata.get("finite_first_measured_source_column"),
        "actual measured source": metadata.get("finite_first_measured_source_is_actual_measured"),
        "source data origin": metadata.get("finite_first_source_data_origin"),
        "source_input_waveform_family": command_profile.get("waveform_type", pd.Series(["triangle"])).iloc[0],
        "target_field_shape": "fixed_rounded_triangle",
        "finite_first_modeling_mode": metadata.get("finite_first_modeling_mode", "phase_synced"),
        "phase sync 기준": metadata.get("phase_sync_peak_reference"),
        "phase sync peak polarity": metadata.get("phase_sync_peak_polarity"),
        "measured peak value mT": metadata.get("measured_peak_value_mT"),
        "voltage peak value V": metadata.get("voltage_peak_value_v"),
        "phase_delay_s": metadata.get("phase_delay_s"),
        "phase_delay_cycles": metadata.get("phase_delay_cycles"),
        "실측 field abs peak (mT)": metadata.get("measured_abs_peak_effective_mT"),
        "실측 field scale": metadata.get("measured_field_scale_to_50mT"),
        "정규화 offset 제거 (mT)": metadata.get("measured_field_total_offset_removed_mT"),
        "정규화 mode": metadata.get("measured_field_normalization_mode"),
        "정규화 후 peak (mT)": metadata.get("measured_aligned_normalized_peak_mT"),
        "residual extra gain": metadata.get("correction_gain_used"),
        "voltage_headroom_v": metadata.get("voltage_headroom_v"),
        "clipping_fraction": metadata.get("clipping_fraction"),
        "active residual finite ratio": metadata.get("active_residual_finite_ratio"),
        "phase support status": metadata.get("phase_support_status"),
        "required source end": metadata.get("required_phase_aligned_source_end_s"),
        "actual source end": metadata.get("actual_source_time_end_s"),
        "phase support margin s": metadata.get("phase_sync_support_margin_s"),
        "active_end_kink_detected": metadata.get("active_end_kink_detected"),
        "target ripple check": metadata.get("target_template_ripple_check_passed"),
        "target linear deviation mT": metadata.get("target_linear_segment_deviation_max_mT"),
    }
    st.dataframe(pd.DataFrame([summary]), use_container_width=True, hide_index=True)
    _render_phase_sync_correction_basis(metadata)
    with st.expander("target template ripple diagnostic", expanded=False):
        diagnostic = {
            "target_template_type": metadata.get("target_template_type"),
            "target_template_ripple_check_passed": metadata.get("target_template_ripple_check_passed"),
            "target_linear_segment_deviation_max_mT": metadata.get("target_linear_segment_deviation_max_mT"),
            "target_peak_positive_mT": metadata.get("target_peak_positive_mT"),
            "target_peak_negative_mT": metadata.get("target_peak_negative_mT"),
        }
        st.dataframe(pd.DataFrame([diagnostic]), use_container_width=True, hide_index=True)
        st.plotly_chart(_finite_first_target_template_plot(command_profile), use_container_width=True)
    if "measured_field_aligned_mT" not in command_profile.columns:
        st.caption("기존 delay 포함 방식, review only: phase sync trace는 생성하지 않습니다.")
        return
    if str(metadata.get("finite_first_modeling_status") or "") == "insufficient_phase_sync_support":
        st.warning(
            "phase sync 이후 active 끝까지 필요한 실측 support가 부족합니다. "
            "이 결과는 1차 command 성공 결과로 사용하지 않습니다."
        )
        return
    st.plotly_chart(_finite_first_phase_sync_plot(command_profile, metadata), use_container_width=True)
    st.plotly_chart(_finite_first_residual_plot(command_profile), use_container_width=True)
    st.caption("다운로드 voltage_v source: limited_voltage_v")
    st.plotly_chart(_finite_first_command_plot(command_profile), use_container_width=True)
    with st.expander("1차 command diagnostic traces", expanded=False):
        st.plotly_chart(_finite_first_command_plot(command_profile, diagnostics=True), use_container_width=True)


def _render_phase_sync_correction_basis(metadata: dict[str, object]) -> None:
    st.markdown("##### Phase sync residual -> 1차 command 반영 기준")
    st.caption(
        "phase-aligned measured field를 추가 정규화 없이 실제 mT 값 그대로 residual에 사용하고, "
        "residual을 ±5V 전압 기준의 unit delta로 변환한 다음 추가 gain 없이 smoothing/stabilization만 적용합니다."
    )
    rows = [
        {"항목": "residual 계산", "계산/의미": "target_normalized_mT - measured_aligned_normalized_mT", "현재값": ""},
        {"항목": "unit delta 변환", "계산/의미": "residual_mT / 50mT * 5V", "현재값": metadata.get("auto_gain_unit_delta_peak_v")},
        {
            "항목": "auto gain",
            "계산/의미": "finite 1차 phase sync에서는 적용하지 않음, 진단값만 표시",
            "현재값": metadata.get("correction_gain_auto"),
        },
        {"항목": "gain clamp", "계산/의미": "finite 1차 phase sync에서는 사용하지 않음", "현재값": metadata.get("auto_gain_clamped")},
        {"항목": "최종 command", "계산/의미": "clip(base_voltage + correction_delta, ±5V)", "현재값": metadata.get("clipping_fraction")},
        {"항목": "실측 min/max", "계산/의미": "smoothing 후 active 구간 min/max, offset 재정렬 없음", "현재값": f"{metadata.get('measured_field_smoothed_active_min_mT')} / {metadata.get('measured_field_smoothed_active_max_mT')}"},
        {"항목": "실측 abs peak", "계산/의미": "max(abs(smoothed measured field)), scale에는 사용하지 않음", "현재값": metadata.get("measured_field_smoothed_abs_peak_mT")},
        {"항목": "offset 제거", "계산/의미": "정규화 단계에서는 0mT, 실측값 위치 보존", "현재값": metadata.get("measured_field_total_offset_removed_mT")},
        {"항목": "실측 field scale", "계산/의미": "1.0 fixed, 실측 mT 그대로 사용", "현재값": metadata.get("measured_field_scale_to_50mT")},
        {"항목": "base voltage peak", "계산/의미": "현재 1차/base command peak", "현재값": metadata.get("auto_gain_first_voltage_peak_v")},
        {"항목": "safe headroom", "계산/의미": "±5V limit 대비 headroom의 20% percentile", "현재값": metadata.get("auto_gain_headroom_safe_v")},
        {"항목": "target delta peak", "계산/의미": "min(0.35*base_peak, 0.70*safe_headroom, 1.0V)", "현재값": metadata.get("auto_gain_target_delta_peak_v")},
        {"항목": "used gain", "계산/의미": "1.0 fixed, residual에는 추가 gain을 곱하지 않음", "현재값": metadata.get("correction_gain_used")},
    ]
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


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


def _finite_first_target_template_plot(command_profile: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    _add_profile_trace(fig, command_profile, "physical_target_output_mT", "analytic fixed rounded triangle target")
    fig.update_layout(
        template="plotly_white",
        height=260,
        title="Target template diagnostic: analytic fixed rounded triangle",
        xaxis_title="time_s",
        yaxis_title="mT",
    )
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
