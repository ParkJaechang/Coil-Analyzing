"""UI helpers for user-triggered finite second modeled LUT generation."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .finite_actual_drive import build_actual_drive_review_case, read_actual_drive_result
from .finite_second_modeling import generate_second_modeled_voltage_lut
from .ui_voltage_lut_review import render_final_voltage_lut_export_panel


TARGET_FIELD_LABEL = "목표 자기장"
MEASURED_FIELD_LABEL = "실측 자기장"
RESIDUAL_LABEL = "오차 (목표 - 실측)"
FIRST_VOLTAGE_LABEL = "1차 모델링 전압"
ACTUAL_VOLTAGE_LABEL = "실제 구동 전압"
SECOND_VOLTAGE_LABEL = "2차 모델링 전압"
CORRECTION_DELTA_LABEL = "보정 전압 변화량"
RAW_HALLBZ_LABEL = "Raw HallBz"
EFFECTIVE_FIELD_LABEL = "부호 보정 자기장 (-HallBz)"
NORMALIZED_FIELD_LABEL = "정규화 자기장 (±50mT)"
BASELINE_REMOVED_FIELD_LABEL = "기준선 제거 후 자기장"
RAW_VOLTAGE_LABEL = "Raw Voltage1_V"
NORMALIZED_VOLTAGE_LABEL = "정규화 전압 (±5V)"


def render_second_modeling_controls(
    *,
    command_profile: pd.DataFrame,
    feedback_selection: dict[str, object] | None,
    freq_hz: float,
    cycle_count: float,
    waveform_type: str | None = None,
) -> None:
    st.markdown("#### 2차 모델링")
    st.caption("사용자가 버튼을 눌렀을 때만 생성합니다. 업로드나 옵션 변경만으로 2차 보정을 자동 생성하지 않습니다.")
    st.caption("Raw peak 값은 참고용입니다. 최종 적합성은 사용자가 그래프를 보고 판단합니다. 자동 합격/불합격 판정은 하지 않습니다.")
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
    selected_file = feedback_selection.get("filename")
    cached = st.session_state.get("quick_lut_second_model_result")
    cached_metadata = dict(cached.get("metadata") or {}) if isinstance(cached, dict) else {}
    dirty = bool(cached_metadata) and (
        cached_metadata.get("quick_lut_actual_drive_selected_file") != selected_file
        or abs(float(cached_metadata.get("requested_cycle_count", cycle_count)) - float(cycle_count)) > 1e-9
    )
    st.session_state["quick_lut_second_model_dirty"] = dirty
    if dirty:
        st.warning("설정 또는 실구동 파일이 변경되었습니다. 2차 모델링 실행을 다시 눌러 갱신하십시오.")
    if not st.button("2차 모델링 전압 LUT 생성", key="generate_second_modeled_voltage_lut"):
        if isinstance(cached, dict) and isinstance(cached.get("command_profile"), pd.DataFrame):
            review_payload = st.session_state.get("quick_lut_actual_drive_review_result")
            native_review = review_payload.get("review_frame") if isinstance(review_payload, dict) else None
            _render_second_modeling_result(
                cached["command_profile"],
                dict(cached.get("metadata") or {}),
                cycle_count=cycle_count,
                native_review_frame=native_review if isinstance(native_review, pd.DataFrame) else None,
            )
        else:
            st.info("실구동 결과 파일 업로드됨: 없음 또는 선택 안 됨. 다음 작업: 2차 모델링 실행을 눌러 실구동 검토와 2차 보정을 시작하십시오.")
        return
    st.caption("2차 모델링 실행을 누르면 실구동 결과 검토 plot과 2차 보정 전압 LUT를 한 번에 생성합니다.")
    suffix = "_" + Path(str(feedback_selection.get("filename") or "actual_drive_result.csv")).name
    with NamedTemporaryFile(prefix="quick_lut_second_model_", suffix=suffix, delete=False) as handle:
        temp_path = Path(handle.name)
        handle.write(bytes(feedback_selection["csv_bytes"]))
    native_review_frame: pd.DataFrame | None = None
    try:
        use_current_metadata = bool(feedback_selection.get("use_current_quick_lut_metadata"))
        native_record = read_actual_drive_result(
            temp_path,
            waveform_type=waveform_type if use_current_metadata else None,
            freq_hz=freq_hz if use_current_metadata else None,
            cycle_count=cycle_count if use_current_metadata else None,
        )
        native_review_frame, native_review_metadata = build_actual_drive_review_case(native_record)
        second_profile, metadata = generate_second_modeled_voltage_lut(
            command_profile,
            native_record,
            freq_hz=freq_hz,
            cycle_count=cycle_count,
            waveform_type=waveform_type if use_current_metadata else None,
            correction_gain=gain,
        )
        metadata = {
            **metadata,
            **{
                f"native_{key}": value
                for key, value in native_review_metadata.items()
                if key in {"timebase_status", "actual_drive_time_unit_detected", "source_time_monotonic", "duplicate_time_count"}
            },
            "quick_lut_actual_drive_selected_file": feedback_selection.get("filename"),
            "requested_cycle_count": float(cycle_count),
        }
    except ValueError as exc:
        second_profile = command_profile.copy()
        metadata = {
            "second_modeling_available": False,
            "second_modeling_status": "actual_drive_source_invalid",
            "second_modeling_unavailable_reason": (
                "unsupported_actual_drive_result_file"
                if "Unsupported finite actual-drive result filename" in str(exc)
                else "invalid_actual_drive_result"
            ),
            "parse_error": str(exc),
            "target_unchanged": True,
            "production_cycle_policy": "1p0_1p5_cycles",
            "supported_production_cycles": [1.0, 1.5],
            "unsupported_cycles": [1.25, 1.75, 2.0],
            "fourier_resynthesis_involved": False,
            "harmonic_export_involved": False,
        }
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
    second_profile["second_modeling_available"] = bool(metadata.get("second_modeling_available", False))
    second_profile["second_modeling_status"] = str(metadata.get("second_modeling_status", "unavailable"))
    run_id = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    metadata["quick_lut_second_model_run_id"] = run_id
    st.session_state["quick_lut_second_model_result"] = {
        "command_profile": second_profile,
        "metadata": metadata,
    }
    st.session_state["quick_lut_actual_drive_selected_file"] = feedback_selection.get("filename")
    st.session_state["quick_lut_actual_drive_review_result"] = {
        "review_loaded": metadata.get("second_modeling_status") == "ok",
        "plot_available": metadata.get("second_modeling_status") == "ok",
        "source_file": feedback_selection.get("filename"),
        "review_frame": native_review_frame if isinstance(native_review_frame, pd.DataFrame) else second_profile,
        "metadata": metadata,
    }
    st.session_state["quick_lut_second_model_run_id"] = run_id
    st.session_state["quick_lut_second_model_status"] = metadata.get("second_modeling_status", "unavailable")
    st.session_state["quick_lut_second_model_dirty"] = False
    st.session_state["quick_lut_final_export_source"] = "second_model" if metadata.get("second_modeling_status") == "ok" else "first_model"
    _render_second_modeling_result(second_profile, metadata, cycle_count=cycle_count, native_review_frame=native_review_frame)


def _render_second_modeling_result(
    command_profile: pd.DataFrame,
    metadata: dict[str, object],
    *,
    cycle_count: float,
    native_review_frame: pd.DataFrame | None = None,
) -> None:
    with st.expander("상세 진단", expanded=False):
        st.dataframe(pd.DataFrame([metadata]), use_container_width=True)
    if metadata.get("second_modeling_status") != "ok":
        st.info(f"2차 모델링 사용 불가: {metadata.get('second_modeling_status', 'unknown')}")
        if isinstance(native_review_frame, pd.DataFrame):
            with st.expander("Raw 데이터 상세 보기", expanded=True):
                st.caption("이 plot은 실구동 result의 native relative timebase를 사용하며, command/target grid interpolation을 사용하지 않습니다.")
                st.plotly_chart(
                    _plot_labeled_frame(
                        build_native_actual_drive_raw_plot_frame(native_review_frame),
                        [
                            RAW_HALLBZ_LABEL,
                            EFFECTIVE_FIELD_LABEL,
                            BASELINE_REMOVED_FIELD_LABEL,
                            NORMALIZED_FIELD_LABEL,
                            RAW_VOLTAGE_LABEL,
                            NORMALIZED_VOLTAGE_LABEL,
                            "Current1_A",
                        ],
                        title="1차 실구동 데이터 원본 확인",
                        yaxis_title="측정값",
                    ),
                    use_container_width=True,
                )
        return
    st.success("2차 모델링 완료")
    st.info(
        "2차 모델링 상태: 완료\n\n"
        f"사용한 실구동 결과 파일: {metadata.get('quick_lut_actual_drive_selected_file') or metadata.get('actual_drive_source_file', 'unknown')}\n\n"
        f"보정 전압 최대값: {metadata.get('correction_delta_peak_v', 'unknown')}\n\n"
        f"전압 제한 상태: {metadata.get('voltage_limit_status', 'unknown')}\n\n"
        f"HallBz 부호 보정 적용: {metadata.get('hallbz_sign_applied', False)}"
    )
    plot_frames = build_second_modeling_plot_frames(command_profile)
    st.markdown("##### 실구동 결과 검토")
    st.plotly_chart(
        _plot_labeled_frame(
            plot_frames["field"],
            [TARGET_FIELD_LABEL, MEASURED_FIELD_LABEL, RESIDUAL_LABEL],
            title="목표 자기장 vs 실측 자기장",
            yaxis_title="자기장 / 오차 (mT)",
        ),
        use_container_width=True,
    )
    st.markdown("##### 1차 전압 vs 2차 보정 전압")
    st.plotly_chart(
        _plot_labeled_frame(
            plot_frames["voltage"],
            [FIRST_VOLTAGE_LABEL, ACTUAL_VOLTAGE_LABEL, SECOND_VOLTAGE_LABEL, CORRECTION_DELTA_LABEL],
            title="1차 전압 vs 2차 보정 전압",
            yaxis_title="전압 (V)",
        ),
        use_container_width=True,
    )
    with st.expander("Raw 데이터 상세 보기", expanded=False):
        raw_plot_frame = (
            build_native_actual_drive_raw_plot_frame(native_review_frame)
            if isinstance(native_review_frame, pd.DataFrame)
            else plot_frames["raw"]
        )
        st.caption("이 plot은 실구동 result의 native relative timebase를 사용하며, command/target grid interpolation을 사용하지 않습니다.")
        st.plotly_chart(
            _plot_labeled_frame(
                raw_plot_frame,
                [
                    RAW_HALLBZ_LABEL,
                    EFFECTIVE_FIELD_LABEL,
                    BASELINE_REMOVED_FIELD_LABEL,
                    NORMALIZED_FIELD_LABEL,
                    RAW_VOLTAGE_LABEL,
                    NORMALIZED_VOLTAGE_LABEL,
                ],
                title="1차 실구동 데이터 원본 확인",
                yaxis_title="측정값",
            ),
            use_container_width=True,
        )
    render_final_voltage_lut_export_panel(
        command_profile=command_profile,
        finite_cycle_mode=True,
        waveform_type=None,
        freq_hz=None,
        cycle_count=cycle_count,
    )


def build_second_modeling_plot_frames(command_profile: pd.DataFrame) -> dict[str, pd.DataFrame]:
    time_s = pd.to_numeric(command_profile["time_s"], errors="coerce")
    field = pd.DataFrame({"time_s": time_s})
    voltage = pd.DataFrame({"time_s": time_s})
    raw = pd.DataFrame({"time_s": time_s})

    _copy_numeric(command_profile, field, "physical_target_output_mT", TARGET_FIELD_LABEL)
    _copy_numeric(command_profile, field, "measured_field_normalized_mT", MEASURED_FIELD_LABEL)
    _copy_numeric(command_profile, field, "first_model_residual_mT", RESIDUAL_LABEL)

    _copy_numeric(command_profile, voltage, "first_modeled_voltage_v", FIRST_VOLTAGE_LABEL)
    _copy_numeric(command_profile, voltage, "actual_drive_voltage_normalized_v", ACTUAL_VOLTAGE_LABEL)
    _copy_numeric(command_profile, voltage, "second_limited_voltage_v", SECOND_VOLTAGE_LABEL)
    _copy_numeric(command_profile, voltage, "second_correction_delta_v", CORRECTION_DELTA_LABEL)

    if "raw_hallbz_mT" in command_profile.columns:
        raw[RAW_HALLBZ_LABEL] = pd.to_numeric(command_profile["raw_hallbz_mT"], errors="coerce")
    elif "measured_field_raw_mT" in command_profile.columns:
        raw[RAW_HALLBZ_LABEL] = pd.to_numeric(command_profile["measured_field_raw_mT"], errors="coerce")
    _copy_numeric(command_profile, raw, "measured_field_effective_mT", EFFECTIVE_FIELD_LABEL)
    _copy_numeric(command_profile, raw, "baseline_removed_effective_field_mT", BASELINE_REMOVED_FIELD_LABEL)
    _copy_numeric(command_profile, raw, "measured_field_normalized_mT", NORMALIZED_FIELD_LABEL)
    _copy_numeric(command_profile, raw, "actual_drive_voltage_v", RAW_VOLTAGE_LABEL)
    _copy_numeric(command_profile, raw, "actual_drive_voltage_normalized_v", NORMALIZED_VOLTAGE_LABEL)
    return {"field": field, "voltage": voltage, "raw": raw}


def build_native_actual_drive_raw_plot_frame(review_frame: pd.DataFrame) -> pd.DataFrame:
    """Build raw actual-drive plot data on native relative time_s without interpolation."""
    raw = pd.DataFrame({"time_s": pd.to_numeric(review_frame["time_s"], errors="coerce")})
    if "raw_hallbz_mT" in review_frame.columns:
        raw[RAW_HALLBZ_LABEL] = pd.to_numeric(review_frame["raw_hallbz_mT"], errors="coerce")
    elif "hallbz_raw_mT" in review_frame.columns:
        raw[RAW_HALLBZ_LABEL] = pd.to_numeric(review_frame["hallbz_raw_mT"], errors="coerce")
    _copy_numeric(review_frame, raw, "measured_field_effective_mT", EFFECTIVE_FIELD_LABEL)
    if "measured_field_baseline_removed_mT" in review_frame.columns:
        _copy_numeric(review_frame, raw, "measured_field_baseline_removed_mT", BASELINE_REMOVED_FIELD_LABEL)
    else:
        _copy_numeric(review_frame, raw, "baseline_removed_effective_field_mT", BASELINE_REMOVED_FIELD_LABEL)
    if "measured_field_normalized_mT" in review_frame.columns:
        _copy_numeric(review_frame, raw, "measured_field_normalized_mT", NORMALIZED_FIELD_LABEL)
    else:
        _copy_numeric(review_frame, raw, "normalized_measured_field_mT", NORMALIZED_FIELD_LABEL)
    _copy_numeric(review_frame, raw, "raw_first_voltage_v", RAW_VOLTAGE_LABEL)
    _copy_numeric(review_frame, raw, "normalized_first_voltage_v", NORMALIZED_VOLTAGE_LABEL)
    if "current_a" in review_frame.columns:
        _copy_numeric(review_frame, raw, "current_a", "Current1_A")
    return raw


def _copy_numeric(source: pd.DataFrame, target: pd.DataFrame, source_column: str, label: str) -> None:
    if source_column in source.columns:
        target[label] = pd.to_numeric(source[source_column], errors="coerce")


def _plot_labeled_frame(frame: pd.DataFrame, columns: list[str], *, title: str, yaxis_title: str) -> go.Figure:
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
