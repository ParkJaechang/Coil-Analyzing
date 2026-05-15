from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .continuous_steady_state_extraction import (
    build_continuous_actual_drive_review_case,
    build_continuous_steady_state_modeling_case,
    evaluate_continuous_steady_state_validation,
)
from .finite_actual_drive import build_actual_drive_review_case, read_actual_drive_result


ModelingCaseBuilder = Callable[..., dict[str, Any]]


def render_continuous_steady_state_runtime_panel(
    *,
    analysis_lookup: dict,
    waveform_type: str | None,
    freq_hz: float | None,
    modeling_case_builder: ModelingCaseBuilder = build_continuous_steady_state_modeling_case,
) -> dict[str, Any] | None:
    st.markdown("#### Continuous steady-state runtime")
    st.caption("Continuous steady-state mode는 안정화된 1cycle만 추출해 반복 출력용 LUT를 생성합니다.")
    st.caption("초반 startup transient 구간은 모델링에 사용하지 않습니다.")
    st.caption("Continuous mode에서는 1.5cycle command를 생성하지 않습니다.")
    st.caption("Continuous mode에서는 zero-return tail을 기본 사용하지 않습니다.")

    candidate_names, candidates = _continuous_candidate_frames(analysis_lookup)
    selected_name = None
    if candidate_names:
        selected_name = st.selectbox("Continuous source dataset", candidate_names, key="continuous_steady_source_dataset")
    else:
        st.info("Continuous extraction에 사용할 time_s/Voltage1_V/HallBz 호환 dataset이 없습니다.")

    if st.button("Steady-state 1cycle 추출", key="continuous_steady_state_extract_button"):
        if selected_name is None:
            st.session_state["continuous_steady_state_metadata"] = {"steady_state_extraction_status": "unavailable_no_source"}
            st.warning("Continuous source dataset이 없어 Steady-state 1cycle 추출을 실행할 수 없습니다.")
        else:
            try:
                case = modeling_case_builder(
                    candidates[selected_name],
                    waveform_type=str(waveform_type or "sine"),
                    freq_hz=float(freq_hz or 1.0),
                )
            except Exception as exc:  # noqa: BLE001 - runtime UI must show parse/extraction failures.
                st.session_state["continuous_steady_state_metadata"] = {
                    "steady_state_extraction_status": "error",
                    "error": str(exc),
                }
                st.error(f"Steady-state 1cycle 추출 실패: {exc}")
            else:
                st.session_state["continuous_steady_state_extraction_result"] = case
                st.session_state["continuous_steady_state_window_frame"] = case["steady_state_one_cycle_frame"]
                st.session_state["continuous_steady_state_metadata"] = case["metadata"]
                st.success("Steady-state 1cycle 추출 완료")

    case = st.session_state.get("continuous_steady_state_extraction_result")
    if isinstance(case, dict):
        window = case.get("steady_state_one_cycle_frame")
        metadata = dict(case.get("metadata") or {})
        if isinstance(window, pd.DataFrame) and not window.empty:
            st.markdown("#### 선택된 steady-state 1cycle")
            st.caption("이 1cycle이 continuous steady-state modeling에 사용됩니다.")
            st.plotly_chart(_plot_continuous_window(window), use_container_width=True)
            st.dataframe(pd.DataFrame([metadata]), use_container_width=True, hide_index=True)
        return case

    st.caption("옵션 변경만으로 heavy calculation을 자동 실행하지 않습니다. `Steady-state 1cycle 추출` 버튼을 누르십시오.")
    return None


def render_continuous_actual_drive_runtime_panel(
    *,
    waveform_type: str | None,
    freq_hz: float | None,
) -> None:
    st.markdown("#### Continuous 1차 실구동 결과 업로드")
    st.caption("Continuous 1차 실구동 결과도 startup transient를 제외하고 안정화된 1cycle만 2차 보정 입력으로 사용합니다.")
    uploaded = st.file_uploader(
        "Continuous 1차 실구동 결과 CSV",
        type=["csv"],
        key="continuous_first_drive_actual_upload",
        help="TimeMs / Voltage1_V / HallBz schema의 장비 측정 CSV를 업로드하십시오.",
    )
    if st.button("실구동 결과에서 안정 1cycle 추출", key="continuous_first_drive_extract_button"):
        if uploaded is None:
            st.warning("Continuous 1차 실구동 결과 CSV를 먼저 업로드하십시오.")
        else:
            try:
                review_frame, review_metadata = _parse_actual_drive_upload(
                    uploaded.name,
                    uploaded.getvalue(),
                    waveform_type=str(waveform_type or "sine"),
                    freq_hz=float(freq_hz or 1.0),
                    cycle_count=1.0,
                )
                result = build_continuous_actual_drive_review_case(
                    review_frame,
                    waveform_type=str(waveform_type or "sine"),
                    freq_hz=float(freq_hz or 1.0),
                    purpose="second_modeling",
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Continuous 1차 실구동 안정 1cycle 추출 실패: {exc}")
            else:
                st.session_state["continuous_first_drive_actual_result"] = {
                    "review_frame": review_frame,
                    "metadata": review_metadata,
                }
                st.session_state["continuous_first_drive_steady_window_frame"] = result["steady_state_one_cycle_frame"]
                st.session_state["continuous_first_drive_steady_metadata"] = result["metadata"]
                st.success("Continuous 1차 실구동 결과에서 안정 1cycle 추출 완료")

    if st.button("Continuous 2차 보정 command 생성", key="continuous_second_command_button"):
        first = st.session_state.get("quick_lut_first_model_result_continuous")
        steady = st.session_state.get("continuous_first_drive_steady_window_frame")
        if not isinstance(first, dict) or not isinstance(steady, pd.DataFrame) or steady.empty:
            st.warning("Continuous 1차 모델링 결과와 실구동 안정 1cycle 추출 결과가 필요합니다.")
        else:
            command_profile = first.get("command_profile")
            metadata = dict(first.get("metadata") or {})
            metadata.update(
                {
                    "second_modeling_input_mode": "continuous_steady_state",
                    "second_drive_actual_data_used": "steady_state_one_cycle_only",
                    "continuous_repeating_lut": True,
                    "continuous_export_cycle_count": 1.0,
                    "continuous_zero_return_tail_enabled": False,
                }
            )
            st.session_state["quick_lut_second_model_result_continuous"] = {
                "command_profile": command_profile.copy(deep=True) if isinstance(command_profile, pd.DataFrame) else command_profile,
                "actual_drive_steady_window_frame": steady.copy(deep=True),
                "metadata": metadata,
            }
            st.success("Continuous 2차 보정 command 생성 결과를 session_state에 저장했습니다.")

    _render_continuous_validation_section(waveform_type=waveform_type, freq_hz=freq_hz)


def _render_continuous_validation_section(*, waveform_type: str | None, freq_hz: float | None) -> None:
    st.markdown("#### Continuous 2차 구동 결과 평가")
    st.caption("평가는 안정화된 1cycle 기준입니다.")
    st.caption("초반 transient cycle은 평가에서 제외되었습니다.")
    uploaded = st.file_uploader(
        "Continuous 2차 구동 결과 CSV",
        type=["csv"],
        key="continuous_second_drive_validation_upload",
    )
    if not st.button("Continuous 2차 구동 결과 평가 실행", key="continuous_second_validation_button"):
        return
    if uploaded is None:
        st.warning("Continuous 2차 구동 결과 CSV를 먼저 업로드하십시오.")
        return
    try:
        review_frame, _metadata = _parse_actual_drive_upload(
            uploaded.name,
            uploaded.getvalue(),
            waveform_type=str(waveform_type or "sine"),
            freq_hz=float(freq_hz or 1.0),
            cycle_count=1.0,
        )
        result = evaluate_continuous_steady_state_validation(
            review_frame,
            waveform_type=str(waveform_type or "sine"),
            freq_hz=float(freq_hz or 1.0),
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Continuous 2차 구동 결과 평가 실패: {exc}")
        return
    st.session_state["continuous_second_drive_validation_result"] = result
    st.dataframe(pd.DataFrame([result["metrics"]]), use_container_width=True, hide_index=True)


def _continuous_candidate_frames(analysis_lookup: dict) -> tuple[list[str], dict[str, pd.DataFrame]]:
    candidates: dict[str, pd.DataFrame] = {}
    for key, analysis in (analysis_lookup or {}).items():
        frame = getattr(getattr(analysis, "parsed", None), "normalized_frame", None)
        if isinstance(frame, pd.DataFrame) and _is_continuous_candidate(frame):
            candidates[str(key)] = frame
    return sorted(candidates.keys()), candidates


def _is_continuous_candidate(frame: pd.DataFrame) -> bool:
    columns = set(frame.columns)
    has_time = bool({"time_s", "time_s_abs", "TimeMs"} & columns)
    has_voltage = bool({"raw_voltage_v", "raw_actual_drive_voltage_v", "Voltage1_V", "command_voltage_v"} & columns)
    has_hall = bool({"raw_hallbz_mT", "hallbz_raw_mT", "HallBz"} & columns)
    return has_time and has_voltage and has_hall


def _parse_actual_drive_upload(
    filename: str,
    csv_bytes: bytes,
    *,
    waveform_type: str,
    freq_hz: float,
    cycle_count: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    with TemporaryDirectory(prefix="continuous_actual_drive_") as temp_dir:
        temp_path = Path(temp_dir) / Path(filename).name
        temp_path.write_bytes(csv_bytes)
        record = read_actual_drive_result(
            temp_path,
            waveform_type=waveform_type,
            freq_hz=freq_hz,
            cycle_count=cycle_count,
        )
        return build_actual_drive_review_case(record)


def _plot_continuous_window(frame: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    traces = [
        ("normalized_physical_target_output_mT", "목표 자기장"),
        ("measured_field_normalized_mT", "선택 steady measured field"),
        ("voltage_normalized_v", "voltage"),
    ]
    for column, label in traces:
        if column not in frame.columns:
            continue
        figure.add_trace(
            go.Scatter(
                x=frame["time_s"],
                y=frame[column],
                mode="lines",
                name=label,
            )
        )
    figure.update_layout(
        template="plotly_white",
        height=360,
        title="선택된 steady-state 1cycle",
        xaxis_title="time_s (s)",
        yaxis_title="value",
    )
    return figure
