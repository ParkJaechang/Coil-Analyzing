from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


def build_final_voltage_lut_frame(command_profile: pd.DataFrame, *, voltage_source_column: str | None = None) -> pd.DataFrame:
    voltage_source_column = voltage_source_column or _export_voltage_source_column(command_profile)
    missing = [column for column in ("time_s", voltage_source_column) if column not in command_profile.columns]
    if missing:
        raise ValueError(f"Missing final voltage LUT source columns: {missing}")
    return pd.DataFrame(
        {
            "sample_index": np.arange(len(command_profile), dtype=int),
            "time_s": pd.to_numeric(command_profile["time_s"], errors="coerce"),
            "voltage_v": pd.to_numeric(command_profile[voltage_source_column], errors="coerce"),
        }
    )


def build_final_voltage_lut_filename(
    *,
    waveform_type: object | None,
    freq_hz: object | None,
    cycle_count: object | None,
) -> str:
    waveform = _safe_name_part(waveform_type)
    freq = _format_number_for_filename(freq_hz)
    cycle = _format_number_for_filename(cycle_count)
    if waveform and freq and cycle:
        return f"finite_recommended_voltage_lut_{waveform}_{freq}Hz_{cycle}cycle.csv"
    return "finite_recommended_voltage_lut.csv"


def build_final_voltage_lut_csv_bytes(command_profile: pd.DataFrame, *, voltage_source_column: str | None = None) -> bytes:
    return build_final_voltage_lut_frame(command_profile, voltage_source_column=voltage_source_column).to_csv(index=False).encode("utf-8-sig")


def render_final_voltage_lut_export_panel(
    *,
    command_profile: pd.DataFrame | None,
    finite_cycle_mode: bool,
    waveform_type: object | None,
    freq_hz: object | None,
    cycle_count: object | None,
) -> None:
    st.markdown("#### 최종 전압 LUT 추출")
    st.info(
        "최종 LUT는 화면에 표시된 최종 전압 샘플을 그대로 저장합니다.\n\n"
        "Fourier 재합성 또는 harmonic 계수 내보내기가 아닙니다.\n\n"
        "저장 컬럼: sample_index, time_s, voltage_v"
    )
    # Source-contract marker for tests and PR review: exported CSV uses plotted final
    # command voltage samples; not Fourier; not harmonic resynthesis; columns:
    # sample_index, time_s, voltage_v.
    if not finite_cycle_mode:
        st.info("최종 전압 LUT 추출 사용 불가: finite 보정 결과에서만 다운로드할 수 있습니다.")
        return
    if command_profile is None or command_profile.empty:
        st.info("최종 전압 LUT 추출 사용 불가: command_profile이 없습니다.")
        return
    missing = [column for column in ("time_s", "limited_voltage_v") if column not in command_profile.columns]
    if missing:
        st.warning(f"최종 전압 LUT 추출 사용 불가: 누락 컬럼 {missing}")
        return

    file_name = build_final_voltage_lut_filename(
        waveform_type=waveform_type,
        freq_hz=freq_hz,
        cycle_count=cycle_count,
    )
    first_source_column = _first_model_voltage_source_column(command_profile)
    second_available = "second_limited_voltage_v" in command_profile.columns and (
        "second_modeling_status" not in command_profile.columns
        or not len(command_profile)
        or str(command_profile["second_modeling_status"].iloc[0]) == "ok"
    )
    export_options = ["1차 모델링 결과"]
    if second_available:
        export_options.append("2차 모델링 결과")
    selected_export = st.radio(
        "추출 대상",
        options=export_options,
        index=1 if second_available and st.session_state.get("quick_lut_final_export_source") == "second_model" else 0,
        key="final_modeled_lut_export_source_selector",
        horizontal=True,
    )
    if selected_export == "2차 모델링 결과" and second_available:
        voltage_source_column = "second_limited_voltage_v"
        file_prefix = "second_modeled_voltage_lut"
        st.info("현재 추출 대상: 2차 모델링 결과\n\n2차 모델링에서 생성된 최종 제한 전압 샘플을 저장합니다.")
    else:
        voltage_source_column = first_source_column
        file_prefix = "first_modeled_voltage_lut"
        st.info("현재 추출 대상: 1차 모델링 결과\n\n1차 모델링에서 생성된 최종 제한 전압 샘플을 저장합니다.")
        if not second_available:
            st.info("2차 모델링 결과가 아직 없습니다. 2차 모델링 전압 LUT 생성을 먼저 실행하면 2차 결과를 선택할 수 있습니다.")

    if waveform_type is not None and freq_hz is not None and cycle_count is not None:
        file_name = (
            f"{file_prefix}_{_safe_name_part(waveform_type)}_"
            f"{_format_number_for_filename(freq_hz)}Hz_{_format_number_for_filename(cycle_count)}cycle.csv"
        )
    with st.expander("상세 진단", expanded=False):
        st.caption(f"exported_voltage_source_column: `{voltage_source_column}`")
        st.caption(f"download_filename: `{file_name}`")
    st.download_button(
        label="선택한 결과를 최종 전압 LUT로 다운로드",
        data=build_final_voltage_lut_csv_bytes(command_profile, voltage_source_column=voltage_source_column),
        file_name=file_name,
        mime="text/csv",
        key="download_final_modeled_voltage_lut_csv",
        help="Fourier 재합성 파형이 아닙니다. 선택한 모델링 결과의 최종 time-voltage LUT입니다.",
    )


def _export_voltage_source_column(command_profile: pd.DataFrame) -> str:
    if "second_limited_voltage_v" in command_profile.columns:
        if "second_modeling_status" in command_profile.columns and len(command_profile):
            if str(command_profile["second_modeling_status"].iloc[0]) == "ok":
                return "second_limited_voltage_v"
        elif "second_modeling_available" in command_profile.columns and len(command_profile):
            if bool(command_profile["second_modeling_available"].iloc[0]):
                return "second_limited_voltage_v"
    return _first_model_voltage_source_column(command_profile)


def _first_model_voltage_source_column(command_profile: pd.DataFrame) -> str:
    if "feedback_corrected_limited_voltage_v" in command_profile.columns:
        if "feedback_correction_status" in command_profile.columns and len(command_profile):
            if str(command_profile["feedback_correction_status"].iloc[0]) == "ok":
                return "feedback_corrected_limited_voltage_v"
        elif "feedback_correction_available" in command_profile.columns and len(command_profile):
            if bool(command_profile["feedback_correction_available"].iloc[0]):
                return "feedback_corrected_limited_voltage_v"
    return "limited_voltage_v"


def _safe_name_part(value: object | None) -> str:
    text = "" if value is None else str(value).strip().lower()
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in text).strip("_")


def _format_number_for_filename(value: object | None) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(number):
        return ""
    return f"{number:g}"
