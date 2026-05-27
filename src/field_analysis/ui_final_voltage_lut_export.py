from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .voltage_policy import COMMAND_VOLTAGE_LIMIT_LABEL

# UI contract marker: 2차 보정 후 ±10V 제한이 적용된 전압 샘플을 저장합니다.


def build_final_voltage_lut_frame(command_profile: pd.DataFrame, *, voltage_source_column: str | None = None) -> pd.DataFrame:
    voltage_source_column = voltage_source_column or _export_voltage_source_column(command_profile)
    missing = [column for column in ("time_s", voltage_source_column) if column not in command_profile.columns]
    if missing:
        raise ValueError(f"Missing final voltage LUT source columns: {missing}")
    source = _loop_safe_command_profile(command_profile)
    return pd.DataFrame(
        {
            "sample_index": np.arange(len(source), dtype=int),
            "time_s": pd.to_numeric(source["time_s"], errors="coerce"),
            "voltage_v": pd.to_numeric(source[voltage_source_column], errors="coerce"),
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
    st.markdown("#### 5. 최종 전압 LUT 추출")
    st.info(
        "최종 LUT는 화면에 표시된 최종 전압 샘플을 그대로 저장합니다.\n\n"
        "1차 모델링 command를 선택하면 1차 추천 전압 command가 저장됩니다.\n\n"
        "2차 보정 command를 선택하면 2차 보정 후 제한 전압이 저장됩니다.\n\n"
        "저장 컬럼은 sample_index, time_s, voltage_v 세 개뿐입니다.\n\n"
        "Fourier 재합성이나 harmonic coefficient export가 아닙니다."
    )
    # Source-contract marker for tests and PR review: exported CSV uses plotted final
    # command voltage samples; not Fourier; not harmonic resynthesis; columns:
    # sample_index, time_s, voltage_v.
    if command_profile is not None and not getattr(command_profile, "empty", True):
        continuous_loop_output = _profile_bool(command_profile, "continuous_loop_output")
    else:
        continuous_loop_output = False
    if not finite_cycle_mode and not continuous_loop_output:
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
    export_options = ["1차 모델링 command"]
    if second_available:
        export_options.append("2차 보정 command")
    selected_export = st.radio(
        "추출 대상",
        options=export_options,
        index=0,
        key="final_modeled_lut_export_source_selector",
        horizontal=True,
    )
    if selected_export == "2차 보정 command" and second_available:
        voltage_source_column = "second_limited_voltage_v"
        file_prefix = "second_modeled_voltage_lut"
        tail_suffix = _tail_suffix(command_profile)
        tail_state = "사용" if _profile_bool(command_profile, "post_cycle_zero_tail_enabled") else "사용 안 함"
        st.info(
            "현재 추출 대상: 2차 보정 command\n\n"
            "voltage_v = second_limited_voltage_v\n\n"
            f"2차 보정 후 {COMMAND_VOLTAGE_LIMIT_LABEL} 제한이 적용된 전압 샘플을 저장합니다.\n\n"
            "2차 보정 command에는 사용자가 지정한 자기장 0 복귀 시간만큼 tail이 포함될 수 있습니다.\n\n"
            "다운로드되는 2차 LUT는 active cycle + tail 구간을 포함합니다."
        )
        st.caption(f"현재 finite tail 상태: {tail_state}")
        st.caption("tail OFF 상태에서는 active cycle 구간만 LUT로 저장됩니다.")
        st.caption("tail ON 상태에서는 active cycle + tail 구간이 LUT에 포함될 수 있습니다.")
    else:
        voltage_source_column = first_source_column
        file_prefix = "first_modeled_voltage_lut"
        tail_suffix = ""
        st.info(
            "현재 추출 대상: 1차 모델링 command\n\n"
            "voltage_v = 1차 모델링 command\n\n"
            "1차 추천 전압 command를 저장합니다."
        )
        if not second_available:
            st.info("2차 보정 command가 아직 없습니다. 2차 보정 command 생성을 먼저 실행하면 2차 결과를 선택할 수 있습니다.")

    if waveform_type is not None and freq_hz is not None and cycle_count is not None:
        file_name = (
            f"{file_prefix}_{_safe_name_part(waveform_type)}_"
            f"{_format_number_for_filename(freq_hz)}Hz_{_format_number_for_filename(cycle_count)}cycle{tail_suffix}.csv"
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
        help="Fourier 재합성 파형이 아닙니다. 선택한 1차/2차 command의 최종 time-voltage LUT입니다.",
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


def _tail_suffix(command_profile: pd.DataFrame) -> str:
    if "post_cycle_zero_tail_enabled" not in command_profile.columns or not len(command_profile):
        return ""
    if not bool(command_profile["post_cycle_zero_tail_enabled"].iloc[0]):
        return "_tailoff"
    if "post_cycle_zero_tail_cycle_count" not in command_profile.columns:
        return "_plustail"
    tail_cycle = _format_number_for_filename(command_profile["post_cycle_zero_tail_cycle_count"].iloc[0])
    return f"_plus{tail_cycle}tail" if tail_cycle else "_plustail"


def _profile_bool(command_profile: pd.DataFrame, column: str) -> bool:
    if column not in command_profile.columns or command_profile.empty:
        return False
    value = command_profile[column].iloc[0]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _loop_safe_command_profile(command_profile: pd.DataFrame) -> pd.DataFrame:
    if not _profile_bool(command_profile, "continuous_loop_output"):
        return command_profile
    if command_profile.empty or str(command_profile.get("loop_endpoint_policy", pd.Series([""])).iloc[0]) != "period_exclusive":
        return command_profile
    freq_hz = _first_numeric(command_profile, "freq_hz")
    if freq_hz is None or freq_hz <= 0.0:
        return command_profile
    period_s = 1.0 / freq_hz
    time_s = pd.to_numeric(command_profile["time_s"], errors="coerce")
    keep = time_s < period_s - max(period_s * 1e-9, 1e-12)
    if not bool(keep.any()):
        return command_profile
    return command_profile.loc[keep].reset_index(drop=True)


def _first_numeric(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns or frame.empty:
        return None
    try:
        value = float(frame[column].iloc[0])
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


# Continuous steady-state export contract marker:
# continuous_loop_output=True exports a loop-safe 1cycle LUT with
# loop_endpoint_policy=period_exclusive and columns sample_index, time_s, voltage_v.
