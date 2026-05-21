from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from .ui_final_voltage_lut_export import build_final_voltage_lut_frame


STAGE_LABELS = {
    "first": "Continuous 1차 modeling command",
    "second": "Continuous 2차 보정 command",
}


def normalize_continuous_result_contract(result: dict[str, Any] | None, stage: str) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    command = result.get("command_profile")
    if not isinstance(command, pd.DataFrame):
        return result
    normalized = dict(result)
    command = command.copy(deep=True)
    metadata = dict(normalized.get("metadata") or {})
    source_column = _continuous_voltage_source_column(command, stage)
    stage_name = "first_model" if stage == "first" else "second_model"
    status_key = "continuous_first_modeling_status" if stage == "first" else "continuous_second_modeling_status"
    metadata.update(
        {
            "continuous_result_stage": stage_name,
            "final_export_voltage_source_column": source_column,
            "continuous_loop_output": True,
            "loop_endpoint_policy": "period_exclusive",
            "continuous_export_cycle_count": 1.0,
        }
    )
    if source_column and metadata.get(status_key) is None:
        metadata[status_key] = "ok" if not command.empty else "not_generated"
    if "continuous_loop_output" not in command.columns:
        command["continuous_loop_output"] = True
    if "loop_endpoint_policy" not in command.columns:
        command["loop_endpoint_policy"] = "period_exclusive"
    if "continuous_export_cycle_count" not in command.columns:
        command["continuous_export_cycle_count"] = 1.0
    normalized["command_profile"] = command
    normalized["metadata"] = metadata
    return normalized


def continuous_result_export_record(stage: str, result: dict[str, Any] | None) -> dict[str, Any]:
    label = STAGE_LABELS.get(stage, str(stage))
    if not isinstance(result, dict):
        return {
            "stage": stage,
            "label": label,
            "available": False,
            "unavailable_reason": "not_generated",
            "voltage_source_column": None,
            "command_profile": None,
            "metadata": {},
        }
    command = result.get("command_profile")
    metadata = dict(result.get("metadata") or {})
    if not isinstance(command, pd.DataFrame) or command.empty:
        return {
            "stage": stage,
            "label": label,
            "available": False,
            "unavailable_reason": "command_profile_empty",
            "voltage_source_column": None,
            "command_profile": command if isinstance(command, pd.DataFrame) else None,
            "metadata": metadata,
        }
    source_column = _continuous_voltage_source_column(command, stage)
    if source_column is None:
        return {
            "stage": stage,
            "label": label,
            "available": False,
            "unavailable_reason": "voltage_source_column_missing",
            "voltage_source_column": None,
            "command_profile": command,
            "metadata": metadata,
        }
    return {
        "stage": stage,
        "label": label,
        "available": True,
        "unavailable_reason": None,
        "voltage_source_column": source_column,
        "command_profile": command,
        "metadata": metadata,
    }


def build_continuous_final_lut_frame(
    command_profile: pd.DataFrame,
    *,
    voltage_source_column: str,
    freq_hz: float | None,
    stage: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = command_profile.copy()
    if freq_hz is not None and "freq_hz" not in source.columns:
        source["freq_hz"] = float(freq_hz)
    if "continuous_loop_output" not in source.columns:
        source["continuous_loop_output"] = True
    if "loop_endpoint_policy" not in source.columns:
        source["loop_endpoint_policy"] = "period_exclusive"
    exported = build_final_voltage_lut_frame(source, voltage_source_column=voltage_source_column)
    _validate_continuous_lut_frame(exported, freq_hz=freq_hz)
    metadata = {
        "continuous_final_lut_export_selected_stage": stage,
        "continuous_final_lut_export_voltage_source_column": voltage_source_column,
        "continuous_final_lut_export_row_count": int(len(exported)),
        "continuous_final_lut_export_columns": "sample_index,time_s,voltage_v",
        "continuous_final_lut_export_loop_safe": True,
        "continuous_final_lut_export_status": "ok",
    }
    return exported, metadata


def build_continuous_final_lut_csv_bytes(
    command_profile: pd.DataFrame,
    *,
    voltage_source_column: str,
    freq_hz: float | None,
    stage: str,
) -> bytes:
    frame, _metadata = build_continuous_final_lut_frame(
        command_profile,
        voltage_source_column=voltage_source_column,
        freq_hz=freq_hz,
        stage=stage,
    )
    return frame.to_csv(index=False).encode("utf-8-sig")


def build_continuous_final_lut_filename(*, stage: str, waveform_type: object | None, freq_hz: object | None) -> str:
    stage_part = "second" if stage == "second" else "first"
    waveform = _safe_name_part(waveform_type) or "continuous"
    freq = _format_number_for_filename(freq_hz) or "unknown"
    return f"continuous_{stage_part}_voltage_lut_{waveform}_{freq}Hz_1cycle_loop.csv"


def render_continuous_final_voltage_lut_export_section(*, waveform_type: object | None, freq_hz: float | None) -> None:
    st.markdown("#### Continuous 최종 전압 LUT 추출")
    st.info(
        "Continuous 1차 또는 2차 modeling 결과 중 하나를 선택하여 반복 출력용 1cycle 전압 LUT로 다운로드합니다.\n\n"
        "다운로드 CSV는 sample_index,time_s,voltage_v 세 컬럼만 포함합니다.\n\n"
        "Continuous mode의 LUT는 endpoint 중복 없는 loop-safe 1cycle입니다."
    )
    first = normalize_continuous_result_contract(st.session_state.get("quick_lut_first_model_result_continuous"), "first")
    second = normalize_continuous_result_contract(st.session_state.get("quick_lut_second_model_result_continuous"), "second")
    if first is not None:
        st.session_state["quick_lut_first_model_result_continuous"] = first
    if second is not None:
        st.session_state["quick_lut_second_model_result_continuous"] = second
    records = [
        continuous_result_export_record("first", first),
        continuous_result_export_record("second", second),
    ]
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "추출 대상": record["label"],
                    "status": "available" if record["available"] else "unavailable",
                    "voltage_source": record["voltage_source_column"] or "unavailable",
                    "reason": record["unavailable_reason"] or "ok",
                }
                for record in records
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    labels = [record["label"] for record in records]
    selected_label = st.radio("추출 대상", options=labels, index=0, key="continuous_final_lut_export_stage_selector")
    selected = next(record for record in records if record["label"] == selected_label)
    if not selected["available"]:
        if selected["stage"] == "second":
            st.warning("Continuous 2차 보정 command가 아직 생성되지 않았습니다.")
            st.caption("먼저 Continuous 1차 실구동 결과 업로드 → 안정 1cycle 추출 → Continuous 2차 보정 command 생성을 실행하십시오.")
        else:
            st.warning("Continuous 1차 modeling command가 아직 생성되지 않았습니다.")
        return
    command = selected["command_profile"]
    source_column = str(selected["voltage_source_column"])
    st.caption(f"다운로드 voltage_v source: {source_column}")
    try:
        export_frame, export_meta = build_continuous_final_lut_frame(
            command,
            voltage_source_column=source_column,
            freq_hz=freq_hz,
            stage=str(selected["stage"]),
        )
    except ValueError as exc:
        st.warning(f"Continuous 최종 전압 LUT 추출 불가: {exc}")
        return
    file_name = build_continuous_final_lut_filename(stage=str(selected["stage"]), waveform_type=waveform_type, freq_hz=freq_hz)
    with st.expander("Continuous export diagnostics", expanded=False):
        st.dataframe(pd.DataFrame([export_meta]), use_container_width=True, hide_index=True)
        st.dataframe(export_frame.head(20), use_container_width=True, hide_index=True)
    st.download_button(
        label="선택한 Continuous 결과를 최종 전압 LUT로 다운로드",
        data=export_frame.to_csv(index=False).encode("utf-8-sig"),
        file_name=file_name,
        mime="text/csv",
        key=f"download_continuous_final_lut_{selected['stage']}",
    )


def _continuous_voltage_source_column(command: pd.DataFrame, stage: str) -> str | None:
    candidates = ("second_limited_voltage_v", "limited_voltage_v") if stage == "second" else ("limited_voltage_v", "first_modeled_voltage_v")
    for column in candidates:
        if column in command.columns:
            return column
    return None


def _validate_continuous_lut_frame(frame: pd.DataFrame, *, freq_hz: float | None) -> None:
    if list(frame.columns) != ["sample_index", "time_s", "voltage_v"]:
        raise ValueError("invalid_continuous_export_columns")
    if frame.empty:
        raise ValueError("empty_continuous_export")
    time_s = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
    voltage = pd.to_numeric(frame["voltage_v"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(time_s).all():
        raise ValueError("nonfinite_time")
    if not np.isfinite(voltage).all():
        raise ValueError("nonfinite_voltage")
    if len(time_s) > 1 and not bool(np.all(np.diff(time_s) > 0.0)):
        raise ValueError("nonmonotonic_time")
    if np.nanmin(time_s) < -1e-12:
        raise ValueError("negative_time")
    try:
        freq = float(freq_hz) if freq_hz is not None else float("nan")
    except (TypeError, ValueError):
        freq = float("nan")
    if np.isfinite(freq) and freq > 0.0:
        period_s = 1.0 / freq
        if np.nanmax(time_s) >= period_s - max(period_s * 1e-9, 1e-12):
            raise ValueError("continuous_endpoint_not_period_exclusive")


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
