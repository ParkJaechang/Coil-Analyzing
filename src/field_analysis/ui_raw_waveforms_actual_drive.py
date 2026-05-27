from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .finite_actual_drive import build_actual_drive_review_case, read_actual_drive_result
from .continuous_steady_state_extraction import build_continuous_steady_state_modeling_case
from .voltage_policy import COMMAND_VOLTAGE_LIMIT_LABEL
from .ui_second_modeling import (
    BASELINE_REMOVED_FIELD_LABEL,
    EFFECTIVE_FIELD_LABEL,
    NORMALIZED_FIELD_LABEL,
    NORMALIZED_VOLTAGE_LABEL,
    RAW_HALLBZ_LABEL,
    RAW_VOLTAGE_LABEL,
    build_native_actual_drive_raw_plot_frame,
)

ACTUAL_DRIVE_REQUIRED_COLUMNS = {"TimeMs", "Voltage1_V", "HallBz"}
FINAL_VOLTAGE_LUT_COLUMNS = {"sample_index", "time_s", "voltage_v"}
ACTUAL_DRIVE_UPLOAD_KEY = "raw_waveform_actual_drive_upload"
REVIEW_RESULT_KEY = "raw_waveform_actual_drive_review_result"
REVIEW_METADATA_KEY = "raw_waveform_actual_drive_review_metadata"
RENDER_KEY = "raw_waveform_actual_drive_render_key"


@dataclass(frozen=True)
class RawActualDriveParsePayload:
    review_frame: pd.DataFrame
    metadata: dict[str, Any]


def classify_raw_waveform_actual_drive_csv(filename: str, csv_bytes: bytes | None) -> dict[str, object]:
    header = _first_csv_header(csv_bytes)
    columns = {part.strip() for part in header.split(",") if part.strip()}
    if FINAL_VOLTAGE_LUT_COLUMNS.issubset(columns):
        return {
            "file_type": "final_voltage_lut",
            "schema_status": "final_voltage_lut_not_actual_drive_result",
            "message": "이 파일은 최종 전압 LUT CSV입니다.",
        }
    if ACTUAL_DRIVE_REQUIRED_COLUMNS.issubset(columns):
        return {
            "file_type": "actual_drive_result",
            "schema_status": "actual_drive_schema",
            "metadata_source": "filename_or_preamble_or_user_fallback",
        }
    return {
        "file_type": "unsupported_schema",
        "schema_status": "unsupported_schema",
        "message": "지원하지 않는 CSV 형식입니다.",
        "filename": filename,
    }


def parse_raw_waveform_actual_drive_upload(
    *,
    filename: str,
    csv_bytes: bytes,
    waveform_type: str | None = None,
    freq_hz: float | None = None,
    cycle_count: float | None = None,
) -> RawActualDriveParsePayload:
    with TemporaryDirectory(prefix="raw_waveform_actual_drive_") as temp_dir:
        temp_path = Path(temp_dir) / Path(filename).name
        temp_path.write_bytes(csv_bytes)
        record = read_actual_drive_result(
            temp_path,
            waveform_type=waveform_type,
            freq_hz=freq_hz,
            cycle_count=cycle_count,
        )
        review_frame, metadata = build_actual_drive_review_case(record)
    metadata = {
        **metadata,
        "source": "Raw Waveforms upload",
        "source_file": filename,
        "schema": "TimeMs / Voltage1_V / HallBz",
        "raw_waveforms_actual_drive_review": True,
    }
    return RawActualDriveParsePayload(review_frame=review_frame, metadata=metadata)


def render_raw_waveforms_actual_drive_upload_section() -> None:
    st.markdown("### 1차 실구동 결과 업로드 확인")
    st.caption("Quick LUT 2차 보정에 사용하는 1차 실구동 결과 CSV를 Raw Waveforms에서도 같은 방식으로 확인합니다.")
    st.caption("TimeMs / Voltage1_V / HallBz 컬럼이 있는 장비 측정 CSV를 업로드하십시오.")
    st.caption("최종 전압 LUT CSV(sample_index / time_s / voltage_v)는 측정 데이터가 아니므로 이 plot 입력으로 사용하지 않습니다.")

    uploaded_files = st.file_uploader(
        "1차 실구동 결과 CSV 업로드",
        type=["csv"],
        accept_multiple_files=True,
        key=ACTUAL_DRIVE_UPLOAD_KEY,
        help="TimeMs / Voltage1_V / HallBz schema의 장비 측정 CSV를 업로드합니다.",
    )
    st.session_state["raw_waveform_actual_drive_files"] = [uploaded.name for uploaded in uploaded_files or []]
    if not uploaded_files:
        st.info("TimeMs / Voltage1_V / HallBz 컬럼이 있는 1차 실구동 결과 CSV를 업로드한 뒤 버튼을 누르십시오.")
        _render_cached_result_if_available(current_render_key=None)
        return

    selected_name = st.selectbox(
        "실구동 데이터 파일 선택",
        options=[uploaded.name for uploaded in uploaded_files],
        key="raw_waveform_actual_drive_selected_file",
    )
    selected_upload = next(uploaded for uploaded in uploaded_files if uploaded.name == selected_name)
    selected_bytes = selected_upload.getvalue()
    classification = classify_raw_waveform_actual_drive_csv(selected_name, selected_bytes)
    _render_schema_message(classification)

    use_user_fallback, waveform_type, freq_hz, cycle_count = _render_metadata_fallback_controls()
    current_render_key = _build_render_key(
        selected_name,
        selected_bytes,
        use_user_fallback=use_user_fallback,
        waveform_type=waveform_type,
        freq_hz=freq_hz,
        cycle_count=cycle_count,
    )
    if st.session_state.get(RENDER_KEY) not in {None, current_render_key}:
        st.warning("설정 또는 파일이 변경되었습니다. 실구동 데이터 plot 생성을 다시 누르십시오.")

    if st.button("실구동 데이터 plot 생성", key="raw_waveform_actual_drive_plot_button"):
        _run_actual_drive_review(
            filename=selected_name,
            csv_bytes=selected_bytes,
            classification=classification,
            use_user_fallback=use_user_fallback,
            waveform_type=waveform_type,
            freq_hz=freq_hz,
            cycle_count=cycle_count,
            render_key=current_render_key,
        )

    _render_cached_result_if_available(current_render_key=current_render_key)


def _run_actual_drive_review(
    *,
    filename: str,
    csv_bytes: bytes,
    classification: dict[str, object],
    use_user_fallback: bool,
    waveform_type: str,
    freq_hz: float,
    cycle_count: float,
    render_key: str,
) -> None:
    if classification.get("file_type") == "final_voltage_lut":
        payload = {
            "review_loaded": False,
            "plot_available": False,
            "source_file": filename,
            "parse_status": "rejected_final_voltage_lut",
            "message": (
                "이 파일은 최종 전압 LUT CSV입니다. Raw Waveforms의 실구동 결과 확인에는 "
                "TimeMs / Voltage1_V / HallBz 컬럼이 있는 장비 측정 CSV가 필요합니다."
            ),
        }
        _save_error_payload(payload, render_key)
        return
    if classification.get("file_type") != "actual_drive_result":
        payload = {
            "review_loaded": False,
            "plot_available": False,
            "source_file": filename,
            "parse_status": "unsupported_schema",
            "message": "지원하지 않는 CSV 형식입니다. 필수 컬럼: TimeMs / Voltage1_V / HallBz",
        }
        _save_error_payload(payload, render_key)
        return

    fallback_kwargs: dict[str, object] = {}
    if use_user_fallback:
        fallback_kwargs = {"waveform_type": waveform_type, "freq_hz": float(freq_hz), "cycle_count": float(cycle_count)}
    try:
        parsed = parse_raw_waveform_actual_drive_upload(filename=filename, csv_bytes=csv_bytes, **fallback_kwargs)
    except ValueError as exc:
        payload = {
            "review_loaded": False,
            "plot_available": False,
            "source_file": filename,
            "parse_status": "parse_error",
            "message": str(exc),
            "next_action": "파일명/metadata에서 조건을 읽지 못했다면 현재 입력값으로 실구동 데이터 조건 지정을 체크하십시오.",
        }
        _save_error_payload(payload, render_key)
        return

    metadata = dict(parsed.metadata)
    if use_user_fallback:
        metadata["raw_waveforms_metadata_source_note"] = "파일명/metadata에서 조건을 읽지 못해 사용자가 지정한 조건으로 처리했습니다."
    payload = {
        "review_loaded": True,
        "plot_available": True,
        "source_file": filename,
        "review_frame": parsed.review_frame,
        "metadata": metadata,
    }
    st.session_state[REVIEW_RESULT_KEY] = payload
    st.session_state[REVIEW_METADATA_KEY] = metadata
    st.session_state[RENDER_KEY] = render_key


def _render_cached_result_if_available(*, current_render_key: str | None) -> None:
    payload = st.session_state.get(REVIEW_RESULT_KEY)
    if not isinstance(payload, dict):
        return
    if current_render_key is not None and st.session_state.get(RENDER_KEY) != current_render_key:
        st.caption("이전 실행 결과를 표시합니다. 현재 설정으로 갱신하려면 실구동 데이터 plot 생성을 다시 누르십시오.")
    if not bool(payload.get("review_loaded")):
        st.error(str(payload.get("message", "실구동 데이터 plot을 생성할 수 없습니다.")))
        if payload.get("next_action"):
            st.caption(str(payload["next_action"]))
        return
    frame = payload.get("review_frame")
    metadata = dict(payload.get("metadata") or {})
    if not isinstance(frame, pd.DataFrame):
        return
    st.success("실구동 데이터 plot 생성 완료")
    _render_actual_drive_status_card(metadata)
    st.caption("이 데이터는 Quick LUT 2차 보정에 사용할 수 있는 1차 실구동 결과 형식입니다.")
    st.caption("Raw plot은 실제 측정 데이터의 native time_s 기준으로 표시됩니다.")
    plot_frame = build_native_actual_drive_raw_plot_frame(frame)
    st.plotly_chart(
        _plot_actual_drive_frame(
            plot_frame,
            [
                RAW_HALLBZ_LABEL,
                EFFECTIVE_FIELD_LABEL,
                BASELINE_REMOVED_FIELD_LABEL,
                NORMALIZED_FIELD_LABEL,
                RAW_VOLTAGE_LABEL,
                NORMALIZED_VOLTAGE_LABEL,
                "Current1_A",
            ],
            title="1차 실구동 데이터 확인",
        ),
        use_container_width=True,
    )
    with st.expander("목표 자기장 vs 실측 자기장", expanded=False):
        field_frame = pd.DataFrame({"time_s": pd.to_numeric(frame["time_s"], errors="coerce")})
        _copy_numeric(frame, field_frame, "normalized_physical_target_output_mT", "normalized_physical_target_output_mT")
        _copy_numeric(frame, field_frame, "normalized_measured_field_mT", "normalized_measured_field_mT")
        _copy_numeric(frame, field_frame, "measured_residual_normalized_mT", "measured_residual_normalized_mT")
        st.plotly_chart(
            _plot_actual_drive_frame(
                field_frame,
                ["normalized_physical_target_output_mT", "normalized_measured_field_mT", "measured_residual_normalized_mT"],
                title="목표 자기장 vs 실측 자기장",
            ),
            use_container_width=True,
        )
    _render_continuous_extraction_preview(frame, metadata)


def _render_continuous_extraction_preview(frame: pd.DataFrame, metadata: dict[str, Any]) -> None:
    st.markdown("#### Continuous steady-state extraction preview")
    st.caption("Quick LUT modeling을 실행하지 않아도 Raw Waveforms에서 extraction 검수 가능합니다.")
    st.caption("이 1cycle이 continuous steady-state modeling에 사용됩니다.")
    if not st.button("Steady-state 1cycle 추출", key="raw_waveform_continuous_steady_extract_button"):
        st.info("옵션 변경만으로 heavy calculation을 자동 실행하지 않습니다. 버튼을 누르면 안정화된 1cycle preview를 생성합니다.")
        return
    try:
        case = build_continuous_steady_state_modeling_case(
            frame,
            waveform_type=str(metadata.get("waveform_type") or "sine"),
            freq_hz=float(metadata.get("freq_hz") or 1.0),
        )
    except Exception as exc:  # noqa: BLE001 - UI should report parse/extraction errors.
        st.error(f"Continuous steady-state extraction 실패: {exc}")
        return
    window = case["steady_state_one_cycle_frame"]
    extract_meta = case["metadata"]
    st.session_state["raw_waveform_continuous_steady_state_extraction_result"] = case
    st.plotly_chart(
        _plot_actual_drive_frame(
            build_native_actual_drive_raw_plot_frame(frame),
            [RAW_HALLBZ_LABEL, EFFECTIVE_FIELD_LABEL, NORMALIZED_FIELD_LABEL, RAW_VOLTAGE_LABEL],
            title="Continuous 원본: startup transient와 steady-state 구간",
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        _plot_actual_drive_frame(
            window.rename(
                columns={
                    "measured_field_normalized_mT": NORMALIZED_FIELD_LABEL,
                    "voltage_normalized_v": NORMALIZED_VOLTAGE_LABEL,
                }
            ),
            ["normalized_physical_target_output_mT", NORMALIZED_FIELD_LABEL, NORMALIZED_VOLTAGE_LABEL],
            title="선택된 steady-state 1cycle",
        ),
        use_container_width=True,
    )
    st.markdown("#### cycle stability metrics")
    st.dataframe(pd.DataFrame([extract_meta]), use_container_width=True)


def _render_actual_drive_status_card(metadata: dict[str, Any]) -> None:
    st.markdown("#### 사용 중인 실구동 데이터")
    rows = [
        ("파일명", metadata.get("source_file")),
        ("source", "Raw Waveforms upload"),
        ("schema", "TimeMs / Voltage1_V / HallBz"),
        ("waveform", metadata.get("waveform_type")),
        ("freq_hz", metadata.get("freq_hz")),
        ("cycle_count", metadata.get("cycle_count")),
        ("metadata source", metadata.get("metadata_source")),
        ("timebase status", metadata.get("timebase_status")),
        ("detected time unit", metadata.get("actual_drive_time_unit_detected", metadata.get("time_unit"))),
        ("dt median", metadata.get("dt_median_s")),
        ("active duration ratio", metadata.get("active_duration_ratio")),
        ("HallBz convention", "effective field = -HallBz raw"),
        ("field normalization", "±50mT"),
        ("voltage normalization", COMMAND_VOLTAGE_LIMIT_LABEL),
    ]
    note = metadata.get("raw_waveforms_metadata_source_note")
    if note:
        rows.append(("metadata note", note))
    st.dataframe(pd.DataFrame(rows, columns=["항목", "값"]), use_container_width=True, hide_index=True)


def _render_schema_message(classification: dict[str, object]) -> None:
    file_type = classification.get("file_type")
    if file_type == "actual_drive_result":
        st.caption("schema: TimeMs / Voltage1_V / HallBz")
        return
    if file_type == "final_voltage_lut":
        st.warning(
            "이 파일은 최종 전압 LUT CSV입니다. Raw Waveforms의 실구동 결과 확인에는 "
            "TimeMs / Voltage1_V / HallBz 컬럼이 있는 장비 측정 CSV가 필요합니다."
        )
        return
    st.warning("지원하지 않는 CSV 형식입니다. 필수 컬럼: TimeMs / Voltage1_V / HallBz")


def _render_metadata_fallback_controls() -> tuple[bool, str, float, float]:
    st.caption("파일명 또는 preamble에서 waveform/freq/cycle을 읽지 못하면 아래 값을 fallback metadata로 사용할 수 있습니다.")
    use_user_fallback = st.checkbox(
        "현재 입력값으로 실구동 데이터 조건 지정",
        value=False,
        key="raw_waveform_actual_drive_use_user_fallback",
    )
    waveform_type = st.selectbox(
        "waveform family",
        options=["sine", "triangle", "rounded_triangle"],
        key="raw_waveform_actual_drive_fallback_waveform",
    )
    freq_hz = st.number_input(
        "freq_hz",
        min_value=0.001,
        value=1.5,
        step=0.1,
        key="raw_waveform_actual_drive_fallback_freq_hz",
    )
    cycle_count = st.selectbox(
        "cycle_count",
        options=[1.0, 1.5, 1.25, 1.75],
        index=1,
        key="raw_waveform_actual_drive_fallback_cycle_count",
    )
    st.caption("1.0 / 1.5 cycle은 production 기준이며, 1.25 / 1.75 cycle은 review-only 조건입니다.")
    return use_user_fallback, str(waveform_type), float(freq_hz), float(cycle_count)


def _plot_actual_drive_frame(frame: pd.DataFrame, columns: list[str], *, title: str) -> go.Figure:
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
            )
        )
    figure.update_layout(
        template="plotly_white",
        height=380,
        title=title,
        xaxis_title="시간 (s)",
        yaxis_title="측정값",
        legend_title="항목",
    )
    return figure


def _copy_numeric(source: pd.DataFrame, target: pd.DataFrame, source_column: str, label: str) -> None:
    if source_column in source.columns:
        target[label] = pd.to_numeric(source[source_column], errors="coerce")


def _save_error_payload(payload: dict[str, object], render_key: str) -> None:
    st.session_state[REVIEW_RESULT_KEY] = payload
    st.session_state[REVIEW_METADATA_KEY] = dict(payload)
    st.session_state[RENDER_KEY] = render_key


def _build_render_key(
    filename: str,
    csv_bytes: bytes,
    *,
    use_user_fallback: bool,
    waveform_type: str,
    freq_hz: float,
    cycle_count: float,
) -> str:
    digest = sha256(csv_bytes).hexdigest()[:16]
    return f"{filename}:{digest}:{use_user_fallback}:{waveform_type}:{freq_hz:g}:{cycle_count:g}"


def _first_csv_header(csv_bytes: bytes | None) -> str:
    if not csv_bytes:
        return ""
    text = bytes(csv_bytes).decode("utf-8-sig", errors="ignore")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return ""
