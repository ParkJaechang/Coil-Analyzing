from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .finite_feedback_peak_correction import apply_finite_feedback_peak_correction
from .ui_raw_waveforms_labels import infer_new_dataset_filename_metadata
from .ui_upload_cache import add_upload_cache_bytes
from .ui_upload_cache import build_upload_cache_records
from .ui_upload_cache import build_upload_cache_selection_options
from .ui_upload_cache import cache_item_bytes
from .ui_upload_cache import fallback_upload_cache_selection


FEEDBACK_CACHE_STATE_KEY = "actual_drive_validation_cache_items"
FEEDBACK_SELECTED_CACHE_KEY = "quick_lut_feedback_selected_cache_id"
FEEDBACK_RUN_LABEL_KEY = "quick_lut_feedback_run_label"


def render_quick_lut_feedback_input_section(*, finite_cycle_mode: bool) -> dict[str, object] | None:
    st.markdown("#### Quick LUT feedback correction")
    st.caption("Physical Target은 유지하고, 실제 구동 결과를 이용해 전압 command만 보정합니다.")
    st.caption("Raw/absolute gain 평가는 하지 않고, ±50mT / ±5V 정규화 기준으로 개형과 타이밍을 봅니다.")
    st.caption("1.25 / 1.75는 phase delay 문제로 peak feedback correction 주력 경로에서 제외합니다.")
    if not finite_cycle_mode:
        st.info("Feedback correction은 Quick LUT finite field compensation 결과에서만 사용할 수 있습니다.")
        return None

    cache_state = st.session_state.setdefault(FEEDBACK_CACHE_STATE_KEY, {})
    if not isinstance(cache_state, dict):
        cache_state = {}
        st.session_state[FEEDBACK_CACHE_STATE_KEY] = cache_state

    uploaded_files = st.file_uploader(
        "actual-drive result files",
        type=["csv"],
        accept_multiple_files=True,
        key="quick_lut_feedback_result_upload",
        help="TimeMs / Voltage1_V / HallBz schema의 실제 구동 결과 CSV를 업로드합니다.",
    )
    for uploaded_file in uploaded_files or []:
        add_upload_cache_bytes(
            cache_state,
            uploaded_file.name,
            uploaded_file.getvalue(),
            cache_type="actual_drive_validation",
            allow_duplicate=False,
        )

    run_label = st.selectbox(
        "feedback run label",
        options=["first_run", "second_run", "unknown"],
        index=0,
        key=FEEDBACK_RUN_LABEL_KEY,
        help="사용자가 feedback source의 run 단계를 구분하기 위한 UI metadata입니다.",
    )
    records = [record for record in build_upload_cache_records(cache_state) if record.cache_type == "actual_drive_validation"]
    if not records:
        st.info("cached feedback files가 없습니다. actual-drive result files를 업로드하면 선택할 수 있습니다.")
        return None

    options, records_by_id, labels_by_id = build_upload_cache_selection_options(records)
    selected_id = fallback_upload_cache_selection(options, st.session_state.get(FEEDBACK_SELECTED_CACHE_KEY))
    if selected_id is None:
        st.info("cached feedback files 선택 항목이 없습니다.")
        return None
    st.session_state[FEEDBACK_SELECTED_CACHE_KEY] = selected_id
    selected_id = st.selectbox(
        "cached feedback files",
        options=options,
        format_func=lambda cache_id: labels_by_id[cache_id],
        key=FEEDBACK_SELECTED_CACHE_KEY,
    )
    selected = records_by_id[selected_id]
    source_bytes = cache_item_bytes(cache_state, selected_id)
    parse_status = "available" if source_bytes else "missing_bytes"
    parsed = infer_new_dataset_filename_metadata(selected.original_filename)
    st.caption(f"filename: `{selected.original_filename}`")
    st.caption(
        "waveform/freq/cycle: "
        f"`{parsed.get('waveform_type') or 'unknown'}` / "
        f"`{parsed.get('freq_hz', 'unknown')}` Hz / "
        f"`{parsed.get('cycle_count', 'unknown')}` cycle"
    )
    st.caption(f"parse status: `{parse_status}` · alignment status: `pending_until_run` · run label: `{run_label}`")
    st.caption(f"internal id: `{selected.cache_item_id}`")
    return {
        "cache_id": selected_id,
        "filename": selected.original_filename,
        "csv_bytes": source_bytes,
        "run_label": run_label,
        "parse_status": parse_status,
        "alignment_status": "pending_until_run",
    }


def apply_feedback_correction_from_selection(
    command_profile: pd.DataFrame,
    feedback_selection: dict[str, object] | None,
    *,
    waveform_type: str,
    freq_hz: float,
    cycle_count: float | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not feedback_selection or not feedback_selection.get("csv_bytes"):
        return command_profile, {
            "feedback_route": "finite_actual_feedback_peak_correction",
            "feedback_correction_available": False,
            "feedback_correction_status": "feedback_source_unavailable",
            "feedback_used_for_correction": False,
            "target_unchanged": True,
        }
    if cycle_count is None:
        return command_profile, {
            "feedback_route": "finite_actual_feedback_peak_correction",
            "feedback_correction_available": False,
            "feedback_correction_status": "missing_cycle_count",
            "feedback_used_for_correction": False,
            "target_unchanged": True,
        }
    suffix = "_" + Path(str(feedback_selection.get("filename") or "feedback.csv")).name
    with NamedTemporaryFile(prefix="quick_lut_feedback_", suffix=suffix, delete=False) as handle:
        temp_path = Path(handle.name)
        handle.write(bytes(feedback_selection["csv_bytes"]))
    try:
        corrected, metadata = apply_finite_feedback_peak_correction(
            command_profile,
            temp_path,
            waveform_type=str(waveform_type),
            freq_hz=float(freq_hz),
            cycle_count=float(cycle_count),
        )
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
    metadata = dict(metadata)
    metadata["feedback_route"] = "finite_actual_feedback_peak_correction"
    metadata["feedback_source_file"] = feedback_selection.get("filename") or metadata.get("feedback_source_file")
    metadata["feedback_run_label"] = feedback_selection.get("run_label") or metadata.get("feedback_run_label")
    return corrected, metadata


def feedback_export_source_column(command_profile: pd.DataFrame) -> str:
    if "feedback_corrected_limited_voltage_v" not in command_profile.columns:
        return "limited_voltage_v"
    if "feedback_correction_status" in command_profile.columns and len(command_profile):
        if str(command_profile["feedback_correction_status"].iloc[0]) != "ok":
            return "limited_voltage_v"
    if "feedback_correction_available" in command_profile.columns and len(command_profile):
        if not bool(command_profile["feedback_correction_available"].iloc[0]):
            return "limited_voltage_v"
    return "feedback_corrected_limited_voltage_v"


def build_feedback_status_rows(metadata: dict[str, object]) -> list[dict[str, object]]:
    route = metadata.get("feedback_route") or "finite_actual_feedback_peak_correction"
    if route == "finite_feedback_symmetric_peak_correction":
        route = "finite_actual_feedback_peak_correction"
    fields = [
        ("route", route),
        ("feedback_correction_available", metadata.get("feedback_correction_available", False)),
        ("feedback_correction_status", metadata.get("feedback_correction_status", "unavailable")),
        ("supported cycles", "1.0, 1.5"),
        ("unsupported cycles", "1.25, 1.75"),
        ("unsupported reason", "unsupported_cycle_phase_delay"),
        ("filename", metadata.get("feedback_source_file", "unavailable")),
        ("parse status", metadata.get("feedback_schema_status", "unavailable")),
        ("alignment status", metadata.get("feedback_alignment_status") or metadata.get("alignment_status", "unavailable")),
        ("target_unchanged", metadata.get("target_unchanged", True)),
    ]
    return [{"field": field, "value": value} for field, value in fields]


def build_feedback_plot_frame(command_profile: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame({"time_s": pd.to_numeric(command_profile["time_s"], errors="coerce")})
    columns = {
        "physical_target_output_mT": "Physical Target",
        "measured_field_normalized_mT": "Normalized measured feedback field",
        "baseline_limited_voltage_v": "Baseline recommended/limited voltage",
        "feedback_correction_delta_v": "Feedback correction delta",
        "feedback_corrected_limited_voltage_v": "Feedback corrected limited voltage",
        "feedback_corrected_predicted_field_mT": "feedback_corrected_predicted_field_mT",
    }
    for source, label in columns.items():
        if source in command_profile.columns:
            frame[label] = pd.to_numeric(command_profile[source], errors="coerce")
    if {"Physical Target", "Normalized measured feedback field"}.issubset(frame.columns):
        frame["Residual"] = frame["Physical Target"] - frame["Normalized measured feedback field"]
    return frame


def render_feedback_correction_review(command_profile: pd.DataFrame, metadata: dict[str, object]) -> None:
    st.markdown("#### Quick LUT feedback correction result")
    st.caption("사용자가 그래프를 보고 판단하는 검토 화면입니다. 모델 품질을 자동 합격 처리하지 않습니다.")
    st.dataframe(pd.DataFrame(build_feedback_status_rows(metadata)), use_container_width=True)
    st.markdown("##### Normalization panel")
    st.caption("HallBz sign applied")
    st.caption("field peak normalized to ±50mT")
    st.caption("voltage normalized/limited to ±5V")
    st.caption("raw peak values shown as informational only")
    norm_rows = [
        {"field": "hallbz_sign_applied", "value": metadata.get("hallbz_sign_applied", "unavailable")},
        {"field": "field_normalization_mode", "value": metadata.get("field_normalization_mode", "unavailable")},
        {"field": "field_normalization_scale_factor", "value": metadata.get("field_normalization_scale_factor", "unavailable")},
        {"field": "voltage_normalization_mode", "value": metadata.get("voltage_normalization_mode", "unavailable")},
        {"field": "voltage_normalization_scale_factor", "value": metadata.get("voltage_normalization_scale_factor", "unavailable")},
        {"field": "raw_field_peak_mT", "value": metadata.get("raw_field_peak_mT", "informational only")},
        {"field": "raw_voltage_peak_v", "value": metadata.get("raw_voltage_peak_v", "informational only")},
    ]
    st.dataframe(pd.DataFrame(norm_rows), use_container_width=True)

    if not bool(metadata.get("feedback_correction_available", False)):
        st.info("feedback correction unavailable: status panel only. Prediction graph is not faked.")
        return

    plot_frame = build_feedback_plot_frame(command_profile)
    _render_plot(plot_frame, ["Physical Target", "Normalized measured feedback field", "Residual"], "Field feedback review")
    _render_plot(
        plot_frame,
        ["Baseline recommended/limited voltage", "Feedback correction delta", "Feedback corrected limited voltage"],
        "Voltage feedback correction review",
    )
    if "feedback_corrected_predicted_field_mT" in plot_frame.columns:
        _render_plot(plot_frame, ["Physical Target", "feedback_corrected_predicted_field_mT"], "Feedback corrected prediction")
    else:
        st.info("feedback_corrected_predicted_field_mT unavailable: prediction panel is unavailable for this result.")

    metrics = [
        "positive_peak_error_before_mT",
        "negative_peak_error_before_mT",
        "peak_symmetry_error_before_mT",
        "positive_peak_error_after_mT",
        "negative_peak_error_after_mT",
        "peak_symmetry_error_after_mT",
        "alignment_time_shift_s",
        "correction_delta_peak_v",
        "voltage_limit_status",
    ]
    st.markdown("##### Feedback metrics")
    st.dataframe(pd.DataFrame([{"metric": key, "value": metadata.get(key, "unavailable")} for key in metrics]), use_container_width=True)


def _render_plot(frame: pd.DataFrame, columns: list[str], title: str) -> None:
    figure = go.Figure()
    for column in columns:
        if column not in frame.columns:
            continue
        figure.add_trace(go.Scatter(x=frame["time_s"], y=frame[column], mode="lines", name=column))
    figure.update_layout(template="plotly_white", height=320, title=title, xaxis_title="time_s")
    st.plotly_chart(figure, use_container_width=True)
