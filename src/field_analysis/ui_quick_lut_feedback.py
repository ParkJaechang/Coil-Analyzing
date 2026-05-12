from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .finite_feedback_peak_correction import apply_finite_feedback_peak_correction
from .finite_actual_drive import build_actual_drive_review_case
from .finite_actual_drive import read_actual_drive_result
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
    st.caption("Production finite feedback correction is 1.0 cycle only. 1.25 / 1.5 / 1.75 / 2.0 are review-only.")
    st.caption("2-cycle policy discarded.")
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


def render_actual_drive_review_from_selection(
    command_profile: pd.DataFrame,
    feedback_selection: dict[str, object] | None,
) -> dict[str, object] | None:
    st.markdown("#### Actual-drive review")
    if not feedback_selection or not feedback_selection.get("csv_bytes"):
        st.info("No actual-drive review result loaded.")
        return None
    status = {
        "uploaded_file_available": True,
        "review_loaded": False,
        "next_action": "Press Load / Review Actual-drive Result",
    }
    if not st.button("Load / Review Actual-drive Result", key="load_review_actual_drive_result"):
        cached = st.session_state.get("quick_lut_actual_drive_review_result")
        if isinstance(cached, dict):
            _render_actual_drive_review_payload(command_profile, cached)
            return cached
        st.dataframe(pd.DataFrame([status]), use_container_width=True)
        return None
    suffix = "_" + Path(str(feedback_selection.get("filename") or "actual_drive_result.csv")).name
    with NamedTemporaryFile(prefix="quick_lut_actual_drive_review_", suffix=suffix, delete=False) as handle:
        temp_path = Path(handle.name)
        handle.write(bytes(feedback_selection["csv_bytes"]))
    try:
        record = read_actual_drive_result(temp_path)
        review_frame, metadata = build_actual_drive_review_case(record)
    except Exception as exc:  # noqa: BLE001 - UI must report parse errors.
        payload = {
            **status,
            "parse_status": "error",
            "parse_error": str(exc),
            "source_file": feedback_selection.get("filename"),
        }
        st.session_state["quick_lut_actual_drive_review_result"] = payload
        st.error(str(exc))
        st.dataframe(pd.DataFrame([payload]), use_container_width=True)
        return payload
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
    payload = {
        "uploaded_file_available": True,
        "review_loaded": True,
        "plot_available": True,
        "source_file": feedback_selection.get("filename") or record.source_file,
        "review_frame": review_frame,
        "metadata": {
            **metadata,
            "hallbz_sign_applied": True,
            "field_normalization_mode": "peak_to_50mT",
            "voltage_normalization_mode": "peak_to_5V_or_limit",
        },
    }
    st.session_state["quick_lut_actual_drive_review_result"] = payload
    _render_actual_drive_review_payload(command_profile, payload)
    return payload


def _render_actual_drive_review_payload(command_profile: pd.DataFrame, payload: dict[str, object]) -> None:
    metadata = dict(payload.get("metadata") or {})
    status = {
        "uploaded_file_available": payload.get("uploaded_file_available", False),
        "review_loaded": payload.get("review_loaded", False),
        "plot_available": payload.get("plot_available", False),
        "hallbz_sign_applied": metadata.get("hallbz_sign_applied", False),
        "field_normalization_mode": metadata.get("field_normalization_mode", "unavailable"),
        "voltage_normalization_mode": metadata.get("voltage_normalization_mode", "unavailable"),
    }
    st.dataframe(pd.DataFrame([status]), use_container_width=True)
    frame = payload.get("review_frame")
    if not isinstance(frame, pd.DataFrame) or not bool(payload.get("plot_available", False)):
        st.info("No actual-drive review result loaded.")
        return
    plot_frame = pd.DataFrame({"time_s": pd.to_numeric(frame["time_s"], errors="coerce")})
    plot_frame["Intended target field normalized to +/-50mT"] = pd.to_numeric(
        frame["normalized_physical_target_output_mT"], errors="coerce"
    )
    plot_frame["Actual measured field normalized to +/-50mT"] = pd.to_numeric(
        frame["normalized_measured_field_mT"], errors="coerce"
    )
    plot_frame["Field residual = target - actual"] = (
        plot_frame["Intended target field normalized to +/-50mT"]
        - plot_frame["Actual measured field normalized to +/-50mT"]
    )
    plot_frame["First modeled voltage command"] = _interp_command_column(command_profile, frame["time_s"], "limited_voltage_v")
    plot_frame["Actual drive voltage from Voltage1_V"] = pd.to_numeric(
        frame.get("normalized_actual_drive_voltage_v", frame.get("normalized_first_voltage_v")), errors="coerce"
    )
    if "second_limited_voltage_v" in command_profile.columns:
        plot_frame["Second modeled voltage"] = _interp_command_column(command_profile, frame["time_s"], "second_limited_voltage_v")
    _render_plot(
        plot_frame,
        [
            "Intended target field normalized to +/-50mT",
            "Actual measured field normalized to +/-50mT",
            "Field residual = target - actual",
            "First modeled voltage command",
            "Actual drive voltage from Voltage1_V",
            "Second modeled voltage",
        ],
        "Intended vs Actual Comparison",
    )
    raw_frame = pd.DataFrame({"time_s": pd.to_numeric(frame["time_s"], errors="coerce")})
    raw_frame["raw HallBz"] = -pd.to_numeric(frame["raw_measured_field_mT"], errors="coerce")
    raw_frame["effective field = -HallBz raw"] = pd.to_numeric(frame["raw_measured_field_mT"], errors="coerce")
    raw_frame["normalized field"] = pd.to_numeric(frame["normalized_measured_field_mT"], errors="coerce")
    raw_frame["raw Voltage1_V"] = pd.to_numeric(frame["raw_first_voltage_v"], errors="coerce")
    raw_frame["normalized/limited voltage"] = pd.to_numeric(frame["normalized_first_voltage_v"], errors="coerce")
    if "current_a" in frame.columns:
        raw_frame["current"] = pd.to_numeric(frame["current_a"], errors="coerce")
    _render_plot(
        raw_frame,
        ["raw HallBz", "effective field = -HallBz raw", "normalized field", "raw Voltage1_V", "normalized/limited voltage", "current"],
        "Raw Actual-drive Visualization",
    )


def _interp_command_column(command_profile: pd.DataFrame, target_time_s: pd.Series, column: str) -> np.ndarray:
    if "time_s" not in command_profile.columns or column not in command_profile.columns:
        return np.full(len(target_time_s), np.nan)
    source_time = pd.to_numeric(command_profile["time_s"], errors="coerce").to_numpy(dtype=float)
    source_value = pd.to_numeric(command_profile[column], errors="coerce").to_numpy(dtype=float)
    target_time = pd.to_numeric(target_time_s, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(source_time) & np.isfinite(source_value)
    if finite.sum() < 2:
        return np.full(len(target_time), np.nan)
    return np.interp(target_time, source_time[finite], source_value[finite], left=np.nan, right=np.nan)


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


def build_command_source_rows(command_profile: pd.DataFrame, metadata: dict[str, object] | None = None) -> list[dict[str, object]]:
    metadata = metadata or {}
    active_source = feedback_export_source_column(command_profile)
    predicted_valid = bool(
        metadata.get("predicted_from_plotted_command", False)
        and "feedback_corrected_predicted_field_mT" in command_profile.columns
    )
    return [
        {"field": "active_command_source", "value": active_source},
        {"field": "plotted_command_source", "value": active_source},
        {"field": "exported_voltage_source_column", "value": active_source},
        {"field": "run_waveform_voltage_source", "value": active_source},
        {"field": "feedback_used_for_correction", "value": bool(metadata.get("feedback_used_for_correction", False))},
        {"field": "predicted_from_plotted_command", "value": bool(metadata.get("predicted_from_plotted_command", False))},
        {"field": "displayed_predicted_valid", "value": predicted_valid},
        {
            "field": "command_prediction_consistency_status",
            "value": metadata.get(
                "command_prediction_consistency_status",
                "ok" if predicted_valid else "forward_prediction_unavailable_for_feedback_corrected_command",
            ),
        },
        {"field": "correction_method", "value": metadata.get("correction_method", "residual_proportional_feedback")},
    ]


def build_feedback_status_rows(metadata: dict[str, object]) -> list[dict[str, object]]:
    route = metadata.get("feedback_route") or "finite_actual_feedback_peak_correction"
    if route == "finite_feedback_symmetric_peak_correction":
        route = "finite_actual_feedback_peak_correction"
    fields = [
        ("route", route),
        ("feedback_correction_available", metadata.get("feedback_correction_available", False)),
        ("feedback_correction_status", metadata.get("feedback_correction_status", "unavailable")),
        ("supported cycles", "1.0"),
        ("unsupported cycles", "1.25, 1.5, 1.75, 2.0"),
        ("unsupported reason", "unsupported_cycle_policy_1cycle_only"),
        ("production cycle policy", "1cycle_only"),
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
        "limited_voltage_v": "active/plotted command",
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
    st.markdown("##### Command source panel")
    st.caption("화면 Command Waveform과 동일한 column을 저장합니다. 실제 active/plotted command source를 아래에서 확인하십시오.")
    st.dataframe(pd.DataFrame(build_command_source_rows(command_profile, metadata)), use_container_width=True)
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
    command_columns = [
        "baseline_limited_voltage_v",
        "Baseline recommended/limited voltage",
        "Feedback correction delta",
        "Feedback corrected limited voltage",
        "active/plotted command",
    ]
    _render_plot(plot_frame, command_columns, "Baseline vs corrected command")
    st.markdown("##### Predicted Output status")
    st.caption("기존 predicted를 corrected prediction처럼 표시하지 않습니다.")
    if "feedback_corrected_predicted_field_mT" in plot_frame.columns:
        _render_plot(plot_frame, ["Physical Target", "feedback_corrected_predicted_field_mT"], "Feedback corrected prediction")
    else:
        st.info("forward prediction unavailable: feedback_corrected_predicted_field_mT가 없어 그래프를 만들지 않습니다.")

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
