from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .finite_actual_drive import build_actual_drive_review_case
from .finite_actual_drive import read_actual_drive_result
from .finite_feedback_peak_correction import apply_finite_feedback_peak_correction
from .ui_quick_lut_feedback_selection import candidate_matches
from .ui_quick_lut_feedback_selection import choose_actual_drive_feedback_candidate
from .ui_quick_lut_feedback_selection import classify_feedback_csv_candidate
from .ui_raw_waveforms_labels import infer_new_dataset_filename_metadata
from .ui_upload_cache import add_upload_cache_bytes
from .ui_upload_cache import build_upload_cache_records
from .ui_upload_cache import build_upload_cache_selection_options
from .ui_upload_cache import cache_item_bytes
from .ui_upload_cache import fallback_upload_cache_selection


FEEDBACK_CACHE_STATE_KEY = "actual_drive_validation_cache_items"
FEEDBACK_SELECTED_CACHE_KEY = "quick_lut_feedback_selected_cache_id"
FEEDBACK_RUN_LABEL_KEY = "quick_lut_feedback_run_label"


FIELD_TARGET_LABEL = "목표 자기장 (±50mT)"
MEASURED_FIELD_LABEL = "실측 자기장 (HallBz 부호 보정, ±50mT)"
RESIDUAL_LABEL = "오차 (목표 - 실측)"
FIRST_VOLTAGE_LABEL = "1차 모델링 전압"
ACTUAL_VOLTAGE_LABEL = "실제 구동 전압"
SECOND_VOLTAGE_LABEL = "2차 모델링 전압"
CORRECTION_DELTA_LABEL = "보정 전압 변화량"
ACTIVE_COMMAND_LABEL = "현재 표시 중인 전압 명령"
BASELINE_VOLTAGE_LABEL = "1차 추천/제한 전압"
FEEDBACK_LIMITED_LABEL = "피드백 보정 후 제한 전압"


def render_quick_lut_feedback_input_section(
    *,
    finite_cycle_mode: bool,
    waveform_type: str | None = None,
    freq_hz: float | None = None,
    cycle_count: float | None = None,
) -> dict[str, object] | None:
    st.markdown("#### Quick LUT 피드백 보정")
    st.caption("목표 자기장은 유지하고, 실제 구동 결과로 전압 명령만 보정합니다.")
    st.caption("Raw/absolute gain 평가는 하지 않고, ±50mT / ±5V 정규화 기준으로 개형과 타이밍을 봅니다.")
    st.caption("Production finite 보정은 1.0 / 1.5 cycle을 지원합니다.")
    st.caption("1.25 / 1.75 / 2.0 cycle은 검토용이며 production 보정/내보내기 대상이 아닙니다.")
    st.caption("2-cycle production 정책은 폐기되었습니다.")
    if not finite_cycle_mode:
        st.info("피드백 보정은 Quick LUT finite field compensation 결과에서만 사용할 수 있습니다.")
        return None

    cache_state = st.session_state.setdefault(FEEDBACK_CACHE_STATE_KEY, {})
    if not isinstance(cache_state, dict):
        cache_state = {}
        st.session_state[FEEDBACK_CACHE_STATE_KEY] = cache_state

    uploaded_files = st.file_uploader(
        "실구동 결과 CSV 업로드",
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
        "피드백 run 단계",
        options=["first_run", "second_run", "unknown"],
        index=0,
        key=FEEDBACK_RUN_LABEL_KEY,
        help="피드백 원본이 1차/2차/unknown run 중 어느 단계인지 구분하는 metadata입니다.",
    )
    records = [record for record in build_upload_cache_records(cache_state) if record.cache_type == "actual_drive_validation"]
    if not records:
        st.info("캐시된 실구동 결과 파일이 없습니다. 실구동 결과 CSV를 업로드하면 선택할 수 있습니다.")
        return None

    options, records_by_id, labels_by_id = build_upload_cache_selection_options(records)
    candidate_payloads = [
        {
            "cache_id": record.cache_item_id,
            "filename": record.original_filename,
            "csv_bytes": cache_item_bytes(cache_state, record.cache_item_id),
            "run_label": run_label,
        }
        for record in records
    ]
    auto_selected, auto_meta = choose_actual_drive_feedback_candidate(
        candidate_payloads,
        waveform_type=waveform_type,
        freq_hz=freq_hz,
        cycle_count=cycle_count,
    )
    if auto_selected is not None:
        selected_id = str(auto_selected.get("cache_id"))
    else:
        selected_id = fallback_upload_cache_selection(options, st.session_state.get(FEEDBACK_SELECTED_CACHE_KEY))
    if selected_id is None:
        st.info("캐시된 실구동 결과 파일 선택 항목이 없습니다.")
        return None
    st.session_state[FEEDBACK_SELECTED_CACHE_KEY] = selected_id
    selected_id = st.selectbox(
        "캐시된 실구동 결과 파일",
        options=options,
        format_func=lambda cache_id: labels_by_id[cache_id],
        key=FEEDBACK_SELECTED_CACHE_KEY,
    )
    selected = records_by_id[selected_id]
    source_bytes = cache_item_bytes(cache_state, selected_id)
    parse_status = "available" if source_bytes else "missing_bytes"
    parsed = infer_new_dataset_filename_metadata(selected.original_filename)
    selected_payload = {
        "cache_id": selected_id,
        "filename": selected.original_filename,
        "csv_bytes": source_bytes,
        "run_label": run_label,
    }
    selected_classification = classify_feedback_csv_candidate(selected.original_filename, source_bytes)
    match_status = "match" if candidate_matches(selected_classification, waveform_type=waveform_type, freq_hz=freq_hz, cycle_count=cycle_count) else "mismatch_or_unknown"

    st.caption(f"파일: `{selected.original_filename}`")
    st.caption(
        "파형/주파수/cycle: "
        f"`{parsed.get('waveform_type') or 'unknown'}` / "
        f"`{parsed.get('freq_hz', 'unknown')}` Hz / "
        f"`{parsed.get('cycle_count', 'unknown')}` cycle"
    )
    st.caption(f"파싱 상태: `{parse_status}` · 정렬 상태: `pending_until_run` · run 단계: `{run_label}`")
    st.caption(
        "자동 선택 상태: "
        f"`{auto_meta.get('selection_status')}` / `{auto_meta.get('selection_reason')}` · "
        f"match 상태: `{match_status}`"
    )
    if selected_classification.get("file_type") == "final_voltage_lut":
        st.warning("이 파일은 최종 전압 LUT CSV입니다. 실구동 결과 CSV가 아닙니다. 피드백 보정에는 TimeMs / Voltage1_V / HallBz 컬럼이 있는 장비 측정 CSV가 필요합니다.")
    elif auto_meta.get("warning"):
        st.warning(str(auto_meta["warning"]))
    with st.expander("파일/캐시 상세", expanded=False):
        st.caption(f"내부 ID: `{selected.cache_item_id}`")
        st.caption("실구동 결과 CSV: TimeMs / Voltage1_V / HallBz")
        st.caption("최종 전압 LUT CSV: sample_index / time_s / voltage_v")
    return {
        "cache_id": selected_id,
        "filename": selected.original_filename,
        "csv_bytes": source_bytes,
        "run_label": run_label,
        "parse_status": parse_status,
        "alignment_status": "pending_until_run",
        "file_type": selected_classification.get("file_type"),
        "schema_status": selected_classification.get("schema_status"),
        "match_status": match_status,
        "selection_reason": auto_meta.get("selection_reason"),
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
    except ValueError as exc:
        message = str(exc)
        reason = "unsupported_actual_drive_result_file" if "Unsupported finite actual-drive result filename" in message else "invalid_actual_drive_result"
        return command_profile, {
            "feedback_route": "finite_actual_feedback_peak_correction",
            "feedback_source_file": feedback_selection.get("filename"),
            "feedback_run_label": feedback_selection.get("run_label") or "unknown",
            "feedback_correction_available": False,
            "feedback_correction_status": "feedback_source_invalid",
            "feedback_correction_unavailable_reason": reason,
            "feedback_parse_error": message,
            "feedback_used_for_correction": False,
            "target_unchanged": True,
            "correction_delta_v_generated": False,
            "second_voltage_v_generated": False,
            "second_lut_generated": False,
        }
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
    st.markdown("#### 1차 실구동 결과 검토")
    if not feedback_selection or not feedback_selection.get("csv_bytes"):
        st.info("실구동 검토 결과가 아직 없습니다.")
        return None
    status = {
        "uploaded_file_available": True,
        "review_loaded": False,
        "next_action": "실구동 결과 검토 버튼을 누르십시오.",
    }
    if not st.button("실구동 결과 검토", key="load_review_actual_drive_result"):
        cached = st.session_state.get("quick_lut_actual_drive_review_result")
        if isinstance(cached, dict):
            _render_actual_drive_review_payload(command_profile, cached)
            return cached
        st.info("업로드 파일이 선택되어 있습니다. 실구동 결과 검토 버튼을 누르면 plot을 생성합니다.")
        with st.expander("상세 진단", expanded=False):
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
        with st.expander("상세 진단", expanded=False):
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
    st.success("실구동 결과 검토 완료")
    st.caption("Raw peak 값은 참고용입니다. 최종 적합성은 사용자가 그래프를 보고 판단합니다.")
    st.caption("자동 합격/불합격 판정은 하지 않습니다.")
    with st.expander("상세 진단", expanded=False):
        st.dataframe(pd.DataFrame([status]), use_container_width=True)

    frame = payload.get("review_frame")
    if not isinstance(frame, pd.DataFrame) or not bool(payload.get("plot_available", False)):
        st.info("실구동 검토 결과가 아직 없습니다.")
        return

    plot_frame = pd.DataFrame({"time_s": pd.to_numeric(frame["time_s"], errors="coerce")})
    plot_frame[FIELD_TARGET_LABEL] = pd.to_numeric(frame["normalized_physical_target_output_mT"], errors="coerce")
    plot_frame[MEASURED_FIELD_LABEL] = pd.to_numeric(frame["normalized_measured_field_mT"], errors="coerce")
    plot_frame[RESIDUAL_LABEL] = plot_frame[FIELD_TARGET_LABEL] - plot_frame[MEASURED_FIELD_LABEL]
    plot_frame[FIRST_VOLTAGE_LABEL] = _interp_command_column(command_profile, frame["time_s"], "limited_voltage_v")
    plot_frame[ACTUAL_VOLTAGE_LABEL] = pd.to_numeric(
        frame.get("normalized_actual_drive_voltage_v", frame.get("normalized_first_voltage_v")), errors="coerce"
    )
    if "second_limited_voltage_v" in command_profile.columns:
        plot_frame[SECOND_VOLTAGE_LABEL] = _interp_command_column(command_profile, frame["time_s"], "second_limited_voltage_v")

    _render_plot(
        plot_frame,
        [FIELD_TARGET_LABEL, MEASURED_FIELD_LABEL, RESIDUAL_LABEL],
        "목표 자기장 vs 실측 자기장",
        yaxis_title="자기장 / 오차 (mT)",
    )
    _render_plot(
        plot_frame,
        [FIRST_VOLTAGE_LABEL, ACTUAL_VOLTAGE_LABEL, SECOND_VOLTAGE_LABEL],
        "명령 전압 vs 실제 구동 전압",
        yaxis_title="전압 (V)",
    )

    raw_frame = pd.DataFrame({"time_s": pd.to_numeric(frame["time_s"], errors="coerce")})
    raw_hallbz = frame.get("raw_hallbz_mT", frame.get("raw_measured_field_mT"))
    raw_frame["Raw HallBz"] = pd.to_numeric(raw_hallbz, errors="coerce")
    raw_frame["부호 보정 자기장 (-HallBz)"] = pd.to_numeric(frame.get("measured_field_effective_mT"), errors="coerce")
    raw_frame["기준선 제거 후 자기장"] = pd.to_numeric(frame.get("baseline_removed_effective_field_mT", frame.get("measured_field_mT")), errors="coerce")
    raw_frame["정규화 자기장 (±50mT)"] = pd.to_numeric(frame["normalized_measured_field_mT"], errors="coerce")
    raw_frame["Raw Voltage1_V"] = pd.to_numeric(frame["raw_first_voltage_v"], errors="coerce")
    raw_frame["정규화 전압 (±5V)"] = pd.to_numeric(frame["normalized_first_voltage_v"], errors="coerce")
    if "current_a" in frame.columns:
        raw_frame["전류"] = pd.to_numeric(frame["current_a"], errors="coerce")
    with st.expander("Raw 데이터 상세 보기", expanded=False):
        _render_plot(
            raw_frame,
            ["Raw HallBz", "부호 보정 자기장 (-HallBz)", "기준선 제거 후 자기장", "정규화 자기장 (±50mT)", "Raw Voltage1_V", "정규화 전압 (±5V)", "전류"],
            "1차 실구동 데이터 원본 확인",
            yaxis_title="측정값",
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
        ("supported cycles", "1.0, 1.5"),
        ("unsupported cycles", "1.25, 1.75, 2.0"),
        ("unsupported reason", "unsupported_cycle_policy_1p0_1p5_only"),
        ("production cycle policy", "1p0_1p5_cycles"),
        ("filename", metadata.get("feedback_source_file", "unavailable")),
        ("parse status", metadata.get("feedback_schema_status", "unavailable")),
        ("alignment status", metadata.get("feedback_alignment_status") or metadata.get("alignment_status", "unavailable")),
        ("target_unchanged", metadata.get("target_unchanged", True)),
    ]
    return [{"field": field, "value": value} for field, value in fields]


def build_feedback_plot_frame(command_profile: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame({"time_s": pd.to_numeric(command_profile["time_s"], errors="coerce")})
    columns = {
        "physical_target_output_mT": FIELD_TARGET_LABEL,
        "measured_field_normalized_mT": MEASURED_FIELD_LABEL,
        "limited_voltage_v": ACTIVE_COMMAND_LABEL,
        "baseline_limited_voltage_v": BASELINE_VOLTAGE_LABEL,
        "feedback_correction_delta_v": CORRECTION_DELTA_LABEL,
        "feedback_corrected_limited_voltage_v": FEEDBACK_LIMITED_LABEL,
        "feedback_corrected_predicted_field_mT": "피드백 보정 예측 자기장",
    }
    for source, label in columns.items():
        if source in command_profile.columns:
            frame[label] = pd.to_numeric(command_profile[source], errors="coerce")
    if {FIELD_TARGET_LABEL, MEASURED_FIELD_LABEL}.issubset(frame.columns):
        frame[RESIDUAL_LABEL] = frame[FIELD_TARGET_LABEL] - frame[MEASURED_FIELD_LABEL]
    return frame


def render_feedback_correction_review(command_profile: pd.DataFrame, metadata: dict[str, object]) -> None:
    st.markdown("#### Quick LUT 피드백 보정 결과")
    st.caption("사용자가 그래프를 보고 판단하는 검토 화면입니다. 자동 합격/불합격 판정은 하지 않습니다.")
    st.caption("화면에 표시된 전압 명령과 같은 column을 저장합니다.")

    if not bool(metadata.get("feedback_correction_available", False)):
        st.info("피드백 보정 사용 불가: 상태만 표시하고 예측 graph를 임의로 만들지 않습니다.")
        with st.expander("상세 진단", expanded=False):
            st.dataframe(pd.DataFrame(build_command_source_rows(command_profile, metadata)), use_container_width=True)
            st.dataframe(pd.DataFrame(build_feedback_status_rows(metadata)), use_container_width=True)
        return

    plot_frame = build_feedback_plot_frame(command_profile)
    _render_plot(
        plot_frame,
        [FIELD_TARGET_LABEL, MEASURED_FIELD_LABEL, RESIDUAL_LABEL],
        "목표 자기장 vs 실측 자기장",
        yaxis_title="자기장 / 오차 (mT)",
    )
    _render_plot(
        plot_frame,
        [BASELINE_VOLTAGE_LABEL, FEEDBACK_LIMITED_LABEL, ACTIVE_COMMAND_LABEL],
        "명령 전압 vs 실제 구동 전압",
        yaxis_title="전압 (V)",
    )
    _render_plot(
        plot_frame,
        [BASELINE_VOLTAGE_LABEL, FEEDBACK_LIMITED_LABEL, CORRECTION_DELTA_LABEL],
        "1차 전압 vs 2차 보정 전압",
        yaxis_title="전압 (V)",
    )

    if "피드백 보정 예측 자기장" in plot_frame.columns:
        with st.expander("상세 플롯 / Debug", expanded=False):
            _render_plot(
                plot_frame,
                [FIELD_TARGET_LABEL, "피드백 보정 예측 자기장"],
                "피드백 보정 예측 출력",
                yaxis_title="자기장 (mT)",
            )
    else:
        st.info("예측 출력 상태: corrected command 기준 forward prediction이 없어 예측 graph를 표시하지 않습니다.")

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
    norm_rows = [
        {"field": "hallbz_sign_applied", "value": metadata.get("hallbz_sign_applied", "unavailable")},
        {"field": "field_normalization_mode", "value": metadata.get("field_normalization_mode", "unavailable")},
        {"field": "field_normalization_scale_factor", "value": metadata.get("field_normalization_scale_factor", "unavailable")},
        {"field": "voltage_normalization_mode", "value": metadata.get("voltage_normalization_mode", "unavailable")},
        {"field": "voltage_normalization_scale_factor", "value": metadata.get("voltage_normalization_scale_factor", "unavailable")},
        {"field": "raw_field_peak_mT", "value": metadata.get("raw_field_peak_mT", "informational only")},
        {"field": "raw_voltage_peak_v", "value": metadata.get("raw_voltage_peak_v", "informational only")},
    ]
    with st.expander("상세 진단", expanded=False):
        st.caption("HallBz 부호 보정 적용")
        st.caption("실측 자기장 peak를 ±50mT 기준으로 정규화")
        st.caption("전압을 ±5V 기준으로 정규화/제한")
        st.caption("Raw peak 값은 참고용입니다.")
        st.markdown("##### 예측 출력 상태")
        st.caption("기존 predicted를 corrected prediction처럼 표시하지 않습니다.")
        st.dataframe(pd.DataFrame(build_command_source_rows(command_profile, metadata)), use_container_width=True)
        st.dataframe(pd.DataFrame(build_feedback_status_rows(metadata)), use_container_width=True)
        st.dataframe(pd.DataFrame(norm_rows), use_container_width=True)
        st.dataframe(pd.DataFrame([{"metric": key, "value": metadata.get(key, "unavailable")} for key in metrics]), use_container_width=True)


def _render_plot(
    frame: pd.DataFrame,
    columns: list[str],
    title: str,
    *,
    yaxis_title: str = "값",
) -> None:
    figure = go.Figure()
    for column in columns:
        if column not in frame.columns:
            continue
        figure.add_trace(go.Scatter(x=frame["time_s"], y=frame[column], mode="lines", name=column))
    figure.update_layout(template="plotly_white", height=320, title=title, xaxis_title="시간 (s)", yaxis_title=yaxis_title)
    st.plotly_chart(figure, use_container_width=True)
