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
from .ui_quick_lut_feedback_contract import ACTIVE_COMMAND_LABEL
from .ui_quick_lut_feedback_contract import ACTUAL_VOLTAGE_LABEL
from .ui_quick_lut_feedback_contract import BASELINE_VOLTAGE_LABEL
from .ui_quick_lut_feedback_contract import CORRECTION_DELTA_LABEL
from .ui_quick_lut_feedback_contract import FEEDBACK_LIMITED_LABEL
from .ui_quick_lut_feedback_contract import FIELD_TARGET_LABEL
from .ui_quick_lut_feedback_contract import FIRST_VOLTAGE_LABEL
from .ui_quick_lut_feedback_contract import MEASURED_FIELD_LABEL
from .ui_quick_lut_feedback_contract import RESIDUAL_LABEL
from .ui_quick_lut_feedback_contract import SECOND_VOLTAGE_LABEL
from .ui_quick_lut_feedback_contract import build_command_source_rows
from .ui_quick_lut_feedback_contract import build_feedback_plot_frame
from .ui_quick_lut_feedback_contract import build_feedback_status_rows
from .ui_quick_lut_feedback_contract import feedback_export_source_column
from .ui_quick_lut_feedback_selection import candidate_matches
from .ui_quick_lut_feedback_selection import choose_actual_drive_feedback_candidate
from .ui_quick_lut_feedback_selection import classify_feedback_csv_candidate
from .ui_quick_lut_feedback_second_sources import count_exact_matches
from .ui_quick_lut_feedback_second_sources import scan_second_actual_drive_upload_folder
from .ui_raw_waveforms_labels import infer_new_dataset_filename_metadata
from .ui_upload_cache import add_upload_cache_bytes
from .ui_upload_cache import build_upload_cache_records
from .ui_upload_cache import cache_item_bytes
from .voltage_policy import COMMAND_VOLTAGE_LIMIT_LABEL, COMMAND_VOLTAGE_NORMALIZATION_OR_LIMIT_MODE


FEEDBACK_CACHE_STATE_KEY = "actual_drive_validation_cache_items"
FEEDBACK_SELECTED_CACHE_KEY = "quick_lut_feedback_selected_cache_id"
FEEDBACK_RUN_LABEL_KEY = "quick_lut_feedback_run_label"
# UI contract marker: 정규화 전압 (±10V)


def select_actual_drive_feedback_candidate_for_target(
    *,
    waveform_type: str | None = None,
    freq_hz: float | None = None,
    cycle_count: float | None = None,
    run_label: str = "first_run",
) -> tuple[dict[str, object] | None, dict[str, object]]:
    """Auto-select a measured first-drive result without rendering the old upload UI."""
    cache_state = st.session_state.setdefault(FEEDBACK_CACHE_STATE_KEY, {})
    if not isinstance(cache_state, dict):
        cache_state = {}
        st.session_state[FEEDBACK_CACHE_STATE_KEY] = cache_state

    records = [record for record in build_upload_cache_records(cache_state) if record.cache_type == "actual_drive_validation"]
    cache_candidates = [
        {
            "candidate_id": f"cache:{record.cache_item_id}",
            "cache_id": record.cache_item_id,
            "source_kind": "upload_cache",
            "source_label": "업로드 메모리",
            "filename": record.original_filename,
            "original_filename": record.original_filename,
            "csv_bytes": cache_item_bytes(cache_state, record.cache_item_id),
            "run_label": run_label,
        }
        for record in records
    ]
    folder_candidates, folder_meta = scan_second_actual_drive_upload_folder(run_label=run_label)
    candidate_payloads = cache_candidates + folder_candidates
    exact_match_count = count_exact_matches(
        candidate_payloads,
        waveform_type=waveform_type,
        freq_hz=freq_hz,
        cycle_count=cycle_count,
    )
    selected, selection_meta = choose_actual_drive_feedback_candidate(
        candidate_payloads,
        waveform_type=waveform_type,
        freq_hz=freq_hz,
        cycle_count=cycle_count,
    )
    metadata = {
        **selection_meta,
        "folder_path": folder_meta.get("folder_path"),
        "folder_file_count": folder_meta.get("file_count", 0),
        "actual_drive_candidate_count": folder_meta.get("actual_drive_candidate_count", 0),
        "final_voltage_lut_count": folder_meta.get("final_voltage_lut_count", 0),
        "exact_match_count": exact_match_count,
        "candidate_count": len(candidate_payloads),
    }
    if selected is not None:
        selected = {
            **selected,
            "selection_reason": metadata.get("selection_reason"),
            "match_status": "match",
        }
    return selected, metadata


def render_quick_lut_feedback_input_section(
    *,
    finite_cycle_mode: bool,
    waveform_type: str | None = None,
    freq_hz: float | None = None,
    cycle_count: float | None = None,
) -> dict[str, object] | None:
    st.markdown("#### 2차 보정 입력 source")
    st.caption("지정된 2nd 폴더 / upload memory에서 실구동 결과를 자동 로드해 2차 보정 입력으로 사용합니다.")
    st.caption("TimeMs / Voltage1_V / HallBz 컬럼이 있으면 실구동 결과 후보로 사용할 수 있습니다.")
    st.caption(f"Raw/absolute gain 평가는 하지 않고, 사용자 목표 피크 자기장 / {COMMAND_VOLTAGE_LIMIT_LABEL} 정규화 기준으로 개형과 타이밍을 봅니다.")
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

    with st.expander("Legacy / 수동 실구동 결과 CSV 업로드", expanded=False):
        st.warning("현재 기본 workflow에서는 사용하지 않습니다. 2nd 폴더 자동 로드를 우선 사용하십시오.")
        uploaded_files = st.file_uploader(
            "실구동 결과 CSV 업로드",
            type=["csv"],
            accept_multiple_files=True,
            key="quick_lut_feedback_result_upload",
            help="TimeMs / Voltage1_V / HallBz schema의 실구동 결과 CSV를 legacy cache에 추가합니다.",
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
        "2차 보정 run 단계",
        options=["first_run", "second_run", "unknown"],
        index=0,
        key=FEEDBACK_RUN_LABEL_KEY,
        help="2nd 폴더 후보를 1차/2차/unknown run metadata로 구분합니다.",
    )
    records = [record for record in build_upload_cache_records(cache_state) if record.cache_type == "actual_drive_validation"]
    cache_candidates = [
        {
            "candidate_id": f"cache:{record.cache_item_id}",
            "cache_id": record.cache_item_id,
            "source_kind": "upload_cache",
            "source_label": "캐시된 실구동 결과 파일",
            "filename": record.original_filename,
            "original_filename": record.original_filename,
            "csv_bytes": cache_item_bytes(cache_state, record.cache_item_id),
            "run_label": run_label,
        }
        for record in records
    ]
    second_folder_candidates, second_folder_meta = scan_second_actual_drive_upload_folder(run_label=run_label)
    candidate_payloads = cache_candidates + second_folder_candidates
    exact_match_count = count_exact_matches(
        candidate_payloads,
        waveform_type=waveform_type,
        freq_hz=freq_hz,
        cycle_count=cycle_count,
    )
    st.caption("2차 모델링용 실구동 결과 폴더")
    st.caption(f"폴더 경로: `{second_folder_meta['folder_path']}`")
    st.caption(
        f"파일 {second_folder_meta['file_count']}개 · actual-drive 후보 {second_folder_meta['actual_drive_candidate_count']}개 · "
        f"최종 전압 LUT 제외 {second_folder_meta['final_voltage_lut_count']}개 · target exact match {exact_match_count}개"
    )
    if not candidate_payloads:
        st.info("실구동 결과 파일이 없습니다. TimeMs / Voltage1_V / HallBz 컬럼이 있는 실구동 결과 CSV를 uploads/2nd에 넣거나 업로드하십시오.")
        return None

    candidates_by_id = {str(candidate["candidate_id"]): candidate for candidate in candidate_payloads}
    options = list(candidates_by_id)
    labels_by_id = {
        candidate_id: f"{candidate.get('filename')} - {candidate.get('source_label')}"
        for candidate_id, candidate in candidates_by_id.items()
    }
    auto_selected, auto_meta = choose_actual_drive_feedback_candidate(
        candidate_payloads,
        waveform_type=waveform_type,
        freq_hz=freq_hz,
        cycle_count=cycle_count,
    )
    selection_widget_rendered = False
    if auto_selected is not None:
        selected_id = str(auto_selected.get("candidate_id"))
    else:
        previous_id = st.session_state.get(FEEDBACK_SELECTED_CACHE_KEY)
        previous_candidate = candidates_by_id.get(str(previous_id)) if previous_id else None
        previous_info = (
            classify_feedback_csv_candidate(str(previous_candidate.get("filename")), previous_candidate.get("csv_bytes") if isinstance(previous_candidate.get("csv_bytes"), bytes) else None)
            if previous_candidate is not None
            else {}
        )
        selected_id = (
            str(previous_id)
            if previous_candidate is not None
            and candidate_matches(previous_info, waveform_type=waveform_type, freq_hz=freq_hz, cycle_count=cycle_count)
            else None
        )
    if selected_id is None:
        st.info("현재 target과 정확히 일치하는 실구동 결과 CSV가 없습니다. 파일을 수동으로 선택하면 Raw preview는 가능하지만 production 2차 모델링은 비활성화됩니다.")
        if auto_meta.get("warning"):
            st.warning(str(auto_meta["warning"]))
        selected_id = st.selectbox(
            "1차 실구동 결과 데이터",
            options=options,
            format_func=lambda cache_id: labels_by_id[cache_id],
            index=None,
            key=FEEDBACK_SELECTED_CACHE_KEY,
        )
        selection_widget_rendered = True
        if selected_id is None:
            return None
    st.session_state[FEEDBACK_SELECTED_CACHE_KEY] = selected_id
    if not selection_widget_rendered:
        selected_id = st.selectbox(
            "1차 실구동 결과 데이터",
            options=options,
            format_func=lambda cache_id: labels_by_id[cache_id],
            key=FEEDBACK_SELECTED_CACHE_KEY,
        )
    selected = candidates_by_id[selected_id]
    source_bytes = selected.get("csv_bytes") if isinstance(selected.get("csv_bytes"), bytes) else None
    parse_status = "available" if source_bytes else "missing_bytes"
    selected_filename = str(selected.get("filename") or "")
    parsed = infer_new_dataset_filename_metadata(selected_filename)
    selected_payload = {
        "cache_id": selected.get("cache_id"),
        "candidate_id": selected_id,
        "source_kind": selected.get("source_kind"),
        "source_path": selected.get("source_path"),
        "filename": selected_filename,
        "csv_bytes": source_bytes,
        "run_label": run_label,
    }
    selected_classification = classify_feedback_csv_candidate(selected_filename, source_bytes)
    match_status = "match" if candidate_matches(selected_classification, waveform_type=waveform_type, freq_hz=freq_hz, cycle_count=cycle_count) else "mismatch_or_unknown"
    metadata_source = str(selected_classification.get("metadata_source") or "")
    can_use_current_metadata = (
        selected_classification.get("file_type") == "actual_drive_result"
        and metadata_source in {"unavailable", ""}
        and waveform_type is not None
        and freq_hz is not None
        and cycle_count is not None
    )
    use_current_metadata = False
    if can_use_current_metadata:
        use_current_metadata = st.checkbox(
            "현재 Quick LUT 설정의 실구동 결과로 사용",
            value=False,
            key="quick_lut_feedback_use_current_metadata",
            help="파일명/프리앰블에서 주파수와 cycle을 읽지 못한 경우에만 사용합니다.",
        )
        if use_current_metadata:
            st.warning("파일명에서 주파수/cycle을 읽지 못해 현재 Quick LUT 설정으로 귀속했습니다. 현재 설정과 다른 실구동 결과 파일이면 2차 보정이 잘못될 수 있습니다.")
            selected_classification = {
                **selected_classification,
                "metadata_source": "current_quick_lut_selection",
                "waveform_type": waveform_type,
                "freq_hz": float(freq_hz),
                "cycle_count": float(cycle_count),
            }
            match_status = "match"

    st.caption(f"파일: `{selected_filename}`")
    st.caption(f"출처: `{selected.get('source_label')}`")
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
        st.caption(f"내부 ID: `{selected_id}`")
        if selected.get("source_path"):
            st.caption(f"경로: `{selected.get('source_path')}`")
        st.caption("실구동 결과 CSV: TimeMs / Voltage1_V / HallBz")
        st.caption("최종 전압 LUT CSV: sample_index / time_s / voltage_v")
    return {
        "cache_id": selected.get("cache_id"),
        "candidate_id": selected_id,
        "source_kind": selected.get("source_kind"),
        "source_path": selected.get("source_path"),
        "filename": selected_filename,
        "csv_bytes": source_bytes,
        "run_label": run_label,
        "parse_status": parse_status,
        "alignment_status": "pending_until_run",
        "file_type": selected_classification.get("file_type"),
        "schema_status": selected_classification.get("schema_status"),
        "metadata_source": selected_classification.get("metadata_source"),
        "use_current_quick_lut_metadata": use_current_metadata,
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
    source_classification = classify_feedback_csv_candidate(
        str(feedback_selection.get("filename") or "feedback.csv"),
        feedback_selection.get("csv_bytes") if isinstance(feedback_selection.get("csv_bytes"), bytes) else None,
    )
    if source_classification.get("file_type") == "final_voltage_lut":
        return command_profile, {
            "feedback_route": "finite_actual_feedback_peak_correction",
            "feedback_source_file": feedback_selection.get("filename"),
            "feedback_run_label": feedback_selection.get("run_label") or "unknown",
            "feedback_correction_available": False,
            "feedback_correction_status": "feedback_source_invalid",
            "feedback_correction_unavailable_reason": "unsupported_actual_drive_result_file",
            "feedback_used_for_correction": False,
            "target_unchanged": True,
            "next_action": "TimeMs / Voltage1_V / HallBz 컬럼이 있는 측정 데이터를 업로드하십시오.",
        }
    suffix = "_" + Path(str(feedback_selection.get("filename") or "feedback.csv")).name
    with NamedTemporaryFile(prefix="quick_lut_feedback_", suffix=suffix, delete=False) as handle:
        temp_path = Path(handle.name)
        handle.write(bytes(feedback_selection["csv_bytes"]))
    try:
        feedback_source: Any = temp_path
        if bool(feedback_selection.get("use_current_quick_lut_metadata")):
            feedback_source = {
                "record": read_actual_drive_result(
                    temp_path,
                    waveform_type=str(waveform_type),
                    freq_hz=float(freq_hz),
                    cycle_count=float(cycle_count),
                )
            }
        corrected, metadata = apply_finite_feedback_peak_correction(
            command_profile,
            feedback_source,
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
            "field_normalization_mode": "peak_to_target_peak_mT",
            "voltage_normalization_mode": COMMAND_VOLTAGE_NORMALIZATION_OR_LIMIT_MODE,
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
        frame.get("actual_drive_voltage_v", frame.get("raw_actual_drive_voltage_v", frame.get("raw_first_voltage_v"))),
        errors="coerce",
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
    raw_frame["정규화 자기장 (target peak mT)"] = pd.to_numeric(frame["normalized_measured_field_mT"], errors="coerce")
    raw_frame["Raw Voltage1_V"] = pd.to_numeric(frame["raw_first_voltage_v"], errors="coerce")
    normalized_voltage_label = f"정규화 전압 ({COMMAND_VOLTAGE_LIMIT_LABEL})"
    raw_frame[normalized_voltage_label] = pd.to_numeric(frame["normalized_first_voltage_v"], errors="coerce")
    if "current_a" in frame.columns:
        raw_frame["전류"] = pd.to_numeric(frame["current_a"], errors="coerce")
    with st.expander("Raw 데이터 상세 보기", expanded=False):
        _render_plot(
            raw_frame,
            ["Raw HallBz", "부호 보정 자기장 (-HallBz)", "기준선 제거 후 자기장", "정규화 자기장 (target peak mT)", "Raw Voltage1_V", normalized_voltage_label, "전류"],
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


def render_feedback_correction_review(command_profile: pd.DataFrame, metadata: dict[str, object]) -> None:
    if not bool(metadata.get("feedback_correction_available", False)):
        return
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
        st.caption("실측 자기장 peak를 사용자 목표 피크 자기장 기준으로 정규화")
        st.caption(f"전압을 {COMMAND_VOLTAGE_LIMIT_LABEL} 기준으로 정규화/제한")
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
