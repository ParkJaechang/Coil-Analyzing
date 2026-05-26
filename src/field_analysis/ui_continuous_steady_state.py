from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .continuous_candidate_discovery import (
    discover_continuous_candidate_frames,
    is_continuous_candidate as _is_continuous_candidate,
    load_dataset_library_continuous_payloads as _load_dataset_library_continuous_payloads,
    load_upload_memory_continuous_payloads as _load_upload_memory_continuous_payloads,
)
from .continuous_steady_state_extraction import (
    build_continuous_actual_drive_review_case,
    build_continuous_phase_aligned_command_profile,
    build_continuous_steady_state_modeling_case,
    evaluate_continuous_steady_state_validation,
)
from .continuous_candidate_frequency import (
    attach_continuous_frequency_attrs,
    continuous_candidate_label,
    infer_continuous_source_frequency,
    matching_candidate_names,
    rank_continuous_candidates_for_target,
)
from .continuous_steady_state_runtime import (
    run_continuous_first_modeling,
    run_continuous_steady_state_extraction,
)
from .ui_continuous_final_lut_export import (
    normalize_continuous_result_contract,
    render_continuous_final_voltage_lut_export_section,
)
from .ui_continuous_first_modeling import render_continuous_first_modeling_controls
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

    target_freq = float(freq_hz or 1.0)
    st.caption("목표 자기장 개형: finite와 동일한 fixed rounded-triangle")
    st.caption("source waveform family는 모델링 입력 데이터 선택용입니다.")
    source_waveform_filter = st.selectbox(
        "Continuous source waveform family",
        ["triangle", "sine", "rounded_triangle", "all"],
        index=0,
        key="continuous_source_waveform_filter",
    )
    candidate_names, candidates, scan = discover_continuous_candidate_frames(
        analysis_lookup,
        target_freq_hz=target_freq,
        source_waveform_filter=source_waveform_filter,
    )
    details = list(scan.get("continuous_candidate_details") or [])
    details_by_name = {str(detail.get("name")): detail for detail in details}
    match_count = int(scan.get("continuous_candidate_matching_count") or 0)
    rejected_count = int(scan.get("continuous_candidate_rejected_count") or 0)
    st.caption(
        f"Continuous 후보: 전체 {len(candidate_names)}개 / target match {match_count}개 / schema rejected {rejected_count}개"
    )
    with st.expander("Continuous 후보 상세 / Debug", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    {"source": key, "candidate_count": value}
                    for key, value in dict(scan.get("continuous_candidate_source_counts") or {}).items()
                ]
                + [{"source": "schema_rejected", "candidate_count": rejected_count}]
            ),
            use_container_width=True,
            hide_index=True,
        )
        if details:
            st.dataframe(pd.DataFrame(details), use_container_width=True, hide_index=True)
    selected_name = None
    if candidate_names:
        previous_target = st.session_state.get("continuous_steady_state_target_freq_hz")
        exact_matches = matching_candidate_names(details)
        current_selection = st.session_state.get("continuous_steady_source_dataset")
        if previous_target is not None and abs(float(previous_target) - target_freq) > 1e-9:
            if exact_matches and current_selection not in exact_matches:
                st.session_state.pop("continuous_steady_source_dataset", None)
        st.session_state["continuous_steady_state_target_freq_hz"] = target_freq
        selected_name = st.selectbox(
            "Continuous source dataset",
            candidate_names,
            key="continuous_steady_source_dataset",
            format_func=lambda name: continuous_candidate_label(details_by_name.get(name, {"name": name})),
        )
        selected_detail = details_by_name.get(selected_name, {})
        if selected_detail.get("frequency_match_status") == "unknown":
            st.checkbox(
                "현재 Quick LUT 주파수로 이 continuous 데이터를 사용",
                key="continuous_unknown_frequency_attribution_enabled",
            )
        scan["continuous_candidate_selected_source"] = selected_name.split(":", 1)[0]
        scan["continuous_candidate_selected_file"] = selected_name.split(":", 1)[1] if ":" in selected_name else selected_name
        scan["continuous_candidate_schema_status"] = selected_detail.get("schema_status")
        signature = {
            "selected_name": selected_name,
            "waveform_type": str(waveform_type or "sine"),
            "freq_hz": target_freq,
            "source_waveform_filter": source_waveform_filter,
        }
        if st.session_state.get("continuous_steady_state_run_signature") not in (None, signature):
            st.session_state["continuous_steady_state_dirty"] = True
        st.session_state["continuous_steady_state_selected_candidate"] = selected_name
        st.session_state["continuous_steady_state_selected_source_freq_hz"] = selected_detail.get("source_freq_hz")
        st.session_state["continuous_steady_state_frequency_match_status"] = selected_detail.get("frequency_match_status")
        st.session_state["continuous_steady_state_selection_signature"] = signature
        st.session_state["continuous_steady_state_selected_frame"] = candidates[selected_name].copy(deep=True)
        st.session_state["continuous_steady_state_standardized_frame"] = candidates[selected_name].copy(deep=True)
        st.session_state["continuous_steady_state_pending_signature"] = signature
        st.session_state["continuous_steady_state_candidate_scan"] = scan
    else:
        if rejected_count:
            st.warning("Continuous 파일은 찾았지만 schema 인식에 실패했습니다.")
            st.caption("Continuous source 파일은 발견되었지만 time/voltage/field 컬럼 매핑에 실패했습니다.")
            st.caption("아래 schema rejected table에서 컬럼명과 reject reason을 확인하십시오.")
        else:
            st.info("Continuous 파일을 찾지 못했습니다.")
        st.session_state["continuous_steady_state_candidate_scan"] = scan
    rejected_reasons = list(scan.get("continuous_candidate_rejection_reasons") or [])
    if rejected_reasons:
        with st.expander("Continuous schema rejected 상세 / Debug", expanded=False):
            st.dataframe(pd.DataFrame({"reason": rejected_reasons}), use_container_width=True, hide_index=True)

    if st.button("Steady-state 1cycle 추출", key="continuous_steady_state_extract_button"):
        if selected_name is None:
            st.session_state["continuous_steady_state_metadata"] = {"steady_state_extraction_status": "unavailable_no_source"}
            st.warning("Continuous source dataset이 없어 Steady-state 1cycle 추출을 실행할 수 없습니다.")
        else:
            selected_detail = details_by_name.get(selected_name, {})
            selected_frame = candidates[selected_name]
            if selected_detail.get("frequency_match_status") == "unknown" and st.session_state.get(
                "continuous_unknown_frequency_attribution_enabled"
            ):
                selected_frame = attach_continuous_frequency_attrs(
                    selected_frame,
                    name=selected_name,
                    user_fallback_freq_hz=target_freq,
                )
            bundle = run_continuous_steady_state_extraction(
                selected_candidate_name=selected_name,
                selected_frame=selected_frame,
                waveform_type=waveform_type,
                freq_hz=freq_hz,
                modeling_case_builder=modeling_case_builder,
            )
            case = bundle.get("extraction_result")
            if bundle.get("status") != "ok" or not isinstance(case, dict):
                metadata = dict(case.get("metadata") or {}) if isinstance(case, dict) else {"steady_state_extraction_status": "error"}
                metadata["error_reason"] = bundle.get("error_reason")
                metadata["matching_candidate_count"] = int(scan.get("matching_candidate_count") or 0)
                metadata["matching_candidate_names"] = list(scan.get("matching_candidate_names") or [])
                st.session_state["continuous_steady_state_metadata"] = metadata
                st.session_state["continuous_steady_state_last_error_metadata"] = metadata
                st.session_state["continuous_steady_state_extraction_status"] = metadata.get("steady_state_extraction_status")
                st.session_state["continuous_steady_state_extraction_blocked_reason"] = metadata.get("extraction_blocked_reason")
                if "continuous_steady_state_extraction_result" in st.session_state:
                    st.session_state["continuous_steady_state_dirty"] = True
                if metadata.get("extraction_blocked_reason") == "frequency_mismatch":
                    _render_frequency_mismatch_detail(metadata, list(scan.get("matching_candidate_names") or []))
                st.warning(f"Steady-state 1cycle 추출 결과가 유효하지 않습니다: {bundle.get('error_reason')}")
            else:
                st.session_state["continuous_steady_state_extraction_result"] = case
                st.session_state["continuous_steady_state_window_frame"] = case["steady_state_one_cycle_frame"]
                st.session_state["continuous_steady_state_support_frame"] = case.get("steady_state_support_frame")
                st.session_state["continuous_steady_state_metadata"] = case["metadata"]
                st.session_state["continuous_steady_state_extraction_status"] = "ok"
                st.session_state["continuous_steady_state_dirty"] = False
                st.session_state["continuous_steady_state_run_signature"] = st.session_state.get("continuous_steady_state_pending_signature")
                st.success("Steady-state 1cycle 추출 완료")

    case = st.session_state.get("continuous_steady_state_extraction_result")
    if st.session_state.get("continuous_steady_state_dirty") and isinstance(case, dict):
        st.warning("설정 또는 데이터가 변경되었습니다. Steady-state 1cycle 추출을 다시 실행하십시오.")
        st.caption("현재 표시 중인 결과는 이전 실행 결과입니다.")
    if isinstance(case, dict):
        window = case.get("steady_state_one_cycle_frame")
        metadata = dict(case.get("metadata") or {})
        if metadata.get("steady_state_extraction_status") != "ok":
            st.warning(f"Steady-state 1cycle 추출 결과가 유효하지 않습니다: {metadata.get('steady_state_extraction_status')}")
        elif isinstance(window, pd.DataFrame) and not window.empty:
            _render_continuous_extraction_result(case, st.session_state.get("continuous_steady_state_standardized_frame"))
            render_continuous_first_modeling_controls(waveform_type=waveform_type, freq_hz=freq_hz)
        else:
            st.warning("Steady-state 1cycle 추출 결과가 비어 있습니다.")
        _render_continuous_runtime_debug(case)
        return case

    _render_continuous_runtime_debug(None)
    st.caption("옵션 변경만으로 heavy calculation을 자동 실행하지 않습니다. `Steady-state 1cycle 추출` 버튼을 누르십시오.")
    return None


def render_continuous_actual_drive_runtime_panel(
    *,
    waveform_type: str | None,
    freq_hz: float | None,
) -> None:
    with st.expander("Legacy / 수동 continuous 실구동 결과 CSV 업로드", expanded=False):
        st.caption("기본 workflow는 지정된 2nd 폴더 / upload memory source scan을 사용합니다. 이 업로드는 호환성 검토용입니다.")
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
            second_profile, second_meta = build_continuous_second_command_profile(
                command_profile,
                steady,
                freq_hz=float(freq_hz or 1.0),
                waveform_type=str(waveform_type or "sine"),
            )
            metadata.update(second_meta)
            st.session_state["quick_lut_second_model_result_continuous"] = normalize_continuous_result_contract(
                {
                    "command_profile": second_profile.copy(deep=True),
                    "actual_drive_steady_window_frame": steady.copy(deep=True),
                    "metadata": metadata,
                },
                "second",
            )
            st.success("Continuous 2차 보정 command 생성 결과를 session_state에 저장했습니다.")

    _render_continuous_validation_section(waveform_type=waveform_type, freq_hz=freq_hz)
    render_continuous_final_voltage_lut_export_section(
        waveform_type=waveform_type,
        freq_hz=freq_hz,
        key_namespace="continuous_actual_drive",
    )


def _render_continuous_validation_section(*, waveform_type: str | None, freq_hz: float | None) -> None:
    with st.expander("Legacy / 수동 continuous 2차 구동 결과 평가", expanded=False):
        st.markdown("#### Continuous 2차 구동 결과 평가")
        st.caption("기본 workflow에서는 2nd 폴더 / upload memory source scan을 사용합니다.")
        st.caption("평가는 안정화된 1cycle 기준입니다.")
        st.caption("초반 transient cycle은 평가에서 제외되었습니다.")
        uploaded = st.file_uploader(
            "Continuous 2차 구동 결과 CSV",
            type=["csv"],
            key="continuous_second_drive_validation_upload",
        )
        run_validation = st.button("Continuous 2차 구동 결과 평가 실행", key="continuous_second_validation_button")
    if not run_validation:
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
    names, candidates, _scan = discover_continuous_candidate_frames(analysis_lookup)
    return sorted(candidates.keys()), candidates


def build_continuous_second_command_profile(
    first_command_profile: pd.DataFrame,
    steady_actual_window_frame: pd.DataFrame,
    *,
    freq_hz: float,
    waveform_type: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not isinstance(first_command_profile, pd.DataFrame) or first_command_profile.empty:
        raise ValueError("continuous first command profile is required")
    actual = steady_actual_window_frame.copy()
    time_s = pd.to_numeric(first_command_profile["time_s"], errors="coerce")
    actual = actual.copy()
    if "time_s" not in actual.columns:
        raise ValueError("steady actual window requires time_s")
    for column in ("normalized_physical_target_output_mT", "measured_field_normalized_mT"):
        if column not in actual.columns:
            raise ValueError(f"steady actual window requires {column}")
    limited_column = "limited_voltage_v" if "limited_voltage_v" in first_command_profile.columns else "first_modeled_voltage_v"
    actual["limited_voltage_v"] = pd.to_numeric(first_command_profile[limited_column], errors="coerce").to_numpy(dtype=float)
    actual["voltage_normalized_v"] = actual["limited_voltage_v"]
    actual["time_s"] = time_s.to_numpy(dtype=float)
    command, metadata = build_continuous_phase_aligned_command_profile(
        actual,
        support_frame=actual,
        freq_hz=float(freq_hz),
        waveform_type=waveform_type,
    )
    if not isinstance(command, pd.DataFrame) or command.empty or "limited_voltage_v" not in command.columns:
        raise ValueError(str(metadata.get("continuous_first_modeling_status") or "continuous_second_command_generation_failed"))
    second = first_command_profile.copy(deep=True).reset_index(drop=True)
    for column in command.columns:
        second[column] = command[column].to_numpy() if len(command[column]) == len(second) else command[column]
    second["second_limited_voltage_v"] = command["limited_voltage_v"].to_numpy(dtype=float)
    metadata.update(
        {
            "continuous_second_modeling_uses_phase_aligned_kernel": True,
            "continuous_second_modeling_status": "ok",
            "continuous_second_modeling_input_window": "steady_state_one_cycle_only",
            "continuous_second_modeling_tail_disabled": True,
            "continuous_second_export_cycle_count": 1.0,
            "continuous_loop_output": True,
            "loop_endpoint_policy": "period_exclusive",
            "continuous_export_cycle_count": 1.0,
            "continuous_result_stage": "second_model",
            "final_export_voltage_source_column": "second_limited_voltage_v",
            "second_modeling_input_mode": "continuous_steady_state",
            "second_drive_actual_data_used": "steady_state_one_cycle_only",
        }
    )
    return second, metadata


def _render_continuous_extraction_result(case: dict[str, Any], source_frame: Any) -> None:
    window = case.get("steady_state_one_cycle_frame")
    metadata = dict(case.get("metadata") or {})
    if not isinstance(window, pd.DataFrame) or window.empty:
        st.warning("Steady-state 1cycle 추출 결과가 비어 있습니다.")
        return
    summary_keys = [
        "continuous_source_file",
        "continuous_source_freq_hz",
        "quick_lut_target_freq_hz",
        "expected_period_s",
        "selected_cycle_index",
        "selected_cycle_start_s",
        "selected_cycle_end_s",
        "selected_cycle_duration_s",
        "selected_cycle_duration_ratio",
        "discarded_startup_cycles",
        "command_stop_s",
        "field_support_end_s",
        "continuous_phase_delay_s",
        "exclude_terminal_cycles",
        "terminal_guard_cycle_count",
        "selected_cycle_is_terminal",
        "selected_cycle_stop_influence_status",
        "selected_cycle_phase_support_clear_of_stop",
        "cycle_boundary_method",
        "steady_state_extraction_status",
    ]
    st.markdown("#### Continuous extraction summary")
    st.dataframe(pd.DataFrame([{key: metadata.get(key) for key in summary_keys}]), use_container_width=True, hide_index=True)
    st.markdown("#### 선택된 steady-state 1cycle")
    st.caption(f"startup transient 제외 cycle: {metadata.get('discarded_startup_cycles')}")
    st.caption("마지막 cycle은 출력 종료 영향 가능성이 있어 기본적으로 제외합니다.")
    st.caption(
        "expected period_s: "
        f"{metadata.get('expected_period_s')} / selected duration_s: "
        f"{metadata.get('selected_cycle_duration_s')} / duration ratio: "
        f"{metadata.get('selected_cycle_duration_ratio')} / boundary method: "
        f"{metadata.get('cycle_boundary_method')}"
    )
    st.plotly_chart(_plot_continuous_window(window), use_container_width=True)
    if isinstance(source_frame, pd.DataFrame) and not source_frame.empty:
        st.markdown("#### Continuous 원본과 선택된 steady-state 구간")
        st.plotly_chart(_plot_continuous_source_preview(source_frame, metadata), use_container_width=True)
    st.markdown("#### cycle stability metrics")
    metrics = case.get("stability_metrics")
    if isinstance(metrics, pd.DataFrame) and not metrics.empty:
        display = metrics.copy(deep=True)
        if "cycle_index" in display.columns:
            display["selected"] = pd.to_numeric(display["cycle_index"], errors="coerce") == int(metadata.get("selected_cycle_index", -1))
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.warning("cycle stability metrics가 비어 있습니다.")


def _frequency_block_metadata(detail: dict[str, Any], *, reason: str = "frequency_mismatch") -> dict[str, Any]:
    return {
        "steady_state_extraction_status": "unavailable_frequency_mismatch",
        "extraction_blocked_reason": reason,
        "continuous_source_file": detail.get("filename"),
        "source_freq_hz": detail.get("source_freq_hz"),
        "target_freq_hz": detail.get("target_freq_hz"),
        "frequency_error_pct": detail.get("frequency_error_pct"),
        "continuous_source_freq_source": detail.get("continuous_source_freq_source"),
        "frequency_match_status": detail.get("frequency_match_status"),
        "frequency_mismatch_blocked": reason == "frequency_mismatch",
    }


def _render_frequency_mismatch_detail(metadata: dict[str, Any], matching_candidates: list[str]) -> None:
    st.warning("선택된 continuous source의 주파수와 현재 Quick LUT target 주파수가 다릅니다.")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "file": metadata.get("continuous_source_file"),
                    "source_freq_hz": metadata.get("source_freq_hz"),
                    "target_freq_hz": metadata.get("target_freq_hz"),
                    "error_pct": metadata.get("frequency_error_pct"),
                    "frequency_source": metadata.get("continuous_source_freq_source"),
                    "blocked_reason": metadata.get("extraction_blocked_reason"),
                }
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        f"source: {metadata.get('source_freq_hz')} Hz / target: {metadata.get('target_freq_hz')} Hz / "
        f"error: {metadata.get('frequency_error_pct')}%"
    )
    if matching_candidates:
        st.caption("현재 target과 일치하는 후보: " + ", ".join(matching_candidates))
    else:
        st.caption(f"현재 target {metadata.get('target_freq_hz')} Hz에 해당하는 continuous source가 없습니다.")


def _render_continuous_runtime_debug(case: dict[str, Any] | None) -> None:
    with st.expander("Continuous runtime debug", expanded=False):
        window = case.get("steady_state_one_cycle_frame") if isinstance(case, dict) else None
        metadata = dict(case.get("metadata") or {}) if isinstance(case, dict) else {}
        first = st.session_state.get("quick_lut_first_model_result_continuous")
        command = first.get("command_profile") if isinstance(first, dict) else None
        scan = st.session_state.get("continuous_steady_state_candidate_scan") or {}
        selected_candidate = st.session_state.get("continuous_steady_state_selected_candidate")
        selected_detail = next(
            (detail for detail in list(scan.get("continuous_candidate_details") or []) if detail.get("name") == selected_candidate),
            {},
        )
        debug = {
            "selected candidate": selected_candidate,
            "target freq_hz": st.session_state.get("continuous_steady_state_target_freq_hz"),
            "selected source freq_hz": st.session_state.get("continuous_steady_state_selected_source_freq_hz"),
            "freq source": selected_detail.get("continuous_source_freq_source"),
            "frequency match status": st.session_state.get("continuous_steady_state_frequency_match_status"),
            "frequency error %": metadata.get("frequency_error_pct"),
            "matching candidate count": scan.get("continuous_candidate_matching_count"),
            "matching candidate names": scan.get("matching_candidate_names"),
            "blocked reason": st.session_state.get("continuous_steady_state_extraction_blocked_reason"),
            "schema status": (st.session_state.get("continuous_steady_state_candidate_scan") or {}).get("continuous_candidate_schema_status"),
            "extraction status": metadata.get("steady_state_extraction_status") or st.session_state.get("continuous_steady_state_extraction_status"),
            "window rows": len(window) if isinstance(window, pd.DataFrame) else 0,
            "window columns": list(window.columns) if isinstance(window, pd.DataFrame) else [],
            "selected duration": metadata.get("selected_cycle_duration_s"),
            "expected period": metadata.get("expected_period_s"),
            "duration ratio": metadata.get("selected_cycle_duration_ratio"),
            "first model status": (first.get("metadata") or {}).get("continuous_first_modeling_status") if isinstance(first, dict) else None,
            "command profile rows": len(command) if isinstance(command, pd.DataFrame) else 0,
            "session_state keys present": [
                key
                for key in (
                    "continuous_steady_state_extraction_result",
                    "continuous_steady_state_window_frame",
                    "continuous_steady_state_metadata",
                    "quick_lut_first_model_result_continuous",
                )
                if key in st.session_state
            ],
        }
        st.dataframe(pd.DataFrame([debug]), use_container_width=True, hide_index=True)


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


def _plot_continuous_source_preview(frame: pd.DataFrame, metadata: dict[str, Any]) -> go.Figure:
    figure = go.Figure()
    x = frame["time_s_abs"] if "time_s_abs" in frame.columns else frame.get("time_s")
    for column, label in [
        ("measured_field_normalized_mT", "전체 normalized measured field"),
        ("voltage_normalized_v", "voltage"),
    ]:
        if column in frame.columns:
            figure.add_trace(go.Scatter(x=x, y=frame[column], mode="lines", name=label))
    start_s = metadata.get("selected_cycle_start_s")
    end_s = metadata.get("selected_cycle_end_s")
    for value, label in [(start_s, "selected start"), (end_s, "selected end")]:
        try:
            x_value = float(value)
        except (TypeError, ValueError):
            continue
        figure.add_vline(x=x_value, line_dash="dash", annotation_text=label)
    for boundary in list(metadata.get("cycle_start_times_s") or [])[:80]:
        try:
            figure.add_vline(x=float(boundary), line_width=1, line_color="rgba(0,0,0,0.12)")
        except (TypeError, ValueError):
            continue
    figure.update_layout(
        template="plotly_white",
        height=320,
        title="Continuous 원본과 선택된 steady-state 구간",
        xaxis_title="absolute time_s (s)",
        yaxis_title="value",
    )
    return figure
