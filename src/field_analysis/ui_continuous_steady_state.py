from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable
from io import StringIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .continuous_steady_state_extraction import (
    adapt_continuous_source_frame,
    build_continuous_actual_drive_review_case,
    build_continuous_phase_aligned_command_profile,
    build_continuous_steady_state_modeling_case,
    evaluate_continuous_steady_state_validation,
)
from .dataset_library import list_manifest_entries, load_dataset_library_settings, read_dataset_entry_bytes
from .finite_actual_drive import build_actual_drive_review_case, read_actual_drive_result
from .ui_upload_state import category_payloads


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

    candidate_names, candidates, scan = discover_continuous_candidate_frames(analysis_lookup)
    st.markdown("##### Continuous 후보 scan 결과")
    st.dataframe(
        pd.DataFrame(
            [
                {"source": key, "candidate_count": value}
                for key, value in dict(scan.get("continuous_candidate_source_counts") or {}).items()
            ]
            + [{"source": "schema_rejected", "candidate_count": int(scan.get("continuous_candidate_rejected_count") or 0)}]
        ),
        use_container_width=True,
        hide_index=True,
    )
    selected_name = None
    if candidate_names:
        selected_name = st.selectbox("Continuous source dataset", candidate_names, key="continuous_steady_source_dataset")
        scan["continuous_candidate_selected_source"] = selected_name.split(":", 1)[0]
        scan["continuous_candidate_selected_file"] = selected_name.split(":", 1)[1] if ":" in selected_name else selected_name
        st.session_state["continuous_steady_state_selected_candidate"] = selected_name
        st.session_state["continuous_steady_state_candidate_scan"] = scan
    else:
        rejected_count = int(scan.get("continuous_candidate_rejected_count") or 0)
        if rejected_count:
            st.warning("Continuous 파일은 찾았지만 schema 인식에 실패했습니다.")
            st.caption("Continuous source 파일은 발견되었지만 time/voltage/field 컬럼 매핑에 실패했습니다.")
            st.caption("아래 schema rejected table에서 컬럼명과 reject reason을 확인하십시오.")
        else:
            st.info("Continuous 파일을 찾지 못했습니다.")
        st.session_state["continuous_steady_state_candidate_scan"] = scan
    rejected_reasons = list(scan.get("continuous_candidate_rejection_reasons") or [])
    if rejected_reasons:
        st.markdown("##### schema rejected")
        st.dataframe(pd.DataFrame({"reason": rejected_reasons}), use_container_width=True, hide_index=True)

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
                st.session_state["continuous_steady_state_dirty"] = False
                st.success("Steady-state 1cycle 추출 완료")

    case = st.session_state.get("continuous_steady_state_extraction_result")
    if isinstance(case, dict):
        window = case.get("steady_state_one_cycle_frame")
        metadata = dict(case.get("metadata") or {})
        if isinstance(window, pd.DataFrame) and not window.empty:
            st.markdown("#### 선택된 steady-state 1cycle")
            st.caption(f"startup transient 제외 cycle: {metadata.get('discarded_startup_cycles')}")
            st.markdown("#### cycle stability metrics")
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
            second_profile, second_meta = build_continuous_second_command_profile(
                command_profile,
                steady,
                freq_hz=float(freq_hz or 1.0),
                waveform_type=str(waveform_type or "sine"),
            )
            metadata.update(second_meta)
            st.session_state["quick_lut_second_model_result_continuous"] = {
                "command_profile": second_profile.copy(deep=True),
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


def discover_continuous_candidate_frames(
    analysis_lookup: dict,
    *,
    upload_payloads: list[tuple[str, bytes]] | None = None,
    dataset_library_payloads: list[tuple[str, bytes]] | None = None,
) -> tuple[list[str], dict[str, pd.DataFrame], dict[str, Any]]:
    candidates: dict[str, pd.DataFrame] = {}
    rejected: list[str] = []
    counts = {"analysis_lookup": 0, "upload_memory_continuous": 0, "dataset_library": 0}
    for key, analysis in (analysis_lookup or {}).items():
        frame = getattr(getattr(analysis, "parsed", None), "normalized_frame", None)
        _try_add_candidate(
            candidates,
            rejected,
            f"analysis_lookup:{key}",
            frame,
            source_key="analysis_lookup",
            counts=counts,
        )
    for name, payload in (upload_payloads if upload_payloads is not None else _load_upload_memory_continuous_payloads()):
        frame, parse_error = _read_csv_payload(name, payload)
        if parse_error is not None:
            rejected.append(f"upload_memory:{name}: {parse_error}")
            continue
        _try_add_candidate(
            candidates,
            rejected,
            f"upload_memory:{name}",
            frame,
            source_key="upload_memory_continuous",
            counts=counts,
        )
    for name, payload in (dataset_library_payloads if dataset_library_payloads is not None else _load_dataset_library_continuous_payloads()):
        frame, parse_error = _read_csv_payload(name, payload)
        if parse_error is not None:
            rejected.append(f"dataset_library:{name}: {parse_error}")
            continue
        _try_add_candidate(
            candidates,
            rejected,
            f"dataset_library:{name}",
            frame,
            source_key="dataset_library",
            counts=counts,
        )
    scan = {
        "continuous_candidate_source_counts": counts,
        "continuous_candidate_rejected_count": len(rejected),
        "continuous_candidate_reject_reasons": rejected,
        "continuous_candidate_rejection_reasons": rejected,
    }
    return sorted(candidates.keys()), candidates, scan


def _continuous_candidate_frames(analysis_lookup: dict) -> tuple[list[str], dict[str, pd.DataFrame]]:
    names, candidates, _scan = discover_continuous_candidate_frames(analysis_lookup)
    return sorted(candidates.keys()), candidates


def _is_continuous_candidate(frame: pd.DataFrame) -> bool:
    try:
        adapt_continuous_source_frame(frame)
    except ValueError:
        return False
    return True


def _try_add_candidate(
    candidates: dict[str, pd.DataFrame],
    rejected: list[str],
    name: str,
    frame: pd.DataFrame | None,
    *,
    source_key: str,
    counts: dict[str, int],
) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return
    try:
        adapted, _metadata = adapt_continuous_source_frame(frame)
    except ValueError as exc:
        rejected.append(f"{name}: {exc}")
        return
    candidates[name] = adapted
    counts[source_key] = int(counts.get(source_key, 0)) + 1


def _read_csv_payload(name: str, payload: bytes) -> tuple[pd.DataFrame | None, str | None]:
    if not str(name).lower().endswith(".csv"):
        return None, None
    try:
        text = payload.decode("utf-8-sig", errors="replace")
        data_lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
        if not data_lines:
            return None, "csv_parse_error:no_data_rows_after_metadata_preamble"
        return pd.read_csv(StringIO("\n".join(data_lines))), None
    except Exception as exc:  # noqa: BLE001 - candidate scan should surface reject reasons in UI.
        return None, f"csv_parse_error:{type(exc).__name__}:{exc}"


def _load_upload_memory_continuous_payloads() -> list[tuple[str, bytes]]:
    try:
        return category_payloads("continuous", None, include_cached_uploads=True)
    except Exception:
        return []


def _load_dataset_library_continuous_payloads() -> list[tuple[str, bytes]]:
    try:
        settings = load_dataset_library_settings()
        dataset_root = str(settings.get("dataset_root") or "").strip()
        if not dataset_root:
            return []
        payloads: list[tuple[str, bytes]] = []
        for entry in list_manifest_entries(dataset_root, dataset_mode="continuous"):
            relative_path = str(entry.get("path") or "")
            payloads.append((relative_path, read_dataset_entry_bytes(dataset_root, relative_path)))
        return payloads
    except Exception:
        return []


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
        freq_hz=float(freq_hz),
        waveform_type=waveform_type,
    )
    second = first_command_profile.copy(deep=True).reset_index(drop=True)
    for column in command.columns:
        second[column] = command[column].to_numpy() if len(command[column]) == len(second) else command[column]
    second["second_limited_voltage_v"] = command["limited_voltage_v"].to_numpy(dtype=float)
    metadata.update(
        {
            "continuous_second_modeling_uses_phase_aligned_kernel": True,
            "continuous_second_modeling_input_window": "steady_state_one_cycle_only",
            "continuous_second_modeling_tail_disabled": True,
            "continuous_second_export_cycle_count": 1.0,
            "second_modeling_input_mode": "continuous_steady_state",
            "second_drive_actual_data_used": "steady_state_one_cycle_only",
        }
    )
    return second, metadata


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
