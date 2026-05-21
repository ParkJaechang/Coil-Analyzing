from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .ui_final_voltage_lut_export import build_final_voltage_lut_csv_bytes
from .ui_final_voltage_lut_export import build_final_voltage_lut_filename
from .ui_final_voltage_lut_export import build_final_voltage_lut_frame
from .ui_final_voltage_lut_export import render_final_voltage_lut_export_panel
from .ui_continuous_final_lut_export import (
    build_continuous_final_lut_filename,
    build_continuous_final_lut_frame,
    continuous_result_export_record,
)

REQUIRED_LUT_COLUMNS = ("sample_index", "time_s", "voltage_v")
DEBUG_VOLTAGE_COLUMNS = (
    "recommended_voltage_v",
    "baseline_recommended_voltage_v",
    "compensated_recommended_voltage_v",
    "startup_compensation_command_delta_v",
    "feedback_corrected_limited_voltage_v",
    "feedback_correction_delta_v",
)
@dataclass(frozen=True)
class ParsedVoltageLut:
    source_name: str
    frame: pd.DataFrame
    ok: bool
    error: str | None = None

def parse_voltage_lut_upload(source_name: str, data: bytes) -> ParsedVoltageLut:
    try:
        frame = pd.read_csv(_bytes_to_buffer(data))
    except Exception as exc:  # pragma: no cover - pandas error details vary
        return ParsedVoltageLut(source_name=source_name, frame=pd.DataFrame(), ok=False, error=str(exc))

    missing = [column for column in REQUIRED_LUT_COLUMNS if column not in frame.columns]
    if missing:
        return ParsedVoltageLut(
            source_name=source_name,
            frame=frame,
            ok=False,
            error=f"Missing required LUT columns: {missing}",
        )
    normalized = _normalize_lut_frame(frame)
    return ParsedVoltageLut(source_name=source_name, frame=normalized, ok=True)

def build_lut_review_options(
    records: list[ParsedVoltageLut],
) -> tuple[list[str], dict[str, ParsedVoltageLut], dict[str, str]]:
    """Build scalar selectbox options while keeping DataFrames in a lookup map."""
    options: list[str] = []
    records_by_id: dict[str, ParsedVoltageLut] = {}
    labels_by_id: dict[str, str] = {}
    for record in records:
        option_id = _unique_option_id(record.source_name, records_by_id)
        options.append(option_id)
        records_by_id[option_id] = record
        labels_by_id[option_id] = record.source_name
    return options, records_by_id, labels_by_id

def add_lut_cache_bytes(*args: object, **kwargs: object) -> str:
    from .ui_voltage_lut_cache import add_lut_cache_bytes as _impl

    return _impl(*args, **kwargs)

def build_lut_cache_records(*args: object, **kwargs: object) -> list[object]:
    from .ui_voltage_lut_cache import build_lut_cache_records as _impl

    return _impl(*args, **kwargs)

def build_lut_cache_selection_options(*args: object, **kwargs: object) -> tuple[list[str], dict[str, object], dict[str, str]]:
    from .ui_voltage_lut_cache import build_lut_cache_selection_options as _impl

    return _impl(*args, **kwargs)

def edit_lut_cache_metadata(*args: object, **kwargs: object) -> bool:
    from .ui_voltage_lut_cache import edit_lut_cache_metadata as _impl

    return _impl(*args, **kwargs)

def delete_lut_cache_item(*args: object, **kwargs: object) -> bool:
    from .ui_voltage_lut_cache import delete_lut_cache_item as _impl

    return _impl(*args, **kwargs)

def fallback_lut_cache_selection(*args: object, **kwargs: object) -> str | None:
    from .ui_voltage_lut_cache import fallback_lut_cache_selection as _impl

    return _impl(*args, **kwargs)


def build_lut_diagnostics(frame: pd.DataFrame) -> dict[str, object]:
    time_s = pd.to_numeric(frame.get("time_s"), errors="coerce").to_numpy(dtype=float)
    voltage = pd.to_numeric(frame.get("voltage_v"), errors="coerce").to_numpy(dtype=float)
    finite_time = time_s[np.isfinite(time_s)]
    finite_voltage = voltage[np.isfinite(voltage)]
    diffs = np.diff(finite_time) if finite_time.size >= 2 else np.array([], dtype=float)
    positive_diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    duplicated_time_count = int(pd.Series(finite_time).duplicated().sum()) if finite_time.size else 0
    time_monotonic = bool(np.all(diffs > 0)) if diffs.size else True
    dt_min = _float_or_nan(np.min(positive_diffs)) if positive_diffs.size else float("nan")
    dt_median = _float_or_nan(np.median(positive_diffs)) if positive_diffs.size else float("nan")
    dt_max = _float_or_nan(np.max(positive_diffs)) if positive_diffs.size else float("nan")
    irregularity = float("nan")
    if np.isfinite(dt_median) and dt_median > 0:
        irregularity = float((dt_max - dt_min) / dt_median)

    duration = float("nan")
    if finite_time.size:
        duration = float(np.nanmax(finite_time) - np.nanmin(finite_time))
    suspected_time_unit = _infer_time_unit(duration, dt_median)
    time_axis_status = _time_axis_status(time_monotonic, duplicated_time_count, irregularity, suspected_time_unit)

    return {
        "sample_count": int(len(frame)),
        "time_start_s": _float_or_nan(np.nanmin(finite_time)) if finite_time.size else float("nan"),
        "time_end_s": _float_or_nan(np.nanmax(finite_time)) if finite_time.size else float("nan"),
        "duration_s": duration,
        "dt_min_s": dt_min,
        "dt_median_s": dt_median,
        "dt_max_s": dt_max,
        "dt_irregularity_ratio": irregularity,
        "time_monotonic": time_monotonic,
        "duplicated_time_count": duplicated_time_count,
        "voltage_min_v": _float_or_nan(np.nanmin(finite_voltage)) if finite_voltage.size else float("nan"),
        "voltage_max_v": _float_or_nan(np.nanmax(finite_voltage)) if finite_voltage.size else float("nan"),
        "voltage_normalization_enabled": bool(frame.attrs.get("voltage_normalization_enabled", False)),
        "voltage_normalization_mode": frame.attrs.get("voltage_normalization_mode"),
        "voltage_normalization_status": frame.attrs.get("voltage_normalization_status"),
        "voltage_normalization_source_peak_v": frame.attrs.get("voltage_normalization_source_peak_v"),
        "voltage_normalization_scale_factor": frame.attrs.get("voltage_normalization_scale_factor"),
        "absolute_gain_evaluation_disabled": bool(frame.attrs.get("absolute_gain_evaluation_disabled", False)),
        "shape_review_only": bool(frame.attrs.get("shape_review_only", False)),
        "suspected_time_unit": suspected_time_unit,
        "time_axis_status": time_axis_status,
    }


def build_normalized_lut_csv_bytes(frame: pd.DataFrame) -> bytes:
    return _normalize_lut_frame(frame).to_csv(index=False).encode("utf-8-sig")


def build_diagnostics_csv_bytes(source_name: str, diagnostics: dict[str, object]) -> bytes:
    row = {"source_name": source_name, **diagnostics}
    return pd.DataFrame([row]).to_csv(index=False).encode("utf-8-sig")


def render_voltage_lut_review_section(default_cache_root: Path | None = None) -> None:
    from .ui_voltage_lut_cache import LUT_CACHE_STATE_KEY
    from .ui_voltage_lut_cache import add_lut_cache_bytes
    from .ui_voltage_lut_cache import build_lut_cache_records
    from .ui_voltage_lut_cache import build_lut_cache_selection_options
    from .ui_voltage_lut_cache import fallback_lut_cache_selection

    st.markdown("### LUT 검수 / LUT Review")
    st.caption("사용자 시간축/전압 파형 검수용 화면입니다. 장비 구동 적합성이나 보정 품질을 자동 판정하지 않습니다.")
    st.caption("필수 schema: sample_index, time_s, voltage_v")

    cache_state = st.session_state.setdefault(LUT_CACHE_STATE_KEY, {})
    if not isinstance(cache_state, dict):
        cache_state = {}
        st.session_state[LUT_CACHE_STATE_KEY] = cache_state
    _sync_session_result_lut_sources_to_cache(cache_state)

    uploaded_files = st.file_uploader(
        "Exported voltage LUT CSV 업로드",
        type=["csv"],
        accept_multiple_files=True,
        key="voltage_lut_review_upload",
    )
    for uploaded_file in uploaded_files or []:
        add_lut_cache_bytes(cache_state, uploaded_file.name, uploaded_file.getvalue())

    cached_files = _discover_cached_lut_files(default_cache_root)
    if cached_files:
        with st.expander("cached/exported LUT file 불러오기", expanded=False):
            cached_ids = [str(path) for path in cached_files]
            cached_by_id = dict(zip(cached_ids, cached_files, strict=True))
            selected_cache_id = st.selectbox(
                "Cached LUT file",
                options=cached_ids,
                format_func=lambda path_id: Path(path_id).name,
                key="voltage_lut_cached_file",
            )
            if st.button("선택한 cached LUT 불러오기", key="load_cached_voltage_lut"):
                selected_cache = cached_by_id[selected_cache_id]
                add_lut_cache_bytes(cache_state, selected_cache.name, selected_cache.read_bytes())

    if not st.button("Load LUT CSV", key="load_lut_csv_for_review"):
        if not st.session_state.get("voltage_lut_review_loaded"):
            st.info("LUT upload/cache selection changed. Press Load LUT CSV to parse diagnostics.")
            return
    else:
        st.session_state["voltage_lut_review_loaded"] = True

    records = build_lut_cache_records(cache_state)
    st.markdown("#### 업로드된 LUT 캐시")
    if not records:
        st.info("업로드된 LUT CSV가 없습니다. Quick LUT에서 다운로드한 최종 모델링 전압 LUT CSV를 업로드하세요.")
        return

    _render_lut_cache_summary(records)
    cache_ids, cache_records_by_id, cache_labels_by_id = build_lut_cache_selection_options(records)
    safe_selected_id = fallback_lut_cache_selection(cache_ids, st.session_state.get("voltage_lut_cache_selected_id"))
    if safe_selected_id is None:
        st.info("업로드된 LUT 캐시가 비어 있습니다.")
        return
    st.session_state["voltage_lut_cache_selected_id"] = safe_selected_id
    selected_cache_id = st.selectbox(
        "업로드된 LUT 캐시 선택",
        options=cache_ids,
        format_func=lambda cache_id: cache_labels_by_id[cache_id],
        key="voltage_lut_cache_selected_id",
    )
    selected_record = cache_records_by_id[selected_cache_id]
    st.caption(f"내부 ID: {selected_record.id}")
    _render_lut_cache_metadata_editor(cache_state, selected_record)
    _render_lut_cache_delete_panel(cache_state, selected_record)

    parsed_items = [record.parsed for record in records]
    successes = [item for item in parsed_items if item.ok]
    failures = [item for item in parsed_items if not item.ok]
    st.write(f"- parsed LUT files: `{len(successes)}`")
    st.write(f"- failed LUT files: `{len(failures)}`")
    for failure in failures:
        st.warning(f"{failure.source_name}: {failure.error or 'parse failed'}")
    if not selected_record.parsed.ok:
        st.warning(f"읽을 수 없음: {selected_record.parsed.error or 'parse failed'}")
        return

    selected = selected_record.parsed
    diagnostics = build_lut_diagnostics(selected.frame)
    if st.button("Render LUT Plot", key="render_lut_plot_button"):
        st.session_state["voltage_lut_render_plot_id"] = selected_record.id
    if st.session_state.get("voltage_lut_render_plot_id") == selected_record.id:
        _render_lut_plots(selected.frame)
    else:
        st.info("Press Render LUT Plot to draw voltage plots.")
    _render_lut_diagnostics(diagnostics)
    _render_lut_warnings(diagnostics)
    st.download_button(
        label="normalized LUT CSV 다운로드",
        data=build_normalized_lut_csv_bytes(selected.frame),
        file_name=f"normalized_{Path(selected.source_name).name}",
        mime="text/csv",
        key="download_normalized_voltage_lut_csv",
    )
    st.download_button(
        label="diagnostics summary CSV 다운로드",
        data=build_diagnostics_csv_bytes(selected.source_name, diagnostics),
        file_name=f"diagnostics_{Path(selected.source_name).stem}.csv",
        mime="text/csv",
        key="download_voltage_lut_diagnostics_csv",
    )


def _sync_session_result_lut_sources_to_cache(cache_state: dict[str, dict[str, object]]) -> None:
    # Final Voltage LUT Export tab session sources:
    # finite_first_voltage_lut / finite_second_voltage_lut
    # continuous_first_voltage_lut / continuous_second_voltage_lut
    finite_sources = [
        ("Finite 1차", "first", "quick_lut_first_model_result"),
        ("Finite 2차", "second", "quick_lut_second_model_result"),
    ]
    for display_prefix, stage, state_key in finite_sources:
        result = st.session_state.get(state_key)
        if not isinstance(result, dict):
            continue
        metadata = dict(result.get("metadata") or {})
        if metadata.get("modeling_input_mode") == "continuous_steady_state":
            continue
        command = result.get("command_profile")
        if not isinstance(command, pd.DataFrame) or command.empty:
            continue
        try:
            source_column = "second_limited_voltage_v" if stage == "second" and "second_limited_voltage_v" in command.columns else None
            frame = build_final_voltage_lut_frame(command, voltage_source_column=source_column)
        except Exception:  # noqa: BLE001 - invalid session result should not break LUT review.
            continue
        filename = build_final_voltage_lut_filename(
            waveform_type=metadata.get("waveform_type") or metadata.get("modeled_target_waveform_family") or "finite",
            freq_hz=metadata.get("freq_hz") or metadata.get("modeled_target_freq_hz"),
            cycle_count=metadata.get("cycle_count") or metadata.get("modeled_target_cycle_count"),
        )
        if stage == "second":
            filename = filename.replace("finite_recommended_voltage_lut", "finite_second_voltage_lut")
        else:
            filename = filename.replace("finite_recommended_voltage_lut", "finite_first_voltage_lut")
        add_lut_cache_bytes(
            cache_state,
            filename,
            frame.to_csv(index=False).encode("utf-8-sig"),
            display_name=f"{display_prefix} - {filename}",
        )
    session_sources = [
        ("Continuous 1차", "first", "quick_lut_first_model_result_continuous"),
        ("Continuous 2차", "second", "quick_lut_second_model_result_continuous"),
    ]
    for display_prefix, stage, state_key in session_sources:
        result = st.session_state.get(state_key)
        record = continuous_result_export_record(stage, result if isinstance(result, dict) else None)
        if not record["available"]:
            continue
        metadata = dict(record.get("metadata") or {})
        command = record["command_profile"]
        try:
            freq_hz = _first_result_freq_hz(metadata, command)
            frame, _export_meta = build_continuous_final_lut_frame(
                command,
                voltage_source_column=str(record["voltage_source_column"]),
                freq_hz=freq_hz,
                stage=stage,
            )
        except Exception:  # noqa: BLE001 - invalid session result should not break LUT review.
            continue
        filename = build_continuous_final_lut_filename(
            stage=stage,
            waveform_type=metadata.get("waveform_type") or metadata.get("target_waveform_family") or "continuous",
            freq_hz=freq_hz,
        )
        add_lut_cache_bytes(
            cache_state,
            filename,
            frame.to_csv(index=False).encode("utf-8-sig"),
            display_name=f"{display_prefix} - {filename}",
        )


def _first_result_freq_hz(metadata: dict[str, object], command: pd.DataFrame) -> float | None:
    for value in (
        metadata.get("freq_hz"),
        metadata.get("modeled_target_freq_hz"),
        metadata.get("target_freq_hz"),
    ):
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number) and number > 0.0:
            return number
    if "freq_hz" in command.columns and not command.empty:
        try:
            number = float(command["freq_hz"].iloc[0])
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) and number > 0.0 else None
    return None


def _normalize_lut_frame(frame: pd.DataFrame) -> pd.DataFrame:
    raw_voltage = pd.to_numeric(frame["voltage_v"], errors="coerce")
    normalized_voltage, voltage_peak, voltage_scale, voltage_status = _review_normalize_voltage(raw_voltage)
    normalized = pd.DataFrame(
        {
            "sample_index": np.arange(len(frame), dtype=int),
            "time_s": pd.to_numeric(frame["time_s"], errors="coerce"),
            "voltage_v": raw_voltage,
            "raw_voltage_v": raw_voltage,
            "normalized_voltage_v": normalized_voltage,
            "voltage_normalization_scale_factor": voltage_scale,
        }
    )
    normalized.attrs["voltage_normalization_enabled"] = True
    normalized.attrs["voltage_normalization_mode"] = "peak_to_5V"
    normalized.attrs["voltage_normalization_status"] = voltage_status
    normalized.attrs["voltage_normalization_source_peak_v"] = voltage_peak
    normalized.attrs["voltage_normalization_scale_factor"] = voltage_scale
    normalized.attrs["absolute_gain_evaluation_disabled"] = True
    normalized.attrs["shape_review_only"] = True
    for column in DEBUG_VOLTAGE_COLUMNS:
        if column in frame.columns:
            normalized[column] = pd.to_numeric(frame[column], errors="coerce")
    return normalized


def _render_lut_plots(frame: pd.DataFrame) -> None:
    voltage_column = "normalized_voltage_v" if "normalized_voltage_v" in frame.columns else "voltage_v"
    time_figure = go.Figure()
    time_figure.add_trace(go.Scatter(x=frame["time_s"], y=frame[voltage_column], mode="lines", name=voltage_column))
    time_figure.update_layout(
        template="plotly_white",
        height=360,
        title="LUT Voltage vs time_s",
        xaxis_title="time_s",
        yaxis_title=voltage_column,
    )
    st.plotly_chart(time_figure, use_container_width=True)

    sample_figure = go.Figure()
    sample_figure.add_trace(
        go.Scatter(x=frame["sample_index"], y=frame[voltage_column], mode="lines", name=voltage_column)
    )
    sample_figure.update_layout(
        template="plotly_white",
        height=320,
        title="LUT Voltage vs sample_index",
        xaxis_title="sample_index",
        yaxis_title=voltage_column,
    )
    st.plotly_chart(sample_figure, use_container_width=True)


def _render_lut_diagnostics(diagnostics: dict[str, object]) -> None:
    st.markdown("#### LUT timebase diagnostics")
    labels = (
        "sample_count",
        "time_start_s",
        "time_end_s",
        "duration_s",
        "dt_min_s",
        "dt_median_s",
        "dt_max_s",
        "dt_irregularity_ratio",
        "time_monotonic",
        "duplicated_time_count",
        "voltage_min_v",
        "voltage_max_v",
        "voltage_normalization_enabled",
        "voltage_normalization_mode",
        "voltage_normalization_status",
        "voltage_normalization_source_peak_v",
        "voltage_normalization_scale_factor",
        "absolute_gain_evaluation_disabled",
        "shape_review_only",
        "suspected_time_unit",
        "time_axis_status",
    )
    st.dataframe(pd.DataFrame([diagnostics], columns=list(labels)), use_container_width=True)


def _render_lut_warnings(diagnostics: dict[str, object]) -> None:
    if diagnostics.get("suspected_time_unit") == "ms_like_seconds_column":
        st.warning("time_s가 ms처럼 보입니다. exported LUT의 time unit을 확인하세요.")
    if diagnostics.get("time_monotonic") is False or int(diagnostics.get("duplicated_time_count") or 0) > 0:
        st.warning("duplicate/non-monotonic time warning: time_s 축을 확인하세요.")
    irregularity = diagnostics.get("dt_irregularity_ratio")
    if isinstance(irregularity, (int, float)) and np.isfinite(irregularity) and irregularity > 0.05:
        st.warning("irregular dt warning: sample interval이 일정하지 않을 수 있습니다.")


def _render_lut_cache_summary(records: list[object]) -> None:
    rows: list[dict[str, object]] = []
    for record in records:
        diagnostics = record.diagnostics
        rows.append(
            {
                "표시 이름": record.display_name,
                "원본 파일명": record.original_filename,
                "uploaded/created time": record.metadata.get("created_time"),
                "sample count": diagnostics.get("sample_count") if record.parsed.ok else None,
                "duration": diagnostics.get("duration_s") if record.parsed.ok else None,
                "time range": _format_range(diagnostics.get("time_start_s"), diagnostics.get("time_end_s"))
                if record.parsed.ok
                else "",
                "voltage range": _format_range(diagnostics.get("voltage_min_v"), diagnostics.get("voltage_max_v"))
                if record.parsed.ok
                else "",
                "timebase status": diagnostics.get("time_axis_status") if record.parsed.ok else "읽을 수 없음",
                "메모": record.user_note,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _render_lut_cache_metadata_editor(
    cache_state: dict[str, dict[str, object]],
    record: object,
) -> None:
    from .ui_voltage_lut_cache import edit_lut_cache_metadata

    with st.expander("LUT 캐시 metadata 편집", expanded=False):
        display_name = st.text_input(
            "표시 이름",
            value=record.display_name,
            key=f"voltage_lut_cache_display_name_{record.id}",
        )
        user_note = st.text_area(
            "메모",
            value=record.user_note,
            key=f"voltage_lut_cache_user_note_{record.id}",
        )
        if st.button("LUT 캐시 metadata 저장", key=f"save_voltage_lut_cache_metadata_{record.id}"):
            edit_lut_cache_metadata(
                cache_state,
                record.id,
                display_name=display_name,
                user_note=user_note,
            )
            st.rerun()


def _render_lut_cache_delete_panel(
    cache_state: dict[str, dict[str, object]],
    record: object,
) -> None:
    from .ui_voltage_lut_cache import delete_lut_cache_item

    with st.expander("LUT 캐시 삭제", expanded=False):
        st.caption("원본 CSV numerical data는 수정하지 않습니다. 앱 캐시 목록에서만 제거합니다.")
        confirm = st.checkbox("삭제 확인", key=f"confirm_delete_voltage_lut_cache_{record.id}")
        if st.button(
            "선택한 LUT 캐시 삭제",
            key=f"delete_voltage_lut_cache_{record.id}",
            disabled=not confirm,
        ):
            delete_lut_cache_item(cache_state, record.id)
            st.session_state.pop("voltage_lut_cache_selected_id", None)
            st.rerun()


def _discover_cached_lut_files(default_cache_root: Path | None) -> list[Path]:
    if default_cache_root is None:
        default_cache_root = Path("outputs") / "field_analysis_app_state"
    root = Path(default_cache_root)
    if not root.exists():
        return []
    patterns = ("finite_recommended_voltage_lut*.csv", "*recommended_voltage_lut*.csv")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path for path in root.rglob(pattern) if path.is_file())
    return sorted(set(files), key=lambda path: str(path).lower())


def _unique_option_id(source_name: str, existing: dict[str, object]) -> str:
    base = str(source_name).strip() or "lut"
    if base not in existing:
        return base
    index = 2
    while f"{base}#{index}" in existing:
        index += 1
    return f"{base}#{index}"


def _format_range(start: object, end: object) -> str:
    start_number = _float_or_nan(start)
    end_number = _float_or_nan(end)
    if not np.isfinite(start_number) or not np.isfinite(end_number):
        return ""
    return f"{start_number:g} .. {end_number:g}"


def _review_normalize_voltage(values: pd.Series) -> tuple[pd.Series, float, float, str]:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    peak = float(np.nanmax(np.abs(finite))) if finite.size else float("nan")
    if not np.isfinite(peak) or peak <= 1e-12:
        return pd.Series(np.zeros(len(numeric), dtype=float), index=numeric.index), peak, float("nan"), "unavailable_zero_peak"
    scale = 5.0 / peak
    return numeric * scale, peak, scale, "ok"


def _bytes_to_buffer(data: bytes) -> object:
    from io import BytesIO

    return BytesIO(data)


def _float_or_nan(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _infer_time_unit(duration_s: float, dt_median_s: float) -> str:
    if np.isfinite(dt_median_s) and dt_median_s >= 1.0:
        return "ms_like_seconds_column"
    if np.isfinite(duration_s) and duration_s > 60.0:
        return "ms_like_seconds_column"
    return "seconds"


def _time_axis_status(
    time_monotonic: bool,
    duplicated_time_count: int,
    irregularity: float,
    suspected_time_unit: str,
) -> str:
    if not time_monotonic or duplicated_time_count > 0:
        return "error_duplicate_or_non_monotonic_time"
    if suspected_time_unit == "ms_like_seconds_column":
        return "warning_time_s_may_be_ms"
    if np.isfinite(irregularity) and irregularity > 0.05:
        return "warning_irregular_dt"
    return "ok"
