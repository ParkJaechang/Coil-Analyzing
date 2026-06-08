from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .finite_actual_drive import build_finite_actual_drive_review_dataset
from .voltage_policy import COMMAND_VOLTAGE_LIMIT_LABEL, COMMAND_VOLTAGE_NORMALIZATION_MODE

# UI contract markers: Voltage는 peak 기준 ±10V 이내로 정규화되어 표시됩니다.
# voltage_normalization_mode = peak_to_10V


DEFAULT_VALIDATION_UPLOAD_DIR = Path("outputs") / "field_analysis_app_state" / "uploads" / "validation"
ACTUAL_DRIVE_CACHE_STATE_KEY = "actual_drive_validation_cache_items"
ACTUAL_DRIVE_SELECTED_CACHE_KEY = "actual_drive_validation_selected_cache_id"


@dataclass(frozen=True)
class ActualDriveReviewCase:
    label: str
    source_file: str
    canonical_source_filename: str
    upload_internal_id: str | None
    waveform_type: str
    freq_hz: float
    cycle_count: float
    review_frame: pd.DataFrame
    metrics: dict[str, Any]
    status: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ActualDriveReviewParseResult:
    cases: list[ActualDriveReviewCase]
    failures: list[dict[str, Any]]
    summary: pd.DataFrame

    @property
    def parsed_count(self) -> int:
        return len(self.cases)

    @property
    def failed_count(self) -> int:
        return len(self.failures)


def build_actual_drive_review_cases_from_paths(paths: Iterable[Path]) -> ActualDriveReviewParseResult:
    dataset = build_finite_actual_drive_review_dataset([Path(path) for path in paths])
    return _review_parse_result_from_dataset(dataset)


def cached_validation_result_paths(
    validation_dir: Path | None = None,
    *,
    project_root: Path | None = None,
) -> list[Path]:
    root = project_root or Path(__file__).resolve().parents[2]
    directory = validation_dir or root / DEFAULT_VALIDATION_UPLOAD_DIR
    if not directory.exists():
        return []
    return sorted(directory.glob("*finite_recommended_voltage_lut_*_result.csv"))


def build_actual_drive_review_cases_from_uploads(uploaded_files: Iterable[Any]) -> ActualDriveReviewParseResult:
    with TemporaryDirectory(prefix="finite_actual_drive_review_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        paths: list[Path] = []
        for index, uploaded_file in enumerate(uploaded_files):
            source_name = str(getattr(uploaded_file, "name", "") or f"uploaded_result_{index}.csv")
            target_path = tmp_root / Path(source_name).name
            target_path.write_bytes(uploaded_file.getvalue())
            paths.append(target_path)
        return build_actual_drive_review_cases_from_paths(paths)


def build_actual_drive_review_cases_from_cache_state(cache_state: dict[str, dict[str, object]]) -> ActualDriveReviewParseResult:
    from .ui_upload_cache import build_upload_cache_records

    records = [
        record
        for record in build_upload_cache_records(cache_state)
        if record.cache_type == "actual_drive_validation"
    ]
    if not records:
        return ActualDriveReviewParseResult(cases=[], failures=[], summary=pd.DataFrame())
    with TemporaryDirectory(prefix="finite_actual_drive_review_cache_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        paths: list[Path] = []
        failures: list[dict[str, Any]] = []
        for record in records:
            item = cache_state.get(record.cache_item_id, {})
            content = item.get("csv_bytes")
            if not isinstance(content, bytes):
                failures.append(
                    {
                        "source_file": record.original_filename,
                        "cache_item_id": record.cache_item_id,
                        "parse_status": "error",
                        "parse_error": "cached actual-drive bytes unavailable",
                    }
                )
                continue
            target_path = tmp_root / f"{record.cache_item_id.replace(':', '_').replace('#', '_')}_{Path(record.original_filename).name}"
            target_path.write_bytes(content)
            paths.append(target_path)
        result = build_actual_drive_review_cases_from_paths(paths) if paths else ActualDriveReviewParseResult([], [], pd.DataFrame())
        return ActualDriveReviewParseResult(
            cases=result.cases,
            failures=[*failures, *result.failures],
            summary=result.summary,
        )


def render_finite_actual_drive_review_section() -> None:
    from .ui_upload_cache import add_upload_cache_bytes
    from .ui_upload_cache import build_upload_cache_records
    from .ui_upload_cache import build_upload_cache_selection_options
    from .ui_upload_cache import delete_upload_cache_item
    from .ui_upload_cache import edit_upload_cache_metadata
    from .ui_upload_cache import fallback_upload_cache_selection
    from .ui_actual_drive_cache import render_actual_drive_cache_manager

    st.markdown("### 실구동 결과 리뷰 / Finite Actual-Drive Review")
    st.caption("이 섹션은 1차 추천 전압을 실제 장비에 넣은 결과를 확인하기 위한 리뷰 화면입니다.")
    st.caption("아직 2차 보정 전압은 계산하지 않습니다.")
    st.caption("사용자가 그래프를 확인한 뒤 2차 보정 방향을 결정합니다.")
    st.caption("이 화면은 절대 gain 평가가 아니라 파형 개형/타이밍 검토용입니다.")
    st.caption("Measured field는 peak 기준 ±50mT로 정규화되어 표시됩니다.")
    st.caption(f"Voltage는 peak 기준 {COMMAND_VOLTAGE_LIMIT_LABEL} 이내로 정규화되어 표시됩니다.")
    st.caption(f"field_normalization_mode = peak_to_50mT · voltage_normalization_mode = {COMMAND_VOLTAGE_NORMALIZATION_MODE}")
    st.caption("Raw 값은 보존되며, 정규화는 review plot/metrics용입니다.")
    st.caption("modeled cycle과 intended drive cycle은 별도 metadata로 표시됩니다.")
    st.caption("This is review, not acceptance. UI does not judge model quality.")
    st.info(
        "이 화면은 절대 gain 평가가 아니라 파형 개형/타이밍 검토용입니다. "
        f"Measured field는 peak 기준 ±50mT, Voltage는 peak 기준 {COMMAND_VOLTAGE_LIMIT_LABEL} 이내로 정규화되어 표시됩니다. "
        "Raw 값은 보존되며 정규화는 review plot/metrics용입니다."
    )

    uploaded_files = st.file_uploader(
        "validation/result CSV 업로드",
        type=["csv"],
        accept_multiple_files=True,
        help="hash/internal id prefix가 붙은 finite_recommended_voltage_lut_*_result.csv 파일도 지원합니다.",
        key="finite_actual_drive_review_upload",
    )
    cache_state = st.session_state.setdefault(ACTUAL_DRIVE_CACHE_STATE_KEY, {})
    if not isinstance(cache_state, dict):
        cache_state = {}
        st.session_state[ACTUAL_DRIVE_CACHE_STATE_KEY] = cache_state
    for uploaded_file in uploaded_files or []:
        add_upload_cache_bytes(
            cache_state,
            uploaded_file.name,
            uploaded_file.getvalue(),
            cache_type="actual_drive_validation",
            allow_duplicate=False,
        )
    use_cached_validation = st.checkbox(
        "캐시된 validation 결과 불러오기",
        value=False,
        help="outputs/field_analysis_app_state/uploads/validation 폴더의 validation/result CSV를 불러옵니다.",
        key="finite_actual_drive_use_cached_validation",
    )
    cached_paths = cached_validation_result_paths() if use_cached_validation else []
    for cached_path in cached_paths:
        add_upload_cache_bytes(
            cache_state,
            cached_path.name,
            cached_path.read_bytes(),
            cache_type="actual_drive_validation",
            source_path=str(cached_path),
            allow_duplicate=False,
        )
    cache_records = [
        record
        for record in build_upload_cache_records(cache_state)
        if record.cache_type == "actual_drive_validation"
    ]
    render_actual_drive_cache_manager(
        cache_state,
        cache_records,
        build_upload_cache_selection_options,
        fallback_upload_cache_selection,
        edit_upload_cache_metadata,
        delete_upload_cache_item,
        selected_cache_key=ACTUAL_DRIVE_SELECTED_CACHE_KEY,
    )
    if not cache_records:
        st.info(
            "파일을 업로드하거나 캐시된 validation 결과를 불러오면 target vs measured / first voltage / residual 그래프를 앱 안에서 바로 볼 수 있습니다."
        )
        if use_cached_validation:
            st.warning("캐시된 validation 결과 파일을 찾지 못했습니다.")
        return

    if not st.button("Review Actual-drive Result", key="review_actual_drive_result_button"):
        cached = st.session_state.get("actual_drive_applied_review_result")
        if isinstance(cached, ActualDriveReviewParseResult):
            parse_result = cached
        else:
            st.info("Actual-drive upload cached. Press Review Actual-drive Result to parse and render plots.")
            return
    else:
        parse_result = build_actual_drive_review_cases_from_cache_state(cache_state)
        st.session_state["actual_drive_applied_review_result"] = parse_result
    _render_parse_summary(parse_result)
    if parse_result.failures:
        st.warning("일부 파일을 파싱하지 못했습니다. 아래 parse error를 확인하십시오.")
        st.dataframe(pd.DataFrame(parse_result.failures), use_container_width=True)
    if not parse_result.cases:
        st.warning("리뷰 가능한 finite actual-drive result CSV가 없습니다.")
        return

    label_by_case = {case.label: case for case in parse_result.cases}
    selected_label = st.selectbox("Actual-drive case 선택", options=list(label_by_case), key="finite_actual_drive_case")
    selected_case = label_by_case[selected_label]
    _render_case_summary(selected_case)
    _render_status_panel(selected_case)
    _render_normalization_status_panel(selected_case)
    _render_cycle_semantics_panel(selected_case)
    _render_review_plots(selected_case)
    _render_all_actual_drive_overlay(parse_result)
    _render_metrics_panel(selected_case)
    _render_review_exports(selected_case, parse_result)


def _review_parse_result_from_dataset(dataset: dict[str, Any]) -> ActualDriveReviewParseResult:
    cases = [_case_from_payload(payload) for payload in dataset.get("cases", [])]
    failures = [dict(error) for error in dataset.get("errors", [])]
    summary = dataset.get("summary")
    if not isinstance(summary, pd.DataFrame):
        summary = pd.DataFrame()
    return ActualDriveReviewParseResult(cases=cases, failures=failures, summary=summary)


def _combine_parse_results(results: list[ActualDriveReviewParseResult | None]) -> ActualDriveReviewParseResult:
    valid_results = [result for result in results if result is not None]
    cases = [case for result in valid_results for case in result.cases]
    failures = [failure for result in valid_results for failure in result.failures]
    summaries = [result.summary for result in valid_results if not result.summary.empty]
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    return ActualDriveReviewParseResult(cases=cases, failures=failures, summary=summary)


def _case_from_payload(payload: dict[str, Any]) -> ActualDriveReviewCase:
    waveform = str(payload.get("waveform_type") or payload.get("waveform") or "unknown")
    freq_hz = float(payload.get("freq_hz", float("nan")))
    cycle_count = float(payload.get("cycle_count", float("nan")))
    canonical_source_filename = str(payload.get("canonical_source_filename") or payload.get("source_file") or "")
    metrics = dict(payload.get("metrics") or {})
    status = dict(payload.get("status") or {})
    metadata = dict(payload.get("metadata") or {})
    return ActualDriveReviewCase(
        label=_format_case_label(waveform, freq_hz, cycle_count, canonical_source_filename),
        source_file=str(payload.get("source_file") or ""),
        canonical_source_filename=canonical_source_filename,
        upload_internal_id=payload.get("upload_internal_id"),
        waveform_type=waveform,
        freq_hz=freq_hz,
        cycle_count=cycle_count,
        review_frame=payload["time_series"],
        metrics=metrics,
        status=status,
        metadata=metadata,
    )


def _format_case_label(waveform: str, freq_hz: float, cycle_count: float, source_file: str) -> str:
    return f"{waveform} | {freq_hz:g} Hz | {cycle_count:g} cycle | {source_file}"


def _render_parse_summary(parse_result: ActualDriveReviewParseResult) -> None:
    columns = st.columns(3)
    columns[0].metric("파싱 성공", parse_result.parsed_count)
    columns[1].metric("파싱 실패", parse_result.failed_count)
    columns[2].metric("parse status", "ready" if parse_result.cases else "unavailable")


def _render_case_summary(case: ActualDriveReviewCase) -> None:
    st.markdown("#### Case Summary")
    columns = st.columns(4)
    columns[0].metric("waveform", case.waveform_type)
    columns[1].metric("freq_hz", f"{case.freq_hz:g}")
    columns[2].metric("cycle_count", f"{case.cycle_count:g}")
    columns[3].metric("field unit inferred", str(case.metadata.get("field_unit", "mT_inferred_from_HallBz")))
    with st.expander("Source / internal identifiers", expanded=False):
        st.write(f"- source_file: `{case.source_file}`")
        st.write(f"- canonical_source_filename: `{case.canonical_source_filename}`")
        st.write(f"- upload_internal_id: `{case.upload_internal_id or 'none'}`")
        st.write(f"- parse_status: `{case.metadata.get('parse_status', 'parsed')}`")


def _render_status_panel(case: ActualDriveReviewCase) -> None:
    st.info("Voltage1_V는 실제 1차 command로 사용됩니다.")
    st.info("HallBz는 measured field로 사용됩니다.")
    st.caption("field unit은 inferred mT로 표시됩니다.")
    st.caption(f"alignment_confidence: {case.metrics.get('alignment_confidence', 'unknown')}")
    if bool(case.metrics.get("possible_polarity_flip_suggested", False)):
        st.warning("possible_polarity_flip_suggested: review polarity before deciding second correction direction.")
    if case.status:
        st.caption("Status flags from backend review payload:")
        st.dataframe(pd.DataFrame([{"status": key, "value": value} for key, value in case.status.items()]))


def _render_normalization_status_panel(case: ActualDriveReviewCase) -> None:
    st.markdown("#### Normalization status")
    st.caption("Shape/timing review only: 절대 gain 평가는 하지 않습니다.")
    st.caption("Command target/gain quality is not automatically judged.")
    rows = []
    fields = [
        "field_normalization_enabled",
        "field_normalization_mode",
        "field_normalization_source_peak_mT",
        "field_normalization_scale_factor",
        "voltage_normalization_enabled",
        "voltage_normalization_mode",
        "voltage_normalization_source_peak_v",
        "voltage_normalization_scale_factor",
        "shape_review_only",
    ]
    missing = []
    for field in fields:
        value = _case_value(case, field)
        if value is None:
            missing.append(field)
            value = "unavailable"
        rows.append({"field": field, "value": value})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    if missing:
        st.info(f"Normalization metadata unavailable for: {', '.join(missing)}")
    st.caption("Raw peak values are informational only; no acceptance decision is made.")


def _render_cycle_semantics_panel(case: ActualDriveReviewCase) -> None:
    st.markdown("#### Cycle semantics")
    st.caption("모델링 cycle label과 실제 구동 의도 cycle은 별도로 표시됩니다. target을 바꾼 것이 아닙니다.")
    rows = []
    fields = [
        "modeled_cycle_count",
        "intended_drive_cycle_count",
        "source_filename_cycle_count",
        "cycle_usage_mode",
    ]
    missing = []
    for field in fields:
        value = _case_value(case, field)
        if value is None:
            missing.append(field)
            value = "unavailable"
        rows.append({"field": field, "value": value})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    if missing:
        st.info(f"Cycle semantics metadata unavailable for: {', '.join(missing)}")


def _render_review_plots(case: ActualDriveReviewCase) -> None:
    frame = case.review_frame
    st.plotly_chart(
        _line_figure(
            frame,
            [
                ("normalized_physical_target_output_mT", "Physical Target normalized"),
                ("normalized_measured_field_mT", "Measured HallBz normalized"),
            ],
            "Normalized Target vs Normalized Measured Field",
            "review-normalized mT",
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        _line_figure(
            frame,
            [("normalized_first_voltage_v", "First Command Voltage normalized")],
            "First Command Voltage (review-normalized)",
            "review-normalized V",
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        _line_figure(
            frame,
            [
                ("normalized_first_voltage_v", "Command Voltage normalized"),
                ("normalized_actual_drive_voltage_v", "Actual Drive Voltage normalized"),
            ],
            "Normalized First/Actual Drive Voltage",
            "review-normalized V",
        ),
        use_container_width=True,
    )
    st.caption("Command vs Actual Drive Voltage is shown with review-normalized values.")
    st.plotly_chart(
        _line_figure(
            frame,
            [("measured_residual_normalized_mT", "Target - Measured normalized")],
            "Normalized Residual",
            "normalized residual mT",
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        _line_figure(
            _terminal_zoom_frame(frame),
            [
                ("normalized_physical_target_output_mT", "Physical Target normalized"),
                ("normalized_measured_field_mT", "Measured HallBz normalized"),
            ],
            "Terminal peak zoom using normalized values",
            "review-normalized mT",
        ),
        use_container_width=True,
    )
    with st.expander("Raw Measured Field", expanded=False):
        st.plotly_chart(
            _line_figure(
                frame,
                [("measured_field_mT", "Raw Measured HallBz")],
                "Raw Measured Field",
                "raw mT",
            ),
            use_container_width=True,
        )
    with st.expander("Raw First/Actual Drive Voltage", expanded=False):
        st.plotly_chart(
            _line_figure(
                frame,
                [
                    ("first_voltage_v", "Raw First Command Voltage"),
                    ("actual_drive_voltage_v", "Raw Actual Drive Voltage"),
                ],
                "Raw First/Actual Drive Voltage",
                "raw V",
            ),
            use_container_width=True,
        )
    if "current_a" in frame.columns:
        st.plotly_chart(
            _line_figure(frame, [("current_a", "Current1_A")], "Current, if available", "A"),
            use_container_width=True,
        )


def _render_all_actual_drive_overlay(parse_result: ActualDriveReviewParseResult) -> None:
    figure = go.Figure()
    for case in parse_result.cases:
        frame = case.review_frame
        if "normalized_measured_field_mT" not in frame.columns:
            continue
        figure.add_trace(
            go.Scatter(x=frame["time_s"], y=frame["normalized_measured_field_mT"], mode="lines", name=case.label, line={"width": 1.0})
        )
    figure.update_layout(
        template="plotly_white",
        height=420,
        title="All Actual-Drive Measured Fields (review-normalized)",
        xaxis_title="time_s",
        yaxis_title="review-normalized mT",
    )
    st.plotly_chart(figure, use_container_width=True)


def _line_figure(frame: pd.DataFrame, columns: list[tuple[str, str]], title: str, yaxis_title: str) -> go.Figure:
    figure = go.Figure()
    for column, label in columns:
        if column not in frame.columns:
            continue
        figure.add_trace(go.Scatter(x=frame["time_s"], y=frame[column], mode="lines", name=label))
    figure.update_layout(template="plotly_white", height=360, title=title, xaxis_title="time_s", yaxis_title=yaxis_title)
    return figure


def _render_metrics_panel(case: ActualDriveReviewCase) -> None:
    st.markdown("#### Metrics")
    st.caption("normalized metrics are for shape/timing review; raw peaks are informational only.")
    metrics = [
        "measured_active_nrmse",
        "measured_shape_corr",
        "measured_peak_error_mT",
        "measured_phase_error_s",
        "measured_terminal_error_mT",
        "measured_tail_residual",
        "measured_startup_residual_mT",
        "raw_field_peak_mT",
        "field_normalization_scale_factor",
        "raw_voltage_peak_v",
        "voltage_normalization_scale_factor",
        "normalized_shape_corr",
        "normalized_nrmse",
        "terminal_peak_time_error_s",
        "shape_review_only",
        "alignment_confidence",
        "possible_polarity_flip_suggested",
    ]
    st.dataframe(pd.DataFrame([{"metric": metric, "value": case.metrics.get(metric)} for metric in metrics]))


def _render_review_exports(case: ActualDriveReviewCase, parse_result: ActualDriveReviewParseResult) -> None:
    st.download_button(
        "normalized review CSV 다운로드",
        data=case.review_frame.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"normalized_actual_drive_review_{case.waveform_type}_{case.freq_hz:g}Hz_{case.cycle_count:g}cycle.csv",
        mime="text/csv",
        key="finite_actual_drive_normalized_review_csv_download",
    )
    st.download_button(
        "리뷰 CSV 다운로드",
        data=case.review_frame.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"actual_drive_review_{case.waveform_type}_{case.freq_hz:g}Hz_{case.cycle_count:g}cycle.csv",
        mime="text/csv",
        key="finite_actual_drive_review_csv_download",
    )
    st.download_button(
        "요약 CSV 다운로드",
        data=parse_result.summary.to_csv(index=False).encode("utf-8-sig"),
        file_name="finite_actual_drive_review_summary.csv",
        mime="text/csv",
        key="finite_actual_drive_summary_csv_download",
    )
    st.download_button(
        "raw-preserved CSV 다운로드",
        data=case.review_frame.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"raw_preserved_actual_drive_review_{case.waveform_type}_{case.freq_hz:g}Hz_{case.cycle_count:g}cycle.csv",
        mime="text/csv",
        key="finite_actual_drive_raw_preserved_csv_download",
    )


def _case_value(case: ActualDriveReviewCase, key: str) -> Any:
    if key in case.metadata:
        return case.metadata.get(key)
    if key in case.metrics:
        return case.metrics.get(key)
    if key in case.status:
        return case.status.get(key)
    return None


def _terminal_zoom_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "time_s" not in frame.columns:
        return frame
    time_values = pd.to_numeric(frame["time_s"], errors="coerce")
    finite_time = time_values.dropna()
    if finite_time.empty:
        return frame
    start = float(finite_time.quantile(0.75))
    return frame.loc[time_values >= start]
