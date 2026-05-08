from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .finite_actual_drive import build_finite_actual_drive_review_dataset


DEFAULT_VALIDATION_UPLOAD_DIR = Path("outputs") / "field_analysis_app_state" / "uploads" / "validation"


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


def render_finite_actual_drive_review_section() -> None:
    st.markdown("### 실구동 결과 리뷰 / Finite Actual-Drive Review")
    st.caption("이 섹션은 1차 추천 전압을 실제 장비에 넣은 결과를 확인하기 위한 리뷰 화면입니다.")
    st.caption("아직 2차 보정 전압은 계산하지 않습니다.")
    st.caption("사용자가 그래프를 확인한 뒤 2차 보정 방향을 결정합니다.")
    st.caption("This is review, not acceptance. UI does not judge model quality.")

    uploaded_files = st.file_uploader(
        "validation/result CSV 업로드",
        type=["csv"],
        accept_multiple_files=True,
        help="hash/internal id prefix가 붙은 finite_recommended_voltage_lut_*_result.csv 파일도 지원합니다.",
        key="finite_actual_drive_review_upload",
    )
    use_cached_validation = st.checkbox(
        "캐시된 validation 결과 불러오기",
        value=False,
        help="outputs/field_analysis_app_state/uploads/validation 폴더의 validation/result CSV를 불러옵니다.",
        key="finite_actual_drive_use_cached_validation",
    )
    cached_paths = cached_validation_result_paths() if use_cached_validation else []
    if not uploaded_files and not cached_paths:
        st.info(
            "파일을 업로드하거나 캐시된 validation 결과를 불러오면 target vs measured / first voltage / residual 그래프를 앱 안에서 바로 볼 수 있습니다."
        )
        if use_cached_validation:
            st.warning("캐시된 validation 결과 파일을 찾지 못했습니다.")
        return

    parse_result = _combine_parse_results(
        [
            build_actual_drive_review_cases_from_uploads(uploaded_files) if uploaded_files else None,
            build_actual_drive_review_cases_from_paths(cached_paths) if cached_paths else None,
        ]
    )
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


def _render_review_plots(case: ActualDriveReviewCase) -> None:
    frame = case.review_frame
    st.plotly_chart(
        _line_figure(
            frame,
            [("physical_target_output_mT", "Physical Target"), ("measured_field_mT", "Measured HallBz")],
            "Target vs Measured Field",
            "mT inferred",
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        _line_figure(frame, [("first_voltage_v", "First Command Voltage")], "First Command Voltage", "V"),
        use_container_width=True,
    )
    st.plotly_chart(
        _line_figure(
            frame,
            [("command_voltage_v", "Command Voltage"), ("actual_drive_voltage_v", "Actual Drive Voltage")],
            "Command vs Actual Drive Voltage",
            "V",
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        _line_figure(frame, [("measured_residual_mT", "Target - Measured")], "Residual", "mT inferred"),
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
        if "measured_field_mT" not in frame.columns:
            continue
        figure.add_trace(
            go.Scatter(x=frame["time_s"], y=frame["measured_field_mT"], mode="lines", name=case.label, line={"width": 1.0})
        )
    figure.update_layout(
        template="plotly_white",
        height=420,
        title="All Actual-Drive Measured Fields",
        xaxis_title="time_s",
        yaxis_title="mT inferred",
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
    metrics = [
        "measured_active_nrmse",
        "measured_shape_corr",
        "measured_peak_error_mT",
        "measured_phase_error_s",
        "measured_terminal_error_mT",
        "measured_tail_residual",
        "measured_startup_residual_mT",
        "alignment_confidence",
        "possible_polarity_flip_suggested",
    ]
    st.dataframe(pd.DataFrame([{"metric": metric, "value": case.metrics.get(metric)} for metric in metrics]))


def _render_review_exports(case: ActualDriveReviewCase, parse_result: ActualDriveReviewParseResult) -> None:
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
