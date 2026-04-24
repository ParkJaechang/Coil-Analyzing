from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from .plotting import plot_waveforms
from .utils import first_number


_OPAQUE_PREFIX_MIN_LEN = 12
_OPAQUE_PREFIX_MAX_LEN = 32
_METADATA_HIDDEN_COLUMNS = {"source_file", "sheet_name", "test_id", "notes", "parse_warnings"}


@dataclass(frozen=True)
class RawWaveformTestRecord:
    test_id: str
    label: str
    source_file: str
    source_file_label: str
    sheet_name: str
    waveform_type: str
    freq_hz: float
    cycle_count: float
    target_current_a: float
    source_type: str
    sample_count: int
    duration_s: float
    sampling_rate_hz: float


def display_source_file_name(file_name: object) -> str:
    raw_name = str(file_name or "").strip()
    if not raw_name:
        return ""
    normalized = raw_name.replace("\\", "/")
    leaf_name = normalized.rsplit("/", 1)[-1]
    parts = leaf_name.split("_", 1)
    if len(parts) == 2 and _looks_like_opaque_prefix(parts[0]):
        leaf_name = parts[1]
    parent = PurePath(normalized).parent.name
    return f"{parent}/{leaf_name}" if parent else leaf_name


def build_raw_waveform_test_records(test_ids: list[str], analysis_lookup: dict) -> list[RawWaveformTestRecord]:
    records = [_build_raw_waveform_test_record(test_id, analysis_lookup[test_id]) for test_id in test_ids]
    return sorted(records, key=_raw_waveform_record_sort_key)


def build_raw_waveform_label_lookup(
    test_ids: list[str],
    analysis_lookup: dict,
) -> tuple[dict[str, str], dict[str, str]]:
    records = build_raw_waveform_test_records(test_ids, analysis_lookup)
    label_by_id = _unique_labels_by_id(records)
    id_by_label = {label: test_id for test_id, label in label_by_id.items()}
    return label_by_id, id_by_label


def format_reference_test_label(test_id: str, analysis_lookup: dict) -> str:
    if test_id not in analysis_lookup:
        return test_id
    label_by_id, _ = build_raw_waveform_label_lookup([test_id], analysis_lookup)
    return label_by_id.get(test_id, test_id)


def render_raw_waveforms_tab(test_ids: list[str], analysis_lookup: dict) -> None:
    records = build_raw_waveform_test_records(test_ids, analysis_lookup)
    if not records:
        st.warning("No parsed tests are available for raw waveform inspection.")
        return

    st.markdown("### Raw Waveforms Data Audit")
    st.caption(
        "Use this screen to manually inspect LUT, transient, and raw waveform data. "
        "Dropdown labels are metadata-based; internal IDs are shown separately."
    )

    filtered_records = _render_raw_waveform_filters(records)
    if not filtered_records:
        st.warning("No tests match the current Raw Waveforms filters.")
        return

    label_by_id = _unique_labels_by_id(filtered_records)
    id_by_label = {label: test_id for test_id, label in label_by_id.items()}
    selected_label = st.selectbox(
        "테스트 선택 (metadata label)",
        options=[label_by_id[record.test_id] for record in filtered_records],
        key="raw_test_audit",
    )
    selected_test_id = id_by_label[selected_label]
    selected_record = next(record for record in filtered_records if record.test_id == selected_test_id)
    selected_analysis = analysis_lookup[selected_test_id]

    dataset_mode = st.radio(
        "Waveform data view",
        options=["corrected", "raw"],
        format_func=lambda value: "Corrected/preprocessed" if value == "corrected" else "Raw normalized parse",
        horizontal=True,
        key="raw_dataset_audit",
    )
    display_frame = (
        selected_analysis.preprocess.corrected_frame
        if dataset_mode == "corrected"
        else selected_analysis.parsed.normalized_frame
    )

    _render_selected_test_summary(selected_record, dataset_mode, display_frame)
    _render_raw_waveform_plot(selected_record, dataset_mode, display_frame)


def _build_raw_waveform_test_record(test_id: str, analysis: Any) -> RawWaveformTestRecord:
    parsed = analysis.parsed
    normalized = parsed.normalized_frame
    corrected = analysis.preprocess.corrected_frame
    metadata = getattr(parsed, "metadata", {}) or {}

    source_file = str(getattr(parsed, "source_file", "") or _first_nonempty(normalized, "source_file") or "")
    source_file_label = display_source_file_name(source_file)
    sheet_name = str(getattr(parsed, "sheet_name", "") or _first_nonempty(normalized, "sheet_name") or "")
    waveform_type = str(
        _first_nonempty(normalized, "waveform_type")
        or _metadata_value(metadata, "waveform", "waveform_type")
        or "unknown"
    )
    freq_hz = _first_numeric(normalized, "freq_hz")
    if not np.isfinite(freq_hz):
        freq_hz = _first_metadata_number(metadata, "frequency(Hz)", "freq_hz", "frequency")
    target_current_a = _first_numeric(normalized, "current_pp_target_a")
    if not np.isfinite(target_current_a):
        target_current_a = _first_metadata_number(metadata, "Target Current(A)", "target_current_a", "current")
    cycle_count = _first_numeric(normalized, "cycle_total_expected")
    if not np.isfinite(cycle_count):
        cycle_count = _first_metadata_number(metadata, "cycle", "cycle_count", "cycle_hint")

    duration_s = _duration_seconds(corrected if not corrected.empty else normalized)
    sample_count = int(len(corrected if not corrected.empty else normalized))
    sampling_rate_hz = _sampling_rate_hz(corrected if not corrected.empty else normalized)
    source_type = _infer_source_type(source_file, sheet_name, cycle_count, duration_s, freq_hz)

    record = RawWaveformTestRecord(
        test_id=str(test_id),
        label="",
        source_file=source_file,
        source_file_label=source_file_label,
        sheet_name=sheet_name,
        waveform_type=waveform_type,
        freq_hz=freq_hz,
        cycle_count=cycle_count,
        target_current_a=target_current_a,
        source_type=source_type,
        sample_count=sample_count,
        duration_s=duration_s,
        sampling_rate_hz=sampling_rate_hz,
    )
    return RawWaveformTestRecord(**{**record.__dict__, "label": _format_raw_waveform_label(record)})


def _render_raw_waveform_filters(records: list[RawWaveformTestRecord]) -> list[RawWaveformTestRecord]:
    with st.container():
        st.markdown("#### Find test data")
        search_text = st.text_input(
            "Search metadata label / source file",
            key="raw_waveform_search",
            placeholder="waveform, frequency, current/App, source file, sheet, internal ID",
        ).strip()
        filter_columns = st.columns(5)
        waveform_filter = filter_columns[0].multiselect(
            "Waveform family",
            options=sorted({record.waveform_type for record in records if record.waveform_type}),
            key="raw_filter_waveform",
        )
        frequency_filter = filter_columns[1].multiselect(
            "Frequency (Hz)",
            options=_unique_number_labels(record.freq_hz for record in records),
            key="raw_filter_frequency",
        )
        cycle_filter = filter_columns[2].multiselect(
            "Cycle count",
            options=_unique_number_labels(record.cycle_count for record in records),
            key="raw_filter_cycle",
        )
        current_filter = filter_columns[3].multiselect(
            "Current/App",
            options=_unique_number_labels(record.target_current_a for record in records),
            key="raw_filter_current",
        )
        source_type_filter = filter_columns[4].multiselect(
            "Source type",
            options=sorted({record.source_type for record in records if record.source_type}),
            key="raw_filter_source_type",
        )

    filtered = records
    if search_text:
        lowered = search_text.lower()
        filtered = [
            record
            for record in filtered
            if lowered
            in " ".join(
                [
                    record.label,
                    record.test_id,
                    record.source_file,
                    record.sheet_name,
                    record.source_type,
                ]
            ).lower()
        ]
    if waveform_filter:
        filtered = [record for record in filtered if record.waveform_type in waveform_filter]
    if frequency_filter:
        allowed = {_number_label(record.freq_hz) for record in records if _number_label(record.freq_hz) in frequency_filter}
        filtered = [record for record in filtered if _number_label(record.freq_hz) in allowed]
    if cycle_filter:
        allowed = {_number_label(record.cycle_count) for record in records if _number_label(record.cycle_count) in cycle_filter}
        filtered = [record for record in filtered if _number_label(record.cycle_count) in allowed]
    if current_filter:
        allowed = {_number_label(record.target_current_a) for record in records if _number_label(record.target_current_a) in current_filter}
        filtered = [record for record in filtered if _number_label(record.target_current_a) in allowed]
    if source_type_filter:
        filtered = [record for record in filtered if record.source_type in source_type_filter]

    st.caption(f"Showing {len(filtered)} / {len(records)} tests. Sorted by waveform, frequency, cycle, current, source file.")
    return filtered


def _render_selected_test_summary(
    record: RawWaveformTestRecord,
    dataset_mode: str,
    display_frame: pd.DataFrame,
) -> None:
    st.markdown("#### Selected Data Summary")
    st.caption(f"Internal ID: `{record.test_id}`")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Source file", record.source_file_label or "unknown")
    metric_columns[1].metric("Waveform family", _display_value(record.waveform_type))
    metric_columns[2].metric("Frequency", _format_number_with_unit(record.freq_hz, "Hz"))
    metric_columns[3].metric("Current/App", _format_number_with_unit(record.target_current_a, "App"))

    detail_columns = st.columns(4)
    detail_columns[0].metric("Cycle count", _format_number_with_unit(record.cycle_count, "cycle"))
    detail_columns[1].metric("Source type", record.source_type)
    detail_columns[2].metric("Rows", str(len(display_frame)))
    detail_columns[3].metric("Sampling rate", _format_number_with_unit(record.sampling_rate_hz, "Hz"))

    st.info(
        "Viewing corrected/preprocessed waveform data."
        if dataset_mode == "corrected"
        else "Viewing raw normalized parse data before preprocessing correction."
    )
    with st.expander("Internal/debug identifiers", expanded=False):
        st.write(f"- source_file: `{record.source_file}`")
        st.write(f"- sheet_name: `{record.sheet_name}`")
        st.write(f"- internal_id: `{record.test_id}`")
        st.write(f"- duration_s: `{_number_label(record.duration_s)}`")


def _render_raw_waveform_plot(
    record: RawWaveformTestRecord,
    dataset_mode: str,
    display_frame: pd.DataFrame,
) -> None:
    default_channels = [
        "daq_input_v",
        "coil1_current_a",
        "coil2_current_a",
        "temperature_c",
        "bx_mT",
        "by_mT",
        "bz_mT",
        "bmag_mT",
    ]
    plottable_columns = [
        column
        for column in display_frame.columns
        if column not in _METADATA_HIDDEN_COLUMNS and pd.api.types.is_numeric_dtype(display_frame[column])
    ]
    selected_channels = st.multiselect(
        "Signals to inspect",
        options=plottable_columns,
        default=[channel for channel in default_channels if channel in plottable_columns],
        key="raw_channels_audit",
    )
    st.caption(
        f"Plot view: {dataset_mode} · source={record.source_file_label or 'unknown'} · "
        f"signals={', '.join(selected_channels) if selected_channels else 'none selected'}"
    )
    if not selected_channels:
        st.warning("Select at least one numeric signal to plot.")
        return
    st.plotly_chart(
        plot_waveforms(display_frame, selected_channels, title=f"{record.label} / {dataset_mode}"),
        use_container_width=True,
    )
    st.dataframe(display_frame.head(200), use_container_width=True)


def _format_raw_waveform_label(record: RawWaveformTestRecord) -> str:
    parts = [
        _display_value(record.source_type),
        _display_value(record.waveform_type).title(),
        _format_number_with_unit(record.freq_hz, "Hz"),
    ]
    if record.source_type == "finite-cycle" and np.isfinite(record.cycle_count) and record.cycle_count > 0:
        parts.append(_format_number_with_unit(record.cycle_count, "cycle"))
    if np.isfinite(record.target_current_a):
        parts.append(_format_number_with_unit(record.target_current_a, "App"))
    if record.source_file_label:
        parts.append(record.source_file_label)
    return " | ".join(part for part in parts if part and part != "unknown")


def _unique_labels_by_id(records: list[RawWaveformTestRecord]) -> dict[str, str]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.label] = counts.get(record.label, 0) + 1

    seen: dict[str, int] = {}
    label_by_id: dict[str, str] = {}
    for record in records:
        label = record.label
        if counts[label] > 1:
            sheet_suffix = f" | sheet {record.sheet_name}" if record.sheet_name else ""
            seen[label] = seen.get(label, 0) + 1
            label = f"{label}{sheet_suffix} | item {seen[record.label]}"
        label_by_id[record.test_id] = label
    return label_by_id


def _raw_waveform_record_sort_key(record: RawWaveformTestRecord) -> tuple:
    return (
        str(record.waveform_type).lower(),
        _finite_or_large(record.freq_hz),
        _finite_or_large(record.cycle_count),
        _finite_or_large(record.target_current_a),
        str(record.source_file_label).lower(),
    )


def _looks_like_opaque_prefix(value: str) -> bool:
    return _OPAQUE_PREFIX_MIN_LEN <= len(value) <= _OPAQUE_PREFIX_MAX_LEN and all(
        character in "0123456789abcdefABCDEF" for character in value
    )


def _first_nonempty(frame: pd.DataFrame, column: str) -> object | None:
    if column not in frame.columns:
        return None
    values = frame[column].dropna()
    if values.empty:
        return None
    value = values.iloc[0]
    return None if str(value).strip() == "" else value


def _first_numeric(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else float("nan")


def _metadata_value(metadata: dict[str, Any], *keys: str) -> object | None:
    normalized = {str(key).strip().lower(): value for key, value in metadata.items()}
    for key in keys:
        value = normalized.get(key.strip().lower())
        if value not in (None, ""):
            return value
    return None


def _first_metadata_number(metadata: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = _metadata_value(metadata, key)
        parsed = first_number(value)
        if parsed is not None and np.isfinite(parsed):
            return float(parsed)
    return float("nan")


def _duration_seconds(frame: pd.DataFrame) -> float:
    if "time_s" not in frame.columns or frame.empty:
        return float("nan")
    values = pd.to_numeric(frame["time_s"], errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.max() - values.min())


def _sampling_rate_hz(frame: pd.DataFrame) -> float:
    if "time_s" not in frame.columns or len(frame) < 2:
        return float("nan")
    time_values = pd.to_numeric(frame["time_s"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(time_values) < 2:
        return float("nan")
    diffs = np.diff(time_values)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return float("nan")
    return float(1.0 / np.median(diffs))


def _infer_source_type(
    source_file: str,
    sheet_name: str,
    cycle_count: float,
    duration_s: float,
    freq_hz: float,
) -> str:
    text = f"{source_file} {sheet_name}".lower()
    if np.isfinite(cycle_count) and 0.0 < cycle_count <= 3.0:
        return "finite-cycle"
    if np.isfinite(cycle_count) and cycle_count == 0.0:
        return "continuous"
    if any(token in text for token in ("finite", "transient", "stop")):
        return "finite-cycle"
    if any(token in text for token in ("continuous", "steady", "steadystate", "steady-state")):
        return "continuous"
    approx_cycles = duration_s * freq_hz if np.isfinite(duration_s) and np.isfinite(freq_hz) else float("nan")
    if np.isfinite(approx_cycles) and approx_cycles <= 3.0:
        return "finite-cycle"
    return "continuous"


def _unique_number_labels(values) -> list[str]:
    labels = {_number_label(value) for value in values if np.isfinite(value)}
    return sorted(labels, key=lambda value: float(value))


def _number_label(value: float) -> str:
    return f"{float(value):g}" if np.isfinite(value) else ""


def _format_number_with_unit(value: float, unit: str) -> str:
    return f"{float(value):g} {unit}" if np.isfinite(value) else "unknown"


def _display_value(value: object) -> str:
    text = str(value or "").strip()
    return text if text else "unknown"


def _finite_or_large(value: float) -> float:
    return float(value) if np.isfinite(value) else float("inf")
