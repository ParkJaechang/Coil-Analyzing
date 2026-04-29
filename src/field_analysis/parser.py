from __future__ import annotations

import csv
from io import BytesIO, StringIO
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from .models import FilePreview, ParsedMeasurement, SchemaConfig, SheetPreview
from .utils import (
    choose_best_match,
    coerce_float,
    column_stats,
    combine_temperature_columns,
    first_number,
    flatten_messages,
    infer_conditions_from_filename,
    infer_current_from_text,
    infer_frequency_from_text,
    infer_waveform_from_text,
    make_test_id,
    normalize_name,
    reconstruct_signed_current_channels,
)

SUPPORTED_FILE_SUFFIXES = {".csv", ".txt", ".xlsx", ".xlsm", ".xls"}
PARSER_VERSION = "parser_timebase_v2"
CONTINUOUS_FILENAME_PATTERN = re.compile(
    r"^continuous_(?P<waveform>sine|triangle)_(?P<freq>\d+(?:[._p]\d+)?)hz$",
    flags=re.IGNORECASE,
)
FINITE_FILENAME_PATTERN = re.compile(
    r"^finite_(?P<waveform>sine|triangle)_(?P<freq>\d+(?:[._p]\d+)?)hz_(?P<cycle>\d+(?:[._p]\d+)?)cycle$",
    flags=re.IGNORECASE,
)


def _parse_filename_decimal(token: str | None) -> float | None:
    if not token:
        return None
    try:
        return float(str(token).lower().replace("p", ".").replace("_", "."))
    except ValueError:
        return None


def infer_dataset_filename_metadata(file_name: str) -> dict[str, Any]:
    """Infer dataset metadata from the new sync-friendly filename patterns."""

    stem = Path(file_name).stem
    finite_match = FINITE_FILENAME_PATTERN.match(stem)
    if finite_match is not None:
        waveform_type = str(finite_match.group("waveform")).lower()
        freq_hz = _parse_filename_decimal(finite_match.group("freq"))
        cycle_count = _parse_filename_decimal(finite_match.group("cycle"))
        return {
            "source_type": "finite_cycle",
            "waveform": waveform_type,
            "waveform_type": waveform_type,
            "freq_hz": freq_hz,
            "cycle": cycle_count,
            "cycle_count": cycle_count,
            "daq_amplitude_v": 5.0,
            "daq_pp_v": 10.0,
            "dcamp_gain_percent": 100.0,
            "gain": 100.0,
            "target_current_a": None,
            "Target Current(A)": None,
            "filename_metadata_inferred": True,
        }

    continuous_match = CONTINUOUS_FILENAME_PATTERN.match(stem)
    if continuous_match is not None:
        waveform_type = str(continuous_match.group("waveform")).lower()
        freq_hz = _parse_filename_decimal(continuous_match.group("freq"))
        return {
            "source_type": "continuous",
            "waveform": waveform_type,
            "waveform_type": waveform_type,
            "freq_hz": freq_hz,
            "cycle": None,
            "cycle_count": None,
            "daq_amplitude_v": 5.0,
            "daq_pp_v": 10.0,
            "dcamp_gain_percent": 100.0,
            "gain": 100.0,
            "target_current_a": None,
            "Target Current(A)": None,
            "filename_metadata_inferred": True,
        }

    return {"filename_metadata_inferred": False}


def decode_text_bytes(data: bytes) -> str:
    """Decode bytes using common UTF-8 and Korean encodings."""

    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def preview_measurement_file(
    file_name: str,
    file_bytes: bytes,
    schema: SchemaConfig,
) -> FilePreview:
    """Create a structure preview for CSV or Excel input."""

    suffix = Path(file_name).suffix.lower()
    if suffix not in SUPPORTED_FILE_SUFFIXES:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {suffix}")

    if suffix in {".csv", ".txt"}:
        sheet_preview = _preview_delimited_text(
            file_name=file_name,
            file_bytes=file_bytes,
            schema=schema,
        )
        return FilePreview(
            file_name=file_name,
            file_type=suffix.lstrip("."),
            sheet_previews=[sheet_preview],
            warnings=list(sheet_preview.warnings),
            logs=list(sheet_preview.logs),
        )

    return _preview_excel_workbook(file_name=file_name, file_bytes=file_bytes, schema=schema)


def parse_measurement_file(
    file_name: str,
    file_bytes: bytes,
    schema: SchemaConfig,
    mapping_overrides: dict[str, dict[str, str | None]] | None = None,
    metadata_overrides: dict[str, dict[str, Any]] | None = None,
    expected_cycles: int | None = None,
    target_current_mode: str | None = None,
) -> list[ParsedMeasurement]:
    """Parse a file into one or more normalized single-test datasets."""

    preview = preview_measurement_file(file_name=file_name, file_bytes=file_bytes, schema=schema)
    results: list[ParsedMeasurement] = []
    cycles_expected = expected_cycles or schema.default_expected_cycles
    current_mode = target_current_mode or schema.target_current_mode

    for sheet_preview in preview.sheet_previews:
        if not sheet_preview.columns:
            continue

        mapping = dict(sheet_preview.recommended_mapping)
        if mapping_overrides and sheet_preview.sheet_name in mapping_overrides:
            mapping.update(mapping_overrides[sheet_preview.sheet_name])

        metadata = dict(sheet_preview.metadata)
        for key, value in infer_dataset_filename_metadata(file_name).items():
            metadata.setdefault(key, value)
        if metadata_overrides and sheet_preview.sheet_name in metadata_overrides:
            for key, value in metadata_overrides[sheet_preview.sheet_name].items():
                if value not in ("", None):
                    metadata[key] = value

        raw_frame = _load_sheet_frame(
            file_name=file_name,
            file_bytes=file_bytes,
            sheet_name=sheet_preview.sheet_name,
            header_row_index=sheet_preview.header_row_index,
            schema=schema,
        )
        normalized_frame, warnings, logs = _normalize_frame(
            raw_frame=raw_frame,
            source_file=file_name,
            sheet_name=sheet_preview.sheet_name,
            metadata=metadata,
            mapping=mapping,
            schema=schema,
            expected_cycles=cycles_expected,
            target_current_mode=current_mode,
        )

        merged_warnings = list(dict.fromkeys(sheet_preview.warnings + warnings))
        merged_logs = sheet_preview.logs + logs
        results.append(
            ParsedMeasurement(
                source_file=file_name,
                file_type=preview.file_type,
                sheet_name=sheet_preview.sheet_name,
                structure_preview=sheet_preview,
                metadata=metadata,
                mapping=mapping,
                raw_frame=raw_frame,
                normalized_frame=normalized_frame,
                warnings=merged_warnings,
                logs=merged_logs,
            )
        )

    return results


def build_mapping_table(
    mapping: dict[str, str | None],
    schema: SchemaConfig,
) -> pd.DataFrame:
    """Return a UI/export-friendly mapping table."""

    rows: list[dict[str, Any]] = []
    for key, spec in schema.field_specs.items():
        rows.append(
            {
                "standard_field": key,
                "라벨": spec.label_ko,
                "unit": spec.unit or "",
                "required": spec.required,
                "source_column": mapping.get(key),
                "description": spec.description,
            }
        )
    return pd.DataFrame(rows)


def _preview_delimited_text(
    file_name: str,
    file_bytes: bytes,
    schema: SchemaConfig,
) -> SheetPreview:
    text = decode_text_bytes(file_bytes)
    delimiter = _guess_delimiter(text)
    reader = csv.reader(StringIO(text), delimiter=delimiter)

    metadata: dict[str, str] = {}
    header: list[str] | None = None
    data_rows: list[list[str]] = []
    header_row_index = 0
    logs = [f"{file_name}: 구분자 `{delimiter}` 로 감지"]

    for row_index, row in enumerate(reader):
        cleaned = [str(cell).strip() for cell in row]
        if not any(cleaned):
            continue
        if cleaned[0].startswith(schema.comment_prefix):
            key = cleaned[0].lstrip(schema.comment_prefix).strip()
            value = delimiter.join(cleaned[1:]).strip()
            metadata[key] = value
            continue
        header = cleaned
        header_row_index = row_index
        break

    if header is None:
        raise ValueError(f"{file_name}: 헤더 행을 찾지 못했습니다.")

    reader = csv.reader(StringIO(text), delimiter=delimiter)
    for row_index, row in enumerate(reader):
        if row_index <= header_row_index:
            continue
        cleaned = [str(cell).strip() for cell in row]
        if not any(cleaned):
            continue
        if len(cleaned) < len(header):
            cleaned = cleaned + [""] * (len(header) - len(cleaned))
        if len(cleaned) > len(header):
            cleaned = cleaned[: len(header)]
        data_rows.append(cleaned)

    preview_frame = pd.DataFrame(data_rows, columns=header)
    preview_frame = preview_frame.dropna(axis=1, how="all")
    preview_frame.columns = [str(column).strip() for column in preview_frame.columns]
    recommended_mapping = _recommend_mapping(preview_frame.columns.tolist(), schema)
    warnings = _structure_warnings(preview_frame)
    logs.append(f"{file_name}: 메타데이터 {len(metadata)}개, 데이터 행 {len(preview_frame)}개")

    return SheetPreview(
        sheet_name="main",
        row_count=len(preview_frame),
        column_count=len(preview_frame.columns),
        columns=preview_frame.columns.tolist(),
        header_row_index=header_row_index,
        metadata=metadata,
        preview_rows=preview_frame.head(5).to_dict(orient="records"),
        recommended_mapping=recommended_mapping,
        warnings=warnings,
        logs=logs,
    )


def _preview_excel_workbook(
    file_name: str,
    file_bytes: bytes,
    schema: SchemaConfig,
) -> FilePreview:
    workbook = pd.ExcelFile(BytesIO(file_bytes))
    sheet_previews: list[SheetPreview] = []
    warnings: list[str] = []
    logs = [f"{file_name}: 시트 {len(workbook.sheet_names)}개 감지"]

    for sheet_name in workbook.sheet_names:
        sample = pd.read_excel(
            BytesIO(file_bytes),
            sheet_name=sheet_name,
            header=None,
            nrows=schema.header_search_rows,
            dtype=object,
        )
        header_row_index = _detect_header_row(sample, schema)
        metadata = _extract_excel_metadata(sample, header_row_index, schema)
        frame = pd.read_excel(
            BytesIO(file_bytes),
            sheet_name=sheet_name,
            header=header_row_index,
            dtype=object,
        )
        frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
        frame.columns = [str(column).strip() for column in frame.columns]
        recommended_mapping = _recommend_mapping(frame.columns.tolist(), schema)
        sheet_warnings = _structure_warnings(frame)
        sheet_logs = [
            f"{sheet_name}: 헤더 행 {header_row_index + 1}",
            f"{sheet_name}: 열 {len(frame.columns)}개, 행 {len(frame)}개",
        ]
        sheet_previews.append(
            SheetPreview(
                sheet_name=sheet_name,
                row_count=len(frame),
                column_count=len(frame.columns),
                columns=frame.columns.tolist(),
                header_row_index=header_row_index,
                metadata=metadata,
                preview_rows=frame.head(5).to_dict(orient="records"),
                recommended_mapping=recommended_mapping,
                warnings=sheet_warnings,
                logs=sheet_logs,
            )
        )

    if not sheet_previews:
        warnings.append("시트에서 분석 가능한 표 구조를 찾지 못했습니다.")

    return FilePreview(
        file_name=file_name,
        file_type=Path(file_name).suffix.lower().lstrip("."),
        sheet_previews=sheet_previews,
        warnings=warnings,
        logs=logs,
    )


def _guess_delimiter(text: str) -> str:
    sample = "\n".join(text.splitlines()[:20])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def _structure_warnings(frame: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    if frame.empty:
        warnings.append("데이터 본문이 비어 있습니다.")
        return warnings
    null_ratio = frame.isna().mean().mean()
    if null_ratio > 0.2:
        warnings.append(f"빈 값 비율이 높습니다. 평균 null ratio={null_ratio:.1%}")
    duplicate_columns = frame.columns[frame.columns.duplicated()].tolist()
    if duplicate_columns:
        warnings.append(f"중복 컬럼이 있습니다: {', '.join(duplicate_columns)}")
    return warnings


def _detect_header_row(sample: pd.DataFrame, schema: SchemaConfig) -> int:
    alias_pool = {
        normalize_name(alias)
        for aliases in schema.column_aliases.values()
        for alias in aliases
    }

    best_index = 0
    best_score = -1
    for row_index in range(len(sample)):
        values = [str(value).strip() for value in sample.iloc[row_index].tolist()]
        score = 0
        for value in values:
            if not value or value == "nan":
                continue
            normalized = normalize_name(value)
            if normalized in alias_pool:
                score += 3
            elif any(alias and alias in normalized for alias in alias_pool):
                score += 1
        if score > best_score:
            best_index = row_index
            best_score = score

    return best_index


def _extract_excel_metadata(
    sample: pd.DataFrame,
    header_row_index: int,
    schema: SchemaConfig,
) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for row_index in range(header_row_index):
        row = [value for value in sample.iloc[row_index].tolist() if pd.notna(value)]
        if not row:
            continue
        key = str(row[0]).strip()
        value = " ".join(str(item).strip() for item in row[1:])
        if key.startswith(schema.comment_prefix):
            key = key.lstrip(schema.comment_prefix).strip()
        metadata[key] = value
    return metadata


def _recommend_mapping(columns: list[str], schema: SchemaConfig) -> dict[str, str | None]:
    mapping: dict[str, str | None] = {}
    for key, aliases in schema.column_aliases.items():
        mapping[key] = choose_best_match(columns, aliases)
    return _sanitize_recommended_mapping(mapping)


def _sanitize_recommended_mapping(mapping: dict[str, str | None]) -> dict[str, str | None]:
    sanitized = dict(mapping)
    for key, value in list(sanitized.items()):
        if key.startswith("temperature") and value is not None and _looks_like_signal_column(value):
            sanitized[key] = None
    return sanitized


def _looks_like_signal_column(column: str) -> bool:
    normalized = normalize_name(column)
    signal_tokens = (
        "current",
        "voltage",
        "daq",
        "hall",
        "bmt",
        "bzm",
        "bxm",
        "bym",
        "bx",
        "by",
        "bz",
    )
    return any(token in normalized for token in signal_tokens)


def _load_sheet_frame(
    file_name: str,
    file_bytes: bytes,
    sheet_name: str,
    header_row_index: int,
    schema: SchemaConfig,
) -> pd.DataFrame:
    suffix = Path(file_name).suffix.lower()
    if suffix in {".csv", ".txt"}:
        preview = _preview_delimited_text(file_name=file_name, file_bytes=file_bytes, schema=schema)
        frame = pd.DataFrame(preview.preview_rows)
        full_text = decode_text_bytes(file_bytes)
        delimiter = _guess_delimiter(full_text)
        reader = csv.reader(StringIO(full_text), delimiter=delimiter)
        header: list[str] | None = None
        rows: list[list[str]] = []
        for row_index, row in enumerate(reader):
            if row_index < header_row_index:
                continue
            cleaned = [str(cell).strip() for cell in row]
            if row_index == header_row_index:
                header = cleaned
                continue
            if not any(cleaned):
                continue
            if header is None:
                raise ValueError("CSV 헤더를 다시 읽는 중 실패했습니다.")
            if len(cleaned) < len(header):
                cleaned = cleaned + [""] * (len(header) - len(cleaned))
            if len(cleaned) > len(header):
                cleaned = cleaned[: len(header)]
            rows.append(cleaned)
        if header is None:
            raise ValueError("CSV 헤더가 없습니다.")
        frame = pd.DataFrame(rows, columns=header)
        frame.columns = [str(column).strip() for column in frame.columns]
        return frame

    frame = pd.read_excel(
        BytesIO(file_bytes),
        sheet_name=sheet_name,
        header=header_row_index,
        dtype=object,
    )
    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _normalize_frame(
    raw_frame: pd.DataFrame,
    source_file: str,
    sheet_name: str,
    metadata: dict[str, Any],
    mapping: dict[str, str | None],
    schema: SchemaConfig,
    expected_cycles: int,
    target_current_mode: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    warnings: list[str] = []
    logs: list[str] = [f"{source_file}/{sheet_name}: 정규화 시작"]
    frame = raw_frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]

    normalized = pd.DataFrame(index=frame.index)
    normalized["source_file"] = source_file
    normalized["sheet_name"] = sheet_name
    normalized["sample_index"] = np.arange(len(frame), dtype=int)

    _assign_time_columns(frame, normalized, mapping, warnings)
    _assign_numeric_column(frame, normalized, mapping, "daq_input_v")
    _assign_numeric_column(frame, normalized, mapping, "daq_input_v_secondary")
    _assign_numeric_column(frame, normalized, mapping, "coil1_current_a")
    _assign_numeric_column(frame, normalized, mapping, "coil2_current_a")
    _assign_numeric_column(frame, normalized, mapping, "coil1_peak_a")
    _assign_numeric_column(frame, normalized, mapping, "coil2_peak_a")
    _assign_numeric_column(frame, normalized, mapping, "temperature_t1_c")
    _assign_numeric_column(frame, normalized, mapping, "temperature_t2_c")
    _assign_numeric_column(frame, normalized, mapping, "temperature_t3_c")
    _assign_numeric_column(frame, normalized, mapping, "temperature_t4_c")
    _assign_numeric_column(frame, normalized, mapping, "temperature_c")
    _assign_numeric_column(frame, normalized, mapping, "source_cycle_no")
    _assign_numeric_column(frame, normalized, mapping, "bx_mT")
    _assign_numeric_column(frame, normalized, mapping, "by_mT")
    _assign_numeric_column(frame, normalized, mapping, "bz_mT")
    _assign_numeric_column(frame, normalized, mapping, "bx_peak_mT")
    _assign_numeric_column(frame, normalized, mapping, "by_peak_mT")
    _assign_numeric_column(frame, normalized, mapping, "bz_peak_mT")
    _assign_numeric_column(frame, normalized, mapping, "amp_gain_setting")

    if normalized["temperature_c"].isna().all():
        normalized["temperature_c"] = combine_temperature_columns(
            normalized,
            ("temperature_t1_c", "temperature_t2_c", "temperature_t3_c", "temperature_t4_c"),
        )

    if normalized["daq_input_v"].isna().all() and "daq_input_v_secondary" in normalized.columns:
        normalized["daq_input_v"] = normalized["daq_input_v_secondary"]

    normalized["i_sum"] = normalized[["coil1_current_a", "coil2_current_a"]].sum(
        axis=1,
        min_count=1,
    )
    normalized["i_diff"] = normalized["coil1_current_a"] - normalized["coil2_current_a"]
    normalized["i_custom"] = normalized["i_sum"]
    signed_current_info = reconstruct_signed_current_channels(normalized)
    normalized["bmag_mT"] = np.sqrt(
        np.square(normalized["bx_mT"].fillna(0.0))
        + np.square(normalized["by_mT"].fillna(0.0))
        + np.square(normalized["bz_mT"].fillna(0.0))
    )
    normalized["bproj_mT"] = normalized["bz_mT"]
    filename_conditions = infer_conditions_from_filename(source_file, sheet_name)

    waveform_type = _extract_metadata_value(metadata, schema.metadata_aliases.get("waveform_type", ()))
    waveform_type = infer_waveform_from_text(waveform_type, source_file, sheet_name)

    metadata_freq = _extract_metadata_float(metadata, schema.metadata_aliases.get("freq_hz", ()))
    file_freq = infer_frequency_from_text(source_file, sheet_name)
    freq_hz = metadata_freq if metadata_freq is not None else file_freq
    if metadata_freq is not None and file_freq is not None and abs(metadata_freq - file_freq) > 0.05:
        warnings.append(
            f"메타데이터 주파수 {metadata_freq:g}Hz 와 파일명 주파수 {file_freq:g}Hz 가 다릅니다."
        )

    raw_target_current = _extract_metadata_float(
        metadata,
        schema.metadata_aliases.get("target_current_a", ()),
    )
    if raw_target_current is None or raw_target_current <= 0:
        raw_target_current = infer_current_from_text(source_file, sheet_name)
    if raw_target_current is None and filename_conditions.get("current_target_a") is not None:
        raw_target_current = float(filename_conditions["current_target_a"])
    effective_target_current_mode = target_current_mode
    if effective_target_current_mode == "auto" and filename_conditions.get("current_target_mode") in {"peak", "pp"}:
        effective_target_current_mode = str(filename_conditions["current_target_mode"])

    current_mode_inferred, current_pk_target, current_pp_target = _infer_target_current_values(
        normalized,
        raw_target_current,
        effective_target_current_mode,
    )

    cycle_hint = _extract_metadata_float(metadata, schema.metadata_aliases.get("cycle_hint", ()))
    if cycle_hint is None or cycle_hint <= 0:
        cycle_hint = coerce_float(filename_conditions.get("cycle_count"))
    cycle_total_expected = float(expected_cycles)
    if cycle_hint is not None and cycle_hint > 0:
        cycle_total_expected = float(cycle_hint)

    amp_gain_setting = _extract_metadata_float(
        metadata,
        schema.metadata_aliases.get("amp_gain_setting", ()),
    )
    if amp_gain_setting is None:
        amp_gain_setting = coerce_float(metadata.get("dcamp_gain_percent"))
    if amp_gain_setting is not None and normalized["amp_gain_setting"].isna().all():
        normalized["amp_gain_setting"] = amp_gain_setting

    source_type = str(metadata.get("source_type") or "").strip() or (
        "finite_cycle" if cycle_hint is not None and cycle_hint > 0 else "continuous"
    )
    daq_amplitude_v = coerce_float(metadata.get("daq_amplitude_v"))
    daq_pp_v = coerce_float(metadata.get("daq_pp_v"))
    dcamp_gain_percent = coerce_float(metadata.get("dcamp_gain_percent"))
    filename_metadata_inferred = bool(metadata.get("filename_metadata_inferred"))

    notes_value = _extract_metadata_value(metadata, schema.metadata_aliases.get("notes", ()))
    test_id = make_test_id(
        source_file=source_file,
        sheet_name=sheet_name,
        waveform_type=waveform_type,
        freq_hz=freq_hz,
        current_pp_target_a=current_pp_target,
    )

    normalized["test_id"] = test_id
    normalized["waveform_type"] = waveform_type
    normalized["freq_hz"] = freq_hz
    normalized["current_pp_target_a"] = current_pp_target
    normalized["current_pk_target_a"] = current_pk_target
    normalized["cycle_total_expected"] = cycle_total_expected
    normalized["source_type"] = source_type
    normalized["cycle_count"] = cycle_hint if cycle_hint is not None and cycle_hint > 0 else np.nan
    normalized["daq_amplitude_v"] = daq_amplitude_v
    normalized["daq_pp_v"] = daq_pp_v
    normalized["dcamp_gain_percent"] = dcamp_gain_percent
    normalized["filename_metadata_inferred"] = filename_metadata_inferred
    normalized["parser_version"] = PARSER_VERSION
    normalized["detected_format"] = _detect_table_format(source_file=source_file, frame=frame, metadata=metadata)
    normalized["sample_rate_hz"] = _estimate_sample_rate_hz(normalized["time_s"])
    if amp_gain_setting is not None:
        normalized["amp_gain_setting"] = normalized["amp_gain_setting"].fillna(amp_gain_setting)
    normalized["notes"] = notes_value or ""
    normalized["target_current_mode_inferred"] = current_mode_inferred

    required_missing = [
        spec.label_ko
        for spec in schema.field_specs.values()
        if spec.required and not mapping.get(spec.key)
    ]
    if required_missing:
        warnings.append(f"필수 매핑 누락: {', '.join(required_missing)}")

    quality_flags = _build_timebase_quality_flags(
        normalized=normalized,
        mapping=mapping,
        freq_hz=freq_hz,
        cycle_count=cycle_hint,
    )
    normalized["parse_quality_flags"] = "|".join(quality_flags)
    warnings.extend(_series_quality_warnings(normalized))
    normalized["parse_warnings"] = flatten_messages(warnings)
    logs.append(
        f"{source_file}/{sheet_name}: waveform={waveform_type}, freq={freq_hz}, source_type={source_type}, target_mode={current_mode_inferred}"
    )
    if signed_current_info["reconstructed_columns"]:
        reconstructed = ", ".join(str(value) for value in signed_current_info["reconstructed_columns"])
        reference_channel = signed_current_info["reference_channel"]
        logs.append(
            f"{source_file}/{sheet_name}: signed current reconstruction on [{reconstructed}] using {reference_channel}"
        )
    logs.append(f"{source_file}/{sheet_name}: 정규화 행 수 {len(normalized)}")
    return normalized, warnings, logs


def _assign_time_columns(
    frame: pd.DataFrame,
    normalized: pd.DataFrame,
    mapping: dict[str, str | None],
    warnings: list[str],
) -> None:
    source_column = mapping.get("timestamp")
    if source_column is None or source_column not in frame.columns:
        normalized["timestamp"] = pd.NaT
        normalized["time_s"] = np.arange(len(frame), dtype=float)
        normalized["timebase_source"] = "sample_index_only"
        normalized["time_unit"] = "sample_index"
        warnings.append("시간 컬럼을 찾지 못해 sample index 를 time_s 로 사용합니다.")
        return

    raw_series = frame[source_column]
    numeric = pd.to_numeric(raw_series, errors="coerce")
    time_unit, scale_to_seconds = _infer_time_unit_from_column(source_column)
    if numeric.notna().any() and scale_to_seconds is not None:
        normalized["timestamp"] = pd.NaT
        normalized["time_s"] = (numeric - numeric.dropna().iloc[0]) * scale_to_seconds
        normalized["timebase_source"] = "explicit_time_column"
        normalized["time_unit"] = time_unit
        return

    dt_series = pd.to_datetime(raw_series, errors="coerce")
    dt_valid_ratio = float(dt_series.notna().mean())
    if dt_valid_ratio >= 0.5:
        normalized["timestamp"] = dt_series
        normalized["time_s"] = (dt_series - dt_series.iloc[0]).dt.total_seconds()
        normalized["timebase_source"] = "explicit_datetime_column"
        normalized["time_unit"] = "datetime"
        return

    normalized["timestamp"] = pd.NaT
    if numeric.notna().any():
        normalized["time_s"] = numeric - numeric.iloc[0]
        normalized["timebase_source"] = "explicit_time_column"
        normalized["time_unit"] = "seconds_assumed"
        return

    normalized["time_s"] = np.arange(len(frame), dtype=float)
    normalized["timebase_source"] = "sample_index_only"
    normalized["time_unit"] = "sample_index"
    warnings.append("시간 컬럼이 datetime/numeric 으로 해석되지 않아 sample index 를 사용합니다.")


def _assign_numeric_column(
    frame: pd.DataFrame,
    normalized: pd.DataFrame,
    mapping: dict[str, str | None],
    target_key: str,
) -> None:
    source_column = mapping.get(target_key)
    if source_column is None or source_column not in frame.columns:
        normalized[target_key] = np.nan
        return
    normalized[target_key] = pd.to_numeric(frame[source_column], errors="coerce")


def _infer_time_unit_from_column(source_column: str) -> tuple[str, float | None]:
    normalized = normalize_name(source_column)
    if normalized in {"timems", "timestampms", "elapsedms", "milliseconds", "millisecond"} or normalized.endswith("ms"):
        return "milliseconds", 0.001
    if normalized in {"timeus", "timestampus", "elapsedus", "microseconds", "microsecond"} or normalized.endswith("us"):
        return "microseconds", 0.000001
    if normalized in {"time", "times", "timestamp", "timestamps", "elapsed", "seconds", "second", "timesec", "timesecs"}:
        return "seconds", 1.0
    return "unknown", None


def _detect_table_format(*, source_file: str, frame: pd.DataFrame, metadata: dict[str, Any]) -> str:
    columns = {normalize_name(column) for column in frame.columns}
    if {"timems", "hallbx", "hallby", "hallbz"} & columns and {"voltage1v", "current1a"} <= columns:
        return "new_lut_csv"
    if bool(metadata.get("filename_metadata_inferred")):
        return "filename_metadata_csv"
    if Path(source_file).suffix.lower() in {".csv", ".txt"}:
        return "legacy_csv"
    if Path(source_file).suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        return "legacy_excel"
    return "unknown"


def _estimate_sample_rate_hz(time_s: pd.Series) -> float:
    values = pd.to_numeric(time_s, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return float("nan")
    diffs = np.diff(values)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return float("nan")
    return float(1.0 / np.median(diffs))


def _build_timebase_quality_flags(
    *,
    normalized: pd.DataFrame,
    mapping: dict[str, str | None],
    freq_hz: float | None,
    cycle_count: float | None,
) -> list[str]:
    flags: set[str] = set()
    time_values = pd.to_numeric(normalized.get("time_s"), errors="coerce").to_numpy(dtype=float)
    finite_time = time_values[np.isfinite(time_values)]
    time_unit = str(_first_column_value(normalized, "time_unit") or "")
    if time_unit == "sample_index":
        flags.add("SAMPLE_INDEX_TIMEBASE")
    if time_unit == "seconds_assumed":
        flags.add("UNKNOWN_TIME_UNIT")
    if len(finite_time) >= 2:
        diffs = np.diff(finite_time)
        valid_diffs = diffs[np.isfinite(diffs)]
        if np.any(valid_diffs <= 0):
            flags.add("TIME_NOT_MONOTONIC")
        positive = valid_diffs[valid_diffs > 0]
        if len(positive) >= 4:
            median = float(np.median(positive))
            if median > 0 and float(np.percentile(positive, 95) - np.percentile(positive, 5)) / median > 1.0:
                flags.add("TIME_INTERVAL_JITTER")
    if mapping.get("timestamp") is None:
        flags.add("NO_EXPLICIT_TIME_COLUMN")
    if np.isfinite(freq_hz or np.nan) and np.isfinite(cycle_count or np.nan) and freq_hz and cycle_count:
        duration = float(np.nanmax(finite_time) - np.nanmin(finite_time)) if len(finite_time) else float("nan")
        expected = float(cycle_count) / float(freq_hz)
        if np.isfinite(duration) and expected > 0 and duration > expected * 3.0:
            flags.add("POST_WINDOW_INCLUDED")
    current_peaks = _estimate_peak_count(normalized.get("i_sum"))
    field_peaks = _estimate_peak_count(normalized.get("bz_mT"))
    if current_peaks is not None and field_peaks is not None and abs(current_peaks - field_peaks) > max(2, current_peaks * 0.35):
        flags.add("FIELD_PEAK_COUNT_MISMATCH")
    return sorted(flags)


def _first_column_value(frame: pd.DataFrame, column: str) -> object | None:
    if column not in frame.columns or frame.empty:
        return None
    values = frame[column].dropna()
    return None if values.empty else values.iloc[0]


def _estimate_peak_count(series: pd.Series | None) -> int | None:
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 8:
        return None
    centered = values - float(np.nanmedian(values))
    amplitude = float(np.nanmax(centered) - np.nanmin(centered))
    if amplitude <= 1e-9:
        return 0
    threshold = max(amplitude * 0.08, 1e-9)
    signs = np.sign(np.where(np.abs(centered) < threshold, 0.0, centered))
    nonzero = signs[signs != 0.0]
    if len(nonzero) < 2:
        return 0
    return int(np.count_nonzero((nonzero[:-1] < 0) & (nonzero[1:] > 0)))


def _extract_metadata_value(metadata: dict[str, Any], aliases: tuple[str, ...]) -> str | None:
    normalized_aliases = {normalize_name(alias): alias for alias in aliases}
    for key, value in metadata.items():
        normalized_key = normalize_name(key)
        if normalized_key in normalized_aliases:
            return str(value).strip()
    return None


def _extract_metadata_float(metadata: dict[str, Any], aliases: tuple[str, ...]) -> float | None:
    value = _extract_metadata_value(metadata, aliases)
    return first_number(value)


def _infer_target_current_values(
    normalized: pd.DataFrame,
    raw_target_current: float | None,
    target_current_mode: str,
) -> tuple[str, float | None, float | None]:
    if raw_target_current is None or raw_target_current <= 0:
        return "unknown", None, None

    candidate_columns = [
        column
        for column in (
            "i_sum_signed",
            "coil2_current_signed_a",
            "coil1_current_signed_a",
            "i_diff_signed",
            "i_sum",
            "coil2_current_a",
            "coil1_current_a",
            "i_diff",
        )
        if column in normalized.columns
    ]
    candidate_stats = {
        column: column_stats(normalized[column])
        for column in candidate_columns
    }
    best_column = max(
        candidate_columns,
        key=lambda column: candidate_stats[column]["peak_to_peak"],
        default=None,
    )
    achieved_pp = candidate_stats[best_column]["peak_to_peak"] if best_column else float("nan")
    achieved_peak = (
        max(
            abs(candidate_stats[best_column]["peak"]),
            abs(candidate_stats[best_column]["valley"]),
        )
        if best_column
        else float("nan")
    )

    if target_current_mode == "peak":
        return "peak", raw_target_current, raw_target_current * 2.0
    if target_current_mode == "pp":
        return "pp", raw_target_current / 2.0, raw_target_current

    if np.isfinite(achieved_peak) and np.isfinite(achieved_pp):
        peak_error = abs(raw_target_current - achieved_peak)
        pp_error = abs(raw_target_current - achieved_pp)
        if peak_error <= pp_error:
            return "peak(auto)", raw_target_current, raw_target_current * 2.0
        return "pp(auto)", raw_target_current / 2.0, raw_target_current

    return "unknown", None, None


def _series_quality_warnings(frame: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    numeric_targets = (
        "coil1_current_a",
        "coil2_current_a",
        "daq_input_v",
        "temperature_c",
        "bx_mT",
        "by_mT",
        "bz_mT",
    )
    for column in numeric_targets:
        if column not in frame.columns:
            continue
        null_ratio = float(frame[column].isna().mean())
        if null_ratio > 0.2:
            warnings.append(f"{column} 결측 비율이 높습니다: {null_ratio:.1%}")

    if "time_s" in frame.columns:
        duplicated = int(frame["time_s"].duplicated().sum())
        if duplicated > 0:
            warnings.append(f"중복 시간축 샘플 {duplicated}개가 있습니다.")
        diffs = frame["time_s"].diff().dropna()
        if (diffs < 0).any():
            warnings.append("시간축이 단조 증가하지 않습니다.")

    if "coil1_current_a" in frame.columns and "coil2_current_a" in frame.columns:
        if frame["coil1_current_a"].std(ddof=0) < 1e-6:
            warnings.append("coil1_current_a 변화가 거의 없습니다.")
        if frame["coil2_current_a"].std(ddof=0) < 1e-6:
            warnings.append("coil2_current_a 변화가 거의 없습니다.")
        for column in ("coil1_current_a", "coil2_current_a"):
            numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
            if len(numeric) >= 8 and float((numeric < 0).mean()) <= 0.02 and float(numeric.abs().max()) > 0:
                warnings.append(f"{column} 값이 거의 양수로만 측정됩니다. signed reconstruction 결과를 확인하십시오.")

    return warnings
