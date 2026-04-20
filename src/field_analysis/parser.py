from __future__ import annotations

import csv
from io import BytesIO, StringIO
from pathlib import Path
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
    return mapping


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
    if amp_gain_setting is not None and normalized["amp_gain_setting"].isna().all():
        normalized["amp_gain_setting"] = amp_gain_setting

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

    warnings.extend(_series_quality_warnings(normalized))
    normalized["parse_warnings"] = flatten_messages(warnings)
    logs.append(
        f"{source_file}/{sheet_name}: waveform={waveform_type}, freq={freq_hz}, target_mode={current_mode_inferred}"
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
        warnings.append("시간 컬럼을 찾지 못해 sample index 를 time_s 로 사용합니다.")
        return

    raw_series = frame[source_column]
    dt_series = pd.to_datetime(raw_series, errors="coerce")
    dt_valid_ratio = float(dt_series.notna().mean())
    if dt_valid_ratio >= 0.5:
        normalized["timestamp"] = dt_series
        normalized["time_s"] = (dt_series - dt_series.iloc[0]).dt.total_seconds()
        return

    numeric = pd.to_numeric(raw_series, errors="coerce")
    normalized["timestamp"] = pd.NaT
    if numeric.notna().any():
        normalized["time_s"] = numeric - numeric.iloc[0]
        return

    normalized["time_s"] = np.arange(len(frame), dtype=float)
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
