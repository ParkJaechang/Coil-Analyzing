"""Input data discovery and loading."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import pandas as pd

from coil_analyzer.models import DatasetMeta
from coil_analyzer.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

ROLE_HINTS = {
    "time": ("time", "timestamp", "sec", "ms"),
    "voltage": ("voltage", "volt", "vout", "coil_v", "vcoil", "diff"),
    "current": ("current", "curr", "icoil", "coil_i", "shunt"),
    "magnetic": ("field", "gauss", "tesla", "hall", "bx", "by", "bz", "mt"),
}


def list_excel_sheets(path: Path) -> list[str]:
    if path.suffix.lower() != ".xlsx":
        return []
    try:
        workbook = pd.ExcelFile(path)
        return workbook.sheet_names
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Failed to inspect Excel sheets for %s: %s", path, exc)
        return []


def load_dataframe(path: Path, sheet_name: str | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_csv_with_fallback(path)
    if suffix == ".xlsx":
        return pd.read_excel(path, sheet_name=sheet_name or 0)
    raise ValueError(f"Unsupported file format: {path.suffix}")


def load_csv_with_fallback(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return pd.DataFrame()

    delimiter, header_row = infer_csv_layout(lines)
    candidates: list[dict[str, Any]] = [
        {"sep": delimiter, "skiprows": header_row, "engine": "python"},
        {"sep": None, "skiprows": header_row, "engine": "python"},
        {"sep": delimiter, "skiprows": 0, "engine": "python"},
        {"sep": None, "skiprows": 0, "engine": "python"},
    ]
    last_error: Exception | None = None
    for options in candidates:
        try:
            df = pd.read_csv(path, on_bad_lines="skip", **options)
            df = drop_empty_columns(df)
            if not df.empty and len(df.columns) >= 2:
                return df
        except Exception as exc:  # pragma: no cover - fallback chain
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise ValueError(f"Unable to parse CSV file: {path.name}")


def infer_csv_layout(lines: list[str]) -> tuple[str, int]:
    delimiters = [",", ";", "\t", "|"]
    best_delimiter = ","
    best_header_row = 0
    best_score = -1
    search_window = min(len(lines), 40)
    for delimiter in delimiters:
        split_rows = [_split_csv_line(line, delimiter) for line in lines[:search_window]]
        for header_row in range(min(12, len(split_rows))):
            header_width = len(split_rows[header_row])
            if header_width < 2:
                continue
            matching_rows = 0
            for row in split_rows[header_row + 1 :]:
                if len(row) == header_width:
                    matching_rows += 1
            score = header_width * 10 + matching_rows
            if matching_rows >= 2 and score > best_score:
                best_score = score
                best_delimiter = delimiter
                best_header_row = header_row
    return best_delimiter, best_header_row


def _split_csv_line(line: str, delimiter: str) -> list[str]:
    return next(csv.reader([line], delimiter=delimiter))


def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.dropna(axis=1, how="all")
    placeholder_columns = [column for column in cleaned.columns if str(column).lower().startswith("unnamed:")]
    return cleaned.drop(columns=placeholder_columns, errors="ignore")


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {
        column: str(column).strip().replace("\n", " ").replace("\r", " ")
        for column in df.columns
    }
    return df.rename(columns=renamed)


def infer_column_roles(columns: list[str]) -> dict[str, list[str]]:
    guesses: dict[str, list[str]] = {"time": [], "voltage": [], "current": [], "magnetic": []}
    for column in columns:
        lowered = column.lower()
        for role, hints in ROLE_HINTS.items():
            if any(hint in lowered for hint in hints):
                guesses[role].append(column)
        if lowered.endswith("_a") or lowered.endswith(" a"):
            guesses["current"].append(column)
        if "b" == lowered.strip():
            guesses["magnetic"].append(column)
    for role in guesses:
        guesses[role] = _dedupe_preserve_order(_sort_role_candidates(role, guesses[role]))
    return guesses


def _sort_role_candidates(role: str, candidates: list[str]) -> list[str]:
    def score(column: str) -> tuple[int, int]:
        lowered = column.lower()
        peak_penalty = 1 if "peak" in lowered else 0
        if role == "current":
            preferred = 0 if lowered.startswith("current") else 1
            return (peak_penalty, preferred)
        if role == "voltage":
            preferred = 0 if lowered.startswith("voltage") else 1
            return (peak_penalty, preferred)
        if role == "magnetic":
            hall_priority = 0 if lowered.startswith("hall") else 1
            return (peak_penalty, hall_priority)
        return (0, 0)

    return sorted(candidates, key=score)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def parse_metadata_from_name(file_name: str) -> dict[str, Any]:
    lowered = file_name.lower()
    payload: dict[str, Any] = {}
    freq_match = re.search(r"(\d+(?:\.\d+)?)\s*hz", lowered)
    current_match = re.search(r"(?:ipp|pp)\s*(\d+(?:\.\d+)?)\s*a", lowered)
    gain_match = re.search(r"(\d+(?:\.\d+)?)\s*v\/v", lowered)
    if freq_match:
        payload["frequency_hz"] = float(freq_match.group(1))
    if current_match:
        payload["target_ipp_a"] = float(current_match.group(1))
    if gain_match:
        payload["gain_mode_v_per_v"] = float(gain_match.group(1))
    if "cv" in lowered:
        payload["amp_mode"] = "CV"
    return payload


def build_dataset_meta(
    dataset_id: str,
    file_name: str,
    stored_path: Path,
    available_sheets: list[str],
) -> DatasetMeta:
    return DatasetMeta(
        dataset_id=dataset_id,
        file_name=file_name,
        stored_path=str(stored_path),
        file_type=stored_path.suffix.lower().lstrip("."),
        available_sheets=available_sheets,
        selected_sheet=available_sheets[0] if available_sheets else None,
        metadata=parse_metadata_from_name(file_name),
    )
