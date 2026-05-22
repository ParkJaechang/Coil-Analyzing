from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def native_measured_support_source(
    frame: pd.DataFrame,
    *,
    fallback_time_s: np.ndarray,
    fallback_measured: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    attr_time = frame.attrs.get("selected_support_source_time_s")
    attr_measured = frame.attrs.get("selected_support_source_mT")
    native_from_attrs = _validate_native_support_arrays(attr_time, attr_measured)
    if native_from_attrs is not None:
        return (*native_from_attrs, "selected_support_source_native_attrs")

    source_time = _first_sequence_value(frame, "selected_support_source_time_s")
    source_measured = _first_sequence_value(frame, "selected_support_source_mT")
    native_from_columns = _validate_native_support_arrays(source_time, source_measured)
    if native_from_columns is not None:
        return (*native_from_columns, "selected_support_source_native")

    column_time = _numeric_column_sequence(frame, "selected_support_source_time_s")
    column_measured = _numeric_column_sequence(frame, "selected_support_source_mT")
    native_from_numeric_columns = _validate_native_support_arrays(column_time, column_measured)
    if native_from_numeric_columns is not None:
        return (*native_from_numeric_columns, "selected_support_source_native_columns")

    return np.asarray(fallback_time_s, dtype=float), np.asarray(fallback_measured, dtype=float), "output_command_grid_fallback"


def _validate_native_support_arrays(
    source_time: Any,
    source_measured: Any,
) -> tuple[np.ndarray, np.ndarray] | None:
    if source_time is None or source_measured is None:
        return None
    source_time_arr = np.asarray(source_time, dtype=float)
    source_measured_arr = np.asarray(source_measured, dtype=float)
    if source_time_arr.size != source_measured_arr.size or source_time_arr.size < 3:
        return None
    finite = np.isfinite(source_time_arr) & np.isfinite(source_measured_arr)
    if finite.sum() < 3:
        return None
    return source_time_arr, source_measured_arr


def _numeric_column_sequence(frame: pd.DataFrame, column: str) -> np.ndarray | None:
    if column not in frame.columns or frame.empty:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return values if np.isfinite(values).sum() >= 3 else None


def _first_sequence_value(frame: pd.DataFrame, column: str) -> list[float] | tuple[float, ...] | np.ndarray | None:
    if column not in frame.columns or frame.empty:
        return None
    for value in frame[column]:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, (list, tuple)):
            return value
        if isinstance(value, str):
            parsed = _parse_sequence_string(value)
            if parsed is not None:
                return parsed
    return None


def _parse_sequence_string(value: str) -> list[float] | None:
    text = value.strip()
    if not text or text.lower() in {"none", "nan"}:
        return None
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    parts = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    if len(parts) < 3:
        return None
    try:
        return [float(part) for part in parts]
    except ValueError:
        return None
