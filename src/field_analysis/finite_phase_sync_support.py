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


def native_voltage_support_source(
    frame: pd.DataFrame,
    *,
    fallback_time_s: np.ndarray,
    fallback_voltage: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    attr_time = frame.attrs.get("selected_support_source_time_s")
    attr_voltage = frame.attrs.get("selected_support_source_voltage_v")
    native_from_attrs = _validate_native_support_arrays(attr_time, attr_voltage)
    if native_from_attrs is not None:
        return (*native_from_attrs, "selected_support_source_voltage_v")

    source_time = _first_sequence_value(frame, "selected_support_source_time_s")
    source_voltage = _first_sequence_value(frame, "selected_support_source_voltage_v")
    native_from_columns = _validate_native_support_arrays(source_time, source_voltage)
    if native_from_columns is not None:
        return (*native_from_columns, "selected_support_source_voltage_v")

    column_time = _numeric_column_sequence(frame, "selected_support_source_time_s")
    column_voltage = _numeric_column_sequence(frame, "selected_support_source_voltage_v")
    native_from_numeric_columns = _validate_native_support_arrays(column_time, column_voltage)
    if native_from_numeric_columns is not None:
        return (*native_from_numeric_columns, "selected_support_source_voltage_v")

    return np.asarray(fallback_time_s, dtype=float), np.asarray(fallback_voltage, dtype=float), "finite_first_base_voltage_v"


def active_end_kink_detected(voltage: np.ndarray, residual: np.ndarray, active_mask: np.ndarray) -> bool:
    active_indices = np.flatnonzero(np.asarray(active_mask, dtype=bool) & np.isfinite(voltage) & np.isfinite(residual))
    if active_indices.size < 4:
        return False
    tail = active_indices[-4:]
    voltage_step = float(np.nanmax(np.abs(np.diff(np.asarray(voltage, dtype=float)[tail]))))
    residual_step = float(np.nanmax(np.abs(np.diff(np.asarray(residual, dtype=float)[tail]))))
    return bool(voltage_step > 1.0 and residual_step > 5.0)


def source_active_start_s(frame: pd.DataFrame, native_time_s: np.ndarray, output_start_s: float) -> float:
    for key in (
        "selected_support_voltage_nonzero_start_s",
        "selected_support_command_nonzero_start_s",
    ):
        value = frame.attrs.get(key)
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            return float(value)
    for column in (
        "selected_support_voltage_nonzero_start_s",
        "selected_support_command_nonzero_start_s",
        "selected_support_original_nonzero_start_s",
        "support_reference_source_window_start_s",
        "selected_support_source_window_start_s",
    ):
        if column in frame.columns:
            value = _first_numeric(frame[column])
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                return float(value)
    finite = np.asarray(native_time_s, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size:
        return float(np.nanmin(finite))
    return float(output_start_s)


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


def _first_numeric(values: pd.Series) -> float | bool | None:
    if values.dtype == bool:
        return bool(values.dropna().iloc[0]) if not values.dropna().empty else None
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.iloc[0]) if not numeric.empty else None


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
