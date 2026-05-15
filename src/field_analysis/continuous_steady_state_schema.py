from __future__ import annotations

from typing import Any

import pandas as pd

TIME_ALIASES = ("time_s", "time_s_abs", "TimeMs", "Time_s", "Time", "timestamp_s")
VOLTAGE_ALIASES = (
    "Voltage1_V",
    "raw_voltage_v",
    "raw_actual_drive_voltage_v",
    "command_voltage_v",
    "actual_drive_voltage_v",
    "first_voltage_v",
)
HALL_ALIASES = (
    "HallBz",
    "HallZ",
    "hallbz_raw_mT",
    "raw_hallbz_mT",
    "measured_field_raw_mT",
    "measured_field_raw",
)


def adapt_continuous_source_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = set(frame.columns)
    if {"sample_index", "time_s", "voltage_v"}.issubset(columns):
        raise ValueError("final_voltage_lut_not_measured_input")
    time_column = first_existing_column(frame, TIME_ALIASES)
    voltage_column = first_existing_column(frame, VOLTAGE_ALIASES)
    hall_column = first_existing_column(frame, HALL_ALIASES)
    missing = []
    if time_column is None:
        missing.append("time")
    if voltage_column is None:
        missing.append("voltage")
    if hall_column is None:
        missing.append("hall")
    if missing:
        raise ValueError(f"continuous_schema_missing_{'_'.join(missing)}")
    time_s = pd.to_numeric(frame[time_column], errors="coerce")
    if time_column == "TimeMs":
        time_s = time_s / 1000.0
    hall = pd.to_numeric(frame[hall_column], errors="coerce")
    voltage = pd.to_numeric(frame[voltage_column], errors="coerce")
    source = pd.DataFrame(
        {
            "time_s_abs": time_s.to_numpy(dtype=float),
            "raw_hallbz_mT": hall.to_numpy(dtype=float),
            "measured_field_effective_mT": -hall.to_numpy(dtype=float),
            "raw_voltage_v": voltage.to_numpy(dtype=float),
        }
    ).dropna(subset=["time_s_abs"])
    metadata = {
        "continuous_schema_status": "ok",
        "continuous_schema_time_column": time_column,
        "continuous_schema_voltage_column": voltage_column,
        "continuous_schema_hall_column": hall_column,
        "continuous_schema_reject_reason": None,
    }
    return source.sort_values("time_s_abs").reset_index(drop=True), metadata


def first_existing_column(frame: pd.DataFrame, columns: tuple[str, ...]) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None
