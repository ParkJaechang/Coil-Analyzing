from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

TIME_ALIASES = (
    "time_s",
    "time_s_abs",
    "TimeMs",
    "Time_ms",
    "Time_s",
    "Time",
    "timestamp_s",
    "elapsed_time_s",
    "sample_time_s",
)
VOLTAGE_ALIASES = (
    "Voltage1_V",
    "voltage_v",
    "raw_voltage_v",
    "raw_actual_drive_voltage_v",
    "command_voltage_v",
    "actual_drive_voltage_v",
    "first_voltage_v",
    "normalized_actual_drive_voltage_v",
    "normalized_first_voltage_v",
)
RAW_HALL_ALIASES = (
    "HallBz",
    "HallZ",
    "hallbz_raw_mT",
    "raw_hallbz_mT",
    "raw_measured_field_mT",
    "measured_field_raw_mT",
    "measured_field_raw",
)
EFFECTIVE_FIELD_ALIASES = ("measured_field_effective_mT",)
NORMALIZED_FIELD_ALIASES = (
    "normalized_measured_field_mT",
    "measured_field_normalized_mT",
    "normalized_effective_field_mT",
)


def adapt_continuous_source_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = set(frame.columns)
    if {"sample_index", "time_s", "voltage_v"}.issubset(columns):
        raise ValueError("final_voltage_lut_not_measured_input")
    time_column = first_existing_column(frame, TIME_ALIASES)
    voltage_column = first_existing_column(frame, VOLTAGE_ALIASES)
    raw_hall_column = first_existing_column(frame, RAW_HALL_ALIASES)
    effective_field_column = first_existing_column(frame, EFFECTIVE_FIELD_ALIASES)
    normalized_field_column = first_existing_column(frame, NORMALIZED_FIELD_ALIASES)
    field_column = raw_hall_column or effective_field_column or normalized_field_column
    missing = []
    if time_column is None:
        missing.append("time")
    if voltage_column is None:
        missing.append("voltage")
    if field_column is None:
        missing.append("field")
    if missing:
        raise ValueError(f"continuous_schema_missing_{'_'.join(missing)}")
    time_s = pd.to_numeric(frame[time_column], errors="coerce")
    if time_column in {"TimeMs", "Time_ms"}:
        time_s = time_s / 1000.0
    voltage = pd.to_numeric(frame[voltage_column], errors="coerce")
    raw_hall_available = raw_hall_column is not None
    normalized_field_available = normalized_field_column is not None
    if raw_hall_column is not None:
        raw_hall = pd.to_numeric(frame[raw_hall_column], errors="coerce")
        effective = -raw_hall
        accepted_source_type = "raw_hallbz"
    elif effective_field_column is not None:
        raw_hall = pd.Series(np.nan, index=frame.index, dtype=float)
        effective = pd.to_numeric(frame[effective_field_column], errors="coerce")
        accepted_source_type = "effective_field"
    else:
        raw_hall = pd.Series(np.nan, index=frame.index, dtype=float)
        effective = pd.to_numeric(frame[normalized_field_column], errors="coerce")
        accepted_source_type = "normalized_field"
    if normalized_field_column is not None:
        normalized_field = pd.to_numeric(frame[normalized_field_column], errors="coerce")
    else:
        baseline = float(np.nanmedian(effective.to_numpy(dtype=float))) if len(effective) else 0.0
        baseline_removed_for_scale = effective - baseline
        peak = float(np.nanmax(np.abs(baseline_removed_for_scale.to_numpy(dtype=float)))) if len(baseline_removed_for_scale) else 0.0
        scale = 50.0 / peak if np.isfinite(peak) and peak > 0.0 else 1.0
        normalized_field = baseline_removed_for_scale * scale
    effective_baseline = float(np.nanmedian(effective.to_numpy(dtype=float))) if len(effective) else 0.0
    baseline_removed = effective - effective_baseline
    if voltage_column in {"normalized_actual_drive_voltage_v", "normalized_first_voltage_v"}:
        voltage_normalized = voltage
    else:
        voltage_peak = float(np.nanmax(np.abs(voltage.to_numpy(dtype=float)))) if len(voltage) else 0.0
        voltage_scale = 5.0 / voltage_peak if np.isfinite(voltage_peak) and voltage_peak > 5.0 else 1.0
        voltage_normalized = voltage * voltage_scale
    source = pd.DataFrame(
        {
            "time_s": time_s.to_numpy(dtype=float),
            "time_s_abs": time_s.to_numpy(dtype=float),
            "raw_hallbz_mT": raw_hall.to_numpy(dtype=float),
            "measured_field_effective_mT": effective.to_numpy(dtype=float),
            "measured_field_baseline_removed_mT": baseline_removed.to_numpy(dtype=float),
            "measured_field_normalized_mT": normalized_field.to_numpy(dtype=float),
            "raw_voltage_v": voltage.to_numpy(dtype=float),
            "voltage_normalized_v": voltage_normalized.to_numpy(dtype=float),
        }
    ).dropna(subset=["time_s_abs"])
    metadata = {
        "continuous_schema_status": "ok",
        "continuous_schema_time_column": time_column,
        "continuous_schema_voltage_column": voltage_column,
        "continuous_schema_hall_column": raw_hall_column,
        "continuous_schema_hall_or_field_column": field_column,
        "continuous_schema_accepted_source_type": accepted_source_type,
        "continuous_schema_reject_reason": None,
        "raw_hallbz_available": raw_hall_available,
        "normalized_field_available": normalized_field_available,
        "continuous_source_file": frame.attrs.get("continuous_source_file"),
        "continuous_source_freq_hz": frame.attrs.get("continuous_source_freq_hz"),
        "continuous_source_freq_source": frame.attrs.get("continuous_source_freq_source"),
        "continuous_source_freq_inferred_from_filename": frame.attrs.get("continuous_source_freq_inferred_from_filename", False),
        "continuous_source_freq_inferred_from_preamble": frame.attrs.get("continuous_source_freq_inferred_from_preamble", False),
        "continuous_source_freq_user_override": frame.attrs.get("continuous_source_freq_user_override", False),
    }
    source.attrs.update({key: value for key, value in metadata.items() if value is not None})
    return source.sort_values("time_s_abs").reset_index(drop=True), metadata


def first_existing_column(frame: pd.DataFrame, columns: tuple[str, ...]) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None
