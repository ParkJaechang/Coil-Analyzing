from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FINAL_MODELED_LUT_COLUMNS = ("sample_index", "time_s", "voltage_v")
FINITE_PRODUCTION_SUPPORTED_CYCLES = (1.0, 1.5)
FINITE_PRODUCTION_UNSUPPORTED_CYCLES = (1.25, 1.75, 2.0)


def build_final_modeled_voltage_lut_export(
    command_profile: pd.DataFrame,
    *,
    route: str | None = None,
    mode: str | None = None,
    freq_hz: float | None = None,
    cycle_count: float | None = None,
    waveform: str | None = None,
    voltage_limit_v: float | None = None,
) -> dict[str, Any]:
    """Export the plotted final modeled voltage samples without re-synthesis."""

    if not isinstance(command_profile, pd.DataFrame) or command_profile.empty:
        raise ValueError("command_profile must be a non-empty DataFrame")
    voltage_source_column = _export_voltage_source_column(command_profile)
    for column in ("time_s", voltage_source_column):
        if column not in command_profile.columns:
            raise ValueError(f"command_profile missing required column: {column}")

    time_s = pd.to_numeric(command_profile["time_s"], errors="coerce").to_numpy(dtype=float)
    voltage_v = pd.to_numeric(command_profile[voltage_source_column], errors="coerce").to_numpy(dtype=float)
    if len(time_s) != len(voltage_v):
        raise ValueError("time_s and limited_voltage_v length mismatch")

    lut_frame = pd.DataFrame(
        {
            "sample_index": np.arange(len(command_profile), dtype=int),
            "time_s": time_s,
            "voltage_v": voltage_v,
        }
    )
    diagnostics = diagnose_modeled_voltage_lut_timebase(lut_frame)
    cycle_policy = _finite_production_cycle_policy(cycle_count)
    metadata = {
        "lut_export_type": "final_modeled_voltage_lut",
        "voltage_source_column": voltage_source_column,
        "exported_voltage_source_column": voltage_source_column,
        "time_source_column": "time_s",
        "fourier_resynthesis_involved": False,
        "harmonic_export_involved": False,
        "finite_only": True,
        "route": route,
        "mode": mode,
        "freq_hz": _optional_float(freq_hz),
        "cycle_count": _optional_float(cycle_count),
        "waveform": waveform,
        "sample_count": int(len(lut_frame)),
        "duration_s": diagnostics["duration_s"],
        "dt_median_s": diagnostics["dt_median_s"],
        "time_monotonic": diagnostics["time_monotonic"],
        "duplicated_time_count": diagnostics["duplicated_time_count"],
        "voltage_min_v": diagnostics["voltage_min_v"],
        "voltage_max_v": diagnostics["voltage_max_v"],
        "voltage_limit_v": _optional_float(voltage_limit_v),
        **cycle_policy,
    }
    return {"frame": lut_frame, "metadata": metadata}


def final_modeled_voltage_lut_to_csv_bytes(export_payload: Mapping[str, Any]) -> bytes:
    frame = export_payload.get("frame")
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("export payload missing frame DataFrame")
    missing = [column for column in FINAL_MODELED_LUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"export frame missing columns: {missing}")
    return frame.loc[:, FINAL_MODELED_LUT_COLUMNS].to_csv(index=False).encode("utf-8-sig")


def load_final_modeled_voltage_lut(source: Any) -> dict[str, Any]:
    frame = _read_lut_csv(source)
    required = {"time_s", "voltage_v"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        return {
            "parse_status": "error",
            "parse_error": f"missing required columns: {', '.join(missing)}",
            "frame": pd.DataFrame(columns=list(FINAL_MODELED_LUT_COLUMNS)),
            "diagnostics": _empty_diagnostics(time_axis_status="missing_time" if "time_s" in missing else "missing_voltage"),
        }

    parsed = pd.DataFrame(
        {
            "sample_index": (
                pd.to_numeric(frame["sample_index"], errors="coerce").astype("Int64")
                if "sample_index" in frame.columns
                else pd.Series(np.arange(len(frame), dtype=int), dtype="Int64")
            ),
            "time_s": pd.to_numeric(frame["time_s"], errors="coerce"),
            "voltage_v": pd.to_numeric(frame["voltage_v"], errors="coerce"),
        }
    )
    diagnostics = diagnose_modeled_voltage_lut_timebase(parsed)
    return {
        "parse_status": "ok",
        "parse_error": None,
        "frame": parsed,
        "diagnostics": diagnostics,
    }


def diagnose_modeled_voltage_lut_timebase(frame: pd.DataFrame) -> dict[str, Any]:
    if "time_s" not in frame.columns:
        return _empty_diagnostics(time_axis_status="missing_time")

    time_s = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
    voltage_v = (
        pd.to_numeric(frame["voltage_v"], errors="coerce").to_numpy(dtype=float)
        if "voltage_v" in frame.columns
        else np.full(len(frame), np.nan)
    )
    finite_time = time_s[np.isfinite(time_s)]
    diffs = np.diff(finite_time) if finite_time.size > 1 else np.array([], dtype=float)
    positive_diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    duplicate_count = int(np.sum(np.isfinite(diffs) & np.isclose(diffs, 0.0, atol=1e-15)))
    non_monotonic = bool(np.any(np.isfinite(diffs) & (diffs < 0)))
    dt_median = float(np.nanmedian(positive_diffs)) if positive_diffs.size else float("nan")
    dt_min = float(np.nanmin(positive_diffs)) if positive_diffs.size else float("nan")
    dt_max = float(np.nanmax(positive_diffs)) if positive_diffs.size else float("nan")
    duration_s = float(np.nanmax(finite_time) - np.nanmin(finite_time)) if finite_time.size else float("nan")
    irregularity = float(dt_max / dt_median) if np.isfinite(dt_max) and np.isfinite(dt_median) and dt_median > 0 else float("nan")
    suspected_unit = _suspect_time_unit(duration_s=duration_s, dt_median_s=dt_median)
    status = _time_axis_status(
        finite_time_count=int(finite_time.size),
        non_monotonic=non_monotonic,
        duplicate_count=duplicate_count,
        suspected_unit=suspected_unit,
        irregularity=irregularity,
    )
    sample_index_order_ok = True
    if "sample_index" in frame.columns:
        sample_index = pd.to_numeric(frame["sample_index"], errors="coerce").to_numpy(dtype=float)
        finite_index = sample_index[np.isfinite(sample_index)]
        sample_index_order_ok = bool(finite_index.size == 0 or np.all(np.diff(finite_index) >= 0))
    return {
        "sample_count": int(len(frame)),
        "time_start_s": float(np.nanmin(finite_time)) if finite_time.size else float("nan"),
        "time_end_s": float(np.nanmax(finite_time)) if finite_time.size else float("nan"),
        "duration_s": duration_s,
        "dt_min_s": dt_min,
        "dt_median_s": dt_median,
        "dt_max_s": dt_max,
        "dt_irregularity_ratio": irregularity,
        "time_monotonic": bool(finite_time.size > 1 and np.all(diffs > 0)),
        "duplicated_time_count": duplicate_count,
        "nonfinite_time_count": int(np.sum(~np.isfinite(time_s))),
        "nonfinite_voltage_count": int(np.sum(~np.isfinite(voltage_v))),
        "voltage_min_v": float(np.nanmin(voltage_v)) if np.isfinite(voltage_v).any() else float("nan"),
        "voltage_max_v": float(np.nanmax(voltage_v)) if np.isfinite(voltage_v).any() else float("nan"),
        "suspected_time_unit": suspected_unit,
        "time_axis_status": status,
        "sample_index_order_ok": sample_index_order_ok,
    }


def _read_lut_csv(source: Any) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if isinstance(source, (str, Path)):
        return pd.read_csv(source)
    return pd.read_csv(source)


def _optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _finite_production_cycle_policy(cycle_count: float | None) -> dict[str, Any]:
    cycle = _optional_float(cycle_count)
    supported = cycle is not None and any(abs(cycle - item) <= 1e-9 for item in FINITE_PRODUCTION_SUPPORTED_CYCLES)
    return {
        "production_cycle_policy": "1p0_1p5_cycles",
        "production_supported_cycles": list(FINITE_PRODUCTION_SUPPORTED_CYCLES),
        "unsupported_cycles": list(FINITE_PRODUCTION_UNSUPPORTED_CYCLES),
        "finite_production_cycle_supported": bool(supported),
        "finite_production_export_status": "ok" if supported else "unsupported_cycle_policy_1p0_1p5_only",
        "finite_production_export_warning": None if supported else "cycle_not_in_1p0_1p5_production_policy",
    }


def _export_voltage_source_column(command_profile: pd.DataFrame) -> str:
    if "second_limited_voltage_v" in command_profile.columns:
        status = None
        if "second_modeling_status" in command_profile.columns and len(command_profile):
            status = str(command_profile["second_modeling_status"].iloc[0])
        available = True
        if "second_modeling_available" in command_profile.columns and len(command_profile):
            available = bool(command_profile["second_modeling_available"].iloc[0])
        if available and status in {None, "ok"}:
            return "second_limited_voltage_v"
    if "feedback_corrected_limited_voltage_v" not in command_profile.columns:
        return "limited_voltage_v"
    status = None
    if "feedback_correction_status" in command_profile.columns and len(command_profile):
        status = str(command_profile["feedback_correction_status"].iloc[0])
    available = True
    if "feedback_correction_available" in command_profile.columns and len(command_profile):
        value = command_profile["feedback_correction_available"].iloc[0]
        available = bool(value)
    return "feedback_corrected_limited_voltage_v" if available and status in {None, "ok"} else "limited_voltage_v"


def _suspect_time_unit(*, duration_s: float, dt_median_s: float) -> str:
    if not np.isfinite(duration_s) or not np.isfinite(dt_median_s):
        return "unknown"
    if duration_s >= 20.0 or dt_median_s >= 1.0:
        return "milliseconds"
    return "seconds"


def _time_axis_status(
    *,
    finite_time_count: int,
    non_monotonic: bool,
    duplicate_count: int,
    suspected_unit: str,
    irregularity: float,
) -> str:
    if finite_time_count == 0:
        return "missing_time"
    if non_monotonic:
        return "non_monotonic"
    if duplicate_count > 0:
        return "duplicate_time"
    if suspected_unit == "milliseconds":
        return "suspect_ms_uploaded_as_s"
    if np.isfinite(irregularity) and irregularity > 1.5:
        return "irregular_sampling"
    return "ok"


def _empty_diagnostics(*, time_axis_status: str) -> dict[str, Any]:
    return {
        "sample_count": 0,
        "time_start_s": float("nan"),
        "time_end_s": float("nan"),
        "duration_s": float("nan"),
        "dt_min_s": float("nan"),
        "dt_median_s": float("nan"),
        "dt_max_s": float("nan"),
        "dt_irregularity_ratio": float("nan"),
        "time_monotonic": False,
        "duplicated_time_count": 0,
        "nonfinite_time_count": 0,
        "nonfinite_voltage_count": 0,
        "voltage_min_v": float("nan"),
        "voltage_max_v": float("nan"),
        "suspected_time_unit": "unknown",
        "time_axis_status": time_axis_status,
        "sample_index_order_ok": False,
    }
