"""Column mapping, units, scaling, and delay correction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from coil_analyzer.constants import TIME_UNIT_FACTORS
from coil_analyzer.models import ChannelConfig, ChannelMapping
from coil_analyzer.preprocessing.alignment import shift_signal


def _apply_channel(series: pd.Series, cfg: ChannelConfig) -> pd.Series:
    values = series.astype(float) * cfg.scale + cfg.offset
    if cfg.invert:
        values = -values
    return values


def standardize_dataset(df: pd.DataFrame, mapping: ChannelMapping) -> pd.DataFrame:
    if not mapping.time.column:
        raise ValueError("Time column must be mapped before analysis.")

    time_s = parse_time_series(df[mapping.time.column], mapping.time.unit or "s")
    output = pd.DataFrame({"time_s": time_s})

    if mapping.voltage.column and mapping.voltage.column in df.columns:
        output["voltage_v"] = _apply_channel(df[mapping.voltage.column], mapping.voltage)
    if mapping.current.column and mapping.current.column in df.columns:
        output["current_a"] = _apply_channel(df[mapping.current.column], mapping.current)
    for key, cfg in mapping.magnetic.items():
        if cfg.column and cfg.column in df.columns:
            output[f"magnetic_{key}"] = _apply_channel(df[cfg.column], cfg)

    output = output.sort_values("time_s").dropna(subset=["time_s"]).reset_index(drop=True)
    for column in [column for column in output.columns if column != "time_s"]:
        cfg = _lookup_config(column, mapping)
        if cfg and cfg.delay_s:
            output[column] = shift_signal(output["time_s"].to_numpy(), output[column].to_numpy(), cfg.delay_s)
    return output


def parse_time_series(series: pd.Series, unit: str) -> pd.Series:
    if unit == "datetime":
        dt = pd.to_datetime(series, errors="coerce")
        if dt.isna().all():
            raise ValueError("시간축 datetime 파싱에 실패했습니다.")
        base = dt.dropna().iloc[0]
        return (dt - base).dt.total_seconds()

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_ratio = float(numeric.notna().mean()) if len(series) else 0.0
    if numeric_ratio > 0.8:
        factor = TIME_UNIT_FACTORS.get(unit or "s", 1.0)
        return numeric * factor

    dt = pd.to_datetime(series, errors="coerce")
    dt_ratio = float(dt.notna().mean()) if len(series) else 0.0
    if dt_ratio > 0.8:
        base = dt.dropna().iloc[0]
        return (dt - base).dt.total_seconds()

    raise ValueError(
        "시간축을 숫자 또는 datetime 으로 해석할 수 없습니다. "
        "시간 컬럼 선택과 단위를 다시 확인하세요."
    )


def _lookup_config(column_name: str, mapping: ChannelMapping) -> ChannelConfig | None:
    if column_name == "voltage_v":
        return mapping.voltage
    if column_name == "current_a":
        return mapping.current
    if column_name.startswith("magnetic_"):
        key = column_name.replace("magnetic_", "", 1)
        return mapping.magnetic.get(key)
    return None


def sample_rate_from_time(time_s: pd.Series) -> float:
    delta = np.diff(time_s.to_numpy(dtype=float))
    delta = delta[np.isfinite(delta) & (delta > 0)]
    if len(delta) == 0:
        raise ValueError("Unable to infer sample rate from time axis.")
    return float(1.0 / np.median(delta))


def summarize_channels(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"rows": int(len(df)), "columns": list(df.columns)}
    if "time_s" in df:
        summary["duration_s"] = float(df["time_s"].iloc[-1] - df["time_s"].iloc[0]) if len(df) > 1 else 0.0
    return summary


def infer_time_unit(series: pd.Series) -> str:
    numeric = pd.to_numeric(series, errors="coerce")
    if len(series) and float(numeric.notna().mean()) > 0.8:
        name = str(series.name).lower()
        if "ms" in name:
            return "ms"
        if "us" in name:
            return "us"
        return "s"
    dt = pd.to_datetime(series, errors="coerce")
    if len(series) and float(dt.notna().mean()) > 0.8:
        return "datetime"
    return "s"
