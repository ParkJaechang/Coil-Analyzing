from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .lut import target_metric_label
from .utils import canonicalize_waveform_type


def _normalize_waveform_type(value: object) -> str:
    normalized = canonicalize_waveform_type(value)
    if normalized:
        return normalized
    text = str(value or "").strip()
    return text or "unknown"


def _normalize_frequency(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return float("nan")
    return float(numeric)


def _frame_has_numeric_signal(frame: pd.DataFrame | None, column: str) -> bool:
    if frame is None or frame.empty or column not in frame.columns:
        return False
    values = pd.to_numeric(frame[column], errors="coerce")
    return bool(values.notna().any())


def _frame_waveform_and_freq(frame: pd.DataFrame, fallback_label: str) -> tuple[str, float, str]:
    if frame.empty:
        return "unknown", float("nan"), fallback_label

    waveform_type = _normalize_waveform_type(frame.get("waveform_type", pd.Series(["unknown"])).iloc[0])
    freq_hz = _normalize_frequency(frame.get("freq_hz", pd.Series([np.nan])).iloc[0])
    source_label = str(
        frame.get("test_id", frame.get("source_file", pd.Series([fallback_label]))).iloc[0]
    )
    return waveform_type, freq_hz, source_label


def _continuous_test_details(
    per_test_summary: pd.DataFrame,
    continuous_frames_by_test_id: dict[str, pd.DataFrame],
    main_field_axis: str,
    voltage_input_column: str,
) -> pd.DataFrame:
    if per_test_summary.empty:
        return pd.DataFrame(
            columns=[
                "test_id",
                "waveform_type",
                "freq_hz",
                "has_main_field_axis",
                "has_voltage_input",
                "field_metric_available",
                "voltage_metric_available",
            ]
        )

    field_metric = f"achieved_{main_field_axis}_pp_mean"
    rows: list[dict[str, Any]] = []
    for row in per_test_summary.to_dict(orient="records"):
        test_id = str(row.get("test_id") or "")
        frame = continuous_frames_by_test_id.get(test_id)
        field_metric_available = bool(
            np.isfinite(pd.to_numeric(pd.Series([row.get(field_metric)]), errors="coerce").iloc[0])
        )
        voltage_metric_available = bool(
            np.isfinite(pd.to_numeric(pd.Series([row.get("daq_input_v_pp_mean")]), errors="coerce").iloc[0])
        )
        if frame is not None:
            has_main_field_axis = _frame_has_numeric_signal(frame, main_field_axis)
            has_voltage_input = _frame_has_numeric_signal(frame, voltage_input_column)
        else:
            has_main_field_axis = field_metric_available
            has_voltage_input = voltage_metric_available
        rows.append(
            {
                "test_id": test_id,
                "waveform_type": _normalize_waveform_type(row.get("waveform_type")),
                "freq_hz": _normalize_frequency(row.get("freq_hz")),
                "has_main_field_axis": has_main_field_axis,
                "has_voltage_input": has_voltage_input,
                "field_metric_available": field_metric_available,
                "voltage_metric_available": voltage_metric_available,
            }
        )

    return pd.DataFrame(rows).sort_values(["waveform_type", "freq_hz", "test_id"]).reset_index(drop=True)


def _transient_test_details(
    transient_frames: list[pd.DataFrame],
    main_field_axis: str,
    voltage_input_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, frame in enumerate(transient_frames, start=1):
        waveform_type, freq_hz, source_label = _frame_waveform_and_freq(frame, fallback_label=f"transient_{index}")
        rows.append(
            {
                "test_id": source_label,
                "waveform_type": waveform_type,
                "freq_hz": freq_hz,
                "has_main_field_axis": _frame_has_numeric_signal(frame, main_field_axis),
                "has_voltage_input": _frame_has_numeric_signal(frame, voltage_input_column),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["test_id", "waveform_type", "freq_hz", "has_main_field_axis", "has_voltage_input"]
        )

    return pd.DataFrame(rows).sort_values(["waveform_type", "freq_hz", "test_id"]).reset_index(drop=True)


def _combo_index(
    continuous_details: pd.DataFrame,
    transient_details: pd.DataFrame,
) -> pd.DataFrame:
    combo_sources: list[pd.DataFrame] = []
    for frame in (continuous_details, transient_details):
        if frame.empty:
            continue
        combo_sources.append(frame[["waveform_type", "freq_hz"]].copy())
    if not combo_sources:
        return pd.DataFrame(columns=["waveform_type", "freq_hz"])

    combined = pd.concat(combo_sources, ignore_index=True).drop_duplicates()
    return combined.sort_values(["waveform_type", "freq_hz"]).reset_index(drop=True)


def _risk_level(
    support_count: int,
    field_ready_count: int,
    voltage_ready_count: int,
) -> str:
    if support_count <= 0:
        return "Missing"
    if field_ready_count <= 0:
        return "Field Missing"
    if voltage_ready_count <= 0:
        return "Voltage Missing"
    if min(field_ready_count, voltage_ready_count) >= 2:
        return "OK"
    return "Weak"


def _summarize_support_by_combo(
    combo_index: pd.DataFrame,
    details: pd.DataFrame,
    support_prefix: str,
) -> pd.DataFrame:
    if combo_index.empty:
        return pd.DataFrame()

    if details.empty:
        aggregated = combo_index.copy()
        aggregated[f"{support_prefix}_test_count"] = 0
        aggregated["field_ready_test_count"] = 0
        aggregated["voltage_ready_test_count"] = 0
    else:
        grouped = (
            details.groupby(["waveform_type", "freq_hz"], dropna=False)
            .agg(
                **{
                    f"{support_prefix}_test_count": ("test_id", "count"),
                    "field_ready_test_count": ("has_main_field_axis", "sum"),
                    "voltage_ready_test_count": ("has_voltage_input", "sum"),
                }
            )
            .reset_index()
        )
        aggregated = combo_index.merge(grouped, how="left", on=["waveform_type", "freq_hz"])
        aggregated[f"{support_prefix}_test_count"] = (
            aggregated[f"{support_prefix}_test_count"].fillna(0).astype(int)
        )
        aggregated["field_ready_test_count"] = aggregated["field_ready_test_count"].fillna(0).astype(int)
        aggregated["voltage_ready_test_count"] = aggregated["voltage_ready_test_count"].fillna(0).astype(int)

    aggregated["risk_level"] = aggregated.apply(
        lambda row: _risk_level(
            int(row[f"{support_prefix}_test_count"]),
            int(row["field_ready_test_count"]),
            int(row["voltage_ready_test_count"]),
        ),
        axis=1,
    )
    aggregated["shape_comparison_possible"] = aggregated["field_ready_test_count"] >= 2
    aggregated["has_support"] = aggregated[f"{support_prefix}_test_count"] > 0
    return aggregated.sort_values(["waveform_type", "freq_hz"]).reset_index(drop=True)


def _target_metric_candidates(per_test_summary: pd.DataFrame, main_field_axis: str) -> pd.DataFrame:
    candidate_metrics = list(
        dict.fromkeys(
            [
                f"achieved_{main_field_axis}_pp_mean",
                "achieved_bz_mT_pp_mean",
                "achieved_bmag_mT_pp_mean",
                "achieved_bproj_mT_pp_mean",
            ]
        )
    )

    rows: list[dict[str, Any]] = []
    for metric in candidate_metrics:
        present = metric in per_test_summary.columns
        non_null_count = (
            int(pd.to_numeric(per_test_summary[metric], errors="coerce").notna().sum())
            if present and not per_test_summary.empty
            else 0
        )
        rows.append(
            {
                "metric": metric,
                "label": target_metric_label(metric),
                "available": bool(present and non_null_count > 0),
                "non_null_test_count": non_null_count,
            }
        )

    return pd.DataFrame(rows)


def _aggregate_by_dimension(
    continuous_support: pd.DataFrame,
    finite_support: pd.DataFrame,
    dimension: str,
) -> pd.DataFrame:
    if continuous_support.empty and finite_support.empty:
        return pd.DataFrame()

    continuous_rows: list[dict[str, Any]] = []
    if not continuous_support.empty:
        for key, group in continuous_support.groupby(dimension, dropna=False):
            continuous_rows.append(
                {
                    dimension: key,
                    "continuous_test_count": int(group["continuous_test_count"].sum()),
                    "ok_combo_count": int((group["risk_level"] == "OK").sum()),
                    "weak_combo_count": int((group["risk_level"] == "Weak").sum()),
                    "missing_combo_count": int((group["risk_level"] == "Missing").sum()),
                    "shape_comparison_combo_count": int(group["shape_comparison_possible"].sum()),
                }
            )

    finite_rows: list[dict[str, Any]] = []
    if not finite_support.empty:
        for key, group in finite_support.groupby(dimension, dropna=False):
            finite_rows.append(
                {
                    dimension: key,
                    "finite_test_count": int(group["finite_test_count"].sum()),
                    "finite_supported_combo_count": int(group["has_support"].sum()),
                    "finite_missing_combo_count": int((~group["has_support"]).sum()),
                }
            )

    merged = pd.DataFrame(continuous_rows)
    if merged.empty:
        merged = pd.DataFrame(columns=[dimension])
    finite_frame = pd.DataFrame(finite_rows)
    merged = merged.merge(finite_frame, how="outer", on=dimension)
    for column in merged.columns:
        if column == dimension:
            continue
        merged[column] = merged[column].fillna(0).astype(int)
    return merged.sort_values(dimension).reset_index(drop=True)


def build_field_waveform_diagnostics(
    per_test_summary: pd.DataFrame,
    *,
    main_field_axis: str = "bz_mT",
    continuous_frames_by_test_id: dict[str, pd.DataFrame] | None = None,
    transient_frames: list[pd.DataFrame] | None = None,
    voltage_input_column: str = "daq_input_v",
) -> dict[str, Any]:
    continuous_frames_by_test_id = dict(continuous_frames_by_test_id or {})
    transient_frames = list(transient_frames or [])

    continuous_details = _continuous_test_details(
        per_test_summary=per_test_summary,
        continuous_frames_by_test_id=continuous_frames_by_test_id,
        main_field_axis=main_field_axis,
        voltage_input_column=voltage_input_column,
    )
    transient_details = _transient_test_details(
        transient_frames=transient_frames,
        main_field_axis=main_field_axis,
        voltage_input_column=voltage_input_column,
    )
    combo_index = _combo_index(continuous_details=continuous_details, transient_details=transient_details)
    continuous_support = _summarize_support_by_combo(
        combo_index=combo_index,
        details=continuous_details,
        support_prefix="continuous",
    )
    finite_support = _summarize_support_by_combo(
        combo_index=combo_index,
        details=transient_details,
        support_prefix="finite",
    )
    target_metric_candidates = _target_metric_candidates(
        per_test_summary=per_test_summary,
        main_field_axis=main_field_axis,
    )
    waveform_counts = _aggregate_by_dimension(
        continuous_support=continuous_support,
        finite_support=finite_support,
        dimension="waveform_type",
    )
    frequency_counts = _aggregate_by_dimension(
        continuous_support=continuous_support,
        finite_support=finite_support,
        dimension="freq_hz",
    )

    summary = {
        "continuous_test_count": int(len(continuous_details)),
        "finite_test_count": int(len(transient_details)),
        "continuous_ok_combo_count": int((continuous_support.get("risk_level", pd.Series(dtype=object)) == "OK").sum()),
        "continuous_weak_combo_count": int((continuous_support.get("risk_level", pd.Series(dtype=object)) == "Weak").sum()),
        "continuous_missing_combo_count": int((continuous_support.get("risk_level", pd.Series(dtype=object)) == "Missing").sum()),
        "finite_supported_combo_count": int(finite_support.get("has_support", pd.Series(dtype=bool)).sum()),
        "finite_missing_combo_count": int((~finite_support.get("has_support", pd.Series(dtype=bool))).sum())
        if not finite_support.empty
        else 0,
        "shape_comparison_combo_count": int(
            continuous_support.get("shape_comparison_possible", pd.Series(dtype=bool)).sum()
        ),
        "main_field_axis_available": bool(continuous_details.get("has_main_field_axis", pd.Series(dtype=bool)).any()),
        "voltage_input_available": bool(continuous_details.get("has_voltage_input", pd.Series(dtype=bool)).any()),
        "target_metric_candidate_count": int(target_metric_candidates["available"].sum()),
    }

    return {
        "summary": summary,
        "waveform_counts": waveform_counts,
        "frequency_counts": frequency_counts,
        "target_metric_candidates": target_metric_candidates,
        "continuous_support": continuous_support,
        "finite_support": finite_support,
        "continuous_test_details": continuous_details,
        "transient_test_details": transient_details,
        "notes": [
            "Primary diagnostics are field-first: voltage input and measured magnetic field coverage drive the risk labels.",
            "Current is intentionally excluded from the primary model-risk summary and should stay debug-only.",
        ],
    }


__all__ = ["build_field_waveform_diagnostics"]
