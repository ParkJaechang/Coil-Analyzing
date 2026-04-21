from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


PRIMARY_METADATA_KEYS = (
    "waveform_type",
    "freq_hz",
    "finite_cycle_mode",
    "target_metric",
    "requested_target_value",
    "used_target_value",
)

PRIMARY_WAVEFORM_COLUMNS = (
    "time_s",
    "recommended_voltage_v",
    "limited_voltage_v",
)

OPTIONAL_PRIMARY_WAVEFORM_COLUMNS = (
    "is_active_target",
    "target_cycle_count",
    "preview_tail_cycles",
)

DEBUG_SCALAR_KEYS = (
    "target_label",
    "in_range",
    "recommendation_mode",
    "requested_freq_hz",
    "used_freq_hz",
    "frequency_mode",
    "available_freq_min",
    "available_freq_max",
    "frequency_in_range",
    "frequency_support_count",
    "support_point_count",
    "distinct_target_count",
    "available_target_min",
    "available_target_max",
    "estimated_voltage_pp",
    "estimated_voltage_peak",
    "estimated_current_pp",
    "estimated_bz_pp",
    "estimated_bmag_pp",
    "max_daq_voltage_pp",
    "amp_gain_at_100_pct",
    "amp_gain_limit_pct",
    "amp_max_output_pk_v",
    "allow_target_extrapolation",
    "within_daq_limit",
    "within_hardware_limits",
    "required_amp_gain_multiplier",
    "required_amp_gain_pct",
    "support_amp_gain_pct",
    "available_amp_gain_pct",
    "amp_output_pp_at_required",
    "amp_output_pk_at_required",
    "limited_voltage_pp",
    "target_output_label",
    "target_output_unit",
    "target_output_pp",
    "primary_output_label",
    "primary_output_unit",
    "primary_output_pp",
    "recommendation_scope",
    "recommendation_scope_label",
    "template_test_id",
)

DEBUG_FRAME_KEYS = (
    "frequency_support_table",
    "lookup_table",
    "neighbor_points",
    "template_waveform",
)


def build_continuous_recommendation_payload(recommendation: Mapping[str, Any]) -> dict[str, Any]:
    return _build_primary_payload(
        recommendation=recommendation,
        output_type="continuous_recommended_voltage_waveform",
        include_finite_cycle_columns=False,
    )


def build_finite_cycle_recommendation_payload(recommendation: Mapping[str, Any]) -> dict[str, Any]:
    return _build_primary_payload(
        recommendation=recommendation,
        output_type="finite_cycle_stop_waveform",
        include_finite_cycle_columns=True,
    )


def build_recommendation_debug_payload(recommendation: Mapping[str, Any]) -> dict[str, Any]:
    debug_payload = {
        "waveform_type": recommendation.get("waveform_type"),
        "freq_hz": _normalize_scalar(recommendation.get("freq_hz")),
        "finite_cycle_mode": bool(recommendation.get("finite_cycle_mode", False)),
        "reference_metrics": {
            key: _normalize_scalar(recommendation.get(key))
            for key in DEBUG_SCALAR_KEYS
            if key in recommendation
        },
        "reference_tables": {
            key: _frame_to_records(recommendation.get(key), include_optional_columns=None)
            for key in DEBUG_FRAME_KEYS
            if key in recommendation
        },
    }
    return debug_payload


def _build_primary_payload(
    *,
    recommendation: Mapping[str, Any],
    output_type: str,
    include_finite_cycle_columns: bool,
) -> dict[str, Any]:
    waveform_columns = list(PRIMARY_WAVEFORM_COLUMNS)
    if include_finite_cycle_columns:
        waveform_columns.extend(OPTIONAL_PRIMARY_WAVEFORM_COLUMNS)
    else:
        waveform_columns.append("is_active_target")

    payload = {
        "output_type": output_type,
        "waveform_type": recommendation.get("waveform_type"),
        "freq_hz": _normalize_scalar(recommendation.get("freq_hz")),
        "finite_cycle_mode": bool(recommendation.get("finite_cycle_mode", False)),
        "target_metric": recommendation.get("target_metric"),
        "target_value": _normalize_scalar(
            recommendation.get("requested_target_value", recommendation.get("target_value"))
        ),
        "used_target_value": _normalize_scalar(recommendation.get("used_target_value")),
        "command_waveform": _frame_to_records(
            recommendation.get("command_waveform"),
            include_optional_columns=waveform_columns,
        ),
    }
    return payload


def _frame_to_records(
    frame: Any,
    *,
    include_optional_columns: list[str] | None,
) -> list[dict[str, Any]]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []

    columns = (
        [column for column in include_optional_columns if column in frame.columns]
        if include_optional_columns is not None
        else frame.columns.tolist()
    )
    if not columns:
        return []

    selected = frame.loc[:, columns].copy()
    selected = selected.where(pd.notna(selected), None)
    return [{key: _normalize_scalar(value) for key, value in row.items()} for row in selected.to_dict("records")]


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value
