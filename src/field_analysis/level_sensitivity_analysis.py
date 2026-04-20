from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _parse_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return {}
        if isinstance(parsed, dict):
            return dict(parsed)
    return {}


def _peak_to_peak(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.nanmax(finite) - np.nanmin(finite))


def _normalized_signal(values: np.ndarray) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    if finite.size == 0:
        return finite
    centered = finite - float(np.nanmean(finite[np.isfinite(finite)])) if np.isfinite(finite).any() else finite
    signal_pp = _peak_to_peak(centered)
    if not np.isfinite(signal_pp) or signal_pp <= 1e-9:
        return centered
    return centered / signal_pp


def normalized_shape_difference(reference: np.ndarray, candidate: np.ndarray) -> float:
    valid = np.isfinite(reference) & np.isfinite(candidate)
    if valid.sum() < 8:
        return float("nan")
    ref = _normalized_signal(reference[valid])
    comp = _normalized_signal(candidate[valid])
    return float(np.sqrt(np.nanmean(np.square(ref - comp))))


def _shape_corr(reference: np.ndarray, candidate: np.ndarray) -> float:
    valid = np.isfinite(reference) & np.isfinite(candidate)
    if valid.sum() < 8:
        return float("nan")
    ref = reference[valid]
    comp = candidate[valid]
    if np.nanstd(ref) <= 1e-12 or np.nanstd(comp) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(ref, comp)[0, 1])


def extract_profile_case(profile_path: str | Path, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    profile = pd.read_csv(profile_path)
    first = profile.iloc[0] if not profile.empty else pd.Series(dtype=object)
    harmonic_weights = _parse_mapping(first.get("harmonic_weights_used"))
    expected_field = pd.to_numeric(profile.get("expected_field_mT", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    expected_current = pd.to_numeric(profile.get("expected_current_a", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    return {
        "profile_path": str(profile_path),
        "recommendation_id": (metadata or {}).get("recommendation_id"),
        "waveform_type": (metadata or {}).get("waveform_type") or first.get("waveform_type"),
        "freq_hz": float(pd.to_numeric(pd.Series([(metadata or {}).get("freq_hz", first.get("freq_hz"))]), errors="coerce").iloc[0]),
        "commanded_cycles": pd.to_numeric(pd.Series([(metadata or {}).get("commanded_cycles", first.get("target_cycle_count"))]), errors="coerce").iloc[0],
        "target_type": (metadata or {}).get("target_type") or first.get("target_output_type"),
        "target_level_value": float(pd.to_numeric(pd.Series([(metadata or {}).get("target_level_value", first.get("target_output_pp"))]), errors="coerce").iloc[0]),
        "request_route": first.get("request_route"),
        "plot_source": first.get("plot_source"),
        "selected_support_id": metadata.get("selected_support_id") if metadata else None,
        "solver_route": metadata.get("solver_route") if metadata else None,
        "field_prediction_source": metadata.get("field_prediction_source") if metadata else None,
        "clipping_flags": {
            "within_hardware_limits": bool(str(first.get("within_hardware_limits", "True")).lower() == "true"),
            "within_daq_limit": bool(str(first.get("within_daq_limit", "True")).lower() == "true"),
        },
        "harmonic_weights": harmonic_weights,
        "predicted_field_pp": _peak_to_peak(expected_field),
        "predicted_current_pp": _peak_to_peak(expected_current),
        "expected_field_mT": expected_field,
    }


def classify_level_switch(reference_case: dict[str, Any], candidate_case: dict[str, Any]) -> dict[str, Any]:
    reference_signal = np.asarray(reference_case.get("expected_field_mT", []), dtype=float)
    candidate_signal = np.asarray(candidate_case.get("expected_field_mT", []), dtype=float)
    normalized_diff = normalized_shape_difference(reference_signal, candidate_signal)
    shape_corr = _shape_corr(reference_signal, candidate_signal)

    switch_flags: list[str] = []
    if reference_case.get("selected_support_id") and candidate_case.get("selected_support_id") and reference_case.get("selected_support_id") != candidate_case.get("selected_support_id"):
        switch_flags.append("support_id_switch")
    if (
        reference_case.get("field_prediction_source") != candidate_case.get("field_prediction_source")
        or reference_case.get("plot_source") != candidate_case.get("plot_source")
        or reference_case.get("solver_route") != candidate_case.get("solver_route")
    ):
        switch_flags.append("prediction_source_switch")
    if reference_case.get("clipping_flags") != candidate_case.get("clipping_flags"):
        switch_flags.append("limit_induced_switch")
    if not switch_flags and np.isfinite(normalized_diff) and normalized_diff > 0.15:
        switch_flags.append("true_nonlinear_shape_change")

    return {
        "reference_id": reference_case.get("recommendation_id"),
        "candidate_id": candidate_case.get("recommendation_id"),
        "waveform_type": candidate_case.get("waveform_type"),
        "freq_hz": candidate_case.get("freq_hz"),
        "reference_level": reference_case.get("target_level_value"),
        "candidate_level": candidate_case.get("target_level_value"),
        "selected_support_id": {
            "reference": reference_case.get("selected_support_id"),
            "candidate": candidate_case.get("selected_support_id"),
        },
        "solver_route": {
            "reference": reference_case.get("solver_route"),
            "candidate": candidate_case.get("solver_route"),
        },
        "field_prediction_source": {
            "reference": reference_case.get("field_prediction_source"),
            "candidate": candidate_case.get("field_prediction_source"),
        },
        "clipping_flags": {
            "reference": reference_case.get("clipping_flags"),
            "candidate": candidate_case.get("clipping_flags"),
        },
        "harmonic_weights": {
            "reference": reference_case.get("harmonic_weights"),
            "candidate": candidate_case.get("harmonic_weights"),
        },
        "predicted_bz_shape_corr": shape_corr,
        "normalized_shape_difference": normalized_diff,
        "switch_types": switch_flags,
    }


def build_level_sensitivity_diagnosis(cases: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for case in cases:
        key = (
            case.get("waveform_type"),
            case.get("freq_hz"),
            case.get("target_type"),
            case.get("commanded_cycles"),
        )
        grouped.setdefault(key, []).append(case)

    comparisons: list[dict[str, Any]] = []
    for group_cases in grouped.values():
        ordered = sorted(group_cases, key=lambda item: float(item.get("target_level_value") or 0.0))
        for reference_case, candidate_case in zip(ordered, ordered[1:], strict=False):
            comparisons.append(classify_level_switch(reference_case, candidate_case))

    summary = {
        "comparison_count": len(comparisons),
        "support_id_switch": sum("support_id_switch" in item["switch_types"] for item in comparisons),
        "prediction_source_switch": sum("prediction_source_switch" in item["switch_types"] for item in comparisons),
        "limit_induced_switch": sum("limit_induced_switch" in item["switch_types"] for item in comparisons),
        "true_nonlinear_shape_change": sum("true_nonlinear_shape_change" in item["switch_types"] for item in comparisons),
    }
    return {
        "cases": cases,
        "comparisons": comparisons,
        "summary": summary,
    }


def render_level_sensitivity_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Level Sensitivity Diagnosis",
        "",
        f"- comparisons: `{payload.get('summary', {}).get('comparison_count', 0)}`",
        f"- support_id_switch: `{payload.get('summary', {}).get('support_id_switch', 0)}`",
        f"- prediction_source_switch: `{payload.get('summary', {}).get('prediction_source_switch', 0)}`",
        f"- limit_induced_switch: `{payload.get('summary', {}).get('limit_induced_switch', 0)}`",
        f"- true_nonlinear_shape_change: `{payload.get('summary', {}).get('true_nonlinear_shape_change', 0)}`",
        "",
    ]
    for item in payload.get("comparisons", [])[:20]:
        lines.extend(
            [
                f"## {item.get('reference_id')} -> {item.get('candidate_id')}",
                "",
                f"- waveform/freq: `{item.get('waveform_type')} / {item.get('freq_hz')} Hz`",
                f"- level: `{item.get('reference_level')}` -> `{item.get('candidate_level')}`",
                f"- switch_types: `{', '.join(item.get('switch_types', [])) or 'none'}`",
                f"- normalized_shape_difference: `{item.get('normalized_shape_difference')}`",
                f"- predicted_bz_shape_corr: `{item.get('predicted_bz_shape_corr')}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
