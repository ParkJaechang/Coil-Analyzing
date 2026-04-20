from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .lut import build_voltage_template, recommend_voltage_waveform


AMPLITUDE_LUT_INPUTS = [
    "waveform_type",
    "freq_hz",
    "target_metric",
    "target_value",
    "frequency_mode",
    "finite_cycle_mode",
    "target_cycle_count",
    "preview_tail_cycles",
    "max_daq_voltage_pp",
    "amp_gain_at_100_pct",
    "amp_gain_limit_pct",
    "amp_max_output_pk_v",
]
AMPLITUDE_LUT_OUTPUTS = [
    "estimated_voltage_pp",
    "limited_voltage_pp",
    "required_amp_gain_pct",
    "support_amp_gain_pct",
    "estimated_current_pp",
    "estimated_bz_pp",
    "estimated_bmag_pp",
    "template_test_id",
    "template_waveform",
    "command_waveform",
]

_TEMPLATE_MIXED_CORR = 0.999
_TEMPLATE_MIXED_RMSE = 0.01


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _finite_values(series: pd.Series) -> list[float]:
    numeric = pd.to_numeric(series, errors="coerce")
    return sorted(float(value) for value in numeric.dropna().tolist() if np.isfinite(value))


def _safe_corr(reference: np.ndarray, candidate: np.ndarray) -> float | None:
    valid = np.isfinite(reference) & np.isfinite(candidate)
    if valid.sum() < 8:
        return None
    left = reference[valid]
    right = candidate[valid]
    if np.allclose(np.nanstd(left), 0.0) or np.allclose(np.nanstd(right), 0.0):
        return None
    return _safe_float(np.corrcoef(left, right)[0, 1])


def _normalized_rmse(reference: np.ndarray, candidate: np.ndarray) -> float | None:
    valid = np.isfinite(reference) & np.isfinite(candidate)
    if valid.sum() < 8:
        return None
    left = reference[valid]
    right = candidate[valid]
    scale = max(float(np.nanmax(np.abs(left))), 1e-12)
    return _safe_float(np.sqrt(np.mean(np.square(left - right))) / scale)


def _template_pairwise_metrics(
    template_profiles: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left_id, right_id in combinations(sorted(template_profiles), 2):
        left = template_profiles[left_id]
        right = template_profiles[right_id]
        rows.append(
            {
                "left_template_id": left_id,
                "right_template_id": right_id,
                "shape_corr": _safe_corr(left, right),
                "shape_nrmse": _normalized_rmse(left, right),
            }
        )
    return rows


def build_amplitude_lut_audit(
    *,
    per_test_summary: pd.DataFrame,
    analyses_by_test_id: dict[str, Any],
    max_freq_hz: float = 5.0,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    filtered = per_test_summary.copy()
    filtered = filtered[pd.to_numeric(filtered["freq_hz"], errors="coerce") <= float(max_freq_hz)].copy()
    filtered = filtered.dropna(subset=["waveform_type", "freq_hz"]).copy()
    for (waveform_type, freq_hz), _group in filtered.groupby(["waveform_type", "freq_hz"], sort=True):
        group = _group.copy()
        for target_metric in ("achieved_current_pp_a_mean", "achieved_bz_mT_pp_mean"):
            if target_metric not in group.columns:
                continue
            target_values = _finite_values(group[target_metric])
            if len(target_values) < 2:
                continue
            evaluations: list[dict[str, Any]] = []
            template_profiles: dict[str, np.ndarray] = {}
            for target_value in target_values:
                result = recommend_voltage_waveform(
                    per_test_summary=per_test_summary,
                    analyses_by_test_id=analyses_by_test_id,
                    waveform_type=str(waveform_type),
                    freq_hz=float(freq_hz),
                    target_metric=target_metric,
                    target_value=float(target_value),
                    frequency_mode="exact",
                )
                if result is None:
                    continue
                template_test_id = str(result.get("template_test_id") or "")
                evaluations.append(
                    {
                        "target_value": float(target_value),
                        "estimated_voltage_pp": _safe_float(result.get("estimated_voltage_pp")),
                        "limited_voltage_pp": _safe_float(result.get("limited_voltage_pp")),
                        "required_amp_gain_pct": _safe_float(result.get("required_amp_gain_pct")),
                        "support_amp_gain_pct": _safe_float(result.get("support_amp_gain_pct")),
                        "template_test_id": template_test_id or None,
                    }
                )
                analysis = analyses_by_test_id.get(template_test_id)
                if template_test_id and template_test_id not in template_profiles and analysis is not None:
                    template = build_voltage_template(
                        analysis,
                        voltage_channel="daq_input_v",
                        fallback_waveform=str(waveform_type),
                        fallback_freq_hz=float(freq_hz),
                        points_per_cycle=256,
                    )
                    template_profiles[template_test_id] = template["voltage_normalized"].to_numpy(dtype=float)
            pairwise = _template_pairwise_metrics(template_profiles)
            distinct_template_ids = sorted(template_profiles)
            observed_shape_meaning = (
                "mixed"
                if any(
                    (metric.get("shape_corr") is not None and float(metric["shape_corr"]) < _TEMPLATE_MIXED_CORR)
                    or (metric.get("shape_nrmse") is not None and float(metric["shape_nrmse"]) > _TEMPLATE_MIXED_RMSE)
                    for metric in pairwise
                )
                else ("amplitude_only" if distinct_template_ids else "unknown")
            )
            rows.append(
                {
                    "waveform_type": str(waveform_type),
                    "freq_hz": float(freq_hz),
                    "target_metric": target_metric,
                    "implementation_meaning": "mixed",
                    "observed_shape_meaning": observed_shape_meaning,
                    "pp_influences_template_selection": len(distinct_template_ids) > 1,
                    "template_test_ids": distinct_template_ids,
                    "evaluation_count": int(len(evaluations)),
                    "evaluations": evaluations,
                    "template_pairwise_metrics": pairwise,
                    "affects_support_selection": len(distinct_template_ids) > 1,
                    "affects_route": False,
                    "affects_shape_prediction": observed_shape_meaning == "mixed",
                }
            )
    return {
        "amplitude_lut_meaning": "mixed" if rows else "unknown",
        "observed_behavior_classification": (
            "mixed"
            if any(row["observed_shape_meaning"] == "mixed" for row in rows)
            else ("amplitude_only_candidate" if rows else "unknown")
        ),
        "current_lut_inputs": AMPLITUDE_LUT_INPUTS,
        "current_lut_outputs": AMPLITUDE_LUT_OUTPUTS,
        "groups": rows,
        "summary": {
            "group_count": int(len(rows)),
            "mixed_observed_group_count": int(sum(row["observed_shape_meaning"] == "mixed" for row in rows)),
            "template_switch_group_count": int(sum(bool(row["pp_influences_template_selection"]) for row in rows)),
        },
    }
