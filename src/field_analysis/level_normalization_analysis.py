from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .analysis import build_shape_phase_comparison


SHAPE_ENGINE_INPUTS = ["regime", "waveform", "freq_hz", "cycle_count"]
SHAPE_ENGINE_OUTPUTS = ["normalized_bz_shape"]

_STABLE_SHAPE_CORR = 0.995
_STABLE_SHAPE_NRMSE = 0.08
_STABLE_PHASE_CYCLES = 0.03


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _resample_signal(
    frame: pd.DataFrame | None,
    *,
    columns: list[str],
    points: int = 256,
) -> np.ndarray:
    if frame is None or frame.empty:
        return np.array([], dtype=float)
    signal_column = next((column for column in columns if column in frame.columns), None)
    if signal_column is None:
        return np.array([], dtype=float)
    if "cycle_progress" in frame.columns:
        x = pd.to_numeric(frame["cycle_progress"], errors="coerce").to_numpy(dtype=float)
    elif "time_s" in frame.columns:
        time_values = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
        valid_time = np.isfinite(time_values)
        if valid_time.sum() < 8:
            return np.array([], dtype=float)
        duration = float(np.nanmax(time_values[valid_time]) - np.nanmin(time_values[valid_time]))
        if not np.isfinite(duration) or duration <= 1e-9:
            return np.array([], dtype=float)
        x = (time_values - float(np.nanmin(time_values[valid_time]))) / duration
    else:
        return np.array([], dtype=float)
    y = pd.to_numeric(frame[signal_column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 8:
        return np.array([], dtype=float)
    order = np.argsort(x[valid])
    x_valid = x[valid][order]
    y_valid = y[valid][order]
    x_unique, indices = np.unique(x_valid, return_index=True)
    y_unique = y_valid[indices]
    if x_unique.size < 8:
        return np.array([], dtype=float)
    if x_unique[0] > 0.0:
        x_unique = np.insert(x_unique, 0, 0.0)
        y_unique = np.insert(y_unique, 0, y_unique[0])
    if x_unique[-1] < 1.0:
        x_unique = np.append(x_unique, 1.0)
        y_unique = np.append(y_unique, y_unique[-1])
    grid = np.linspace(0.0, 1.0, max(int(points), 32))
    return np.interp(grid, x_unique, y_unique)


def _normalize_signal(values: np.ndarray) -> np.ndarray:
    signal = np.asarray(values, dtype=float)
    finite = np.isfinite(signal)
    if finite.sum() < 8:
        return np.array([], dtype=float)
    signal = signal[finite]
    centered = signal - float(np.nanmean(signal))
    pp = float(np.nanmax(centered) - np.nanmin(centered))
    if not np.isfinite(pp) or pp <= 1e-9:
        return np.array([], dtype=float)
    return centered / (pp / 2.0)


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


def _best_alignment_shift(reference: np.ndarray, candidate: np.ndarray) -> int:
    valid = np.isfinite(reference) & np.isfinite(candidate)
    if valid.sum() < 8:
        return 0
    ref = reference[valid]
    cand = candidate[valid]
    max_shift = max(1, len(ref) // 4)
    best_shift = 0
    best_score = -np.inf
    for shift in range(-max_shift, max_shift + 1):
        shifted = np.roll(cand, shift)
        score = _safe_corr(ref, shifted)
        if score is not None and score > best_score:
            best_shift = shift
            best_score = score
    return int(best_shift)


def _aligned_shape_metrics(reference: np.ndarray, candidate: np.ndarray, freq_hz: float | None) -> dict[str, float | None]:
    if reference.size == 0 or candidate.size == 0:
        return {
            "shape_corr": None,
            "nrmse": None,
            "phase_lag_samples": None,
            "phase_lag_cycles": None,
            "phase_lag_s": None,
        }
    shift = _best_alignment_shift(reference, candidate)
    aligned = np.roll(candidate, shift)
    phase_lag_samples = int(-shift)
    phase_lag_cycles = phase_lag_samples / max(len(candidate), 1)
    phase_lag_s = phase_lag_cycles / float(freq_hz) if _safe_float(freq_hz) not in (None, 0.0) else None
    return {
        "shape_corr": _safe_corr(reference, aligned),
        "nrmse": _normalized_rmse(reference, aligned),
        "phase_lag_samples": phase_lag_samples,
        "phase_lag_cycles": phase_lag_cycles,
        "phase_lag_s": phase_lag_s,
    }


def _shape_stability_flag(corr_min: float | None, nrmse_max: float | None, phase_cycles_max: float | None) -> bool | str:
    if corr_min is None or nrmse_max is None or phase_cycles_max is None:
        return "unknown"
    stable = bool(
        corr_min >= _STABLE_SHAPE_CORR
        and nrmse_max <= _STABLE_SHAPE_NRMSE
        and phase_cycles_max <= _STABLE_PHASE_CYCLES
    )
    return False if stable else True


def _preview_shape(overlay: pd.DataFrame) -> list[float]:
    if overlay.empty:
        return []
    grouped = overlay.groupby("cycle_progress", sort=True)["normalized_signal_aligned"].mean().reset_index(drop=False)
    return [float(value) for value in grouped["normalized_signal_aligned"].head(16).tolist()]


def _continuous_group_summary(
    *,
    analyses: list[Any],
    waveform_type: str,
    freq_hz: float,
    current_channel: str,
    field_channel: str,
) -> dict[str, Any]:
    signal_summaries: dict[str, Any] = {}
    overlay_bz = pd.DataFrame()
    for signal_channel in (field_channel, current_channel, "daq_input_v"):
        overlay, summary = build_shape_phase_comparison(
            analyses=analyses,
            waveform_type=waveform_type,
            freq_hz=freq_hz,
            signal_channel=signal_channel,
            current_channel=current_channel,
            main_field_axis=field_channel,
            normalization_mode="peak_to_peak",
            points_per_cycle=256,
        )
        if signal_channel == field_channel:
            overlay_bz = overlay
        if summary.empty:
            continue
        phase_cycles = summary["phase_lag_seconds"].abs().fillna(np.nan) * float(freq_hz)
        signal_summaries[signal_channel] = {
            "reference_test_id": str(summary["reference_test_id"].iloc[0]),
            "level_count": int(len(summary)),
            "level_values": [float(value) for value in summary["current_pp_target_a"].dropna().tolist()],
            "shape_corr_min": _safe_float(summary["shape_corr_aligned"].min()),
            "shape_corr_median": _safe_float(summary["shape_corr_aligned"].median()),
            "shape_nrmse_max": _safe_float(summary["shape_nrmse_aligned"].max()),
            "phase_lag_cycles_max": _safe_float(phase_cycles.max()),
            "phase_lag_s_max": _safe_float(summary["phase_lag_seconds"].abs().max()),
            "preview": _preview_shape(overlay),
        }
    bz_summary = signal_summaries.get(field_channel, {})
    return {
        "regime": "continuous_exact",
        "waveform_type": waveform_type,
        "freq_hz": float(freq_hz),
        "signal_summaries": signal_summaries,
        "shape_engine_source": "normalized_exact_support_mean",
        "pp_affects_shape": _shape_stability_flag(
            bz_summary.get("shape_corr_min"),
            bz_summary.get("shape_nrmse_max"),
            bz_summary.get("phase_lag_cycles_max"),
        ),
        "normalized_bz_shape_preview": bz_summary.get("preview", []),
    }


def _finite_group_summary(
    *,
    entries: list[dict[str, Any]],
    waveform_type: str,
    freq_hz: float,
    cycle_count: float,
    current_channel: str,
    field_channel: str,
) -> dict[str, Any]:
    ordered = sorted(
        entries,
        key=lambda item: (_safe_float(item.get("requested_current_pp")) or np.inf, str(item.get("test_id") or "")),
    )
    resampled: list[dict[str, Any]] = []
    for entry in ordered:
        active_frame = entry.get("active_frame") if isinstance(entry.get("active_frame"), pd.DataFrame) else entry.get("frame")
        current_norm = _normalize_signal(
            _resample_signal(active_frame, columns=[str(entry.get("resolved_current_channel") or ""), current_channel, "i_sum_signed"])
        )
        field_norm = _normalize_signal(
            _resample_signal(active_frame, columns=[str(entry.get("resolved_field_channel") or ""), field_channel, "bz_mT", "bproj_mT", "bmag_mT"])
        )
        if current_norm.size == 0 and field_norm.size == 0:
            continue
        resampled.append(
            {
                "test_id": str(entry.get("test_id") or ""),
                "requested_level_pp": _safe_float(entry.get("requested_current_pp")),
                "current_norm": current_norm,
                "field_norm": field_norm,
            }
        )
    signal_summaries: dict[str, Any] = {}
    if not resampled:
        return {
            "regime": "finite_exact",
            "waveform_type": waveform_type,
            "freq_hz": float(freq_hz),
            "cycle_count": float(cycle_count),
            "signal_summaries": signal_summaries,
            "shape_engine_source": "normalized_exact_support_mean",
            "pp_affects_shape": "unknown",
            "normalized_bz_shape_preview": [],
        }
    reference = resampled[0]
    for label, key in (("bz_mT", "field_norm"), ("i_sum_signed", "current_norm")):
        reference_signal = np.asarray(reference[key], dtype=float)
        if reference_signal.size == 0:
            continue
        rows: list[dict[str, Any]] = []
        for item in resampled:
            rows.append(
                {
                    "test_id": item["test_id"],
                    "requested_level_pp": item["requested_level_pp"],
                    **_aligned_shape_metrics(reference_signal, np.asarray(item[key], dtype=float), freq_hz),
                }
            )
        frame = pd.DataFrame(rows)
        signal_summaries[label] = {
            "reference_test_id": reference["test_id"],
            "level_count": int(len(frame)),
            "level_values": [value for value in frame["requested_level_pp"].tolist() if value is not None],
            "shape_corr_min": _safe_float(frame["shape_corr"].min()),
            "shape_corr_median": _safe_float(frame["shape_corr"].median()),
            "shape_nrmse_max": _safe_float(frame["nrmse"].max()),
            "phase_lag_cycles_max": _safe_float(frame["phase_lag_cycles"].abs().max()),
            "phase_lag_s_max": _safe_float(frame["phase_lag_s"].abs().max()),
            "preview": [float(value) for value in reference_signal[:16].tolist()],
        }
    bz_summary = signal_summaries.get("bz_mT", {})
    return {
        "regime": "finite_exact",
        "waveform_type": waveform_type,
        "freq_hz": float(freq_hz),
        "cycle_count": float(cycle_count),
        "signal_summaries": signal_summaries,
        "shape_engine_source": "normalized_exact_support_mean",
        "pp_affects_shape": _shape_stability_flag(
            bz_summary.get("shape_corr_min"),
            bz_summary.get("shape_nrmse_max"),
            bz_summary.get("phase_lag_cycles_max"),
        ),
        "normalized_bz_shape_preview": bz_summary.get("preview", []),
    }


def build_shape_engine_audit(
    *,
    analyses_by_test_id: dict[str, Any],
    per_test_summary: pd.DataFrame,
    finite_entries: list[dict[str, Any]],
    current_channel: str = "i_sum_signed",
    field_channel: str = "bz_mT",
    max_freq_hz: float = 5.0,
) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    analyses = list(analyses_by_test_id.values())
    continuous = per_test_summary[pd.to_numeric(per_test_summary["freq_hz"], errors="coerce") <= float(max_freq_hz)].copy()
    for (waveform_type, freq_hz), group in continuous.groupby(["waveform_type", "freq_hz"], sort=True):
        if len(group) < 2:
            continue
        groups.append(
            _continuous_group_summary(
                analyses=analyses,
                waveform_type=str(waveform_type),
                freq_hz=float(freq_hz),
                current_channel=current_channel,
                field_channel=field_channel,
            )
        )
    finite_frame = pd.DataFrame(finite_entries)
    if not finite_frame.empty:
        finite_frame = finite_frame[pd.to_numeric(finite_frame["freq_hz"], errors="coerce") <= float(max_freq_hz)].copy()
        for (waveform_type, freq_hz, cycle_count), group in finite_frame.groupby(["waveform_type", "freq_hz", "requested_cycle_count"], sort=True):
            if len(group) < 2:
                continue
            groups.append(
                _finite_group_summary(
                    entries=group.to_dict(orient="records"),
                    waveform_type=str(waveform_type),
                    freq_hz=float(freq_hz),
                    cycle_count=float(cycle_count),
                    current_channel=current_channel,
                    field_channel=field_channel,
                )
            )
    return {
        "prototype": {
            "shape_engine_source": "normalized_exact_support_mean",
            "inputs": SHAPE_ENGINE_INPUTS,
            "outputs": SHAPE_ENGINE_OUTPUTS,
            "normalization": "peak_to_peak",
            "alignment": "best_correlation_shift",
        },
        "groups": groups,
        "summary": {
            "group_count": int(len(groups)),
            "stable_group_count": int(sum(group.get("pp_affects_shape") is False for group in groups)),
            "unstable_group_count": int(sum(group.get("pp_affects_shape") is True for group in groups)),
        },
    }


def build_same_freq_level_sensitivity(*, shape_engine_audit: dict[str, Any]) -> dict[str, Any]:
    groups = list(shape_engine_audit.get("groups", []))
    return {
        "comparison_basis": {
            "normalization": "peak_to_peak",
            "alignment": "best_correlation_shift",
        },
        "continuous_exact": [group for group in groups if group.get("regime") == "continuous_exact"],
        "finite_exact": [group for group in groups if group.get("regime") == "finite_exact"],
        "summary": {
            "continuous_group_count": int(sum(group.get("regime") == "continuous_exact" for group in groups)),
            "finite_group_count": int(sum(group.get("regime") == "finite_exact" for group in groups)),
            "stable_group_count": int(sum(group.get("pp_affects_shape") is False for group in groups)),
            "unstable_group_count": int(sum(group.get("pp_affects_shape") is True for group in groups)),
        },
    }


def build_support_route_level_influence(*, probe_groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for label, probes in probe_groups.items():
        ordered = sorted(probes, key=lambda item: _safe_float(item.get("target_level_pp")) or np.inf)
        support_ids = {str(item.get("selected_support_id")) for item in ordered if item.get("selected_support_id")}
        prediction_sources = {
            (str(item.get("field_prediction_source") or ""), str(item.get("field_prediction_status") or ""))
            for item in ordered
        }
        shape_sources = {str(item.get("shape_engine_source") or "") for item in ordered if item.get("shape_engine_source")}
        amplitude_sources = {str(item.get("amplitude_engine_source") or "") for item in ordered if item.get("amplitude_engine_source")}
        clipping_or_limit = any(bool(item.get("predicted_clipping")) or not bool(item.get("within_hardware_limits", True)) for item in ordered)
        corr_values = [_safe_float(item.get("predicted_shape_corr")) for item in ordered]
        corr_values = [value for value in corr_values if value is not None]
        predicted_shape_corr_span = float(max(corr_values) - min(corr_values)) if len(corr_values) >= 2 else None
        reason_codes: list[str] = []
        if len(support_ids) > 1:
            reason_codes.append("support_id_switch")
        if len(prediction_sources) > 1:
            reason_codes.append("prediction_source_switch")
        if clipping_or_limit:
            reason_codes.append("limit_induced_switch")
        if predicted_shape_corr_span is not None and predicted_shape_corr_span > 0.25 and not reason_codes:
            reason_codes.append("true_nonlinear_shape_change")
        rows.append(
            {
                "probe_group": label,
                "levels": [_safe_float(item.get("target_level_pp")) for item in ordered],
                "probes": ordered,
                "selected_support_ids": sorted(support_ids),
                "prediction_sources": sorted(f"{source}:{status}" for source, status in prediction_sources),
                "shape_engine_sources": sorted(shape_sources),
                "amplitude_engine_sources": sorted(amplitude_sources),
                "clipping_or_limit_detected": clipping_or_limit,
                "predicted_shape_corr_span": predicted_shape_corr_span,
                "pp_affects_shape": bool(reason_codes) if ordered else "unknown",
                "reason_codes": reason_codes,
            }
        )
    return {
        "probe_groups": rows,
        "summary": {
            "group_count": int(len(rows)),
            "shape_affected_group_count": int(sum(group.get("pp_affects_shape") is True for group in rows)),
            "support_switch_group_count": int(sum("support_id_switch" in group.get("reason_codes", []) for group in rows)),
            "source_switch_group_count": int(sum("prediction_source_switch" in group.get("reason_codes", []) for group in rows)),
        },
    }
