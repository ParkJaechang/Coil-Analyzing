from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .canonical_runs import CanonicalRun
from .models import CycleDetectionConfig, ParsedMeasurement, PreprocessResult
from .segmentation import build_analysis_frame_from_canonical, segment_canonical_run
from .utils import canonicalize_waveform_type


def _resolve_signal_column(frame: pd.DataFrame, candidates: list[str], default: str) -> str:
    for column in candidates:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            return column
    return default


def build_finite_support_entries_from_canonical(
    transient_measurements: list[ParsedMeasurement],
    transient_preprocess_results: list[PreprocessResult],
    transient_canonical_runs: list[CanonicalRun],
    current_channel: str = "i_sum_signed",
    field_channel: str = "bz_mT",
) -> list[dict[str, Any]]:
    """Build finite support entries using canonical segmentation instead of raw duration heuristics."""

    entries: list[dict[str, Any]] = []
    for parsed, preprocess, canonical_run in zip(
        transient_measurements,
        transient_preprocess_results,
        transient_canonical_runs,
        strict=False,
    ):
        corrected_projected = build_analysis_frame_from_canonical(
            parsed,
            canonical_run,
            source_frame=preprocess.corrected_frame,
        )
        resolved_current_channel = _resolve_signal_column(
            corrected_projected,
            [current_channel, "signed_current_a", "i_custom_signed", "coil1_current_a"],
            current_channel,
        )
        resolved_field_channel = _resolve_signal_column(
            corrected_projected,
            [field_channel, "bz_mT", "bproj_mT", "bmag_mT"],
            field_channel,
        )
        cycle_result = segment_canonical_run(
            canonical_run,
            corrected_projected,
            CycleDetectionConfig(
                reference_channel=resolved_current_channel,
                expected_cycles=max(int(np.ceil(canonical_run.commanded_cycles or 1.0)), 1),
            ),
        )
        annotated = cycle_result.annotated_frame.copy()
        active_mask = annotated["cycle_index"].notna() if "cycle_index" in annotated.columns else pd.Series(False, index=annotated.index)
        if not active_mask.any() and canonical_run.active_window_s is not None:
            start_s, end_s = canonical_run.active_window_s
            active_mask = annotated["time_s"].between(start_s, end_s, inclusive="both")
        active_frame = annotated.loc[active_mask].copy() if active_mask.any() else annotated.copy()
        if active_frame.empty:
            active_frame = annotated.copy()

        duration_s = (
            float(pd.to_numeric(active_frame["time_s"], errors="coerce").max() - pd.to_numeric(active_frame["time_s"], errors="coerce").min())
            if "time_s" in active_frame.columns and not active_frame.empty
            else float("nan")
        )
        approx_cycle_span = (
            float(canonical_run.commanded_cycles)
            if canonical_run.commanded_cycles is not None and np.isfinite(canonical_run.commanded_cycles)
            else (
                duration_s * float(canonical_run.freq_hz)
                if np.isfinite(duration_s) and canonical_run.freq_hz is not None and np.isfinite(canonical_run.freq_hz)
                else float("nan")
            )
        )
        entries.append(
            {
                "test_id": _frame_first_value(active_frame, "test_id", canonical_run.run_id),
                "source_file": parsed.source_file,
                "sheet_name": parsed.sheet_name,
                "waveform_type": canonicalize_waveform_type(canonical_run.command_waveform),
                "freq_hz": float(canonical_run.freq_hz) if canonical_run.freq_hz is not None else float("nan"),
                "duration_s": duration_s,
                "approx_cycle_span": approx_cycle_span,
                "estimated_cycle_span": duration_s * float(canonical_run.freq_hz)
                if np.isfinite(duration_s) and canonical_run.freq_hz is not None and np.isfinite(canonical_run.freq_hz)
                else float("nan"),
                "requested_cycle_count": float(canonical_run.commanded_cycles)
                if canonical_run.commanded_cycles is not None
                else float("nan"),
                "target_current_a": _frame_first_value(active_frame, "current_pk_target_a", np.nan),
                "requested_current_pp": _frame_first_value(active_frame, "current_pp_target_a", np.nan),
                "requested_current_pk": _frame_first_value(active_frame, "current_pk_target_a", np.nan),
                "notes": _frame_first_value(active_frame, "notes", ""),
                "current_pp": _signal_peak_to_peak(active_frame, resolved_current_channel),
                "field_pp": _signal_peak_to_peak(active_frame, resolved_field_channel),
                "daq_voltage_pp": _signal_peak_to_peak(active_frame, "daq_input_v"),
                "frame": annotated,
                "active_frame": active_frame,
                "segmentation_mode": "canonical_transient",
                "boundary_count": len(cycle_result.boundaries),
                "quality_flags": sorted(canonical_run.quality_flags),
                "resolved_current_channel": resolved_current_channel,
                "resolved_field_channel": resolved_field_channel,
            }
        )
    return entries


def _signal_peak_to_peak(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan")
    return float(np.nanmax(finite) - np.nanmin(finite))


def _frame_first_value(frame: pd.DataFrame, column: str, default: object) -> object:
    if frame.empty or column not in frame.columns:
        return default
    series = frame[column].dropna()
    if series.empty:
        return default
    return series.iloc[0]
