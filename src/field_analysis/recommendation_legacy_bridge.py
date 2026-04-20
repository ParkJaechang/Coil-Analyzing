from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .canonical_runs import CanonicalRun
from .compensation import (
    build_finite_support_entries as _build_finite_support_entries,
    build_harmonic_transfer_lut as _build_harmonic_transfer_lut,
    run_validation_recommendation_loop as _run_validation_recommendation_loop,
    synthesize_current_waveform_compensation as _synthesize_current_waveform_compensation,
    synthesize_finite_empirical_compensation as _synthesize_finite_empirical_compensation,
)
from .lut import recommend_voltage_waveform as _recommend_voltage_waveform
from .support_extraction import build_finite_support_entries_from_canonical


def canonical_run_to_legacy_frame(run: CanonicalRun) -> pd.DataFrame:
    """Minimal bridge for future legacy adapter removal."""

    frame = pd.DataFrame({"time_s": run.time_s, "daq_input_v": run.input_v, "i_sum_signed": run.signed_current_a})
    if run.bx_mT is not None:
        frame["bx_mT"] = run.bx_mT
    if run.by_mT is not None:
        frame["by_mT"] = run.by_mT
    if run.bz_mT is not None:
        frame["bz_mT"] = run.bz_mT
    frame["waveform_type"] = run.command_waveform
    frame["freq_hz"] = run.freq_hz
    frame["cycle_total_expected"] = run.commanded_cycles
    return frame


def build_finite_support_entries(
    transient_measurements: list[Any],
    transient_preprocess_results: list[Any],
    current_channel: str = "i_sum_signed",
    field_channel: str = "bz_mT",
    transient_canonical_runs: list[CanonicalRun] | None = None,
) -> list[dict[str, Any]]:
    """Finite-support bridge that prefers canonical segmented runs and falls back to legacy extraction."""

    if transient_canonical_runs:
        canonical_entries = build_finite_support_entries_from_canonical(
            transient_measurements=transient_measurements,
            transient_preprocess_results=transient_preprocess_results,
            transient_canonical_runs=transient_canonical_runs,
            current_channel=current_channel,
            field_channel=field_channel,
        )
        if canonical_entries:
            return canonical_entries
    return _build_finite_support_entries(
        transient_measurements=transient_measurements,
        transient_preprocess_results=transient_preprocess_results,
        current_channel=current_channel,
        field_channel=field_channel,
    )


def build_harmonic_transfer_lut(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Legacy harmonic LUT bridge exported for UI migration."""

    return _build_harmonic_transfer_lut(*args, **kwargs)


def run_validation_recommendation_loop(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _run_validation_recommendation_loop(*args, **kwargs)


def recommend_voltage_waveform(*args: Any, **kwargs: Any) -> dict[str, Any] | None:
    return _recommend_voltage_waveform(*args, **kwargs)


def synthesize_current_waveform_compensation(*args: Any, **kwargs: Any) -> dict[str, Any] | None:
    return _synthesize_current_waveform_compensation(*args, **kwargs)


def synthesize_finite_empirical_compensation(*args: Any, **kwargs: Any) -> dict[str, Any] | None:
    return _synthesize_finite_empirical_compensation(*args, **kwargs)


def _extract_command_profile(payload: dict[str, Any] | None) -> pd.DataFrame:
    return payload.get("command_profile", pd.DataFrame()) if payload else pd.DataFrame()


def _extract_lookup_table(payload: dict[str, Any] | None) -> pd.DataFrame:
    return payload.get("lookup_table", pd.DataFrame()) if payload else pd.DataFrame()


def _extract_support_table(payload: dict[str, Any] | None) -> pd.DataFrame:
    return payload.get("support_table", pd.DataFrame()) if payload else pd.DataFrame()


def _extract_series(frame: pd.DataFrame | None, candidates: list[str]) -> np.ndarray | None:
    if frame is None or frame.empty:
        return None
    for column in candidates:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return None
