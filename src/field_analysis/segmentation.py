from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .canonical_runs import CanonicalRun
from .cycle_detection import annotate_cycles, build_boundaries_from_times, detect_cycles
from .models import CycleDetectionConfig, CycleDetectionResult, ParsedMeasurement
from .utils import apply_bz_effective_convention


def _resolve_source_signal(source: pd.DataFrame, candidates: list[str], target_time: np.ndarray) -> np.ndarray | None:
    source_time = pd.to_numeric(source.get("time_s"), errors="coerce").to_numpy(dtype=float) if "time_s" in source.columns else np.array([], dtype=float)
    for column in candidates:
        if column not in source.columns:
            continue
        values = pd.to_numeric(source[column], errors="coerce")
        if values.notna().sum() < 2:
            continue
        return _interp_numeric(source_time, values, target_time)
    return None


def build_analysis_frame_from_canonical(
    parsed: ParsedMeasurement,
    canonical_run: CanonicalRun,
    source_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Project a canonical run into the legacy analysis frame contract."""

    source = source_frame.copy() if source_frame is not None else parsed.normalized_frame.copy()
    if source.empty:
        source = pd.DataFrame({"time_s": canonical_run.time_s})
    if "time_s" not in source.columns:
        source["time_s"] = np.arange(len(source), dtype=float)

    source["time_s"] = pd.to_numeric(source["time_s"], errors="coerce")
    source = source.dropna(subset=["time_s"]).sort_values("time_s").reset_index(drop=True)
    source = source.loc[~source["time_s"].duplicated(keep="first")].reset_index(drop=True)

    frame = pd.DataFrame({"time_s": np.asarray(canonical_run.time_s, dtype=float)})
    source_time = pd.to_numeric(source["time_s"], errors="coerce").to_numpy(dtype=float)
    target_time = frame["time_s"].to_numpy(dtype=float)

    for column in source.columns:
        if column == "time_s":
            continue
        if pd.api.types.is_numeric_dtype(source[column]):
            frame[column] = _interp_numeric(source_time, source[column], target_time)
        else:
            non_null = source[column].dropna()
            frame[column] = non_null.iloc[0] if not non_null.empty else None

    frame["daq_input_v"] = _fit_signal(canonical_run.input_v, len(frame))
    signed_current = _fit_signal(canonical_run.signed_current_a, len(frame))
    if pd.to_numeric(pd.Series(signed_current), errors="coerce").notna().sum() < 2:
        fallback_current = _resolve_source_signal(
            source,
            ["i_sum_signed", "signed_current_a", "i_custom_signed", "coil1_current_a"],
            target_time,
        )
        if fallback_current is not None:
            signed_current = fallback_current
    frame["i_sum_signed"] = signed_current
    frame["i_custom_signed"] = frame["i_sum_signed"]
    if canonical_run.bx_mT is not None:
        frame["bx_mT"] = _fit_signal(canonical_run.bx_mT, len(frame))
    if canonical_run.by_mT is not None:
        frame["by_mT"] = _fit_signal(canonical_run.by_mT, len(frame))
    if canonical_run.bz_mT is not None:
        frame["bz_mT"] = _fit_signal(canonical_run.bz_mT, len(frame))
    if pd.to_numeric(frame["bz_mT"], errors="coerce").notna().sum() < 2:
        fallback_field = _resolve_source_signal(source, ["bz_mT", "bproj_mT", "bmag_mT"], target_time)
        if fallback_field is not None:
            frame["bz_mT"] = fallback_field

    for column in ("bx_mT", "by_mT", "bz_mT", "temperature_c", "coil1_current_a", "coil2_current_a"):
        if column not in frame.columns:
            frame[column] = np.nan

    frame["bmag_mT"] = np.sqrt(
        np.square(pd.to_numeric(frame["bx_mT"], errors="coerce").fillna(0.0))
        + np.square(pd.to_numeric(frame["by_mT"], errors="coerce").fillna(0.0))
        + np.square(pd.to_numeric(frame["bz_mT"], errors="coerce").fillna(0.0))
    )
    primary_column = f"{canonical_run.primary_field_axis}_mT" if canonical_run.primary_field_axis else "bz_mT"
    frame["bproj_mT"] = pd.to_numeric(frame.get(primary_column, frame["bz_mT"]), errors="coerce")

    frame["source_file"] = parsed.source_file
    frame["sheet_name"] = parsed.sheet_name
    frame["test_id"] = _first_or_default(frame, "test_id", canonical_run.run_id)
    frame["waveform_type"] = _first_or_default(frame, "waveform_type", canonical_run.command_waveform)
    frame["freq_hz"] = _first_or_default(frame, "freq_hz", canonical_run.freq_hz)
    frame["current_pp_target_a"] = _first_or_default(
        frame,
        "current_pp_target_a",
        canonical_run.target_level_value if canonical_run.target_level_kind == "pp" else np.nan,
    )
    frame["current_pk_target_a"] = _first_or_default(
        frame,
        "current_pk_target_a",
        canonical_run.target_level_value if canonical_run.target_level_kind == "peak" else np.nan,
    )
    frame["cycle_total_expected"] = _first_or_default(
        frame,
        "cycle_total_expected",
        canonical_run.commanded_cycles if canonical_run.commanded_cycles is not None else np.nan,
    )
    frame["notes"] = _first_or_default(frame, "notes", "")
    apply_bz_effective_convention(frame)
    return frame


def segment_canonical_run(
    canonical_run: CanonicalRun,
    corrected_frame: pd.DataFrame,
    config: CycleDetectionConfig,
) -> CycleDetectionResult:
    """Dispatch segmentation based on canonical regime."""

    if canonical_run.regime == "transient":
        return segment_transient_run(canonical_run, corrected_frame, config)
    return segment_continuous_cycles(canonical_run, corrected_frame, config)


def segment_continuous_cycles(
    canonical_run: CanonicalRun,
    corrected_frame: pd.DataFrame,
    config: CycleDetectionConfig,
) -> CycleDetectionResult:
    """Use the existing detector on canonicalized continuous frames."""

    expected_cycles = config.expected_cycles
    if canonical_run.commanded_cycles is not None:
        rounded = int(round(canonical_run.commanded_cycles))
        if np.isclose(canonical_run.commanded_cycles, rounded, atol=1e-9) and rounded > 0:
            expected_cycles = rounded

    return detect_cycles(
        corrected_frame,
        replace(
            config,
            expected_cycles=max(1, expected_cycles),
            reference_channel=config.reference_channel or "i_sum_signed",
        ),
    )


def segment_transient_run(
    canonical_run: CanonicalRun,
    corrected_frame: pd.DataFrame,
    config: CycleDetectionConfig,
) -> CycleDetectionResult:
    """Build finite-cycle boundaries from canonical metadata instead of CycleNo labels."""

    time = pd.to_numeric(corrected_frame["time_s"], errors="coerce").to_numpy(dtype=float)
    if len(time) < 2 or canonical_run.freq_hz is None or canonical_run.freq_hz <= 0:
        fallback = segment_continuous_cycles(canonical_run, corrected_frame, config)
        fallback.warnings.append("finite metadata 부족으로 continuous detector fallback을 사용했습니다.")
        return fallback

    period_s = 1.0 / float(canonical_run.freq_hz)
    start_s = float(canonical_run.active_window_s[0]) if canonical_run.active_window_s else float(time[0])
    requested_cycles = canonical_run.commanded_cycles if canonical_run.commanded_cycles and canonical_run.commanded_cycles > 0 else 1.0
    requested_end_s = start_s + float(requested_cycles) * period_s
    end_s = min(requested_end_s, float(time[-1]))

    boundary_times = [start_s]
    whole_cycles = int(np.floor(float(requested_cycles)))
    for cycle_index in range(1, whole_cycles + 1):
        candidate = start_s + cycle_index * period_s
        if candidate < end_s - 1e-9:
            boundary_times.append(candidate)
    if end_s > boundary_times[-1]:
        boundary_times.append(end_s)

    boundaries = build_boundaries_from_times(time, np.asarray(boundary_times, dtype=float))
    annotated = annotate_cycles(corrected_frame, boundaries, config.reference_channel or "i_sum_signed")

    warnings: list[str] = []
    if end_s < requested_end_s - max(period_s * 0.05, 1e-6):
        warnings.append("finite active window이 기대 duration보다 짧아 마지막 cycle이 부분적으로 잘렸습니다.")
    logs = [
        "segmentation_mode=transient",
        f"freq_hz={canonical_run.freq_hz}",
        f"requested_cycles={requested_cycles}",
        f"start_s={start_s}",
        f"end_s={end_s}",
    ]

    return CycleDetectionResult(
        annotated_frame=annotated,
        boundaries=boundaries,
        estimated_period_s=period_s,
        estimated_frequency_hz=float(canonical_run.freq_hz),
        reference_channel=config.reference_channel or "i_sum_signed",
        warnings=warnings,
        logs=logs,
    )


def _interp_numeric(source_time: np.ndarray, series: pd.Series, target_time: np.ndarray) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(source_time) & np.isfinite(numeric)
    if valid.sum() < 2:
        return np.full(len(target_time), np.nan, dtype=float)
    return np.interp(
        target_time,
        source_time[valid],
        numeric[valid],
        left=numeric[valid][0],
        right=numeric[valid][-1],
    )


def _fit_signal(values: np.ndarray, length: int) -> np.ndarray:
    signal = np.asarray(values, dtype=float)
    if len(signal) == length:
        return signal
    if len(signal) == 0:
        return np.full(length, np.nan, dtype=float)
    source_axis = np.linspace(0.0, 1.0, len(signal))
    target_axis = np.linspace(0.0, 1.0, length)
    return np.interp(target_axis, source_axis, signal)


def _first_or_default(frame: pd.DataFrame, column: str, default: object) -> object:
    if column not in frame.columns:
        return default
    series = frame[column].dropna()
    if series.empty:
        return default
    return series.iloc[0]
