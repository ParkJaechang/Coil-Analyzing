from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from .canonical_runs import (
    CANONICAL_SCHEMA_VERSION,
    CanonicalRun,
    FieldAxis,
    Regime,
    Role,
    validate_canonical_run,
)
from .models import ParsedMeasurement
from .utils import (
    apply_bz_effective_convention,
    canonicalize_waveform_type,
    first_number,
    normalize_name,
    reconstruct_signed_current_channels,
)


@dataclass(slots=True)
class CanonicalizeConfig:
    """Configuration for converting parsed uploads into canonical runs."""

    preferred_field_axis: str = "bz_mT"
    uniform_resample: bool = True
    resample_dt_s: float | None = None
    active_threshold_ratio: float = 0.02
    axis_ambiguity_ratio: float = 0.15
    low_field_pp_threshold_mT: float = 1e-3
    custom_current_alpha: float = 1.0
    custom_current_beta: float = 1.0


def canonicalize_batch(
    parsed_measurements: list[ParsedMeasurement],
    *,
    regime: Regime,
    role: Role,
    config: CanonicalizeConfig | None = None,
) -> list[CanonicalRun]:
    """Canonicalize a batch of parsed runs."""

    active_config = config or CanonicalizeConfig()
    return [
        canonicalize_run(parsed, regime=regime, role=role, config=active_config)
        for parsed in parsed_measurements
    ]


def canonicalize_run(
    parsed: ParsedMeasurement,
    *,
    regime: Regime,
    role: Role,
    config: CanonicalizeConfig | None = None,
) -> CanonicalRun:
    """Convert a parsed upload into a fixed canonical contract."""

    active_config = config or CanonicalizeConfig()
    frame = parsed.normalized_frame.copy()
    frame = _prepare_frame(frame)

    signed_info = reconstruct_signed_current_channels(
        frame,
        custom_current_alpha=active_config.custom_current_alpha,
        custom_current_beta=active_config.custom_current_beta,
    )

    time_s = _series_to_numpy(frame.get("time_s"))
    waveform = _first_non_null(frame, "waveform_type")
    waveform = canonicalize_waveform_type(waveform)
    freq_hz = _first_finite(frame, "freq_hz")
    target_level_value, target_level_kind, target_type = _extract_target_definition(frame)
    source_cycle_quality = _assess_source_cycle_labels(frame)

    quality_flags: set[str] = set()
    if signed_info.get("reconstructed_columns"):
        quality_flags.add("current_sign_inferred")
    if freq_hz is None:
        quality_flags.add("missing_freq_metadata")
    if source_cycle_quality != "trusted":
        quality_flags.add("cycle_label_untrusted")

    uniform_time_s, resampled_frame, resample_scores, was_resampled = _build_uniform_frame(
        frame,
        active_config,
    )
    if was_resampled:
        quality_flags.add("resampled_uniform")

    signed_current_a = _extract_signed_current(resampled_frame)
    bx_mT = _extract_optional_signal(resampled_frame, "bx_mT")
    by_mT = _extract_optional_signal(resampled_frame, "by_mT")
    bz_mT = _extract_optional_signal(resampled_frame, "bz_mT")
    input_v = _series_to_numpy(resampled_frame.get("daq_input_v"))

    primary_axis, axis_scores = _select_primary_field_axis(
        bx_mT=bx_mT,
        by_mT=by_mT,
        bz_mT=bz_mT,
        preferred_field_axis=active_config.preferred_field_axis,
        ambiguity_ratio=active_config.axis_ambiguity_ratio,
    )
    if axis_scores["ambiguous"]:
        quality_flags.add("axis_ambiguous")
    if axis_scores["peak_to_peak_mT"] <= active_config.low_field_pp_threshold_mT:
        quality_flags.add("field_low_snr")

    sample_rate_hz = float(1.0 / resample_scores["dt_s"]) if resample_scores["dt_s"] > 0 else float("nan")
    active_window_samples, active_window_s = _detect_active_window(
        time_s=uniform_time_s,
        input_v=input_v,
        signed_current_a=signed_current_a,
        threshold_ratio=active_config.active_threshold_ratio,
    )
    commanded_cycles = _infer_commanded_cycles(
        parsed=parsed,
        frame=resampled_frame,
        freq_hz=freq_hz,
        active_window_s=active_window_s,
    )

    inference_scores = {
        "time_uniformity_cv": resample_scores["dt_cv"],
        "primary_axis_confidence": axis_scores["confidence"],
        "field_peak_to_peak_mT": axis_scores["peak_to_peak_mT"],
        "source_cycle_label_score": 1.0 if source_cycle_quality == "trusted" else 0.0,
    }

    raw_meta = {
        "metadata": dict(parsed.metadata),
        "mapping": dict(parsed.mapping),
        "warnings": list(parsed.warnings),
        "logs": list(parsed.logs),
        "canonicalize_config": asdict(active_config),
        "source_cycle_quality": source_cycle_quality,
    }

    canonical_run = CanonicalRun(
        run_id=_build_run_id(parsed, waveform, freq_hz, role),
        regime=regime,
        role=role,
        command_waveform=waveform,
        freq_hz=freq_hz,
        commanded_cycles=commanded_cycles,
        target_type=target_type,
        target_level_value=target_level_value,
        target_level_kind=target_level_kind,
        time_s=uniform_time_s,
        input_v=input_v,
        signed_current_a=signed_current_a,
        bx_mT=bx_mT,
        by_mT=by_mT,
        bz_mT=bz_mT,
        primary_field_axis=primary_axis,
        sample_rate_hz=sample_rate_hz,
        active_window_samples=active_window_samples,
        active_window_s=active_window_s,
        quality_flags=frozenset(sorted(quality_flags)),
        inference_scores=inference_scores,
        raw_meta=raw_meta,
        source_file=parsed.source_file,
        source_sheet=parsed.sheet_name,
        source_hash=_build_source_hash(parsed),
    )
    validate_canonical_run(canonical_run)
    return canonical_run


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    if "time_s" not in working.columns:
        working["time_s"] = np.arange(len(working), dtype=float)
    working["time_s"] = pd.to_numeric(working["time_s"], errors="coerce")
    working = working.dropna(subset=["time_s"]).sort_values("time_s").reset_index(drop=True)
    if working.empty:
        working = pd.DataFrame({"time_s": np.array([0.0], dtype=float)})
    working = working.loc[~working["time_s"].duplicated(keep="first")].reset_index(drop=True)
    working["time_s"] = working["time_s"] - float(working["time_s"].iloc[0])
    apply_bz_effective_convention(working)
    return working


def _build_uniform_frame(
    frame: pd.DataFrame,
    config: CanonicalizeConfig,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, float], bool]:
    time_s = _series_to_numpy(frame.get("time_s"))
    if len(time_s) < 2:
        return time_s, frame.reset_index(drop=True), {"dt_s": 1.0, "dt_cv": 0.0}, False

    diffs = np.diff(time_s)
    positive_diffs = diffs[diffs > 0]
    if len(positive_diffs) == 0:
        return time_s, frame.reset_index(drop=True), {"dt_s": 1.0, "dt_cv": 0.0}, False

    dt_s = float(config.resample_dt_s or np.median(positive_diffs))
    dt_cv = float(np.std(positive_diffs) / np.mean(positive_diffs)) if np.mean(positive_diffs) > 0 else 0.0
    if not config.uniform_resample or dt_s <= 0:
        return time_s, frame.reset_index(drop=True), {"dt_s": dt_s, "dt_cv": dt_cv}, False

    uniform_time = np.arange(0.0, time_s[-1] + dt_s * 0.5, dt_s, dtype=float)
    if len(uniform_time) < 2:
        uniform_time = time_s.copy()
    if len(uniform_time) == len(time_s) and np.allclose(uniform_time, time_s, atol=max(dt_s * 1e-6, 1e-12)):
        return time_s, frame.reset_index(drop=True), {"dt_s": dt_s, "dt_cv": dt_cv}, False

    resampled = pd.DataFrame({"time_s": uniform_time})
    for column in frame.columns:
        if column == "time_s":
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            resampled[column] = _interpolate_numeric_series(time_s, frame[column], uniform_time)
        else:
            non_null = frame[column].dropna()
            resampled[column] = non_null.iloc[0] if not non_null.empty else None
    return uniform_time, resampled.reset_index(drop=True), {"dt_s": dt_s, "dt_cv": dt_cv}, True


def _interpolate_numeric_series(
    time_s: np.ndarray,
    series: pd.Series,
    uniform_time: np.ndarray,
) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(time_s) & np.isfinite(numeric)
    if valid.sum() < 2:
        return np.full(len(uniform_time), np.nan, dtype=float)
    return np.interp(
        uniform_time,
        time_s[valid],
        numeric[valid],
        left=numeric[valid][0],
        right=numeric[valid][-1],
    )


def _extract_signed_current(frame: pd.DataFrame) -> np.ndarray:
    for column in (
        "i_sum_signed",
        "i_custom_signed",
        "coil2_current_signed_a",
        "coil1_current_signed_a",
        "i_sum",
        "coil2_current_a",
        "coil1_current_a",
    ):
        if column in frame.columns:
            return _series_to_numpy(frame[column])
    return np.full(len(frame), np.nan, dtype=float)


def _extract_optional_signal(frame: pd.DataFrame, column: str) -> np.ndarray | None:
    if column not in frame.columns:
        return None
    values = _series_to_numpy(frame[column])
    return None if np.isnan(values).all() else values


def _select_primary_field_axis(
    *,
    bx_mT: np.ndarray | None,
    by_mT: np.ndarray | None,
    bz_mT: np.ndarray | None,
    preferred_field_axis: str,
    ambiguity_ratio: float,
) -> tuple[FieldAxis | None, dict[str, float | bool]]:
    candidates = {
        "bx": _peak_to_peak(bx_mT),
        "by": _peak_to_peak(by_mT),
        "bz": _peak_to_peak(bz_mT),
    }
    finite_candidates = {axis: value for axis, value in candidates.items() if np.isfinite(value)}
    if not finite_candidates:
        return None, {"confidence": 0.0, "peak_to_peak_mT": 0.0, "ambiguous": False}

    preferred_axis = preferred_field_axis.replace("_mT", "")
    ordered = sorted(finite_candidates.items(), key=lambda item: item[1], reverse=True)
    best_axis, best_value = ordered[0]
    second_value = ordered[1][1] if len(ordered) > 1 else 0.0
    ambiguous = best_value > 0 and second_value >= best_value * (1.0 - ambiguity_ratio)
    if ambiguous and preferred_axis in finite_candidates and finite_candidates[preferred_axis] > 0:
        selected_axis = preferred_axis
    else:
        selected_axis = best_axis

    confidence = float(best_value / (best_value + second_value)) if (best_value + second_value) > 0 else 0.0
    return selected_axis, {"confidence": confidence, "peak_to_peak_mT": best_value, "ambiguous": ambiguous}


def _detect_active_window(
    *,
    time_s: np.ndarray,
    input_v: np.ndarray,
    signed_current_a: np.ndarray,
    threshold_ratio: float,
) -> tuple[tuple[int, int] | None, tuple[float, float] | None]:
    if len(time_s) == 0:
        return None, None

    envelope = np.nan_to_num(np.abs(input_v), nan=0.0)
    if float(np.nanmax(envelope)) <= 0:
        envelope = np.nan_to_num(np.abs(signed_current_a), nan=0.0)

    peak = float(np.nanmax(envelope))
    if peak <= 0:
        return (0, len(time_s) - 1), (float(time_s[0]), float(time_s[-1]))

    mask = envelope >= peak * max(threshold_ratio, 1e-6)
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return (0, len(time_s) - 1), (float(time_s[0]), float(time_s[-1]))

    start_index = int(indices[0])
    end_index = int(indices[-1])
    return (start_index, end_index), (float(time_s[start_index]), float(time_s[end_index]))


def _infer_commanded_cycles(
    *,
    parsed: ParsedMeasurement,
    frame: pd.DataFrame,
    freq_hz: float | None,
    active_window_s: tuple[float, float] | None,
) -> float | None:
    metadata_cycles = _extract_metadata_cycle_count(parsed.metadata)
    if metadata_cycles is not None:
        return metadata_cycles

    frame_cycle_hint = _first_finite(frame, "cycle_total_expected")
    if frame_cycle_hint is not None:
        return frame_cycle_hint

    if freq_hz is None or active_window_s is None:
        return None

    duration_s = max(active_window_s[1] - active_window_s[0], 0.0)
    if duration_s <= 0:
        return None
    return float(duration_s * freq_hz)


def _extract_metadata_cycle_count(metadata: dict[str, Any]) -> float | None:
    direct_keys = {
        "cycle",
        "cycles",
        "cyclecount",
        "cycle_count",
        "commandedcycles",
        "targetcyclecount",
        "cycletotalexpected",
    }
    for key, value in metadata.items():
        if normalize_name(key) in direct_keys:
            cycle_value = first_number(value)
            if cycle_value is not None and cycle_value > 0:
                return float(cycle_value)
    return None


def _extract_target_definition(frame: pd.DataFrame) -> tuple[float | None, str | None, str]:
    current_pp = _first_finite(frame, "current_pp_target_a")
    if current_pp is not None:
        return current_pp, "pp", "current"
    current_peak = _first_finite(frame, "current_pk_target_a")
    if current_peak is not None:
        return current_peak, "peak", "current"
    return None, None, "unknown"


def _assess_source_cycle_labels(frame: pd.DataFrame) -> str:
    if "source_cycle_no" not in frame.columns:
        return "missing"
    numeric = pd.to_numeric(frame["source_cycle_no"], errors="coerce").dropna()
    if numeric.empty:
        return "missing"
    if numeric.nunique() <= 1:
        return "flat"
    return "trusted"


def _series_to_numpy(series: pd.Series | np.ndarray | None) -> np.ndarray:
    if series is None:
        return np.array([], dtype=float)
    return pd.to_numeric(pd.Series(series), errors="coerce").to_numpy(dtype=float)


def _first_non_null(frame: pd.DataFrame, column: str) -> Any:
    if column not in frame.columns:
        return None
    values = frame[column].dropna()
    if values.empty:
        return None
    return values.iloc[0]


def _first_finite(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    numeric = pd.to_numeric(frame[column], errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return None
    return float(finite.iloc[0])


def _peak_to_peak(values: np.ndarray | None) -> float:
    if values is None or len(values) == 0:
        return float("nan")
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan")
    return float(np.max(finite) - np.min(finite))


def _build_run_id(parsed: ParsedMeasurement, waveform: str | None, freq_hz: float | None, role: str) -> str:
    source_hash = _build_source_hash(parsed)[:12]
    parts = [CANONICAL_SCHEMA_VERSION, source_hash, parsed.sheet_name, waveform or "unknown", role]
    if freq_hz is not None:
        parts.append(f"{freq_hz:g}Hz")
    return "__".join(part.replace("\\", "_").replace("/", "_") for part in parts if part)


def _build_source_hash(parsed: ParsedMeasurement) -> str:
    digest = hashlib.sha256()
    digest.update(parsed.source_file.encode("utf-8", errors="ignore"))
    digest.update(str(parsed.sheet_name).encode("utf-8", errors="ignore"))
    digest.update(json.dumps(parsed.metadata, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))
    digest.update(json.dumps(parsed.mapping, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))
    digest.update(",".join(str(column) for column in parsed.raw_frame.columns).encode("utf-8", errors="ignore"))
    frame_hash = pd.util.hash_pandas_object(parsed.raw_frame.fillna(""), index=True)
    digest.update(frame_hash.to_numpy().tobytes())
    return digest.hexdigest()
