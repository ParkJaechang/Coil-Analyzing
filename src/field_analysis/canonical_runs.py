from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd


CANONICAL_SCHEMA_VERSION = "v1"
CANONICAL_QUALITY_FLAGS = frozenset(
    {
        "current_sign_inferred",
        "resampled_uniform",
        "cycle_label_untrusted",
        "missing_freq_metadata",
        "axis_ambiguous",
        "field_low_snr",
        "input_clipped",
    }
)


Regime = Literal["continuous", "transient"]
Role = Literal["train", "validation", "unknown"]
TargetType = Literal["current", "field", "unknown"]
TargetLevelKind = Literal["pp", "peak", "rms"]
FieldAxis = Literal["bx", "by", "bz"]


@dataclass(slots=True)
class CanonicalRun:
    """Standardized single-run record shared across modeling and recommendation layers."""

    run_id: str
    regime: Regime
    role: Role
    command_waveform: str | None
    freq_hz: float | None
    commanded_cycles: float | None
    target_type: TargetType
    target_level_value: float | None
    target_level_kind: TargetLevelKind | None
    time_s: np.ndarray
    input_v: np.ndarray
    signed_current_a: np.ndarray
    bx_mT: np.ndarray | None
    by_mT: np.ndarray | None
    bz_mT: np.ndarray | None
    primary_field_axis: FieldAxis | None
    sample_rate_hz: float
    active_window_samples: tuple[int, int] | None = None
    active_window_s: tuple[float, float] | None = None
    quality_flags: frozenset[str] = field(default_factory=frozenset)
    inference_scores: dict[str, float] = field(default_factory=dict)
    raw_meta: dict[str, Any] = field(default_factory=dict)
    source_file: str = ""
    source_sheet: str | None = None
    source_hash: str = ""


def validate_canonical_run(run: CanonicalRun) -> None:
    """Raise ValueError when canonical invariants are violated."""

    lengths = {len(run.time_s), len(run.input_v), len(run.signed_current_a)}
    if len(lengths) != 1:
        raise ValueError("CanonicalRun arrays must have identical lengths.")
    if len(run.time_s) >= 2 and not np.all(np.diff(run.time_s) > 0):
        raise ValueError("CanonicalRun.time_s must be strictly increasing.")
    if not np.isfinite(run.sample_rate_hz) or run.sample_rate_hz <= 0:
        raise ValueError("CanonicalRun.sample_rate_hz must be positive.")
    if run.primary_field_axis == "bx" and run.bx_mT is None:
        raise ValueError("CanonicalRun.primary_field_axis=bx requires bx_mT.")
    if run.primary_field_axis == "by" and run.by_mT is None:
        raise ValueError("CanonicalRun.primary_field_axis=by requires by_mT.")
    if run.primary_field_axis == "bz" and run.bz_mT is None:
        raise ValueError("CanonicalRun.primary_field_axis=bz requires bz_mT.")
    if run.commanded_cycles is not None and run.commanded_cycles <= 0:
        raise ValueError("CanonicalRun.commanded_cycles must be positive or None.")
    unknown_flags = set(run.quality_flags) - set(CANONICAL_QUALITY_FLAGS)
    if unknown_flags:
        raise ValueError(f"CanonicalRun.quality_flags contains unsupported values: {sorted(unknown_flags)}")


def summarize_canonical_runs(runs: Sequence[CanonicalRun]) -> pd.DataFrame:
    """Build a compact UI/export summary for canonical runs."""

    rows: list[dict[str, object]] = []
    for run in runs:
        rows.append(
            {
                "run_id": run.run_id,
                "regime": run.regime,
                "role": run.role,
                "command_waveform": run.command_waveform,
                "freq_hz": run.freq_hz,
                "commanded_cycles": run.commanded_cycles,
                "target_type": run.target_type,
                "target_level_value": run.target_level_value,
                "target_level_kind": run.target_level_kind,
                "sample_rate_hz": run.sample_rate_hz,
                "primary_field_axis": run.primary_field_axis,
                "active_start_s": run.active_window_s[0] if run.active_window_s else None,
                "active_end_s": run.active_window_s[1] if run.active_window_s else None,
                "quality_flags": ", ".join(sorted(run.quality_flags)),
                "source_file": run.source_file,
                "source_sheet": run.source_sheet,
            }
        )

    return pd.DataFrame(rows)
