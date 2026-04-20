from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .canonical_runs import CanonicalRun
from .recommendation_constants import (
    EXACT_MATRIX_ARTIFACT_PATH,
    FINITE_PROVISIONAL_RECIPE_CANDIDATES,
    OFFICIAL_OPERATION_MAX_FREQ_HZ,
)
from .recommendation_models import TargetRequest
from .utils import canonicalize_waveform_type
from .validation import ValidationReport


def _safe_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _key_float(value: object) -> float | None:
    numeric = _safe_float(value)
    return round(float(numeric), 9) if numeric is not None else None


def _resolve_exact_matrix_artifact_path(target: TargetRequest | None = None) -> Path | None:
    context = target.context if target is not None else {}
    explicit_path = context.get("exact_matrix_artifact_path") if isinstance(context, dict) else None
    if explicit_path:
        return Path(str(explicit_path))
    env_path = os.environ.get("FIELD_ANALYSIS_EXACT_MATRIX_PATH")
    if env_path:
        return Path(env_path)
    return EXACT_MATRIX_ARTIFACT_PATH


@lru_cache(maxsize=4)
def _load_exact_matrix_index_cached(path_str: str, mtime_ns: int) -> dict[str, Any]:
    payload = json.loads(Path(path_str).read_text(encoding="utf-8"))
    continuous_current_exact = {
        (
            canonicalize_waveform_type(cell.get("waveform")) or str(cell.get("waveform") or ""),
            _key_float(cell.get("freq_hz")),
            _key_float(cell.get("level_a", cell.get("level_pp_a"))),
        )
        for cell in payload.get("continuous_current_exact_matrix", {}).get("cells", [])
    }
    continuous_field_exact = {
        (
            canonicalize_waveform_type(row.get("waveform")) or str(row.get("waveform") or ""),
            _key_float(row.get("freq_hz")),
        )
        for row in payload.get("continuous_field_exact_matrix", {}).get("summary", [])
    }
    finite_exact = {
        (
            canonicalize_waveform_type(cell.get("waveform")) or str(cell.get("waveform") or ""),
            _key_float(cell.get("freq_hz")),
            _key_float(cell.get("cycles")),
            _key_float(cell.get("level_pp_a")),
        )
        for cell in payload.get("finite_exact_matrix", {}).get("cells", [])
    }
    provisional = {
        (
            canonicalize_waveform_type(cell.get("waveform")) or str(cell.get("waveform") or ""),
            _key_float(cell.get("freq_hz")),
            _key_float(cell.get("cycles")),
            _key_float(cell.get("level_pp_a")),
        )
        for cell in payload.get("provisional_cell", {}).get("cells", [])
    }
    missing = {
        (
            canonicalize_waveform_type(cell.get("waveform")) or str(cell.get("waveform") or ""),
            _key_float(cell.get("freq_hz")),
            _key_float(cell.get("cycles")),
            _key_float(cell.get("level_pp_a")),
        )
        for cell in payload.get("missing_exact_cell", {}).get("cells", [])
    }
    reference_only = {
        (
            canonicalize_waveform_type(cell.get("waveform")) or str(cell.get("waveform") or ""),
            _key_float(cell.get("freq_hz")),
            _key_float(cell.get("level_a", cell.get("level_pp_a"))),
        )
        for cell in payload.get("reference_only", {}).get("cells", [])
    }
    return {
        "path": path_str,
        "mtime_ns": mtime_ns,
        "continuous_current_exact": continuous_current_exact,
        "continuous_field_exact": continuous_field_exact,
        "finite_exact": finite_exact,
        "provisional": provisional,
        "missing": missing,
        "reference_only": reference_only,
    }


def _load_exact_matrix_index(target: TargetRequest | None = None) -> dict[str, Any] | None:
    path = _resolve_exact_matrix_artifact_path(target)
    if path is None or not path.exists():
        return None
    try:
        return _load_exact_matrix_index_cached(str(path), path.stat().st_mtime_ns)
    except (OSError, json.JSONDecodeError):
        return None


def _resolve_artifact_scope_lock(target: TargetRequest) -> dict[str, Any] | None:
    index = _load_exact_matrix_index(target)
    waveform = canonicalize_waveform_type(target.command_waveform or target.target_waveform)
    freq_hz = _key_float(target.freq_hz)
    cycle_count = _key_float(target.commanded_cycles)
    level = _key_float(target.target_level_value)
    if index is None or waveform is None or freq_hz is None:
        return None

    if target.regime == "transient" and target.target_type == "current" and cycle_count is not None and level is not None:
        finite_key = (waveform, freq_hz, cycle_count, level)
        if finite_key in index["finite_exact"]:
            return {
                "bucket": "exact",
                "support_state": "exact",
                "request_route": "exact",
                "status": "certified_exact",
                "path": index["path"],
            }
        if finite_key in index["provisional"]:
            return {
                "bucket": "provisional_preview",
                "support_state": "provisional_preview",
                "request_route": "provisional",
                "status": "provisional_preview",
                "path": index["path"],
            }
        if finite_key in index["missing"]:
            return {
                "bucket": "missing_exact",
                "support_state": "unsupported",
                "request_route": "unsupported",
                "status": "missing_exact",
                "path": index["path"],
            }
        if float(freq_hz) > float(OFFICIAL_OPERATION_MAX_FREQ_HZ):
            return {
                "bucket": "reference_only",
                "support_state": "unsupported",
                "request_route": "reference_only",
                "status": "reference_only",
                "path": index["path"],
            }
        return None

    if target.regime == "continuous":
        if target.target_type == "current" and level is not None:
            continuous_key = (waveform, freq_hz, level)
            if continuous_key in index["continuous_current_exact"]:
                return {
                    "bucket": "exact",
                    "support_state": "exact",
                    "request_route": "exact",
                    "status": "certified_exact",
                    "path": index["path"],
                }
            if continuous_key in index["reference_only"] or float(freq_hz) > float(OFFICIAL_OPERATION_MAX_FREQ_HZ):
                return {
                    "bucket": "reference_only",
                    "support_state": "unsupported",
                    "request_route": "reference_only",
                    "status": "reference_only",
                    "path": index["path"],
                }
        if target.target_type == "field":
            field_key = (waveform, freq_hz)
            if field_key in index["continuous_field_exact"]:
                return {
                    "bucket": "exact",
                    "support_state": "exact",
                    "request_route": "exact",
                    "status": "software_ready_bench_pending",
                    "path": index["path"],
                }
            if float(freq_hz) > float(OFFICIAL_OPERATION_MAX_FREQ_HZ):
                return {
                    "bucket": "reference_only",
                    "support_state": "unsupported",
                    "request_route": "reference_only",
                    "status": "reference_only",
                    "path": index["path"],
                }
    return None


def _filter_finite_support_entries_for_scope_lock(
    support_entries: list[dict[str, Any]],
    *,
    target: TargetRequest,
    artifact_scope_lock: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if artifact_scope_lock is None or artifact_scope_lock.get("bucket") not in {"provisional_preview", "missing_exact", "reference_only"}:
        return support_entries
    if target.regime != "transient" or target.target_type != "current" or target.target_level_value is None or not np.isfinite(target.target_level_value):
        return support_entries

    target_waveform = canonicalize_waveform_type(target.command_waveform or target.target_waveform)
    target_freq = _safe_float(target.freq_hz)
    target_cycles = _safe_float(target.commanded_cycles)
    target_level = float(target.target_level_value)
    if target_waveform is None or target_freq is None or target_cycles is None:
        return support_entries

    level_tolerance = max(abs(target_level) * 0.15, 1.0)
    filtered: list[dict[str, Any]] = []
    for entry in support_entries:
        entry_waveform = canonicalize_waveform_type(entry.get("waveform_type"))
        entry_freq = _safe_float(entry.get("freq_hz"))
        entry_cycles = _safe_float(entry.get("requested_cycle_count", entry.get("approx_cycle_span")))
        entry_level = _safe_float(entry.get("requested_current_pp", entry.get("current_pp")))
        is_exact_target_entry = (
            entry_waveform == target_waveform
            and entry_freq is not None
            and np.isclose(entry_freq, target_freq, atol=1e-9)
            and entry_cycles is not None
            and np.isclose(entry_cycles, target_cycles, atol=1e-9)
            and entry_level is not None
            and abs(entry_level - target_level) <= level_tolerance
        )
        if is_exact_target_entry:
            continue
        filtered.append(entry)
    return filtered


def build_support_report(
    continuous_runs: list[CanonicalRun],
    transient_runs: list[CanonicalRun],
    target: TargetRequest,
) -> ValidationReport:
    """Build a minimal support report from canonical runs only."""

    target_waveform = canonicalize_waveform_type(target.target_waveform) or target.target_waveform
    requested_runs = transient_runs if target.regime == "transient" else continuous_runs
    waveform_runs = [
        run
        for run in requested_runs
        if (canonicalize_waveform_type(run.command_waveform) or run.command_waveform) == target_waveform
    ]
    exact_freq_runs = [
        run
        for run in waveform_runs
        if target.freq_hz is not None and run.freq_hz is not None and np.isclose(float(run.freq_hz), float(target.freq_hz), atol=1e-9)
    ]
    exact_cycle_runs = [
        run
        for run in exact_freq_runs
        if target.commanded_cycles is not None
        and run.commanded_cycles is not None
        and np.isclose(float(run.commanded_cycles), float(target.commanded_cycles), atol=1e-9)
    ]

    reasons: list[str] = []
    if not waveform_runs:
        reasons.append("waveform support ?놁쓬")
    if target.freq_hz is not None and not exact_freq_runs:
        reasons.append("exact frequency support ?놁쓬")
    if target.regime == "transient" and target.commanded_cycles is not None and not exact_cycle_runs:
        reasons.append("exact cycle support ?놁쓬")

    quality_candidates = exact_cycle_runs or exact_freq_runs or waveform_runs
    blocking_flags = {"missing_freq_metadata"}
    soft_flags = {"field_low_snr", "axis_ambiguous", "cycle_label_untrusted"}
    quality_blocked = bool(quality_candidates) and all(run.quality_flags & blocking_flags for run in quality_candidates)
    if quality_blocked:
        reasons.append("quality flag濡??먮룞 異붿쿇 湲덉?")

    quality_scores = []
    for run in quality_candidates:
        penalty = 0.15 * len(run.quality_flags & soft_flags) + 0.35 * len(run.quality_flags & blocking_flags)
        quality_scores.append(max(0.0, 1.0 - penalty))
    shape_quality = float(np.mean(quality_scores)) if quality_scores else 0.0

    exact_freq_match = bool(exact_freq_runs)
    exact_cycle_match = bool(exact_cycle_runs) if target.regime == "transient" and target.commanded_cycles is not None else True
    in_support = exact_freq_match and exact_cycle_match
    expected_error_band = 0.05 if in_support else 0.25 if waveform_runs else 1.0
    allow_auto_recommendation = in_support and not quality_blocked

    return ValidationReport(
        in_support=in_support,
        exact_freq_match=exact_freq_match,
        exact_cycle_match=exact_cycle_match,
        shape_quality=shape_quality,
        expected_error_band=expected_error_band,
        allow_auto_recommendation=allow_auto_recommendation,
        reasons=reasons,
    )


def _has_exact_level_support_for_request(
    runs: list[CanonicalRun],
    target: TargetRequest,
) -> bool:
    if target.target_level_value is None or not np.isfinite(target.target_level_value):
        return False
    if target.regime != "transient":
        return False
    for run in runs:
        if run.target_level_value is None or not np.isfinite(run.target_level_value):
            continue
        if run.command_waveform is not None and target.target_waveform is not None:
            if (canonicalize_waveform_type(run.command_waveform) or run.command_waveform) != (
                canonicalize_waveform_type(target.target_waveform) or target.target_waveform
            ):
                continue
        if target.freq_hz is not None:
            if run.freq_hz is None or not np.isclose(float(run.freq_hz), float(target.freq_hz), atol=1e-9):
                continue
        if target.regime == "transient" and target.commanded_cycles is not None:
            if run.commanded_cycles is None or not np.isclose(float(run.commanded_cycles), float(target.commanded_cycles), atol=1e-9):
                continue
        if np.isclose(float(run.target_level_value), float(target.target_level_value), atol=1e-9):
            return True
    return False


def _find_provisional_finite_recipe(
    runs: list[CanonicalRun],
    target: TargetRequest,
) -> dict[str, Any] | None:
    if target.regime != "transient" or target.target_type != "current":
        return None
    if target.target_level_value is None or not np.isfinite(target.target_level_value):
        return None
    target_waveform = canonicalize_waveform_type(target.command_waveform or target.target_waveform)
    if target_waveform is None or target.freq_hz is None or not np.isfinite(target.freq_hz):
        return None
    if target.commanded_cycles is None or not np.isfinite(target.commanded_cycles):
        return None
    target_level = float(target.target_level_value)
    requested_freq = float(target.freq_hz)
    requested_cycles = float(target.commanded_cycles)

    for candidate in FINITE_PROVISIONAL_RECIPE_CANDIDATES:
        if target_waveform != candidate["waveform"]:
            continue
        if not np.isclose(requested_freq, float(candidate["freq_hz"]), atol=1e-9):
            continue
        if not np.isclose(requested_cycles, float(candidate["cycles"]), atol=1e-9):
            continue
        if not np.isclose(target_level, float(candidate["target_level_pp"]), atol=1e-9):
            continue
        source_level = float(candidate["source_level_pp"])
        for run in runs:
            run_waveform = canonicalize_waveform_type(run.command_waveform)
            if run_waveform is None and isinstance(getattr(run, "raw_meta", None), dict):
                run_waveform = canonicalize_waveform_type(str(run.raw_meta.get("waveform", "")))
            if run_waveform != target_waveform:
                continue
            if run.freq_hz is None or not np.isclose(float(run.freq_hz), requested_freq, atol=1e-9):
                continue
            if run.commanded_cycles is None or not np.isclose(float(run.commanded_cycles), requested_cycles, atol=1e-9):
                continue
            if run.target_level_value is None or not np.isfinite(run.target_level_value):
                continue
            if not np.isclose(float(run.target_level_value), source_level, atol=1e-9):
                continue
            return {
                "waveform": target_waveform,
                "freq_hz": requested_freq,
                "cycles": requested_cycles,
                "target_level_pp": target_level,
                "source_level_pp": source_level,
                "scale_ratio": target_level / source_level if source_level else np.nan,
                "mode": "scaled_level_surrogate_preview",
                "label": f"{requested_freq:g} Hz / {requested_cycles:g} cycle / {source_level:g} pp -> {target_level:g} pp",
            }
    return None
