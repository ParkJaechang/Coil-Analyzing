from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


FREQUENCY_MATCH_TOLERANCE = 0.02
_HZ_PATTERN = re.compile(r"(?<![A-Za-z0-9.])(?P<freq>\d+(?:\.\d+)?)\s*Hz(?![A-Za-z])", re.IGNORECASE)
_WAVEFORM_PATTERNS = (
    ("rounded_triangle", re.compile(r"rounded[_-]?triangle", re.IGNORECASE)),
    ("triangle", re.compile(r"(?<!rounded[_-])triangle", re.IGNORECASE)),
    ("sine", re.compile(r"sine|sinus", re.IGNORECASE)),
)


def infer_continuous_source_frequency(name: str | None) -> tuple[float | None, str | None]:
    text = str(name or "")
    for match in _HZ_PATTERN.finditer(text):
        try:
            return float(match.group("freq")), "filename"
        except ValueError:
            continue
    return None, None


def infer_continuous_source_waveform(name: str | None) -> tuple[str | None, str | None]:
    text = str(name or "")
    for family, pattern in _WAVEFORM_PATTERNS:
        if pattern.search(text):
            return family, "filename"
    return None, None


def attach_continuous_frequency_attrs(
    frame: pd.DataFrame,
    *,
    name: str | None,
    user_fallback_freq_hz: float | None = None,
) -> pd.DataFrame:
    output = frame.copy(deep=True)
    attrs = dict(getattr(frame, "attrs", {}) or {})
    freq = _safe_float(attrs.get("continuous_source_freq_hz"))
    source = attrs.get("continuous_source_freq_source")
    if np.isfinite(freq) and source == "preamble":
        attrs["continuous_source_freq_hz"] = float(freq)
        attrs["continuous_source_freq_source"] = "preamble"
    else:
        filename_freq, filename_source = infer_continuous_source_frequency(name)
        if filename_freq is not None:
            attrs["continuous_source_freq_hz"] = filename_freq
            attrs["continuous_source_freq_source"] = filename_source
            attrs["continuous_source_freq_inferred_from_filename"] = True
        elif np.isfinite(freq):
            attrs["continuous_source_freq_hz"] = float(freq)
            attrs["continuous_source_freq_source"] = str(source or "frame_attrs")
        else:
            column_freq = _frequency_from_column(output)
            if column_freq is not None:
                attrs["continuous_source_freq_hz"] = column_freq
                attrs["continuous_source_freq_source"] = "column"
            elif user_fallback_freq_hz is not None and np.isfinite(float(user_fallback_freq_hz)):
                attrs["continuous_source_freq_hz"] = float(user_fallback_freq_hz)
                attrs["continuous_source_freq_source"] = "user_attribution"
                attrs["continuous_source_freq_user_override"] = True
    waveform = attrs.get("continuous_source_waveform_family")
    waveform_source = attrs.get("continuous_source_waveform_source")
    filename_waveform, filename_waveform_source = infer_continuous_source_waveform(name)
    if filename_waveform is not None:
        attrs["continuous_source_waveform_family"] = filename_waveform
        attrs["continuous_source_waveform_source"] = filename_waveform_source
    elif waveform:
        attrs["continuous_source_waveform_family"] = str(waveform)
        attrs["continuous_source_waveform_source"] = str(waveform_source or "frame_attrs")
    elif "waveform_type" in output.columns:
        values = output["waveform_type"].dropna()
        if not values.empty:
            attrs["continuous_source_waveform_family"] = str(values.iloc[0])
            attrs["continuous_source_waveform_source"] = "column"
    output.attrs.update(attrs)
    return output


def build_continuous_candidate_details(
    candidates: dict[str, pd.DataFrame],
    *,
    target_freq_hz: float | None,
    source_waveform_filter: str | None = None,
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    target = _safe_float(target_freq_hz)
    for name, frame in candidates.items():
        attrs = dict(getattr(frame, "attrs", {}) or {})
        source_freq = _safe_float(attrs.get("continuous_source_freq_hz"))
        status = "unknown"
        error_pct: float | None = None
        if np.isfinite(source_freq) and np.isfinite(target):
            error_pct = abs(source_freq - target) / max(abs(target), 1e-12) * 100.0
            status = "match" if error_pct <= FREQUENCY_MATCH_TOLERANCE * 100.0 else "mismatch"
        elif np.isfinite(source_freq):
            status = "no_target_frequency"
        filename = str(attrs.get("continuous_source_file") or name.split(":", 1)[-1])
        source_waveform = attrs.get("continuous_source_waveform_family") or "unknown"
        waveform_filter = str(source_waveform_filter or "all")
        waveform_status = (
            "match"
            if waveform_filter == "all" or str(source_waveform) == waveform_filter
            else "mismatch"
        )
        label = f"{name} [{source_waveform}, {_format_freq(source_freq)}, {status}]"
        details.append(
            {
                "name": name,
                "filename": filename,
                "source_category": name.split(":", 1)[0] if ":" in name else "unknown",
                "source_freq_hz": float(source_freq) if np.isfinite(source_freq) else None,
                "target_freq_hz": float(target) if np.isfinite(target) else None,
                "frequency_error_pct": error_pct,
                "frequency_match_status": status,
                "continuous_source_freq_source": attrs.get("continuous_source_freq_source") or "unknown",
                "schema_status": attrs.get("continuous_schema_status", "ok"),
                "continuous_source_waveform_family": source_waveform,
                "continuous_source_waveform_source": attrs.get("continuous_source_waveform_source") or "unknown",
                "continuous_source_waveform_filter": waveform_filter,
                "continuous_source_waveform_match_status": waveform_status,
                "continuous_target_field_waveform": "fixed_rounded_triangle",
                "continuous_candidate_label": label,
            }
        )
    return sorted(details, key=_candidate_sort_key)


def rank_continuous_candidates_for_target(
    candidates: dict[str, pd.DataFrame],
    *,
    target_freq_hz: float | None,
) -> list[dict[str, Any]]:
    return build_continuous_candidate_details(candidates, target_freq_hz=target_freq_hz)


def continuous_candidate_label(detail: dict[str, Any]) -> str:
    source_freq = detail.get("source_freq_hz")
    status = detail.get("frequency_match_status") or "unknown"
    waveform = detail.get("continuous_source_waveform_family") or "unknown"
    return f"{detail.get('name')} [{waveform}, {_format_freq(source_freq)}, {status}]"


def matching_candidate_names(details: list[dict[str, Any]]) -> list[str]:
    return [str(detail["name"]) for detail in details if detail.get("frequency_match_status") == "match"]


def candidate_detail_by_name(details: list[dict[str, Any]], name: str | None) -> dict[str, Any] | None:
    for detail in details:
        if detail.get("name") == name:
            return detail
    return None


def _candidate_sort_key(detail: dict[str, Any]) -> tuple[int, int, float, str]:
    status_order = {"match": 0, "unknown": 1, "no_target_frequency": 1, "mismatch": 2}
    waveform_order = {"match": 0, "mismatch": 2}
    status = str(detail.get("frequency_match_status") or "unknown")
    waveform_status = str(detail.get("continuous_source_waveform_match_status") or "match")
    error = detail.get("frequency_error_pct")
    return (
        waveform_order.get(waveform_status, 1),
        status_order.get(status, 9),
        float(error) if error is not None else 0.0,
        str(detail.get("name") or ""),
    )


def _format_freq(value: Any) -> str:
    freq = _safe_float(value)
    return "unknown Hz" if not np.isfinite(freq) else f"{freq:g}Hz"


def _frequency_from_column(frame: pd.DataFrame) -> float | None:
    if "freq_hz" not in frame.columns:
        return None
    values = pd.to_numeric(frame["freq_hz"], errors="coerce").dropna()
    if values.empty:
        return None
    value = float(values.iloc[0])
    return value if np.isfinite(value) and value > 0.0 else None


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
