from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


FREQUENCY_MATCH_TOLERANCE = 0.02
_HZ_PATTERN = re.compile(r"(?<![A-Za-z0-9.])(?P<freq>\d+(?:\.\d+)?)\s*Hz(?![A-Za-z])", re.IGNORECASE)


def infer_continuous_source_frequency(name: str | None) -> tuple[float | None, str | None]:
    text = str(name or "")
    for match in _HZ_PATTERN.finditer(text):
        try:
            return float(match.group("freq")), "filename"
        except ValueError:
            continue
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
    if np.isfinite(freq):
        attrs["continuous_source_freq_hz"] = float(freq)
        attrs["continuous_source_freq_source"] = str(source or "preamble")
    else:
        column_freq = _frequency_from_column(output)
        if column_freq is not None:
            attrs["continuous_source_freq_hz"] = column_freq
            attrs["continuous_source_freq_source"] = "column"
        else:
            filename_freq, filename_source = infer_continuous_source_frequency(name)
            if filename_freq is not None:
                attrs["continuous_source_freq_hz"] = filename_freq
                attrs["continuous_source_freq_source"] = filename_source
                attrs["continuous_source_freq_inferred_from_filename"] = True
            elif user_fallback_freq_hz is not None and np.isfinite(float(user_fallback_freq_hz)):
                attrs["continuous_source_freq_hz"] = float(user_fallback_freq_hz)
                attrs["continuous_source_freq_source"] = "user_attribution"
                attrs["continuous_source_freq_user_override"] = True
    output.attrs.update(attrs)
    return output


def build_continuous_candidate_details(
    candidates: dict[str, pd.DataFrame],
    *,
    target_freq_hz: float | None,
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
    freq_text = "unknown Hz" if source_freq is None else f"{float(source_freq):g}Hz"
    return f"{detail.get('name')} [{freq_text}, {status}]"


def matching_candidate_names(details: list[dict[str, Any]]) -> list[str]:
    return [str(detail["name"]) for detail in details if detail.get("frequency_match_status") == "match"]


def candidate_detail_by_name(details: list[dict[str, Any]], name: str | None) -> dict[str, Any] | None:
    for detail in details:
        if detail.get("name") == name:
            return detail
    return None


def _candidate_sort_key(detail: dict[str, Any]) -> tuple[int, float, str]:
    status_order = {"match": 0, "unknown": 1, "no_target_frequency": 1, "mismatch": 2}
    status = str(detail.get("frequency_match_status") or "unknown")
    error = detail.get("frequency_error_pct")
    return (status_order.get(status, 9), float(error) if error is not None else 0.0, str(detail.get("name") or ""))


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
