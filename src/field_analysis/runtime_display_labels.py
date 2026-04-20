from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

from .utils import (
    canonicalize_waveform_type,
    coerce_float,
    infer_conditions_from_filename,
    infer_current_from_text,
    infer_frequency_from_text,
    infer_waveform_from_text,
)


HASH_LIKE_TOKEN_RE = re.compile(
    r"^(?:[0-9a-f]{8,64}|[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}|[A-Za-z0-9]{24,})$",
    flags=re.IGNORECASE,
)
INTERNAL_ID_LEAK_RE = re.compile(
    r"(::|__corrected_iter\d+|\b(?:steady_state_harmonic|control_formula|validation_auto|runtime_identity|selection_id|source_hash|lut_id)\b|[A-Za-z0-9]+(?:_[A-Za-z0-9]+){3,})",
    flags=re.IGNORECASE,
)
CYCLE_TOKEN_RE = re.compile(r"(?P<cycle>\d+(?:[.p]\d+)?)\s*cycle", flags=re.IGNORECASE)
TARGET_TYPE_RE = re.compile(r"\b(?P<target_type>current|field)\b", flags=re.IGNORECASE)
CORRECTED_ITER_RE = re.compile(r"corrected[_\-\s]*iter(?P<index>\d+)", flags=re.IGNORECASE)
MULTI_SEPARATOR_RE = re.compile(r"[_\-\s]+")
FILE_SUFFIXES = {".csv", ".json", ".md", ".txt", ".xlsx", ".xls", ".yaml", ".yml"}


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        path = Path(text)
        if path.suffix.lower() in FILE_SUFFIXES:
            text = path.stem
    except OSError:
        pass
    return text.strip()


def _is_hash_like_token(token: str) -> bool:
    cleaned = str(token or "").strip()
    if not cleaned:
        return False
    if not HASH_LIKE_TOKEN_RE.fullmatch(cleaned):
        return False
    lower = cleaned.lower()
    if re.fullmatch(r"[0-9a-f]{8,64}", lower):
        return True
    if re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", lower):
        return True
    return any(char.isdigit() for char in cleaned) and any(char.isalpha() for char in cleaned)


def strip_hash_like_prefix(value: object) -> str:
    text = _normalize_text(value)
    if not text:
        return ""
    text = text.split("::", 1)[0].strip()
    while True:
        match = re.match(r"^(?P<prefix>[A-Za-z0-9-]+)(?P<sep>[_\-\s]+)(?P<rest>.+)$", text)
        if match is None or not _is_hash_like_token(match.group("prefix")):
            return text
        text = match.group("rest").strip()


def has_hash_like_prefix(value: object) -> bool:
    text = _normalize_text(value)
    if not text:
        return False
    return strip_hash_like_prefix(text) != text.split("::", 1)[0].strip()


def has_internal_display_leak(value: object) -> bool:
    text = _normalize_text(value)
    if not text:
        return False
    if has_hash_like_prefix(text):
        return True
    return INTERNAL_ID_LEAK_RE.search(text) is not None


def sanitize_display_text(value: object) -> str:
    text = strip_hash_like_prefix(value)
    if not text:
        return ""
    text = MULTI_SEPARATOR_RE.sub(" ", text).strip()
    return re.sub(r"\s+", " ", text)


def _format_number(value: object) -> str:
    numeric = coerce_float(value)
    if numeric is None or not math.isfinite(numeric):
        return ""
    if abs(numeric - round(numeric)) < 1e-9:
        return str(int(round(numeric)))
    return f"{numeric:g}"


def _infer_target_type_from_text(*values: object) -> str | None:
    for value in values:
        if value is None:
            continue
        match = TARGET_TYPE_RE.search(str(value))
        if match is not None:
            return str(match.group("target_type")).lower()
    return None


def _infer_cycle_count_from_text(*values: object) -> float | None:
    inferred = infer_conditions_from_filename(*values)
    if inferred.get("cycle_count") is not None:
        return float(inferred["cycle_count"])
    for value in values:
        if value is None:
            continue
        match = CYCLE_TOKEN_RE.search(str(value))
        if match is not None:
            return float(match.group("cycle").replace("p", "."))
    return None


def _infer_level_with_kind(*values: object) -> tuple[float | None, str | None]:
    inferred = infer_conditions_from_filename(*values)
    if inferred.get("current_target_a") is not None:
        kind = "pp" if inferred.get("current_target_mode") == "pp" else None
        return float(inferred["current_target_a"]), kind
    for value in values:
        if value is None:
            continue
        match = re.search(r"(?<![0-9a-z])(?P<level>\d+(?:[.p]\d+)?)(?P<kind>pp|a|app)\b", str(value), flags=re.IGNORECASE)
        if match is not None:
            kind_token = match.group("kind").lower()
            return float(match.group("level").replace("p", ".")), "pp" if "pp" in kind_token else "a"
    numeric = infer_current_from_text(*values)
    return numeric, None


def infer_iteration_index(*values: object) -> int | None:
    for value in values:
        if value is None:
            continue
        match = CORRECTED_ITER_RE.search(str(value))
        if match is not None:
            return int(match.group("index"))
    return None


def build_display_name(
    *,
    target_type: str | None = None,
    waveform: str | None = None,
    freq_hz: float | None = None,
    cycle_count: float | None = None,
    level: float | None = None,
    level_kind: str | None = None,
    fallback_texts: Iterable[object] = (),
) -> str:
    fallback_values = [value for value in fallback_texts if value not in (None, "")]
    inferred = infer_conditions_from_filename(*fallback_values)

    resolved_target_type = str(target_type).strip().lower() if str(target_type or "").strip() else _infer_target_type_from_text(*fallback_values)
    resolved_waveform = canonicalize_waveform_type(waveform) or infer_waveform_from_text(waveform, *fallback_values)
    resolved_freq_hz = coerce_float(freq_hz, default=None)
    if resolved_freq_hz is None:
        structured_freq = inferred.get("freq_hz")
        resolved_freq_hz = float(structured_freq) if structured_freq is not None else infer_frequency_from_text(*fallback_values)
    resolved_cycle_count = coerce_float(cycle_count, default=None)
    if resolved_cycle_count is None:
        structured_cycle = inferred.get("cycle_count")
        resolved_cycle_count = float(structured_cycle) if structured_cycle is not None else _infer_cycle_count_from_text(*fallback_values)

    resolved_level = coerce_float(level, default=None)
    resolved_level_kind = str(level_kind or "").strip().lower() or None
    if resolved_level is None:
        resolved_level, inferred_kind = _infer_level_with_kind(*fallback_values)
        resolved_level_kind = resolved_level_kind or inferred_kind

    parts: list[str] = []
    if resolved_target_type and resolved_target_type != "unknown":
        parts.append(resolved_target_type)
    if resolved_waveform:
        parts.append(str(resolved_waveform))
    if resolved_freq_hz is not None and math.isfinite(resolved_freq_hz):
        parts.append(f"{_format_number(resolved_freq_hz)} Hz")
    if resolved_cycle_count is not None and math.isfinite(resolved_cycle_count):
        parts.append(f"{_format_number(resolved_cycle_count)} cycle")
    if resolved_level is not None and math.isfinite(resolved_level):
        normalized_kind = resolved_level_kind or ("pp" if resolved_cycle_count is not None else "A")
        unit = "pp" if normalized_kind == "pp" else "A"
        parts.append(f"{_format_number(resolved_level)} {unit}")
    if parts:
        return " / ".join(parts)
    return sanitize_display_text(next(iter(fallback_values), ""))


def build_display_label(
    *,
    display_name: str,
    source_kind: str | None = None,
    iteration_index: int | None = None,
    status: str | None = None,
    include_source_context: bool = False,
) -> str:
    name = sanitize_display_text(display_name)
    if not name:
        return ""
    tags: list[str] = []
    if include_source_context:
        source = str(source_kind or "").strip().lower()
        if source == "corrected":
            if iteration_index is not None:
                tags.append(f"corrected iter{int(iteration_index):02d}")
            else:
                tags.append("corrected")
        elif source in {"recommendation", "export"}:
            tags.append(source)
    if not tags and str(status or "").strip():
        normalized_status = sanitize_display_text(status)
        if normalized_status and normalized_status not in name:
            tags.append(normalized_status)
    return f"{name} | {' / '.join(tags)}" if tags else name


def build_display_object_key(
    *,
    target_type: str | None = None,
    waveform: str | None = None,
    freq_hz: float | None = None,
    cycle_count: float | None = None,
    level: float | None = None,
    level_kind: str | None = None,
) -> str:
    waveform_value = canonicalize_waveform_type(waveform) or ""
    parts = [
        str(target_type or "").strip().lower() or "unknown",
        waveform_value,
        _format_number(freq_hz),
        _format_number(cycle_count),
        _format_number(level),
        str(level_kind or "").strip().lower() or "",
    ]
    return "::".join(parts)
