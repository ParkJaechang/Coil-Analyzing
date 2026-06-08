from __future__ import annotations

import re
from pathlib import Path
from typing import Any

RESULT_FILENAME_RE = re.compile(
    r"(?P<canonical>finite_recommended_voltage_lut_(?P<waveform>[A-Za-z]+)_(?P<freq>[0-9]+(?:\.[0-9]+)?)Hz_(?P<cycle>[0-9]+(?:\.[0-9]+)?)cycle(?:_result)?\.csv)$",
    re.IGNORECASE,
)
DEFAULT_TRIANGLE_RESULT_FILENAME_RE = re.compile(
    r"(?P<canonical>(?:finite_)?(?P<freq>[0-9]+(?:[._p][0-9]+)?)hz_(?P<cycle>[0-9]+(?:[._p][0-9]+)?)cycle\.csv)$",
    re.IGNORECASE,
)


def parse_finite_actual_drive_filename(path: str | Path) -> dict[str, Any]:
    name = Path(path).name
    match = RESULT_FILENAME_RE.search(name)
    waveform = None
    if match is None:
        match = DEFAULT_TRIANGLE_RESULT_FILENAME_RE.search(name)
        waveform = "triangle" if match is not None else None
    if match is None:
        raise ValueError(f"Unsupported finite actual-drive result filename: {name}")
    prefix = name[: match.start("canonical")]
    if prefix.endswith("_"):
        prefix = prefix[:-1]
    waveform = waveform or match.group("waveform").lower()
    return {
        "source_type": "finite_actual_drive_result",
        "source_file": name,
        "canonical_source_filename": match.group("canonical"),
        "upload_internal_id": prefix or None,
        "waveform": waveform,
        "waveform_type": waveform,
        "freq_hz": _filename_float(match.group("freq")),
        "cycle_count": _filename_float(match.group("cycle")),
    }


def _filename_float(value: str) -> float:
    return float(str(value).replace("_", ".").replace("p", ".").replace("P", "."))


def parse_preamble(lines: list[str]) -> tuple[dict[str, Any], int]:
    metadata: dict[str, Any] = {}
    header_index = -1
    for index, line in enumerate(lines):
        stripped = line.strip()
        header_parts = {part.strip() for part in stripped.split(",")}
        if stripped.startswith("Row,") or {"TimeMs", "Voltage1_V", "HallBz"}.issubset(header_parts):
            header_index = index
            break
        if not stripped.startswith("#"):
            continue
        parts = [part.strip() for part in stripped[1:].split(",")]
        if not parts or not parts[0]:
            continue
        metadata[parts[0]] = parts[1] if len(parts) == 2 else parts[1:]
    if header_index < 0:
        raise ValueError("Could not find actual-drive result table header")
    return metadata, header_index


def numeric_metadata(metadata: dict[str, Any], key: str) -> float | None:
    value = metadata.get(key)
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return None
    match = re.search(r"[-+]?[0-9]*\.?[0-9]+", str(value))
    return float(match.group(0)) if match else None


def text_metadata(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_auto_sync_lag_ms(metadata: dict[str, Any]) -> float | None:
    value = metadata.get("AutoSyncHallLag")
    if value is None:
        return None
    match = re.search(r"applied\s+([-+]?[0-9]*\.?[0-9]+)ms", str(value), flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def resolve_actual_drive_metadata(
    source_path: Path,
    preamble: dict[str, Any],
    *,
    waveform_type: str | None,
    freq_hz: float | None,
    cycle_count: float | None,
) -> dict[str, Any]:
    try:
        meta = parse_finite_actual_drive_filename(source_path)
        return {**meta, "metadata_source": "filename"}
    except ValueError:
        pass
    preamble_freq = numeric_metadata(preamble, "Frequency(Hz)")
    preamble_cycle = numeric_metadata(preamble, "Cycles")
    preamble_waveform = text_metadata(preamble, "Waveform") or text_metadata(preamble, "WaveformFamily")
    if preamble_freq is not None and preamble_freq > 0.0 and preamble_cycle is not None and preamble_cycle > 0.0:
        waveform = (preamble_waveform or waveform_type or "sine").lower()
        return {
            "source_type": "finite_actual_drive_result",
            "source_file": source_path.name,
            "canonical_source_filename": None,
            "upload_internal_id": None,
            "waveform": waveform,
            "waveform_type": waveform,
            "freq_hz": float(preamble_freq),
            "cycle_count": float(preamble_cycle),
            "metadata_source": "preamble",
        }
    if waveform_type is not None and freq_hz is not None and cycle_count is not None:
        waveform = str(waveform_type).lower()
        return {
            "source_type": "finite_actual_drive_result",
            "source_file": source_path.name,
            "canonical_source_filename": None,
            "upload_internal_id": None,
            "waveform": waveform,
            "waveform_type": waveform,
            "freq_hz": float(freq_hz),
            "cycle_count": float(cycle_count),
            "metadata_source": "current_quick_lut_selection",
        }
    raise ValueError(
        "Actual-drive result metadata unavailable: filename/preamble does not provide waveform/freq/cycle "
        "and no current Quick LUT fallback metadata was supplied"
    )
