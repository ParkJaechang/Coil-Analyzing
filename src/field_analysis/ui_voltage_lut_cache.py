from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .ui_voltage_lut_review import ParsedVoltageLut
from .ui_voltage_lut_review import build_lut_diagnostics
from .ui_voltage_lut_review import parse_voltage_lut_upload


LUT_CACHE_STATE_KEY = "voltage_lut_cache_items"


@dataclass(frozen=True)
class VoltageLutCacheRecord:
    id: str
    original_filename: str
    display_name: str
    user_note: str
    parsed: ParsedVoltageLut
    diagnostics: dict[str, object]
    metadata: dict[str, object]

    @property
    def source_name(self) -> str:
        return self.original_filename


def add_lut_cache_bytes(
    cache_state: dict[str, dict[str, object]],
    original_filename: str,
    data: bytes,
    *,
    created_time: str | None = None,
    display_name: str | None = None,
    user_note: str = "",
) -> str:
    content = bytes(data)
    digest = hashlib.sha256(content).hexdigest()[:16]
    stem = _safe_name_part(Path(original_filename).stem) or "lut"
    timestamp = _safe_name_part(created_time) if created_time else "no_time"
    cache_id = f"lut:{stem}:{timestamp}:{digest}"
    if cache_id not in cache_state:
        cache_state[cache_id] = {
            "id": cache_id,
            "original_filename": str(original_filename),
            "display_name": display_name or str(original_filename),
            "user_note": str(user_note),
            "created_time": created_time,
            "content_sha256": digest,
            "csv_bytes": content,
        }
    return cache_id


def build_lut_cache_records(cache_state: dict[str, dict[str, object]]) -> list[VoltageLutCacheRecord]:
    records: list[VoltageLutCacheRecord] = []
    for cache_id, item in cache_state.items():
        original_filename = str(item.get("original_filename") or item.get("source_name") or cache_id)
        parsed = _parse_cache_item(original_filename, item)
        diagnostics = build_lut_diagnostics(parsed.frame) if parsed.ok else _empty_lut_diagnostics()
        display_name = str(item.get("display_name") or original_filename)
        user_note = str(item.get("user_note") or "")
        metadata = {
            "id": str(item.get("id") or cache_id),
            "original_filename": original_filename,
            "display_name": display_name,
            "user_note": user_note,
            "created_time": item.get("created_time"),
            "sample_count": diagnostics.get("sample_count"),
            "duration": diagnostics.get("duration_s"),
            "duration_s": diagnostics.get("duration_s"),
            "time_start": diagnostics.get("time_start_s"),
            "time_start_s": diagnostics.get("time_start_s"),
            "time_end": diagnostics.get("time_end_s"),
            "time_end_s": diagnostics.get("time_end_s"),
            "voltage_min": diagnostics.get("voltage_min_v"),
            "voltage_min_v": diagnostics.get("voltage_min_v"),
            "voltage_max": diagnostics.get("voltage_max_v"),
            "voltage_max_v": diagnostics.get("voltage_max_v"),
            "timebase_status": diagnostics.get("time_axis_status"),
            "parse_status": "ok" if parsed.ok else "unavailable",
            "parse_error": parsed.error,
        }
        records.append(
            VoltageLutCacheRecord(
                id=cache_id,
                original_filename=original_filename,
                display_name=display_name,
                user_note=user_note,
                parsed=parsed,
                diagnostics=diagnostics,
                metadata=metadata,
            )
        )
    return sorted(records, key=lambda record: record.id)


def build_lut_cache_selection_options(
    records: list[VoltageLutCacheRecord],
) -> tuple[list[str], dict[str, VoltageLutCacheRecord], dict[str, str]]:
    options = [record.id for record in records]
    records_by_id = {record.id: record for record in records}
    labels_by_id = {record.id: _lut_cache_label(record) for record in records}
    return options, records_by_id, labels_by_id


def edit_lut_cache_metadata(
    cache_state: dict[str, dict[str, object]],
    cache_id: str,
    *,
    display_name: str | None = None,
    user_note: str | None = None,
) -> bool:
    item = cache_state.get(cache_id)
    if item is None:
        return False
    if display_name is not None:
        item["display_name"] = str(display_name)
    if user_note is not None:
        item["user_note"] = str(user_note)
    return True


def delete_lut_cache_item(cache_state: dict[str, dict[str, object]], cache_id: str) -> bool:
    return cache_state.pop(cache_id, None) is not None


def fallback_lut_cache_selection(options: list[str], selected_id: str | None) -> str | None:
    if not options:
        return None
    if selected_id in options:
        return selected_id
    return options[0]


def _parse_cache_item(original_filename: str, item: dict[str, object]) -> ParsedVoltageLut:
    raw_bytes = item.get("csv_bytes")
    if isinstance(raw_bytes, bytes):
        return parse_voltage_lut_upload(original_filename, raw_bytes)
    file_path = item.get("file_path")
    if file_path:
        path = Path(str(file_path))
        if path.exists() and path.is_file():
            return parse_voltage_lut_upload(original_filename, path.read_bytes())
        return ParsedVoltageLut(
            source_name=original_filename,
            frame=pd.DataFrame(),
            ok=False,
            error="cached LUT file unavailable",
        )
    return ParsedVoltageLut(
        source_name=original_filename,
        frame=pd.DataFrame(),
        ok=False,
        error="cached LUT bytes unavailable",
    )


def _lut_cache_label(record: VoltageLutCacheRecord) -> str:
    status = "ok" if record.parsed.ok else "읽을 수 없음"
    return f"{record.display_name} · {status} · samples={record.metadata.get('sample_count')}"


def _empty_lut_diagnostics() -> dict[str, object]:
    return {
        "sample_count": 0,
        "time_start_s": float("nan"),
        "time_end_s": float("nan"),
        "duration_s": float("nan"),
        "dt_min_s": float("nan"),
        "dt_median_s": float("nan"),
        "dt_max_s": float("nan"),
        "dt_irregularity_ratio": float("nan"),
        "time_monotonic": False,
        "duplicated_time_count": 0,
        "voltage_min_v": float("nan"),
        "voltage_max_v": float("nan"),
        "suspected_time_unit": "unknown",
        "time_axis_status": "unavailable",
    }


def _safe_name_part(value: object | None) -> str:
    text = "" if value is None else str(value).strip().lower()
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in text).strip("_")
