from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .ui_upload_cache import add_upload_cache_bytes
from .ui_upload_cache import build_upload_cache_records
from .ui_upload_cache import delete_upload_cache_item
from .ui_upload_cache import edit_upload_cache_metadata
from .ui_upload_cache import fallback_upload_cache_selection
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
    duplicate_of: str | None = None

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
    return add_upload_cache_bytes(
        cache_state,
        original_filename,
        data,
        cache_type="final_voltage_lut",
        upload_time=created_time,
        display_name=display_name,
        user_note=user_note,
        allow_duplicate=False,
    )


def build_lut_cache_records(cache_state: dict[str, dict[str, object]]) -> list[VoltageLutCacheRecord]:
    records: list[VoltageLutCacheRecord] = []
    for cache_record in build_upload_cache_records(cache_state):
        cache_id = cache_record.cache_item_id
        item = cache_state.get(cache_id, {})
        parsed = _parse_cache_item(cache_record.original_filename, item)
        diagnostics = build_lut_diagnostics(parsed.frame) if parsed.ok else _empty_lut_diagnostics()
        metadata = {
            "cache_item_id": cache_id,
            "id": cache_id,
            "cache_type": "final_voltage_lut",
            "original_filename": cache_record.original_filename,
            "display_name": cache_record.display_name,
            "user_note": cache_record.user_note,
            "upload_time": cache_record.upload_time,
            "created_time": cache_record.upload_time,
            "discovered_time": cache_record.discovered_time,
            "source_path": cache_record.source_path,
            "duplicate_of": cache_record.duplicate_of,
            "row_count": diagnostics.get("sample_count"),
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
            "dt_median_s": diagnostics.get("dt_median_s"),
            "sample_rate_hz": _sample_rate_from_diagnostics(diagnostics),
            "timebase_status": diagnostics.get("time_axis_status"),
            "validation_status": "ok" if parsed.ok else "unavailable",
            "parse_status": "ok" if parsed.ok else "unavailable",
            "parse_error": parsed.error,
            "normalization_status": diagnostics.get("voltage_normalization_status"),
        }
        records.append(
            VoltageLutCacheRecord(
                id=cache_id,
                original_filename=cache_record.original_filename,
                display_name=cache_record.display_name,
                user_note=cache_record.user_note,
                parsed=parsed,
                diagnostics=diagnostics,
                metadata=metadata,
                duplicate_of=cache_record.duplicate_of,
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
    return edit_upload_cache_metadata(cache_state, cache_id, display_name=display_name, user_note=user_note)


def delete_lut_cache_item(cache_state: dict[str, dict[str, object]], cache_id: str) -> bool:
    return delete_upload_cache_item(cache_state, cache_id)


def fallback_lut_cache_selection(options: list[str], selected_id: str | None) -> str | None:
    return fallback_upload_cache_selection(options, selected_id)


def _parse_cache_item(original_filename: str, item: dict[str, object]) -> ParsedVoltageLut:
    raw_bytes = item.get("csv_bytes")
    if isinstance(raw_bytes, bytes):
        return parse_voltage_lut_upload(original_filename, raw_bytes)
    file_path = item.get("file_path") or item.get("source_path")
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
    duplicate = f" duplicate_of={record.duplicate_of}" if record.duplicate_of else ""
    return f"{record.display_name} · {status} · samples={record.metadata.get('sample_count')}{duplicate}"


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
        "voltage_normalization_status": "unavailable",
    }


def _sample_rate_from_diagnostics(diagnostics: dict[str, object]) -> float:
    try:
        dt = float(diagnostics.get("dt_median_s", float("nan")))
    except (TypeError, ValueError):
        return float("nan")
    return 1.0 / dt if dt > 0 else float("nan")
