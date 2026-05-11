from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


UploadCacheType = Literal["final_voltage_lut", "actual_drive_validation", "raw_waveform", "unknown"]


@dataclass(frozen=True)
class UploadCacheRecord:
    cache_item_id: str
    cache_type: UploadCacheType
    original_filename: str
    display_name: str
    user_note: str
    upload_time: str | None
    discovered_time: str | None
    source_path: str | None
    content_sha256: str | None
    duplicate_of: str | None
    metadata: dict[str, object]

    @property
    def id(self) -> str:
        return self.cache_item_id


def add_upload_cache_bytes(
    cache_state: dict[str, dict[str, object]],
    original_filename: str,
    data: bytes,
    *,
    cache_type: UploadCacheType = "unknown",
    upload_time: str | None = None,
    discovered_time: str | None = None,
    source_path: str | None = None,
    display_name: str | None = None,
    user_note: str = "",
    allow_duplicate: bool = True,
) -> str:
    content = bytes(data)
    digest = hashlib.sha256(content).hexdigest()
    if not allow_duplicate:
        existing_id = _find_same_upload(cache_state, digest, cache_type, original_filename)
        if existing_id is not None:
            return existing_id
    duplicate_of = _find_duplicate_content(cache_state, digest, cache_type)
    cache_id = _unique_cache_id(cache_state, cache_type, original_filename, upload_time or discovered_time, digest)
    cache_state[cache_id] = {
        "cache_item_id": cache_id,
        "id": cache_id,
        "cache_type": cache_type,
        "original_filename": str(original_filename),
        "display_name": display_name or str(original_filename),
        "user_note": str(user_note),
        "upload_time": upload_time,
        "discovered_time": discovered_time,
        "source_path": source_path,
        "content_sha256": digest,
        "content_sha256_short": digest[:16],
        "duplicate_of": duplicate_of,
        "csv_bytes": content,
    }
    return cache_id


def build_upload_cache_records(cache_state: dict[str, dict[str, object]]) -> list[UploadCacheRecord]:
    records = [_record_from_item(cache_id, item) for cache_id, item in cache_state.items()]
    return sorted(records, key=lambda record: record.cache_item_id)


def build_upload_cache_selection_options(
    records: list[UploadCacheRecord],
) -> tuple[list[str], dict[str, UploadCacheRecord], dict[str, str]]:
    options = [record.cache_item_id for record in records]
    records_by_id = {record.cache_item_id: record for record in records}
    labels_by_id = {record.cache_item_id: _record_label(record) for record in records}
    return options, records_by_id, labels_by_id


def edit_upload_cache_metadata(
    cache_state: dict[str, dict[str, object]],
    cache_item_id: str,
    *,
    display_name: str | None = None,
    user_note: str | None = None,
) -> bool:
    item = cache_state.get(cache_item_id)
    if item is None:
        return False
    if display_name is not None:
        item["display_name"] = str(display_name)
    if user_note is not None:
        item["user_note"] = str(user_note)
    return True


def delete_upload_cache_item(cache_state: dict[str, dict[str, object]], cache_item_id: str) -> bool:
    return cache_state.pop(cache_item_id, None) is not None


def fallback_upload_cache_selection(options: list[str], selected_id: str | None) -> str | None:
    if not options:
        return None
    if selected_id in options:
        return selected_id
    return options[0]


def cache_item_bytes(cache_state: dict[str, dict[str, object]], cache_item_id: str) -> bytes | None:
    item = cache_state.get(cache_item_id)
    if item is None:
        return None
    value = item.get("csv_bytes")
    return value if isinstance(value, bytes) else None


def _record_from_item(cache_id: str, item: dict[str, object]) -> UploadCacheRecord:
    metadata = {
        "cache_item_id": str(item.get("cache_item_id") or item.get("id") or cache_id),
        "cache_type": str(item.get("cache_type") or "unknown"),
        "original_filename": str(item.get("original_filename") or item.get("source_name") or cache_id),
        "display_name": str(item.get("display_name") or item.get("original_filename") or cache_id),
        "user_note": str(item.get("user_note") or ""),
        "upload_time": item.get("upload_time"),
        "discovered_time": item.get("discovered_time"),
        "source_path": item.get("source_path") or item.get("file_path"),
        "duplicate_of": item.get("duplicate_of"),
        "parse_status": item.get("parse_status", "pending"),
        "validation_status": item.get("validation_status", "pending"),
        "normalization_status": item.get("normalization_status"),
    }
    return UploadCacheRecord(
        cache_item_id=str(metadata["cache_item_id"]),
        cache_type=_normalize_cache_type(metadata["cache_type"]),
        original_filename=str(metadata["original_filename"]),
        display_name=str(metadata["display_name"]),
        user_note=str(metadata["user_note"]),
        upload_time=str(metadata["upload_time"]) if metadata["upload_time"] is not None else None,
        discovered_time=str(metadata["discovered_time"]) if metadata["discovered_time"] is not None else None,
        source_path=str(metadata["source_path"]) if metadata["source_path"] is not None else None,
        content_sha256=str(item.get("content_sha256")) if item.get("content_sha256") is not None else None,
        duplicate_of=str(metadata["duplicate_of"]) if metadata["duplicate_of"] is not None else None,
        metadata=metadata,
    )


def _unique_cache_id(
    cache_state: dict[str, dict[str, object]],
    cache_type: str,
    original_filename: str,
    timestamp: str | None,
    digest: str,
) -> str:
    stem = _safe_name_part(Path(original_filename).stem) or "upload"
    time_part = _safe_name_part(timestamp) if timestamp else "session"
    type_part = _safe_name_part(cache_type) or "unknown"
    base = f"{type_part}:{stem}:{time_part}:{digest[:16]}"
    if base not in cache_state:
        return base
    index = 2
    while f"{base}#{index}" in cache_state:
        index += 1
    return f"{base}#{index}"


def _find_duplicate_content(cache_state: dict[str, dict[str, object]], digest: str, cache_type: str) -> str | None:
    for item_id, item in cache_state.items():
        if item.get("content_sha256") == digest and item.get("cache_type") == cache_type:
            return str(item.get("cache_item_id") or item.get("id") or item_id)
    return None


def _find_same_upload(
    cache_state: dict[str, dict[str, object]],
    digest: str,
    cache_type: str,
    original_filename: str,
) -> str | None:
    for item_id, item in cache_state.items():
        if item.get("content_sha256") != digest:
            continue
        if item.get("cache_type") != cache_type:
            continue
        if str(item.get("original_filename") or "") != str(original_filename):
            continue
        return str(item.get("cache_item_id") or item.get("id") or item_id)
    return None


def _record_label(record: UploadCacheRecord) -> str:
    suffix = f" duplicate_of={record.duplicate_of}" if record.duplicate_of else ""
    return f"{record.display_name} - {record.cache_type}{suffix}"


def _normalize_cache_type(value: object) -> UploadCacheType:
    text = str(value)
    if text in {"final_voltage_lut", "actual_drive_validation", "raw_waveform", "unknown"}:
        return text  # type: ignore[return-value]
    return "unknown"


def _safe_name_part(value: object | None) -> str:
    text = "" if value is None else str(value).strip().lower()
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in text).strip("_")
