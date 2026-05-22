from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .upload_category_aliases import normalize_upload_category
from .upload_filename import canonicalize_upload_filename


def list_active_upload_payload_records_from_manifest(
    category: str,
    *,
    paths: Any,
    manifest: dict[str, Any],
    list_persisted_uploads: Callable[..., list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    category = normalize_upload_category(category)
    active_names = {str(item) for item in manifest.get("active_uploads", {}).get(category, []) if str(item)}
    if not active_names:
        return []
    records = _direct_active_records(category, active_names, paths=paths, manifest=manifest)
    if records:
        return sorted(records, key=lambda item: str(item.get("cache_name") or ""))
    return sorted(
        _canonical_fallback_records(
            category,
            active_names,
            paths=paths,
            manifest=manifest,
            list_persisted_uploads=list_persisted_uploads,
        ),
        key=lambda item: str(item.get("cache_name") or ""),
    )


def _direct_active_records(
    category: str,
    active_names: set[str],
    *,
    paths: Any,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    category_dir = paths.category_dir(category)
    for entry in manifest["files"].get(category, []):
        if not isinstance(entry, dict):
            continue
        cache_name = str(entry.get("cache_name") or entry.get("stored_filename") or entry.get("file_name") or "")
        if cache_name not in active_names:
            continue
        path = Path(str(entry.get("stored_path") or entry.get("path") or category_dir / cache_name))
        if path.exists() and path.is_file():
            filename_meta = canonicalize_upload_filename(
                cache_name,
                original_filename=entry.get("original_filename") or entry.get("display_name") or entry.get("file_name"),
            )
            records.append(
                {
                    "cache_name": cache_name,
                    "canonical_filename": filename_meta["upload_canonical_filename"],
                    "path": str(path),
                    "payload_source": "remembered_upload",
                }
            )
    return records


def _canonical_fallback_records(
    category: str,
    active_names: set[str],
    *,
    paths: Any,
    manifest: dict[str, Any],
    list_persisted_uploads: Callable[..., list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    active_canonical_names = _active_canonical_names(category, active_names, manifest=manifest)
    for record in list_persisted_uploads(category, paths=paths):
        canonical_name = str(record.get("canonical_filename") or record.get("original_filename") or "")
        cache_name = str(record.get("cache_name") or record.get("stored_filename") or "")
        path = Path(str(record.get("path") or record.get("stored_path") or ""))
        if cache_name not in active_names and canonical_name not in active_canonical_names:
            continue
        if path.exists() and path.is_file():
            records.append(
                {
                    "cache_name": cache_name,
                    "canonical_filename": canonical_name,
                    "path": str(path),
                    "payload_source": "remembered_upload_canonical_fallback",
                }
            )
    return records


def _active_canonical_names(category: str, active_names: set[str], *, manifest: dict[str, Any]) -> set[str]:
    names: set[str] = {
        str(canonicalize_upload_filename(active_name).get("upload_canonical_filename") or "")
        for active_name in active_names
    } - {""}
    for entry in manifest["files"].get(category, []):
        if not isinstance(entry, dict):
            continue
        cache_name = str(entry.get("cache_name") or entry.get("stored_filename") or entry.get("file_name") or "")
        if cache_name not in active_names:
            continue
        filename_meta = canonicalize_upload_filename(
            cache_name,
            original_filename=entry.get("original_filename") or entry.get("display_name") or entry.get("file_name"),
        )
        canonical_name = str(filename_meta.get("upload_canonical_filename") or "")
        if canonical_name:
            names.add(canonical_name)
    return names
