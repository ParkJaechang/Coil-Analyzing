from __future__ import annotations

from typing import Any

from .upload_category_aliases import UPLOAD_CATEGORIES, normalize_upload_category, upload_category_alias_metadata


def normalize_upload_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    files = payload.get("files")
    if not isinstance(files, dict):
        files = {}
    legacy_manifest_without_active_set = "active_uploads" not in payload
    active_uploads = payload.get("active_uploads")
    if not isinstance(active_uploads, dict):
        active_uploads = {}
    normalized = {"files": {}, "active_uploads": {}}
    for category in UPLOAD_CATEGORIES:
        normalized["files"][category] = _merged_entries(files, category)
        normalized["active_uploads"][category] = _merged_active_uploads(
            active_uploads,
            category,
            legacy_manifest_without_active_set=legacy_manifest_without_active_set,
            entries=normalized["files"][category],
        )
    return normalized


def _merged_entries(files: dict[str, Any], category: str) -> list[dict[str, Any]]:
    merged_entries: list[dict[str, Any]] = []
    for manifest_key, raw_entries in files.items():
        if normalize_upload_category(manifest_key) != category or not isinstance(raw_entries, list):
            continue
        for entry in raw_entries:
            if not isinstance(entry, dict):
                continue
            normalized_entry = dict(entry)
            manifest_alias_meta = upload_category_alias_metadata(manifest_key)
            normalized_entry.setdefault("upload_category_original", manifest_alias_meta["upload_category_original"])
            normalized_entry.setdefault("upload_category_canonical", manifest_alias_meta["upload_category_canonical"])
            normalized_entry["upload_category_alias_applied"] = bool(
                normalized_entry.get("upload_category_alias_applied")
                or manifest_alias_meta["upload_category_alias_applied"]
            )
            normalized_entry["category"] = category
            merged_entries.append(normalized_entry)
    return merged_entries


def _merged_active_uploads(
    active_uploads: dict[str, Any],
    category: str,
    *,
    legacy_manifest_without_active_set: bool,
    entries: list[dict[str, Any]],
) -> list[str]:
    merged_active: list[str] = []
    for active_key, raw_active in active_uploads.items():
        if normalize_upload_category(active_key) == category and isinstance(raw_active, list):
            merged_active.extend(str(item) for item in raw_active if str(item))
    if merged_active:
        return sorted(set(merged_active))
    if legacy_manifest_without_active_set:
        return [
            str(entry.get("cache_name") or entry.get("stored_filename") or entry.get("file_name") or "")
            for entry in entries
            if entry.get("cache_name") or entry.get("stored_filename") or entry.get("file_name")
        ]
    return []
