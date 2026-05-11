from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def uploaded_file_keys(uploaded_files: list[Any] | tuple[Any, ...]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for uploaded in uploaded_files:
        display_name = Path(str(getattr(uploaded, "name", "") or "")).name or "upload.bin"
        raw_bytes = bytes(uploaded.getvalue())
        keys.add((display_name, hashlib.sha256(raw_bytes).hexdigest()))
    return keys


def exact_upload_key(entry: dict[str, Any]) -> tuple[str, str] | None:
    digest = str(entry.get("content_sha256") or "").strip()
    if not digest:
        return None
    original = str(entry.get("original_filename") or entry.get("display_name") or entry.get("file_name") or "").strip()
    if not original:
        return None
    return (Path(original).name, digest)


def dedupe_exact_upload_entries(entries: list[dict[str, Any]], *, category_dir: Path) -> list[dict[str, Any]]:
    retained: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        key = exact_upload_key(entry)
        if key is None or key not in seen:
            retained.append(entry)
            if key is not None:
                seen.add(key)
            continue
        delete_duplicate_upload_file(entry, category_dir=category_dir)
    return retained


def list_exact_upload_records_from_manifest(
    manifest: dict[str, Any],
    category: str,
    exact_keys: set[tuple[str, str]],
    *,
    category_dir: Path,
) -> list[dict[str, Any]]:
    if not exact_keys:
        return []
    records: list[dict[str, Any]] = []
    for entry in manifest.get("files", {}).get(category, []):
        if not isinstance(entry, dict) or exact_upload_key(entry) not in exact_keys:
            continue
        cache_name = str(entry.get("cache_name") or entry.get("stored_filename") or entry.get("file_name") or "").strip()
        if not cache_name:
            continue
        path = Path(str(entry.get("stored_path") or entry.get("path") or category_dir / cache_name))
        if path.exists() and path.is_file():
            records.append({"cache_name": cache_name, "path": str(path), "payload_source": "current_upload"})
    return sorted(records, key=lambda item: str(item.get("cache_name") or ""))


def delete_duplicate_upload_file(entry: dict[str, Any], *, category_dir: Path) -> None:
    path = Path(str(entry.get("stored_path") or entry.get("path") or category_dir / str(entry.get("cache_name") or "")))
    try:
        resolved = path.resolve()
        root = category_dir.resolve()
    except OSError:
        return
    if not is_relative_to(resolved, root):
        return
    if resolved.exists() and resolved.is_file():
        resolved.unlink()


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
