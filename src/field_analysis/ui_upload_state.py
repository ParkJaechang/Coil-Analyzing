from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from .ui_raw_waveforms_labels import infer_new_dataset_filename_metadata

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_STATE_DIRNAME = "field_analysis_app_state"
UPLOAD_CATEGORIES = ("continuous", "transient", "validation", "lcr")
CATEGORY_LABELS = {
    "continuous": "연속 cycle",
    "transient": "finite-cycle",
    "validation": "2차 보정 검증 run",
    "lcr": "LCR",
}
UPLOADER_SESSION_KEYS = {
    "continuous": "continuous_uploads",
    "transient": "transient_uploads",
    "validation": "validation_uploads",
    "lcr": "lcr_uploads",
}
UPLOAD_MEMORY_LABEL_BY_CATEGORY = {
    "continuous": "continuous_cycle",
    "transient": "finite_cycle",
    "validation": "actual_drive_validation_run",
    "lcr": "lcr",
}
UPLOAD_MEMORY_CATEGORY_BY_LABEL = {value: key for key, value in UPLOAD_MEMORY_LABEL_BY_CATEGORY.items()}


@dataclass(frozen=True)
class UploadStatePaths:
    repo_root: Path
    app_state_dir: Path
    uploads_dir: Path
    upload_manifest_path: Path
    recommendation_library_dir: Path
    validation_retune_history_path: Path

    def category_dir(self, category: str) -> Path:
        return self.uploads_dir / str(category)


def build_upload_state_paths(repo_root: Path | None = None) -> UploadStatePaths:
    root = (repo_root or REPO_ROOT).resolve()
    app_state_dir = root / "outputs" / APP_STATE_DIRNAME
    return UploadStatePaths(
        repo_root=root,
        app_state_dir=app_state_dir,
        uploads_dir=app_state_dir / "uploads",
        upload_manifest_path=app_state_dir / "upload_manifest.json",
        recommendation_library_dir=app_state_dir / "recommendation_library",
        validation_retune_history_path=app_state_dir / "validation_retune_history.json",
    )


def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    files = payload.get("files")
    if not isinstance(files, dict):
        files = {}
    normalized = {"files": {}}
    for category in UPLOAD_CATEGORIES:
        entries = files.get(category)
        if not isinstance(entries, list):
            entries = []
        normalized["files"][category] = [entry for entry in entries if isinstance(entry, dict)]
    return normalized


def load_upload_manifest(*, paths: UploadStatePaths | None = None) -> dict[str, Any]:
    resolved_paths = paths or build_upload_state_paths()
    return _normalize_manifest(_load_json(resolved_paths.upload_manifest_path, {"files": {}}))


def _stable_cache_name(file_name: str, raw_bytes: bytes) -> str:
    leaf = Path(str(file_name or "")).name or "upload.bin"
    if len(leaf) > 17 and leaf[16] == "_" and all(char in "0123456789abcdefABCDEF" for char in leaf[:16]):
        return leaf
    digest = hashlib.sha256(raw_bytes).hexdigest()[:16]
    return f"{digest}_{leaf}"


def _manifest_record(
    *,
    category: str,
    display_name: str,
    cache_name: str,
    size_bytes: int,
    path: Path,
    source: str,
    upload_item_id: str | None = None,
    duplicate_of: str | None = None,
) -> dict[str, Any]:
    return {
        "upload_item_id": upload_item_id or _upload_item_id(category, cache_name),
        "label": UPLOAD_MEMORY_LABEL_BY_CATEGORY.get(category, "unknown"),
        "category": category,
        "original_filename": display_name,
        "display_name": display_name,
        "file_name": display_name,
        "cache_name": cache_name,
        "stored_filename": cache_name,
        "size_bytes": int(size_bytes),
        "file_size": int(size_bytes),
        "path": str(path),
        "stored_path": str(path),
        "source": source,
        "duplicate_of": duplicate_of,
    }


def list_persisted_uploads(category: str, *, paths: UploadStatePaths | None = None) -> list[dict[str, Any]]:
    resolved_paths = paths or build_upload_state_paths()
    category_dir = resolved_paths.category_dir(category)
    manifest = load_upload_manifest(paths=resolved_paths)
    records: list[dict[str, Any]] = []
    seen_cache_names: set[str] = set()

    for entry in manifest["files"].get(category, []):
        cache_name = str(entry.get("cache_name") or entry.get("file_name") or "").strip()
        if not cache_name:
            continue
        path = category_dir / cache_name
        if not path.exists():
            continue
        display_name = str(entry.get("display_name") or entry.get("file_name") or cache_name)
        records.append(
            _manifest_record(
                category=category,
                display_name=display_name,
                cache_name=cache_name,
                size_bytes=int(entry.get("size_bytes") or path.stat().st_size),
                path=path,
                source="manifest",
                upload_item_id=str(entry.get("upload_item_id") or _upload_item_id(category, cache_name)),
                duplicate_of=str(entry.get("duplicate_of")) if entry.get("duplicate_of") else None,
            )
        )
        seen_cache_names.add(cache_name)

    if category_dir.exists():
        for path in sorted(child for child in category_dir.iterdir() if child.is_file()):
            if path.name in seen_cache_names:
                continue
            records.append(
                _manifest_record(
                    category=category,
                    display_name=path.name,
                    cache_name=path.name,
                    size_bytes=path.stat().st_size,
                    path=path,
                    source="scan",
                    upload_item_id=_upload_item_id(category, path.name),
                )
            )

    return sorted(records, key=lambda item: (str(item.get("display_name") or ""), str(item.get("cache_name") or "")))


def persist_uploaded_files(
    category: str,
    uploaded_files: list[Any] | tuple[Any, ...] | None,
    *,
    paths: UploadStatePaths | None = None,
) -> list[dict[str, Any]]:
    resolved_paths = paths or build_upload_state_paths()
    category_dir = resolved_paths.category_dir(category)
    category_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_upload_manifest(paths=resolved_paths)
    existing_entries = [dict(entry) for entry in manifest["files"].get(category, []) if isinstance(entry, dict)]
    existing_cache_names = {str(entry.get("cache_name") or "") for entry in existing_entries if entry.get("cache_name")}
    existing_digest_to_id = {
        str(entry.get("content_sha256")): str(entry.get("upload_item_id") or _upload_item_id(category, str(entry.get("cache_name") or "")))
        for entry in existing_entries
        if entry.get("content_sha256") and entry.get("cache_name")
    }

    for uploaded in uploaded_files or []:
        display_name = Path(str(getattr(uploaded, "name", "") or "")).name or "upload.bin"
        raw_bytes = bytes(uploaded.getvalue())
        digest = hashlib.sha256(raw_bytes).hexdigest()
        cache_name = _unique_cache_name(_stable_cache_name(display_name, raw_bytes), existing_cache_names)
        target_path = category_dir / cache_name
        if not target_path.exists() or target_path.read_bytes() != raw_bytes:
            target_path.write_bytes(raw_bytes)
        upload_item_id = _upload_item_id(category, cache_name)
        existing_entries.append(
            {
                "upload_item_id": upload_item_id,
                "label": UPLOAD_MEMORY_LABEL_BY_CATEGORY.get(category, "unknown"),
                "category": category,
                "file_name": display_name,
                "original_filename": display_name,
                "display_name": display_name,
                "cache_name": cache_name,
                "stored_filename": cache_name,
                "path": str(target_path),
                "stored_path": str(target_path),
                "size_bytes": len(raw_bytes),
                "file_size": len(raw_bytes),
                "content_sha256": digest,
                "duplicate_of": existing_digest_to_id.get(digest),
            }
        )
        existing_cache_names.add(cache_name)
        existing_digest_to_id.setdefault(digest, upload_item_id)

    manifest["files"][category] = sorted(
        existing_entries,
        key=lambda item: (str(item.get("display_name") or item.get("file_name") or ""), str(item.get("cache_name") or "")),
    )
    _write_json(resolved_paths.upload_manifest_path, manifest)
    return list_persisted_uploads(category, paths=resolved_paths)


def category_payloads(
    category: str,
    uploaded_files: list[Any] | tuple[Any, ...] | None,
    *,
    paths: UploadStatePaths | None = None,
) -> list[tuple[str, bytes]]:
    resolved_paths = paths or build_upload_state_paths()
    if uploaded_files:
        persist_uploaded_files(category, uploaded_files, paths=resolved_paths)
    payloads: list[tuple[str, bytes]] = []
    for record in list_persisted_uploads(category, paths=resolved_paths):
        path = Path(str(record["path"]))
        payloads.append((str(record["cache_name"]), path.read_bytes()))
    return payloads


def category_summary_rows(*, paths: UploadStatePaths | None = None) -> list[dict[str, Any]]:
    resolved_paths = paths or build_upload_state_paths()
    item_summary = build_upload_memory_group_summary(build_upload_memory_items(paths=resolved_paths))
    rows: list[dict[str, Any]] = []
    for category in UPLOAD_CATEGORIES:
        memory_label = UPLOAD_MEMORY_LABEL_BY_CATEGORY.get(category, category)
        group = item_summary.get(memory_label, {"count": 0, "files": []})
        rows.append(
            {
                "category": category,
                "label": CATEGORY_LABELS.get(category, category),
                "count": int(group["count"]),
                "files": ", ".join(str(file_name) for file_name in group["files"][:4]),
                "dir": str(resolved_paths.category_dir(category)),
            }
        )
    return rows


def clear_category_uploads(category: str, *, paths: UploadStatePaths | None = None) -> None:
    delete_upload_memory_group(category, paths=paths)


def clear_all_uploads(*, paths: UploadStatePaths | None = None) -> None:
    resolved_paths = paths or build_upload_state_paths()
    for category in UPLOAD_CATEGORIES:
        clear_category_uploads(category, paths=resolved_paths)


def build_upload_memory_items(*, paths: UploadStatePaths | None = None) -> list[dict[str, Any]]:
    resolved_paths = paths or build_upload_state_paths()
    manifest = load_upload_manifest(paths=resolved_paths)
    items: list[dict[str, Any]] = []
    for category in UPLOAD_CATEGORIES:
        category_dir = resolved_paths.category_dir(category)
        seen_cache_names: set[str] = set()
        for entry in manifest["files"].get(category, []):
            if not isinstance(entry, dict):
                continue
            item = _upload_memory_item_from_entry(category, entry, category_dir=category_dir)
            if item is not None:
                items.append(item)
                seen_cache_names.add(str(item["stored_filename"]))
        if category_dir.exists():
            for path in sorted(child for child in category_dir.iterdir() if child.is_file()):
                if path.name in seen_cache_names:
                    continue
                item = _upload_memory_item_from_entry(
                    category,
                    {
                        "cache_name": path.name,
                        "file_name": path.name,
                        "display_name": path.name,
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                    },
                    category_dir=category_dir,
                )
                if item is not None:
                    items.append(item)
    return sorted(
        items,
        key=lambda item: (
            str(item["label"]),
            str(item["original_filename"]),
            bool(item.get("duplicate_of")),
            str(item["stored_filename"]),
            str(item["upload_item_id"]),
        ),
    )


def build_upload_memory_group_summary(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary = {
        label: {"label": label, "count": 0, "files": [], "items": []}
        for label in UPLOAD_MEMORY_CATEGORY_BY_LABEL
    }
    for item in items:
        label = str(item.get("label") or "unknown")
        group = summary.setdefault(label, {"label": label, "count": 0, "files": [], "items": []})
        group["items"].append(item)
        group["files"].append(str(item.get("original_filename") or item.get("stored_filename") or "unknown"))
        group["count"] = int(group["count"]) + 1
    return summary


def delete_upload_memory_item(
    upload_item_id: str,
    *,
    paths: UploadStatePaths | None = None,
    delete_physical: bool = True,
) -> dict[str, Any]:
    return delete_upload_memory_items([upload_item_id], paths=paths, delete_physical=delete_physical)


def delete_upload_memory_items(
    upload_item_ids: list[str] | tuple[str, ...] | set[str],
    *,
    paths: UploadStatePaths | None = None,
    delete_physical: bool = True,
) -> dict[str, Any]:
    resolved_paths = paths or build_upload_state_paths()
    target_ids = {str(item_id) for item_id in upload_item_ids if str(item_id)}
    if not target_ids:
        return _delete_result([], [], [])
    manifest = load_upload_manifest(paths=resolved_paths)
    deleted: list[dict[str, Any]] = []
    retained_by_category: dict[str, list[dict[str, Any]]] = {}
    for category in UPLOAD_CATEGORIES:
        retained: list[dict[str, Any]] = []
        category_dir = resolved_paths.category_dir(category)
        manifest_cache_names: set[str] = set()
        for entry in manifest["files"].get(category, []):
            if not isinstance(entry, dict):
                continue
            item = _upload_memory_item_from_entry(category, entry, category_dir=category_dir)
            if item is not None:
                manifest_cache_names.add(str(item["stored_filename"]))
            if item is not None and item["upload_item_id"] in target_ids:
                deleted.append(item)
            else:
                retained.append(entry)
        if category_dir.exists():
            for path in sorted(child for child in category_dir.iterdir() if child.is_file()):
                if path.name in manifest_cache_names:
                    continue
                item = _upload_memory_item_from_entry(
                    category,
                    {
                        "cache_name": path.name,
                        "file_name": path.name,
                        "display_name": path.name,
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                    },
                    category_dir=category_dir,
                )
                if item is not None and item["upload_item_id"] in target_ids:
                    deleted.append(item)
        retained_by_category[category] = retained
    for category, retained in retained_by_category.items():
        manifest["files"][category] = retained
    _write_json(resolved_paths.upload_manifest_path, manifest)
    physical_deleted = _delete_physical_files(deleted, resolved_paths=resolved_paths, delete_physical=delete_physical)
    invalidated = sorted({UPLOADER_SESSION_KEYS.get(str(item.get("category")), "") for item in deleted} - {""})
    return _delete_result(deleted, physical_deleted, invalidated)


def delete_upload_memory_group(
    label_or_category: str,
    *,
    paths: UploadStatePaths | None = None,
    delete_physical: bool = True,
) -> dict[str, Any]:
    category = UPLOAD_MEMORY_CATEGORY_BY_LABEL.get(label_or_category, label_or_category)
    ids = [
        str(item["upload_item_id"])
        for item in build_upload_memory_items(paths=paths)
        if item.get("category") == category or item.get("label") == label_or_category
    ]
    return delete_upload_memory_items(ids, paths=paths, delete_physical=delete_physical)


def reset_uploader_session_state(*, session_keys: tuple[str, ...] = ()) -> None:
    for key in tuple(UPLOADER_SESSION_KEYS.values()) + tuple(session_keys):
        if key in st.session_state:
            del st.session_state[key]


def render_sidebar_memory_panel(*, paths: UploadStatePaths | None = None) -> None:
    from .ui_upload_memory_management import render_upload_memory_management

    render_upload_memory_management(paths=paths)


def render_workspace_panel(*, paths: UploadStatePaths | None = None) -> None:
    import pandas as pd

    resolved_paths = paths or build_upload_state_paths()
    with st.expander("작업 공간 / 기기별 데이터 / 업로드 폴더", expanded=False):
        summary_frame = pd.DataFrame(category_summary_rows(paths=resolved_paths))[["label", "count", "dir"]]
        st.dataframe(summary_frame, hide_index=True, use_container_width=True)
        st.markdown("#### 경로")
        st.code(
            "\n".join(
                [
                    f"app_state: {resolved_paths.app_state_dir}",
                    f"upload_manifest: {resolved_paths.upload_manifest_path}",
                    f"recommendation_library: {resolved_paths.recommendation_library_dir}",
                    f"validation_retune_history: {resolved_paths.validation_retune_history_path}",
                ]
            )
        )


def _upload_item_id(category: str, cache_name: str) -> str:
    digest = hashlib.sha256(f"{category}/{cache_name}".encode("utf-8")).hexdigest()[:16]
    return f"{UPLOAD_MEMORY_LABEL_BY_CATEGORY.get(category, 'unknown')}:{digest}"


def _unique_cache_name(base_name: str, existing_names: set[str]) -> str:
    if base_name not in existing_names:
        return base_name
    path = Path(base_name)
    stem = path.stem
    suffix = path.suffix
    index = 2
    while True:
        candidate = f"{stem}_{index}{suffix}"
        if candidate not in existing_names:
            return candidate
        index += 1


def _upload_memory_item_from_entry(category: str, entry: dict[str, Any], *, category_dir: Path) -> dict[str, Any] | None:
    cache_name = str(entry.get("cache_name") or entry.get("stored_filename") or entry.get("file_name") or "").strip()
    if not cache_name:
        return None
    stored_path = Path(str(entry.get("stored_path") or entry.get("path") or category_dir / cache_name))
    original_filename = str(entry.get("original_filename") or entry.get("display_name") or entry.get("file_name") or cache_name)
    file_exists = stored_path.exists() and stored_path.is_file()
    size = int(entry.get("file_size") or entry.get("size_bytes") or (stored_path.stat().st_size if file_exists else 0))
    filename_meta = infer_new_dataset_filename_metadata(original_filename)
    row_count = _row_count(stored_path) if file_exists else None
    parse_status = str(entry.get("parse_status") or ("ok" if file_exists else "missing_file"))
    return {
        "upload_item_id": str(entry.get("upload_item_id") or _upload_item_id(category, cache_name)),
        "label": str(entry.get("label") or UPLOAD_MEMORY_LABEL_BY_CATEGORY.get(category, "unknown")),
        "category": category,
        "original_filename": original_filename,
        "stored_filename": cache_name,
        "stored_path": str(stored_path),
        "uploaded_at": entry.get("uploaded_at"),
        "discovered_at": entry.get("discovered_at"),
        "file_size": size,
        "parse_status": parse_status,
        "detected_format": str(entry.get("detected_format") or filename_meta.get("source_type") or "unknown"),
        "source_type": str(entry.get("source_type") or filename_meta.get("source_type") or "unknown"),
        "waveform_family": entry.get("waveform_family") or filename_meta.get("waveform_type"),
        "freq_hz": entry.get("freq_hz") if entry.get("freq_hz") is not None else filename_meta.get("freq_hz"),
        "cycle_count": entry.get("cycle_count") if entry.get("cycle_count") is not None else filename_meta.get("cycle_count"),
        "row_count": row_count,
        "validation_status": str(entry.get("validation_status") or ("available" if file_exists else "unavailable")),
        "duplicate_of": entry.get("duplicate_of"),
        "content_sha256": entry.get("content_sha256"),
        "file_exists": file_exists,
    }


def _row_count(path: Path) -> int | None:
    try:
        return max(len(path.read_text(encoding="utf-8-sig").splitlines()) - 1, 0)
    except OSError:
        return None


def _delete_physical_files(
    items: list[dict[str, Any]],
    *,
    resolved_paths: UploadStatePaths,
    delete_physical: bool,
) -> list[str]:
    deleted: list[str] = []
    if not delete_physical:
        return deleted
    upload_root = resolved_paths.uploads_dir.resolve()
    for item in items:
        path = Path(str(item.get("stored_path") or ""))
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if not _is_relative_to(resolved, upload_root):
            continue
        if resolved.exists() and resolved.is_file():
            resolved.unlink()
            deleted.append(str(resolved))
    return deleted


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _delete_result(deleted: list[dict[str, Any]], physical_deleted: list[str], invalidated: list[str]) -> dict[str, Any]:
    return {
        "deleted_count": len(deleted),
        "deleted_item_ids": [str(item.get("upload_item_id")) for item in deleted],
        "physical_deleted_count": len(physical_deleted),
        "physical_deleted_paths": physical_deleted,
        "invalidated_session_keys": invalidated,
    }


__all__ = [name for name in globals() if not name.startswith("_")]
