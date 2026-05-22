from __future__ import annotations

import json
from typing import Any

from .upload_category_aliases import UPLOAD_CATEGORIES, normalize_upload_category
from .ui_upload_state import UploadStatePaths
from .ui_upload_state import build_upload_state_paths
from .ui_upload_state import list_persisted_uploads
from .ui_upload_state import load_upload_manifest


def upload_memory_status(*, paths: UploadStatePaths | None = None) -> dict[str, Any]:
    resolved_paths = paths or build_upload_state_paths()
    manifest = load_upload_manifest(paths=resolved_paths)
    active = manifest.get("active_uploads", {})
    category_counts = {
        category: {
            "cached_count": len(list_persisted_uploads(category, paths=resolved_paths)),
            "remembered_count": len(active.get(category, []) or []),
            "manifest_count": len(manifest.get("files", {}).get(category, []) or []),
        }
        for category in UPLOAD_CATEGORIES
    }
    missing_remembered_counts = {
        category: max(int(counts["remembered_count"]) - int(counts["cached_count"]), 0)
        for category, counts in category_counts.items()
    }
    return {
        "upload_memory_manifest_exists": resolved_paths.upload_manifest_path.exists(),
        "cache_directory_path": str(resolved_paths.uploads_dir),
        "upload_manifest_path": str(resolved_paths.upload_manifest_path),
        "upload_memory_restore_status": "ok" if any(counts["cached_count"] for counts in category_counts.values()) else "no_cached_files",
        "continuous_upload_restore_status": "ok"
        if category_counts["continuous"]["cached_count"]
        else ("remembered_but_missing_files" if category_counts["continuous"]["remembered_count"] else "no_continuous_files"),
        "upload_memory_restore_source": "disk_upload_cache_manifest" if resolved_paths.upload_manifest_path.exists() else "none",
        "upload_memory_category_counts": category_counts,
        "upload_memory_missing_remembered_counts": missing_remembered_counts,
        "upload_memory_restore_error": "",
        "cached_continuous_count": category_counts["continuous"]["cached_count"],
        "cached_finite_count": category_counts["transient"]["cached_count"],
        "remembered_continuous_count": category_counts["continuous"]["remembered_count"],
        "remembered_finite_count": category_counts["transient"]["remembered_count"],
        "missing_remembered_continuous_count": missing_remembered_counts["continuous"],
        "missing_remembered_finite_count": missing_remembered_counts["transient"],
    }


def activate_cached_uploads(category: str, *, paths: UploadStatePaths | None = None) -> dict[str, Any]:
    category = normalize_upload_category(category)
    if category not in UPLOAD_CATEGORIES:
        raise ValueError(f"Unsupported upload category: {category}")
    resolved_paths = paths or build_upload_state_paths()
    manifest = load_upload_manifest(paths=resolved_paths)
    category_dir = resolved_paths.category_dir(category)
    active_names: list[str] = []
    for entry in manifest["files"].get(category, []):
        cache_name = str(entry.get("cache_name") or entry.get("stored_filename") or entry.get("file_name") or "").strip()
        if not cache_name:
            continue
        path = category_dir / cache_name
        if path.exists() and path.is_file():
            active_names.append(cache_name)
    manifest.setdefault("active_uploads", {})[category] = sorted(set(active_names))
    resolved_paths.upload_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_paths.upload_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"category": category, "activated_count": len(set(active_names))}


__all__ = ["activate_cached_uploads", "upload_memory_status"]
