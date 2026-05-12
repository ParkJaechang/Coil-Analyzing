from __future__ import annotations

import json
from typing import Any

from .ui_upload_state import UPLOAD_CATEGORIES
from .ui_upload_state import UploadStatePaths
from .ui_upload_state import build_upload_state_paths
from .ui_upload_state import list_persisted_uploads
from .ui_upload_state import load_upload_manifest


def upload_memory_status(*, paths: UploadStatePaths | None = None) -> dict[str, Any]:
    resolved_paths = paths or build_upload_state_paths()
    active = load_upload_manifest(paths=resolved_paths).get("active_uploads", {})
    return {
        "upload_memory_manifest_exists": resolved_paths.upload_manifest_path.exists(),
        "cache_directory_path": str(resolved_paths.uploads_dir),
        "cached_continuous_count": len(list_persisted_uploads("continuous", paths=resolved_paths)),
        "cached_finite_count": len(list_persisted_uploads("transient", paths=resolved_paths)),
        "remembered_continuous_count": len(active.get("continuous", []) or []),
        "remembered_finite_count": len(active.get("transient", []) or []),
    }


def activate_cached_uploads(category: str, *, paths: UploadStatePaths | None = None) -> dict[str, Any]:
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
