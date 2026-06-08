from __future__ import annotations

from pathlib import Path
from typing import Any

from .upload_state_dedupe import is_relative_to


def delete_physical_upload_files(
    items: list[dict[str, Any]],
    *,
    upload_root: Path,
    delete_physical: bool,
) -> list[str]:
    deleted: list[str] = []
    if not delete_physical:
        return deleted
    resolved_root = upload_root.resolve()
    for item in items:
        path = Path(str(item.get("stored_path") or ""))
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if not is_relative_to(resolved, resolved_root):
            continue
        if resolved.exists() and resolved.is_file():
            resolved.unlink()
            deleted.append(str(resolved))
    return deleted


def delete_upload_result(deleted: list[dict[str, Any]], physical_deleted: list[str], invalidated: list[str]) -> dict[str, Any]:
    return {
        "deleted_count": len(deleted),
        "deleted_item_ids": [str(item.get("upload_item_id")) for item in deleted],
        "physical_deleted_count": len(physical_deleted),
        "physical_deleted_paths": physical_deleted,
        "invalidated_session_keys": invalidated,
    }
