from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


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


def _manifest_record(*, category: str, display_name: str, cache_name: str, size_bytes: int, path: Path, source: str) -> dict[str, Any]:
    return {
        "category": category,
        "display_name": display_name,
        "file_name": display_name,
        "cache_name": cache_name,
        "size_bytes": int(size_bytes),
        "path": str(path),
        "source": source,
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
    existing_entries = {
        str(entry.get("cache_name") or ""): dict(entry)
        for entry in manifest["files"].get(category, [])
        if isinstance(entry, dict) and entry.get("cache_name")
    }

    for uploaded in uploaded_files or []:
        display_name = Path(str(getattr(uploaded, "name", "") or "")).name or "upload.bin"
        raw_bytes = bytes(uploaded.getvalue())
        cache_name = _stable_cache_name(display_name, raw_bytes)
        target_path = category_dir / cache_name
        if not target_path.exists() or target_path.read_bytes() != raw_bytes:
            target_path.write_bytes(raw_bytes)
        existing_entries[cache_name] = {
            "file_name": display_name,
            "display_name": display_name,
            "cache_name": cache_name,
            "size_bytes": len(raw_bytes),
        }

    manifest["files"][category] = sorted(
        existing_entries.values(),
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
    rows: list[dict[str, Any]] = []
    for category in UPLOAD_CATEGORIES:
        records = list_persisted_uploads(category, paths=resolved_paths)
        rows.append(
            {
                "category": category,
                "label": CATEGORY_LABELS.get(category, category),
                "count": int(len(records)),
                "files": ", ".join(str(record["display_name"]) for record in records[:4]),
                "dir": str(resolved_paths.category_dir(category)),
            }
        )
    return rows


def clear_category_uploads(category: str, *, paths: UploadStatePaths | None = None) -> None:
    resolved_paths = paths or build_upload_state_paths()
    category_dir = resolved_paths.category_dir(category)
    if category_dir.exists():
        shutil.rmtree(category_dir)
    manifest = load_upload_manifest(paths=resolved_paths)
    manifest["files"][category] = []
    _write_json(resolved_paths.upload_manifest_path, manifest)


def clear_all_uploads(*, paths: UploadStatePaths | None = None) -> None:
    resolved_paths = paths or build_upload_state_paths()
    for category in UPLOAD_CATEGORIES:
        clear_category_uploads(category, paths=resolved_paths)


def reset_uploader_session_state(*, session_keys: tuple[str, ...] = ()) -> None:
    for key in tuple(UPLOADER_SESSION_KEYS.values()) + tuple(session_keys):
        if key in st.session_state:
            del st.session_state[key]


def render_sidebar_memory_panel(*, paths: UploadStatePaths | None = None) -> None:
    resolved_paths = paths or build_upload_state_paths()
    st.caption("업로드 기억")
    action_left, action_right = st.columns(2)
    if action_left.button("기억 초기화", use_container_width=True, key="upload_memory_reset"):
        reset_uploader_session_state()
        st.rerun()
    if action_right.button("전체삭제", use_container_width=True, key="upload_memory_delete"):
        clear_all_uploads(paths=resolved_paths)
        reset_uploader_session_state()
        st.rerun()

    rows = category_summary_rows(paths=resolved_paths)
    if any(row["count"] for row in rows):
        summary_frame = pd.DataFrame(rows)[["label", "count", "files"]]
        st.dataframe(summary_frame, hide_index=True, use_container_width=True)
    else:
        st.caption("기억된 업로드가 없습니다.")


def render_workspace_panel(*, paths: UploadStatePaths | None = None) -> None:
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


__all__ = [name for name in globals() if not name.startswith("_")]
