from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from .ui_upload_state import CATEGORY_LABELS
from .ui_upload_state import UPLOAD_CATEGORIES
from .ui_upload_state import UPLOAD_MEMORY_LABEL_BY_CATEGORY
from .ui_upload_state import UploadStatePaths
from .ui_upload_state import build_upload_memory_group_summary
from .ui_upload_state import build_upload_memory_items
from .ui_upload_state import build_upload_state_paths
from .ui_upload_state import delete_upload_memory_group
from .ui_upload_state import delete_upload_memory_items as delete_upload_memory_items_by_id
from .ui_upload_state import persist_uploaded_files
from .ui_upload_state import reset_uploader_session_state


GROUP_UPLOAD_LABELS = {
    "continuous": "Add continuous-cycle files",
    "transient": "Add finite-cycle files",
    "validation": "Add actual-drive validation files",
    "lcr": "Add LCR files",
}


def build_upload_memory_summary_rows(
    *,
    paths: UploadStatePaths | None = None,
    preview_limit: int = 4,
) -> list[dict[str, Any]]:
    resolved_paths = paths or build_upload_state_paths()
    items = build_upload_memory_items(paths=resolved_paths)
    summary = build_upload_memory_group_summary(items)
    rows: list[dict[str, Any]] = []
    for category in UPLOAD_CATEGORIES:
        memory_label = UPLOAD_MEMORY_LABEL_BY_CATEGORY.get(category, category)
        group = summary.get(memory_label, {"count": 0, "files": [], "items": []})
        files = [str(name) for name in group.get("files", [])]
        preview_names = files[:preview_limit]
        remaining = max(len(files) - len(preview_names), 0)
        preview = ", ".join(preview_names)
        if remaining:
            preview = f"{preview} ... +{remaining}" if preview else f"+{remaining}"
        rows.append(
            {
                "category": category,
                "label": CATEGORY_LABELS.get(category, category),
                "memory_label": memory_label,
                "count": int(group.get("count") or 0),
                "files preview": preview,
                "item_ids": [str(item.get("upload_item_id")) for item in group.get("items", [])],
            }
        )
    return rows


def build_upload_memory_group_records(
    category: str,
    *,
    paths: UploadStatePaths | None = None,
) -> list[dict[str, Any]]:
    resolved_paths = paths or build_upload_state_paths()
    records: list[dict[str, Any]] = []
    for item in build_upload_memory_items(paths=resolved_paths):
        if item.get("category") != category:
            continue
        records.append(
            {
                "select": False,
                "category": category,
                "upload_item_id": str(item.get("upload_item_id") or ""),
                "original filename": str(item.get("original_filename") or ""),
                "stored filename": str(item.get("stored_filename") or ""),
                "stored path": str(item.get("stored_path") or ""),
                "internal id": str(item.get("upload_item_id") or ""),
                "file size": int(item.get("file_size") or 0),
                "parsed waveform": item.get("waveform_family") or "",
                "parsed freq_hz": item.get("freq_hz"),
                "parsed cycle_count": item.get("cycle_count"),
                "row count": item.get("row_count"),
                "parse status": str(item.get("parse_status") or "unknown"),
                "validation status": str(item.get("validation_status") or "unknown"),
                "duplicate_of": item.get("duplicate_of"),
            }
        )
    return records


def find_duplicate_upload_names(
    category: str,
    upload_names: list[str],
    *,
    paths: UploadStatePaths | None = None,
) -> list[str]:
    resolved_paths = paths or build_upload_state_paths()
    existing = {
        str(item.get("original_filename") or "")
        for item in build_upload_memory_items(paths=resolved_paths)
        if item.get("category") == category
    }
    return [Path(str(name)).name for name in upload_names if Path(str(name)).name in existing]


def delete_upload_memory_items(
    category: str,
    item_ids_or_stored_names: list[str],
    *,
    paths: UploadStatePaths | None = None,
) -> list[str]:
    resolved_paths = paths or build_upload_state_paths()
    requested = {str(value) for value in item_ids_or_stored_names if str(value)}
    resolved_ids: list[str] = []
    for item in build_upload_memory_items(paths=resolved_paths):
        if item.get("category") != category:
            continue
        item_id = str(item.get("upload_item_id") or "")
        stored_name = str(item.get("stored_filename") or "")
        if item_id in requested or stored_name in requested:
            resolved_ids.append(item_id)
    result = delete_upload_memory_items_by_id(resolved_ids, paths=resolved_paths)
    return [str(item_id) for item_id in result.get("deleted_item_ids", [])]


def render_upload_memory_management(*, paths: UploadStatePaths | None = None) -> None:
    resolved_paths = paths or build_upload_state_paths()
    st.caption("Upload memory")
    st.caption("Manage cached uploads per file. Selection values are stable scalar upload_item_id values.")
    _render_danger_actions(resolved_paths)
    _render_summary(resolved_paths)
    for category in UPLOAD_CATEGORIES:
        _render_group(category, resolved_paths)


def _render_danger_actions(paths: UploadStatePaths) -> None:
    with st.expander("Global upload memory actions", expanded=False):
        st.caption("Reset only clears Streamlit uploader state. Delete all removes upload cache manifest entries.")
        confirm_all = st.checkbox("Confirm delete all upload memory", key="upload_memory_confirm_delete_all")
        action_left, action_right = st.columns(2)
        if action_left.button("Reset uploader state", use_container_width=True, key="upload_memory_reset"):
            reset_uploader_session_state()
            st.rerun()
        if action_right.button("Delete all", use_container_width=True, key="upload_memory_delete", disabled=not confirm_all):
            for category in UPLOAD_CATEGORIES:
                delete_upload_memory_group(category, paths=paths)
            reset_uploader_session_state()
            st.rerun()


def _render_summary(paths: UploadStatePaths) -> None:
    rows = build_upload_memory_summary_rows(paths=paths)
    if any(row["count"] for row in rows):
        summary = pd.DataFrame(rows)[["label", "count", "files preview"]]
        st.dataframe(summary, hide_index=True, use_container_width=True)
        with st.expander("Upload memory item ids", expanded=False):
            st.dataframe(pd.DataFrame(rows)[["label", "count", "item_ids"]], hide_index=True, use_container_width=True)
    else:
        st.caption("No cached uploads.")


def _render_group(category: str, paths: UploadStatePaths) -> None:
    label = CATEGORY_LABELS.get(category, category)
    with st.expander(label, expanded=False):
        uploaded_files = st.file_uploader(
            GROUP_UPLOAD_LABELS.get(category, f"Add {label} files"),
            accept_multiple_files=True,
            key=f"upload_memory_add_{category}",
        )
        if uploaded_files:
            names = [Path(str(getattr(uploaded, "name", "") or "")).name for uploaded in uploaded_files]
            duplicates = find_duplicate_upload_names(category, names, paths=paths)
            if duplicates:
                st.warning(
                    "Duplicate filename detected. It will be kept as a separate cache item: "
                    + ", ".join(duplicates)
                )
            persist_uploaded_files(category, uploaded_files, paths=paths)
            st.success(f"Added {len(uploaded_files)} file(s).")
            st.rerun()

        rows = build_upload_memory_group_records(category, paths=paths)
        if not rows:
            st.info(f"No files in {label}.")
            return
        selected_ids: list[str] = []
        for row in rows:
            row_id = str(row["upload_item_id"])
            cols = st.columns([0.12, 0.52, 0.36])
            selected = cols[0].checkbox(
                "Select file",
                key=f"upload_memory_select_{category}_{_key_part(row_id)}",
                label_visibility="collapsed",
            )
            if selected:
                selected_ids.append(row_id)
            cols[1].write(f"**{row['original filename']}**")
            cols[1].caption(f"upload_item_id: {row_id}")
            cols[2].caption(
                f"size={row['file size']} bytes | waveform={row['parsed waveform'] or 'n/a'} | "
                f"freq={_format_optional(row['parsed freq_hz'])} | cycle={_format_optional(row['parsed cycle_count'])} | "
                f"rows={row['row count']} | parse={row['parse status']} | validation={row['validation status']}"
            )
            confirm_single = cols[2].checkbox("Confirm delete", key=f"upload_memory_confirm_one_{category}_{_key_part(row_id)}")
            if cols[2].button("Delete", key=f"upload_memory_delete_one_{category}_{_key_part(row_id)}", disabled=not confirm_single):
                delete_upload_memory_items_by_id([row_id], paths=paths)
                st.rerun()

        confirm_selected = st.checkbox("Confirm selected delete", key=f"upload_memory_confirm_selected_{category}")
        if st.button("Delete selected files", key=f"upload_memory_delete_selected_{category}", disabled=not (confirm_selected and selected_ids)):
            delete_upload_memory_items_by_id(selected_ids, paths=paths)
            st.rerun()
        confirm_group = st.checkbox("Confirm group delete", key=f"upload_memory_confirm_group_{category}")
        if st.button(f"Delete all {label} files", key=f"upload_memory_delete_group_{category}", disabled=not confirm_group):
            delete_upload_memory_group(category, paths=paths)
            st.rerun()


def _format_optional(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{numeric:g}" if pd.notna(numeric) else "n/a"


def _key_part(value: object) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() else "_" for ch in text)
