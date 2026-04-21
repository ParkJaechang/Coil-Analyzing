from __future__ import annotations

import streamlit as st

from .dataset_library import (
    build_dataset_payloads,
    build_dataset_manifest,
    get_dataset_manifest_path,
    get_default_settings_path,
    list_manifest_entries,
    load_dataset_library_settings,
    load_dataset_manifest,
    save_dataset_library_settings,
)


def render_dataset_library_panel() -> None:
    settings = load_dataset_library_settings()
    session_key = "dataset_library_root"
    if session_key not in st.session_state:
        st.session_state[session_key] = str(settings.get("dataset_root") or "")

    with st.expander("Dataset Library", expanded=False):
        dataset_root_value = str(
            st.text_input(
                "Dataset root path",
                key=session_key,
                placeholder="D:/OneDrive/CoilDatasets",
                help="Save a shared sync-folder path such as OneDrive, Google Drive, Dropbox, NAS, or external SSD.",
            )
            or ""
        ).strip()
        current_manifest = None
        action_left, action_right = st.columns(2)
        if action_left.button("Save Root", use_container_width=True, key="dataset_library_save"):
            save_dataset_library_settings({"dataset_root": dataset_root_value})
            st.success("Dataset root saved.")
        if action_right.button("Manifest Refresh", use_container_width=True, key="dataset_library_refresh"):
            if not dataset_root_value:
                st.warning("Enter a dataset root path first.")
            else:
                try:
                    with st.spinner("Building dataset manifest..."):
                        current_manifest = build_dataset_manifest(dataset_root_value)
                    save_dataset_library_settings({"dataset_root": dataset_root_value})
                    st.success("Dataset manifest refreshed.")
                except (FileNotFoundError, NotADirectoryError) as exc:
                    st.error(str(exc))

        saved_settings_path = get_default_settings_path()
        st.caption(f"settings: {saved_settings_path}")
        if not dataset_root_value:
            st.caption("No dataset root is saved.")
            return

        manifest_path = get_dataset_manifest_path(dataset_root_value)
        if current_manifest is None:
            current_manifest = load_dataset_manifest(dataset_root_value)

        count_left, count_right = st.columns(2)
        count_left.metric("Registered Files", int(current_manifest.get("file_count") or 0))
        count_right.metric("Continuous", int(current_manifest.get("counts", {}).get("continuous") or 0))
        count_left.metric("Finite Cycle", int(current_manifest.get("counts", {}).get("finite_cycle") or 0))
        count_right.metric("Unknown", int(current_manifest.get("counts", {}).get("unknown") or 0))
        st.caption(f"root: {dataset_root_value}")
        if manifest_path.exists():
            st.caption(f"manifest: {manifest_path}")
        else:
            st.caption("Manifest has not been generated yet.")


def render_dataset_library_file_selector(
    *,
    dataset_mode: str,
    key_prefix: str,
) -> list[tuple[str, bytes]]:
    settings = load_dataset_library_settings()
    dataset_root = str(settings.get("dataset_root") or "").strip()
    mode_label = {
        "continuous": "Continuous",
        "finite_cycle": "Finite-cycle",
        "unknown": "Unknown",
    }.get(dataset_mode, dataset_mode)

    with st.expander(f"Dataset Library: {mode_label}", expanded=False):
        if not dataset_root:
            st.caption("Save a dataset root first.")
            return []

        manifest_path = get_dataset_manifest_path(dataset_root)
        if not manifest_path.exists():
            st.caption("Manifest Refresh first.")
            return []

        entries = list_manifest_entries(dataset_root, dataset_mode=dataset_mode)
        if not entries:
            st.caption(f"No {mode_label.lower()} files are available in the manifest.")
            return []

        option_lookup = {
            str(entry["path"]): entry
            for entry in entries
        }
        selected_paths = st.multiselect(
            f"{mode_label} library files",
            options=list(option_lookup.keys()),
            format_func=lambda path: _format_dataset_entry_option(option_lookup[path]),
            key=f"{key_prefix}_dataset_library_paths",
        )
        st.caption(f"{len(selected_paths)} selected / {len(option_lookup)} available")

        if not selected_paths:
            return []

        try:
            return build_dataset_payloads(dataset_root, selected_paths)
        except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
            st.error(str(exc))
            return []


def _format_dataset_entry_option(entry: dict[str, object]) -> str:
    path = str(entry.get("path") or "")
    size_bytes = int(entry.get("size_bytes") or 0)
    return f"{path} ({size_bytes} bytes)"


__all__ = ["render_dataset_library_file_selector", "render_dataset_library_panel"]
