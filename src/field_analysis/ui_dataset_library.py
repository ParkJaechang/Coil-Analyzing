from __future__ import annotations

import streamlit as st

from .dataset_library import (
    build_dataset_manifest,
    get_dataset_manifest_path,
    get_default_settings_path,
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


__all__ = ["render_dataset_library_panel"]
