from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import streamlit as st

from .dataset_access_preflight import build_dataset_access_preflight
from .dataset_library import get_dataset_manifest_path, load_dataset_library_settings, load_dataset_manifest
from .ui_run_readiness_exports import render_run_readiness_export_panel


def build_run_readiness_summary(
    *,
    dataset_root: str | None,
    dataset_root_exists: bool,
    manifest_exists: bool,
    manifest_counts: Mapping[str, Any] | None = None,
    selected_library_counts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_root = str(dataset_root or "").strip()
    counts = {
        "continuous": int((manifest_counts or {}).get("continuous") or 0),
        "finite_cycle": int((manifest_counts or {}).get("finite_cycle") or 0),
        "unknown": int((manifest_counts or {}).get("unknown") or 0),
    }
    selected_counts = {
        "continuous": int((selected_library_counts or {}).get("continuous") or 0),
        "finite_cycle": int((selected_library_counts or {}).get("finite_cycle") or 0),
    }
    selected_counts["total"] = selected_counts["continuous"] + selected_counts["finite_cycle"]

    return {
        "dataset_root": normalized_root,
        "dataset_root_saved": bool(normalized_root),
        "dataset_root_exists": bool(dataset_root_exists),
        "manifest_exists": bool(manifest_exists),
        "manifest_counts": counts,
        "selected_library_counts": selected_counts,
    }


def render_run_readiness_section() -> None:
    settings = load_dataset_library_settings()
    dataset_root = str(settings.get("dataset_root") or "").strip()
    dataset_root_path = Path(dataset_root).expanduser() if dataset_root else None
    dataset_root_exists = bool(dataset_root_path and dataset_root_path.is_dir())

    manifest_path = get_dataset_manifest_path(dataset_root_path) if dataset_root_path else None
    manifest_exists = bool(manifest_path and manifest_path.is_file())
    manifest_counts = None
    if manifest_exists and dataset_root_path is not None:
        manifest_counts = load_dataset_manifest(dataset_root_path).get("counts", {})

    summary = build_run_readiness_summary(
        dataset_root=dataset_root,
        dataset_root_exists=dataset_root_exists,
        manifest_exists=manifest_exists,
        manifest_counts=manifest_counts,
        selected_library_counts={
            "continuous": len(_selected_paths("continuous")),
            "finite_cycle": len(_selected_paths("transient")),
        },
    )
    access_preflight = build_dataset_access_preflight(
        dataset_root=dataset_root,
        selected_paths_by_mode={
            "continuous": _selected_paths("continuous"),
            "finite_cycle": _selected_paths("transient"),
        },
        manifest_entries=(load_dataset_manifest(dataset_root_path).get("files", []) if manifest_exists and dataset_root_path is not None else []),
    )

    st.markdown("#### Run Readiness")
    st.caption("Quick preflight for local testability before manual browser or hardware checks.")

    status_left, status_right, status_manifest = st.columns(3)
    status_left.metric("Saved Dataset Root", "Yes" if summary["dataset_root_saved"] else "No")
    status_right.metric("Dataset Directory", "Ready" if summary["dataset_root_exists"] else "Missing")
    status_manifest.metric("Manifest", "Ready" if summary["manifest_exists"] else "Missing")

    count1, count2, count3 = st.columns(3)
    count1.metric("Continuous Files", summary["manifest_counts"]["continuous"])
    count2.metric("Finite-Cycle Files", summary["manifest_counts"]["finite_cycle"])
    count3.metric("Unknown Files", summary["manifest_counts"]["unknown"])

    selected_left, selected_right, selected_total = st.columns(3)
    selected_left.metric("Selected Continuous", summary["selected_library_counts"]["continuous"])
    selected_right.metric("Selected Finite-Cycle", summary["selected_library_counts"]["finite_cycle"])
    selected_total.metric("Selected Library Files", summary["selected_library_counts"]["total"])

    access_left, access_right, access_total = st.columns(3)
    access_left.metric("Accessible Continuous", access_preflight["selected_by_mode"]["continuous"]["ok_count"])
    access_right.metric("Accessible Finite-Cycle", access_preflight["selected_by_mode"]["finite_cycle"]["ok_count"])
    access_total.metric("Selected Missing / Unavailable", access_preflight["selected"]["unavailable_count"])

    if summary["dataset_root_saved"]:
        st.caption(f"dataset_root: {summary['dataset_root']}")
    else:
        st.info("No dataset root is saved yet. Dataset Library-backed inputs are not ready.")

    if manifest_path is not None:
        st.caption(f"manifest: {manifest_path}")

    if summary["dataset_root_saved"] and not summary["dataset_root_exists"]:
        st.warning("The saved dataset root does not exist on this PC. Check the sync-drive or external storage path.")
    elif summary["dataset_root_exists"] and not summary["manifest_exists"]:
        st.warning("Dataset root exists but no manifest was found. Open `Dataset Library` and run `Manifest Refresh`.")
    elif summary["manifest_exists"]:
        st.success("Dataset Library manifest is available for local test inputs.")

    manifest_summary = access_preflight["manifest"]
    st.caption(
        "Manifest entry access: "
        f"{manifest_summary['ok_count']} ok / "
        f"{manifest_summary['missing_count']} missing / "
        f"{manifest_summary['unreadable_count']} unreadable / "
        f"{manifest_summary['blocked_count']} blocked"
    )

    if summary["dataset_root_exists"] and manifest_summary["missing_count"] > 0:
        st.warning("Dataset root is present on this PC, but some manifest-backed files are missing or no longer synced.")
    if access_preflight["selected"]["unavailable_count"] > 0:
        st.warning("Some selected Dataset Library files are not accessible on this PC.")
        for sample in access_preflight["selected"]["problem_samples"]:
            st.write(f"- {sample['path']} ({sample['status']})")
    elif access_preflight["selected"]["checked_count"] > 0:
        st.success("Selected Dataset Library files are accessible on this PC.")

    render_run_readiness_export_panel(
        summary=summary,
        access_preflight=access_preflight,
    )

    st.write("- CI coverage in this repo is unit/smoke level only.")
    st.write("- Browser clickthrough, Streamlit interaction, and hardware-linked checks still need manual confirmation.")


def _selected_paths(key_prefix: str) -> list[str]:
    session_key = f"{key_prefix}_dataset_library_paths"
    selected = st.session_state.get(session_key, [])
    if not isinstance(selected, list):
        return []
    return [str(path) for path in selected if str(path).strip()]


__all__ = ["build_run_readiness_summary", "render_run_readiness_section"]
