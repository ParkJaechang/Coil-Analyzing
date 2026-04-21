from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Mapping

import pandas as pd
import streamlit as st


def build_run_readiness_report_payload(
    *,
    summary: Mapping[str, Any],
    access_preflight: Mapping[str, Any],
    generated_at: str | None = None,
) -> dict[str, Any]:
    return {
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "readiness_summary": dict(summary),
        "access_preflight": dict(access_preflight),
    }


def build_problem_rows(
    summaries_by_source: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source, check_summary in summaries_by_source.items():
        for check in check_summary.get("checks", []):
            status = str(check.get("status") or "")
            if status == "ok":
                continue
            rows.append(
                {
                    "source": str(source),
                    "path": str(check.get("path") or ""),
                    "status": status,
                    "message": str(check.get("message") or ""),
                }
            )
    return rows


def build_problem_frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["source", "path", "status", "message"])
    return pd.DataFrame(rows, columns=["source", "path", "status", "message"])


def build_problem_csv_bytes(rows: list[dict[str, str]]) -> bytes:
    frame = build_problem_frame(rows)
    return frame.to_csv(index=False).encode("utf-8")


def build_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def render_run_readiness_export_panel(
    *,
    summary: Mapping[str, Any],
    access_preflight: Mapping[str, Any],
) -> None:
    report_payload = build_run_readiness_report_payload(
        summary=summary,
        access_preflight=access_preflight,
    )
    selected_problem_rows = build_problem_rows(
        {
            "selected_continuous": access_preflight.get("selected_by_mode", {}).get("continuous", {}),
            "selected_finite_cycle": access_preflight.get("selected_by_mode", {}).get("finite_cycle", {}),
        }
    )
    manifest_problem_rows = build_problem_rows(
        {
            "manifest": access_preflight.get("manifest", {}),
        }
    )

    st.markdown("#### Run Readiness Report")
    st.caption("Export the current-PC readiness snapshot as JSON/CSV before a manual test run.")

    download_left, download_right = st.columns(2)
    download_left.download_button(
        label="Readiness Summary JSON",
        data=build_json_bytes(report_payload["readiness_summary"]),
        file_name="run_readiness_summary.json",
        mime="application/json",
        use_container_width=True,
        key="run_readiness_summary_json",
    )
    download_right.download_button(
        label="Access Preflight JSON",
        data=build_json_bytes(report_payload["access_preflight"]),
        file_name="run_readiness_access_preflight.json",
        mime="application/json",
        use_container_width=True,
        key="run_readiness_access_preflight_json",
    )

    selected_csv_left, selected_csv_right = st.columns(2)
    selected_csv_left.download_button(
        label="Selected Problem Files CSV",
        data=build_problem_csv_bytes(selected_problem_rows),
        file_name="run_readiness_selected_problem_files.csv",
        mime="text/csv",
        use_container_width=True,
        key="run_readiness_selected_problem_csv",
        disabled=not selected_problem_rows,
    )
    selected_csv_right.download_button(
        label="Manifest Problem Files CSV",
        data=build_problem_csv_bytes(manifest_problem_rows),
        file_name="run_readiness_manifest_problem_files.csv",
        mime="text/csv",
        use_container_width=True,
        key="run_readiness_manifest_problem_csv",
        disabled=not manifest_problem_rows,
    )

    selected_problem_frame = build_problem_frame(selected_problem_rows)
    manifest_problem_frame = build_problem_frame(manifest_problem_rows)

    with st.expander("Selected Inaccessible Files", expanded=False):
        if selected_problem_frame.empty:
            st.caption("No selected file access problems were detected.")
        else:
            st.dataframe(selected_problem_frame, use_container_width=True, hide_index=True)

    with st.expander("Manifest Problem Samples", expanded=False):
        if manifest_problem_frame.empty:
            st.caption("No manifest-backed file access problems were detected.")
        else:
            st.dataframe(manifest_problem_frame, use_container_width=True, hide_index=True)


__all__ = [
    "build_json_bytes",
    "build_problem_csv_bytes",
    "build_problem_frame",
    "build_problem_rows",
    "build_run_readiness_report_payload",
    "render_run_readiness_export_panel",
]
