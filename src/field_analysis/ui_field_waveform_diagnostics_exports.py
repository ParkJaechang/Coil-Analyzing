from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Mapping
from typing import Any

import pandas as pd
import streamlit as st


DIAGNOSTICS_EXPORT_TABLES = (
    "target_metric_candidates",
    "waveform_counts",
    "frequency_counts",
    "continuous_support",
    "finite_support",
    "continuous_test_details",
    "transient_test_details",
)


def build_field_waveform_diagnostics_export_payloads(
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    summary_payload = {
        "summary": dict(diagnostics.get("summary", {})),
        "notes": list(diagnostics.get("notes", [])),
        "available_tables": [
            table_name
            for table_name in DIAGNOSTICS_EXPORT_TABLES
            if isinstance(diagnostics.get(table_name), pd.DataFrame)
        ],
    }
    table_exports = {
        table_name: diagnostics[table_name]
        for table_name in DIAGNOSTICS_EXPORT_TABLES
        if isinstance(diagnostics.get(table_name), pd.DataFrame)
    }
    return {
        "summary_json": summary_payload,
        "tables": table_exports,
    }


def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    export_frame = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    return export_frame.to_csv(index=False).encode("utf-8-sig")


def payload_to_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def build_field_waveform_diagnostics_artifact_map(
    diagnostics: Mapping[str, Any],
    *,
    file_stem: str = "field_waveform_diagnostics",
) -> dict[str, bytes]:
    export_payloads = build_field_waveform_diagnostics_export_payloads(diagnostics)
    summary_payload = export_payloads["summary_json"]
    table_exports = export_payloads["tables"]

    artifacts = {
        f"{file_stem}_summary.json": payload_to_json_bytes(summary_payload),
    }
    for table_name, frame in table_exports.items():
        artifacts[f"{file_stem}_{table_name}.csv"] = dataframe_to_csv_bytes(frame)
    return artifacts


def build_field_waveform_diagnostics_bundle_zip_bytes(artifacts: Mapping[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_name, content in artifacts.items():
            archive.writestr(file_name, content)
    return buffer.getvalue()


def render_field_waveform_diagnostics_export_panel(
    diagnostics: Mapping[str, Any],
    *,
    file_stem: str = "field_waveform_diagnostics",
    key_prefix: str = "field_waveform_diagnostics_export",
) -> None:
    export_payloads = build_field_waveform_diagnostics_export_payloads(diagnostics)
    summary_payload = export_payloads["summary_json"]
    table_exports = export_payloads["tables"]
    artifacts = build_field_waveform_diagnostics_artifact_map(diagnostics, file_stem=file_stem)
    bundle_bytes = build_field_waveform_diagnostics_bundle_zip_bytes(artifacts)

    with st.expander("Diagnostics Exports", expanded=False):
        st.caption("Download the current diagnostics snapshot as one summary JSON plus table-level CSV files.")
        st.download_button(
            label="Diagnostics ZIP 다운로드",
            data=bundle_bytes,
            file_name=f"{file_stem}_artifacts.zip",
            mime="application/zip",
            key=f"{key_prefix}_artifact_bundle_zip",
        )

        st.markdown("#### Summary JSON")
        st.code(json.dumps(summary_payload, ensure_ascii=False, indent=2), language="json")
        st.download_button(
            label="Summary JSON 다운로드",
            data=payload_to_json_bytes(summary_payload),
            file_name=f"{file_stem}_summary.json",
            mime="application/json",
            key=f"{key_prefix}_summary_json",
        )

        st.markdown("#### Table CSV Downloads")
        for table_name, frame in table_exports.items():
            row_left, row_right = st.columns([3, 1])
            with row_left:
                st.write(f"- `{table_name}` ({len(frame)} rows)")
            with row_right:
                st.download_button(
                    label=f"{table_name} CSV",
                    data=dataframe_to_csv_bytes(frame),
                    file_name=f"{file_stem}_{table_name}.csv",
                    mime="text/csv",
                    key=f"{key_prefix}_{table_name}_csv",
                )


__all__ = [
    "build_field_waveform_diagnostics_artifact_map",
    "build_field_waveform_diagnostics_bundle_zip_bytes",
    "build_field_waveform_diagnostics_export_payloads",
    "dataframe_to_csv_bytes",
    "payload_to_json_bytes",
    "render_field_waveform_diagnostics_export_panel",
]
