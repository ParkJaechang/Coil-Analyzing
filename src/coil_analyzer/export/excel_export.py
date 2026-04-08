"""Excel and HTML export helpers."""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any

import pandas as pd
import plotly.graph_objects as go


def build_excel_bytes(sheet_map: dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, df in sheet_map.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    buffer.seek(0)
    return buffer.read()


def build_export_bundle(
    workbook_sheets: dict[str, pd.DataFrame],
    figures: dict[str, go.Figure],
    settings_payload: dict[str, Any],
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("coil_analysis_export.xlsx", build_excel_bytes(workbook_sheets))
        bundle.writestr("analysis_settings.json", json.dumps(settings_payload, ensure_ascii=False, indent=2))
        for figure_name, figure in figures.items():
            bundle.writestr(f"{figure_name}.html", figure.to_html(include_plotlyjs="cdn"))
    buffer.seek(0)
    return buffer.read()
