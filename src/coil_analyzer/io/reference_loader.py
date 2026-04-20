"""Reference file discovery and loading with graceful fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from coil_analyzer.constants import REFERENCE_FILE_CANDIDATES
from coil_analyzer.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class ReferenceFileStatus:
    name: str
    path: str | None
    exists: bool
    notes: str = ""


def discover_reference_files(root: Path) -> list[ReferenceFileStatus]:
    statuses: list[ReferenceFileStatus] = []
    for candidate in REFERENCE_FILE_CANDIDATES:
        matches = list(root.rglob(candidate))
        if matches:
            statuses.append(
                ReferenceFileStatus(
                    name=candidate,
                    path=str(matches[0]),
                    exists=True,
                    notes="Detected in workspace.",
                )
            )
        else:
            statuses.append(
                ReferenceFileStatus(
                    name=candidate,
                    path=None,
                    exists=False,
                    notes="Not found. The app will continue with fallback behavior.",
                )
            )
    return statuses


def load_reference_workbook(path: Path) -> tuple[list[str], dict[str, pd.DataFrame]]:
    if not path.exists():
        return [], {}
    try:
        workbook = pd.ExcelFile(path)
        sheets = {sheet: pd.read_excel(path, sheet_name=sheet) for sheet in workbook.sheet_names}
        return workbook.sheet_names, sheets
    except Exception as exc:
        LOGGER.warning("Failed to load reference workbook %s: %s", path, exc)
        return [], {}


def infer_reference_columns(df: pd.DataFrame) -> dict[str, str | None]:
    lowered = {str(column).lower(): column for column in df.columns}
    result: dict[str, str | None] = {"frequency": None, "l": None, "r": None, "z": None}
    for lowered_name, original in lowered.items():
        if result["frequency"] is None and "freq" in lowered_name:
            result["frequency"] = original
        if result["l"] is None and lowered_name.startswith("l"):
            result["l"] = original
        if result["r"] is None and lowered_name.startswith("r"):
            result["r"] = original
        if result["z"] is None and "z" in lowered_name:
            result["z"] = original
    return result


def summarize_reference_sheet(df: pd.DataFrame) -> dict[str, Any]:
    columns = infer_reference_columns(df)
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "detected_columns": columns,
    }


def build_reference_impedance_table(
    df: pd.DataFrame,
    frequency_col: str,
    z_col: str | None = None,
    r_col: str | None = None,
    l_col: str | None = None,
    inductance_multiplier: float = 1.0,
) -> pd.DataFrame:
    result = pd.DataFrame({"frequency_hz": pd.to_numeric(df[frequency_col], errors="coerce")})
    if z_col:
        result["lcr_z_ohm"] = pd.to_numeric(df[z_col], errors="coerce")
    elif r_col and l_col:
        r = pd.to_numeric(df[r_col], errors="coerce")
        l = pd.to_numeric(df[l_col], errors="coerce") * inductance_multiplier
        result["lcr_z_ohm"] = np.sqrt(np.square(r) + np.square(2.0 * np.pi * result["frequency_hz"] * l))
    else:
        result["lcr_z_ohm"] = np.nan
    if r_col:
        result["lcr_r_ohm"] = pd.to_numeric(df[r_col], errors="coerce")
    if l_col:
        result["lcr_l_h"] = pd.to_numeric(df[l_col], errors="coerce") * inductance_multiplier
    return result.dropna(subset=["frequency_hz"]).sort_values("frequency_hz").reset_index(drop=True)
