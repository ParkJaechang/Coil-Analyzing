from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(slots=True)
class FieldSpec:
    """Canonical normalized field definition used by the parser and UI."""

    key: str
    label_ko: str
    aliases: tuple[str, ...]
    unit: str | None = None
    required: bool = False
    description: str = ""


@dataclass(slots=True)
class SchemaConfig:
    """Fixed-format schema configuration loaded from YAML."""

    comment_prefix: str = "#"
    header_search_rows: int = 25
    default_expected_cycles: int = 10
    target_current_mode: str = "auto"
    preferred_sheet_names: tuple[str, ...] = ("data", "raw", "sheet1")
    metadata_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    column_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    field_specs: dict[str, FieldSpec] = field(default_factory=dict)
    default_main_field_axis: str = "bz_mT"
    default_current_axis: str = "i_sum"


@dataclass(slots=True)
class SheetPreview:
    """Structural preview for a single CSV pseudo-sheet or workbook sheet."""

    sheet_name: str
    row_count: int
    column_count: int
    columns: list[str]
    header_row_index: int
    metadata: dict[str, str]
    preview_rows: list[dict[str, Any]]
    recommended_mapping: dict[str, str | None]
    warnings: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FilePreview:
    """Preview summary for an uploaded file."""

    file_name: str
    file_type: str
    sheet_previews: list[SheetPreview]
    warnings: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ParsedMeasurement:
    """Normalized single-test dataset built from one file/sheet pair."""

    source_file: str
    file_type: str
    sheet_name: str
    structure_preview: SheetPreview
    metadata: dict[str, Any]
    mapping: dict[str, str | None]
    raw_frame: pd.DataFrame
    normalized_frame: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ChannelLag:
    """Estimated inter-channel lag relative to a reference signal."""

    channel: str
    lag_seconds: float
    lag_samples: int
    correlation: float


@dataclass(slots=True)
class PreprocessConfig:
    """User-configurable preprocessing switches for corrected data."""

    baseline_seconds: float = 0.0
    smoothing_method: str = "none"
    smoothing_window: int = 11
    savgol_polyorder: int = 2
    alignment_reference: str | None = None
    alignment_targets: tuple[str, ...] = ()
    apply_alignment: bool = False
    outlier_zscore_threshold: float = 0.0
    sign_flips: dict[str, int] = field(default_factory=dict)
    custom_current_alpha: float = 1.0
    custom_current_beta: float = 1.0
    projection_vector: tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass(slots=True)
class PreprocessResult:
    """Corrected dataset plus transparent preprocessing bookkeeping."""

    corrected_frame: pd.DataFrame
    offsets: dict[str, float]
    lags: list[ChannelLag]
    warnings: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CycleDetectionConfig:
    """Cycle detection settings."""

    reference_channel: str = "i_sum"
    expected_cycles: int = 10
    manual_start_s: float | None = None
    manual_period_s: float | None = None
    use_corrected_data: bool = True


@dataclass(slots=True)
class CycleBoundary:
    """Single detected cycle boundary interval."""

    cycle_index: int
    start_index: int
    end_index: int
    start_s: float
    end_s: float


@dataclass(slots=True)
class CycleDetectionResult:
    """Detected cycle boundaries and annotated frame."""

    annotated_frame: pd.DataFrame
    boundaries: list[CycleBoundary]
    estimated_period_s: float | None
    estimated_frequency_hz: float | None
    reference_channel: str
    warnings: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DatasetAnalysis:
    """Complete per-test analysis bundle used by the UI."""

    parsed: ParsedMeasurement
    preprocess: PreprocessResult
    cycle_detection: CycleDetectionResult
    per_cycle_summary: pd.DataFrame
    per_test_summary: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExportArtifacts:
    """Generated export paths."""

    root_dir: Path
    normalized_data_path: Path
    per_test_summary_path: Path
    per_cycle_summary_path: Path
    report_path: Path
    config_snapshot_path: Path
    plots_dir: Path
    excel_formatting_report_path: Path | None = None
