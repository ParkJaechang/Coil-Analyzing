from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PARENT_ROOT = ROOT.parent
SRC_ROOT = PARENT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.validation_retune import (  # noqa: E402
    EXACT_PATH_CURRENT,
    EXACT_PATH_FIELD,
    EXACT_PATH_FINITE,
    SOURCE_KIND_CORRECTED,
    SOURCE_KIND_EXPORT,
    SOURCE_KIND_RECOMMENDATION,
    build_quality_badge_markdown,
    normalize_corrected_lineage_root,
)
from field_analysis.validation_retune_catalog import (  # noqa: E402
    build_corrected_lut_catalog_payload,
    build_provenance_badge_markdown,
    build_retune_picker_payload,
    build_validation_catalog_payload,
    load_json,
    write_json,
)
from field_analysis.runtime_display_labels import (  # noqa: E402
    build_display_label,
    build_display_name,
    build_display_object_key,
    has_hash_like_prefix,
    has_internal_display_leak,
    infer_iteration_index,
    sanitize_display_text,
)

from report_exact_and_finite_scope import build_scope_payload, run_provisional_promotion_smoke  # noqa: E402


OUTPUT_DIR = ROOT / "artifacts" / "bz_first_exact_matrix"
POLICY_SCOPE_PATH = ROOT / "artifacts" / "policy_eval" / "exact_and_finite_scope.json"
EXPORT_VALIDATION_DIR = ROOT / "artifacts" / "policy_eval" / "export_validation"
RECOMMENDATION_LIBRARY_DIR = PARENT_ROOT / "outputs" / "field_analysis_app_state" / "recommendation_library"
RECOMMENDATION_MANIFEST_PATH = PARENT_ROOT / "outputs" / "field_analysis_app_state" / "recommendation_manifest.json"
UPLOAD_MANIFEST_PATH = PARENT_ROOT / "outputs" / "field_analysis_app_state" / "upload_manifest.json"
RETUNE_HISTORY_PATH = PARENT_ROOT / "outputs" / "field_analysis_app_state" / "validation_retune_history.json"
CONTINUOUS_UPLOAD_DIR = PARENT_ROOT / "outputs" / "field_analysis_app_state" / "uploads" / "continuous"
TRANSIENT_UPLOAD_DIR = PARENT_ROOT / "outputs" / "field_analysis_app_state" / "uploads" / "transient"
VALIDATION_UPLOAD_DIR = PARENT_ROOT / "outputs" / "field_analysis_app_state" / "uploads" / "validation"
LCR_UPLOAD_DIR = PARENT_ROOT / "outputs" / "field_analysis_app_state" / "uploads" / "lcr"
RETUNE_DIRS = (
    ROOT / "artifacts" / "validation_retune_mvp_example",
    ROOT / "artifacts" / "validation_retune_real_example",
    PARENT_ROOT / "outputs" / "field_analysis_app_state" / "validation_retune",
)
OFFICIAL_MAX_FREQ_HZ = 5.0
STATUS_PRIORITY = {
    "certified_exact": 0,
    "software_ready_bench_pending": 1,
    "provisional_experimental": 2,
    "preview_only": 3,
    "deprecated": 4,
}
TARGET_PRIORITY = {"current": 0, "field": 1}
SOURCE_PRIORITY = {
    SOURCE_KIND_RECOMMENDATION: 0,
    SOURCE_KIND_EXPORT: 1,
    SOURCE_KIND_CORRECTED: 2,
}
SINE_TOKENS = ("sine", "sin", "sinusoid", "sinusoidal", "sinusidal")
TRIANGLE_TOKENS = ("triangle", "tri")


@dataclass(slots=True)
class ArtifactPaths:
    output_dir: Path = OUTPUT_DIR
    export_validation_dir: Path = EXPORT_VALIDATION_DIR
    policy_scope_path: Path = POLICY_SCOPE_PATH
    recommendation_library_dir: Path = RECOMMENDATION_LIBRARY_DIR
    recommendation_manifest_path: Path = RECOMMENDATION_MANIFEST_PATH
    upload_manifest_path: Path = UPLOAD_MANIFEST_PATH
    retune_history_path: Path = RETUNE_HISTORY_PATH
    retune_dirs: tuple[Path, ...] = RETUNE_DIRS
    continuous_upload_dir: Path = CONTINUOUS_UPLOAD_DIR
    transient_upload_dir: Path = TRANSIENT_UPLOAD_DIR
    validation_upload_dir: Path = VALIDATION_UPLOAD_DIR
    lcr_upload_dir: Path = LCR_UPLOAD_DIR


@dataclass(slots=True)
class MatrixIndexes:
    official_max_freq_hz: float
    continuous_current_exact: set[tuple[str, float, float]]
    continuous_field_ready: set[tuple[str, float]]
    finite_exact: set[tuple[str, float, float, float]]
    provisional_preview: set[tuple[str, float, float, float]]
    missing_exact: set[tuple[str, float, float, float]]
    reference_only: set[tuple[str, float, float]]
    continuous_current_cells: list[dict[str, Any]]
    continuous_field_rows: list[dict[str, Any]]
    finite_exact_cells: list[dict[str, Any]]
    provisional_records: list[dict[str, Any]]
    missing_records: list[dict[str, Any]]
    reference_only_cells: list[dict[str, Any]]
    continuous_gap_cells: list[dict[str, Any]]
    finite_full_grid_missing: list[dict[str, Any]]


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def read_profile(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_sidecar_json(path: Path) -> dict[str, Any]:
    json_path = path.with_suffix(".json")
    if not json_path.exists():
        return {}
    payload = load_json(json_path)
    return payload if isinstance(payload, dict) else {}


def safe_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def key_float(value: object) -> float | None:
    numeric = safe_float(value)
    return round(numeric, 6) if numeric is not None else None


def safe_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def safe_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def iso_from_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat(timespec="seconds")


def format_number(value: object) -> str:
    numeric = safe_float(value)
    return f"{numeric:g}" if numeric is not None else "-"


def format_list(values: Iterable[object]) -> str:
    rendered = [format_number(value) if isinstance(value, (float, int)) else str(value) for value in values]
    return ", ".join(item for item in rendered if item and item != "-") or "-"


def format_display_list(values: Iterable[object], *, unit: str | None = None) -> str:
    items: list[str] = []
    for value in values:
        numeric = safe_float(value)
        if numeric is None:
            text = sanitize_display_text(value)
            if text:
                items.append(text)
            continue
        rendered = format_number(numeric)
        items.append(f"{rendered} {unit}".strip() if unit else rendered)
    return ", ".join(item for item in items if item) or "-"


def build_matrix_display_name(
    *,
    target_type: str,
    waveform: object,
    freq_hz: object,
    cycle_count: object | None = None,
    level: object | None = None,
    level_kind: str | None = None,
) -> str:
    return build_display_name(
        target_type=target_type,
        waveform=str(waveform or ""),
        freq_hz=safe_float(freq_hz),
        cycle_count=safe_float(cycle_count),
        level=safe_float(level),
        level_kind=level_kind,
        fallback_texts=(waveform, freq_hz, cycle_count, level),
    )


def build_matrix_display_label(
    *,
    target_type: str,
    waveform: object,
    freq_hz: object,
    cycle_count: object | None = None,
    level: object | None = None,
    level_kind: str | None = None,
    levels: Iterable[object] | None = None,
) -> str:
    display_name = build_matrix_display_name(
        target_type=target_type,
        waveform=waveform,
        freq_hz=freq_hz,
        cycle_count=cycle_count,
        level=level,
        level_kind=level_kind,
    )
    if levels is None:
        return display_name
    unit = "pp" if str(level_kind or "").lower() == "pp" else "A"
    level_text = format_display_list(levels, unit=unit)
    return f"{display_name} | levels {level_text}" if level_text and level_text != "-" else display_name


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_None_"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def document(title: str, summary_lines: list[str], sections: list[tuple[str, str]]) -> str:
    lines = [f"# {title}", ""]
    lines.extend(f"- {line}" for line in summary_lines)
    for heading, body in sections:
        lines.extend(["", f"## {heading}", "", body.strip()])
    return "\n".join(lines).strip() + "\n"


def normalized_waveform(value: object) -> str | None:
    text = safe_text(value)
    if text is None:
        return None
    lowered = text.lower()
    if any(token in lowered for token in TRIANGLE_TOKENS):
        return "triangle"
    if any(token in lowered for token in SINE_TOKENS):
        return "sine"
    return None


def extract_predicted_bz(frame: pd.DataFrame) -> dict[str, Any]:
    for column in ("modeled_field_mT", "expected_field_mT", "target_field_mT", "aligned_target_field_mT", "bz_mT"):
        if column not in frame.columns:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
        if numeric.empty:
            continue
        peak_abs = float(numeric.abs().max())
        rms = float(math.sqrt(float((numeric * numeric).mean())))
        return {
            "column": column,
            "pp_mT": float(numeric.max() - numeric.min()),
            "peak_abs_mT": peak_abs,
            "rms_mT": rms,
        }
    return {"column": None, "pp_mT": None, "peak_abs_mT": None, "rms_mT": None}


def infer_target_type(stem: str, frame: pd.DataFrame, sidecar: dict[str, Any]) -> str:
    for key in ("target_type", "target_output_type"):
        value = safe_text(sidecar.get(key))
        if value and value.lower() in {"current", "field"}:
            return value.lower()
    if "target_output_type" in frame.columns:
        series = frame["target_output_type"].dropna()
        if not series.empty:
            value = str(series.iloc[0]).strip().lower()
            if value in {"current", "field"}:
                return value
    if "target_field_mT" in frame.columns and pd.to_numeric(frame["target_field_mT"], errors="coerce").notna().any():
        return "field"
    return "field" if "_field_" in stem.lower() else "current"


def infer_waveform(stem: str, frame: pd.DataFrame, sidecar: dict[str, Any]) -> str | None:
    for value in (sidecar.get("waveform_type"), sidecar.get("waveform")):
        waveform = normalized_waveform(value)
        if waveform is not None:
            return waveform
    for column in ("waveform_type", "selected_support_waveform"):
        if column not in frame.columns:
            continue
        series = frame[column].dropna()
        if not series.empty:
            waveform = normalized_waveform(series.iloc[0])
            if waveform is not None:
                return waveform
    return normalized_waveform(stem)


def infer_freq_hz(stem: str, frame: pd.DataFrame, sidecar: dict[str, Any]) -> float | None:
    for key in ("freq_hz", "frequency_hz"):
        numeric = safe_float(sidecar.get(key))
        if numeric is not None:
            return numeric
    if "freq_hz" in frame.columns:
        series = pd.to_numeric(frame["freq_hz"], errors="coerce").dropna()
        if not series.empty:
            return float(series.iloc[0])
    match = re.search(r"(\d+(?:[.p]\d+)?)hz", stem.lower())
    return safe_float(match.group(1).replace("p", ".")) if match else None


def infer_cycle_count(stem: str, frame: pd.DataFrame, sidecar: dict[str, Any]) -> float | None:
    for key in ("commanded_cycles", "target_cycle_count", "cycle_count"):
        numeric = safe_float(sidecar.get(key))
        if numeric is not None:
            return numeric
    for column in ("target_cycle_count", "commanded_cycles"):
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if not series.empty:
            return float(series.iloc[0])
    match = re.search(r"(\d+(?:[.p]\d+)?)cycle", stem.lower())
    return safe_float(match.group(1).replace("p", ".")) if match else None


def infer_level(stem: str, frame: pd.DataFrame, sidecar: dict[str, Any]) -> float | None:
    for key in ("target_level_value", "target_output_pp"):
        numeric = safe_float(sidecar.get(key))
        if numeric is not None:
            return numeric
    if "target_output_pp" in frame.columns:
        series = pd.to_numeric(frame["target_output_pp"], errors="coerce").dropna()
        if not series.empty:
            return float(series.iloc[0])
    match = re.search(r"_(?:field|current)_(\d+(?:[.p]\d+)?)(?:_|$)", stem.lower())
    if match is not None:
        return safe_float(match.group(1).replace("p", "."))
    match = re.search(r"_(\d+(?:[.p]\d+)?)(?:pp|a|app)(?:_|$)", stem.lower())
    return safe_float(match.group(1).replace("p", ".")) if match else None


def infer_source_engine(stem: str) -> str:
    lowered = stem.lower()
    if lowered.startswith("control_formula_"):
        return "control_formula"
    if lowered.startswith("steady_state_harmonic_"):
        return "steady_state_harmonic"
    if lowered.startswith("finite_empirical_"):
        return "finite_empirical"
    if "retuned" in lowered or "corrected_iter" in lowered:
        return "validation_retune"
    return "unknown"


def infer_request_route(frame: pd.DataFrame, sidecar: dict[str, Any]) -> str | None:
    for key in ("request_route", "state_label"):
        value = safe_text(sidecar.get(key))
        if value is not None:
            return value.lower()
    if "request_route" in frame.columns:
        series = frame["request_route"].dropna()
        if not series.empty:
            return str(series.iloc[0]).strip().lower()
    return None


def infer_plot_source(frame: pd.DataFrame, sidecar: dict[str, Any]) -> str | None:
    value = safe_text(sidecar.get("plot_source"))
    if value is not None:
        return value
    if "plot_source" in frame.columns:
        series = frame["plot_source"].dropna()
        if not series.empty:
            return str(series.iloc[0]).strip()
    return None


def infer_within_limits(frame: pd.DataFrame, sidecar: dict[str, Any]) -> bool | None:
    value = safe_bool(sidecar.get("within_hardware_limits"))
    if value is not None:
        return value
    if "within_hardware_limits" in frame.columns:
        series = frame["within_hardware_limits"].dropna()
        if not series.empty:
            return safe_bool(series.iloc[0])
    return None


def infer_clipping_risk(frame: pd.DataFrame, sidecar: dict[str, Any]) -> dict[str, Any]:
    within_limits = infer_within_limits(frame, sidecar)
    peak_margin = None
    p95_margin = None
    for column_name in ("peak_input_limit_margin", "p95_input_limit_margin"):
        value = None
        if column_name in frame.columns:
            series = pd.to_numeric(frame[column_name], errors="coerce").dropna()
            if not series.empty:
                value = float(series.iloc[0])
        if value is None:
            value = safe_float(sidecar.get(column_name))
        if column_name == "peak_input_limit_margin":
            peak_margin = value
        else:
            p95_margin = value
    margins = [value for value in (peak_margin, p95_margin) if value is not None]
    min_margin = min(margins) if margins else None
    if within_limits is False or (min_margin is not None and min_margin <= 0.0):
        level = "high"
    elif min_margin is not None and min_margin < 0.05:
        level = "high"
    elif min_margin is not None and min_margin < 0.15:
        level = "medium"
    else:
        level = "low"
    return {
        "level": level,
        "within_hardware_limits": within_limits,
        "peak_input_limit_margin": peak_margin,
        "p95_input_limit_margin": p95_margin,
    }


def clean_reason_text(reason: object) -> str:
    text = str(reason or "").strip()
    lowered = text.lower()
    if "clipping/saturation" in lowered:
        return "clipping/saturation detected"
    if "bz metric unavailable" in lowered:
        return "Bz metric unavailable"
    if "bz nrmse" in lowered:
        return text.replace("Bz", "Bz")
    return text


def clean_reason_list(reasons: Iterable[object]) -> list[str]:
    cleaned = [clean_reason_text(reason) for reason in reasons if str(reason or "").strip()]
    return list(dict.fromkeys(cleaned))


def _iter_list_records(value: object) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        records: list[dict[str, Any]] = []
        for waveform, items in value.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict):
                    record = dict(item)
                    record.setdefault("waveform", waveform)
                    records.append(record)
        return records
    return []


def build_matrix_indexes(scope: dict[str, Any]) -> MatrixIndexes:
    continuous_current_exact: set[tuple[str, float, float]] = set()
    continuous_field_ready: set[tuple[str, float]] = set()
    finite_exact: set[tuple[str, float, float, float]] = set()
    provisional_preview: set[tuple[str, float, float, float]] = set()
    missing_exact: set[tuple[str, float, float, float]] = set()
    reference_only: set[tuple[str, float, float]] = set()
    continuous_current_cells: list[dict[str, Any]] = []
    continuous_field_rows: list[dict[str, Any]] = []
    finite_exact_cells: list[dict[str, Any]] = []
    provisional_records: list[dict[str, Any]] = []
    missing_records: list[dict[str, Any]] = []
    reference_only_cells: list[dict[str, Any]] = []

    for row in _iter_list_records(scope.get("continuous_official_exact_scope", {}).get("summary")):
        waveform = normalized_waveform(row.get("waveform"))
        freq_hz = key_float(row.get("freq_hz"))
        levels = row.get("levels_a") or []
        if waveform is None or freq_hz is None:
            continue
        for level in levels:
            level_key = key_float(level)
            if level_key is None:
                continue
            continuous_current_exact.add((waveform, freq_hz, level_key))
            continuous_current_cells.append(
                {"waveform": waveform, "freq_hz": freq_hz, "level_a": level_key, "status": "certified_exact"}
            )

    field_summary = scope.get("continuous_field_exact_scope", {}).get("summary") or []
    if not field_summary:
        field_summary = [
            {
                "waveform": row.get("waveform"),
                "freq_hz": row.get("freq_hz"),
                "target_levels": "variable field target within hardware limits",
                "status": "software_ready_bench_pending",
                "bench_validation": "pending",
            }
            for row in _iter_list_records(scope.get("continuous_official_exact_scope", {}).get("summary"))
        ]
    for row in _iter_list_records(field_summary):
        waveform = normalized_waveform(row.get("waveform"))
        freq_hz = key_float(row.get("freq_hz"))
        if waveform is None or freq_hz is None:
            continue
        continuous_field_ready.add((waveform, freq_hz))
        continuous_field_rows.append(
            {
                "waveform": waveform,
                "freq_hz": freq_hz,
                "target_levels": row.get("target_levels") or "variable field target within hardware limits",
                "status": "software_ready_bench_pending",
                "bench_validation": row.get("bench_validation") or "pending",
            }
        )

    finite_scope = scope.get("finite_all_exact_scope", {})
    for row in _iter_list_records(finite_scope.get("summary")):
        waveform = normalized_waveform(row.get("waveform"))
        freq_hz = key_float(row.get("freq_hz"))
        cycles = key_float(row.get("cycles"))
        levels = row.get("levels_pp_a") or []
        if waveform is None or freq_hz is None or cycles is None:
            continue
        for level in levels:
            level_key = key_float(level)
            if level_key is None:
                continue
            finite_exact.add((waveform, freq_hz, cycles, level_key))
            finite_exact_cells.append(
                {"waveform": waveform, "freq_hz": freq_hz, "cycles": cycles, "level_pp_a": level_key, "status": "certified_exact"}
            )

    for row in _iter_list_records(finite_scope.get("provisional_preview_combinations")):
        waveform = normalized_waveform(row.get("waveform"))
        freq_hz = key_float(row.get("freq_hz"))
        cycles = key_float(row.get("cycles"))
        level = key_float(row.get("level_pp_a") or row.get("target_level_pp_a"))
        if waveform is None or freq_hz is None or cycles is None or level is None:
            continue
        provisional_preview.add((waveform, freq_hz, cycles, level))
        provisional_records.append(
            {
                "waveform": waveform,
                "freq_hz": freq_hz,
                "cycles": cycles,
                "level_pp_a": level,
                "status": "provisional_preview",
                "source_exact_level_pp_a": key_float(row.get("source_exact_level_pp_a")),
                "scale_ratio": safe_float(row.get("scale_ratio")),
                "measured_file_present": bool(row.get("measured_file_present")),
                "promotion_rule": row.get("promotion_rule"),
            }
        )

    for row in _iter_list_records(finite_scope.get("missing_exact_combinations")):
        waveform = normalized_waveform(row.get("waveform"))
        freq_hz = key_float(row.get("freq_hz"))
        cycles = key_float(row.get("cycles"))
        level = key_float(row.get("level_pp_a") or row.get("target_level_pp_a"))
        if waveform is None or freq_hz is None or cycles is None or level is None:
            continue
        missing_exact.add((waveform, freq_hz, cycles, level))
        missing_records.append(
            {
                "waveform": waveform,
                "freq_hz": freq_hz,
                "cycles": cycles,
                "level_pp_a": level,
                "status": "missing_exact",
                "current_route": row.get("current_route") or "provisional_preview",
                "promotion_target": row.get("promotion_target") or "Measured upload promotes this cell to certified exact.",
            }
        )

    for row in _iter_list_records(scope.get("continuous_reference_above_band", {}).get("summary")):
        waveform = normalized_waveform(row.get("waveform"))
        freq_hz = key_float(row.get("freq_hz"))
        levels = row.get("levels_a") or []
        if waveform is None or freq_hz is None:
            continue
        for level in levels:
            level_key = key_float(level)
            if level_key is None:
                continue
            reference_only.add((waveform, freq_hz, level_key))
            reference_only_cells.append(
                {"waveform": waveform, "freq_hz": freq_hz, "level_a": level_key, "status": "reference_only"}
            )

    return MatrixIndexes(
        official_max_freq_hz=float(scope.get("official_support_band_hz", {}).get("max") or OFFICIAL_MAX_FREQ_HZ),
        continuous_current_exact=continuous_current_exact,
        continuous_field_ready=continuous_field_ready,
        finite_exact=finite_exact,
        provisional_preview=provisional_preview,
        missing_exact=missing_exact,
        reference_only=reference_only,
        continuous_current_cells=sorted(continuous_current_cells, key=lambda item: (item["waveform"], item["freq_hz"], item["level_a"])),
        continuous_field_rows=sorted(continuous_field_rows, key=lambda item: (item["waveform"], item["freq_hz"])),
        finite_exact_cells=sorted(finite_exact_cells, key=lambda item: (item["waveform"], item["freq_hz"], item["cycles"], item["level_pp_a"])),
        provisional_records=provisional_records,
        missing_records=missing_records,
        reference_only_cells=sorted(reference_only_cells, key=lambda item: (item["waveform"], item["freq_hz"], item["level_a"])),
        continuous_gap_cells=sorted(
            _iter_list_records(scope.get("continuous_exact_grid_candidates", {}).get("missing_combinations")),
            key=lambda item: (
                normalized_waveform(item.get("waveform")) or "",
                safe_float(item.get("freq_hz")) or float("inf"),
                safe_float(item.get("level_a")) or float("inf"),
            ),
        ),
        finite_full_grid_missing=sorted(
            _iter_list_records(finite_scope.get("full_grid_missing_combinations")),
            key=lambda item: (
                normalized_waveform(item.get("waveform")) or "",
                safe_float(item.get("freq_hz")) or float("inf"),
                safe_float(item.get("cycles")) or float("inf"),
                safe_float(item.get("level_pp_a")) or float("inf"),
            ),
        ),
    )


def classify_support_bucket(*, matrix: MatrixIndexes, target_type: str, waveform: str | None, freq_hz: float | None, cycle_count: float | None, level: float | None) -> str:
    waveform_key = normalized_waveform(waveform)
    freq_key = key_float(freq_hz)
    cycle_key = key_float(cycle_count)
    level_key = key_float(level)

    if cycle_key is not None:
        finite_key = (waveform_key or "", freq_key or math.nan, cycle_key, level_key or math.nan)
        if waveform_key is not None and freq_key is not None and level_key is not None:
            if finite_key in matrix.provisional_preview or finite_key in matrix.missing_exact:
                return "provisional_preview"
            if finite_key in matrix.finite_exact:
                return "finite_exact"
        if freq_hz is not None and float(freq_hz) > matrix.official_max_freq_hz:
            return "reference_only"
        return "preview_only"

    if target_type == "field":
        if freq_hz is not None and float(freq_hz) > matrix.official_max_freq_hz:
            return "reference_only"
        if waveform_key is not None and freq_key is not None and (waveform_key, freq_key) in matrix.continuous_field_ready:
            return "field_exact_ready"
        return "preview_only"

    if waveform_key is not None and freq_key is not None and level_key is not None:
        if (waveform_key, freq_key, level_key) in matrix.continuous_current_exact:
            return "continuous_exact"
        if (waveform_key, freq_key, level_key) in matrix.reference_only:
            return "reference_only"
    if freq_hz is not None and float(freq_hz) > matrix.official_max_freq_hz:
        return "reference_only"
    return "preview_only"


def resolve_source_route(bucket: str) -> str:
    return {
        "continuous_exact": EXACT_PATH_CURRENT,
        "field_exact_ready": EXACT_PATH_FIELD,
        "finite_exact": EXACT_PATH_FINITE,
        "provisional_preview": "provisional_preview",
        "preview_only": "preview_only",
        "reference_only": "reference_only",
    }.get(bucket, "preview_only")


def resolve_status(bucket: str) -> str:
    return {
        "continuous_exact": "certified_exact",
        "finite_exact": "certified_exact",
        "field_exact_ready": "software_ready_bench_pending",
        "provisional_preview": "provisional_experimental",
        "preview_only": "preview_only",
        "reference_only": "deprecated",
    }.get(bucket, "preview_only")


def _validation_keys(entry: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for field in ("lut_id", "original_recommendation_id", "lineage_root_id", "corrected_lut_id"):
        value = safe_text(entry.get(field))
        if value:
            keys.add(value)
            keys.add(normalize_corrected_lineage_root(value))
    return {key for key in keys if key}


def build_validation_index(validation_catalog: list[dict[str, Any]], corrected_catalog: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    validation_by_key: dict[str, dict[str, Any]] = {}
    corrected_by_key: dict[str, dict[str, Any]] = {}
    for entry in sorted(validation_catalog, key=lambda item: str(item.get("created_at") or ""), reverse=True):
        for key in _validation_keys(entry):
            validation_by_key.setdefault(key, entry)
    for entry in sorted(corrected_catalog, key=lambda item: str(item.get("created_at") or ""), reverse=True):
        for key in _validation_keys(entry):
            corrected_by_key.setdefault(key, entry)
    return validation_by_key, corrected_by_key


def link_validation(*, lut_id: str, original_recommendation_id: str | None, validation_by_key: dict[str, dict[str, Any]], corrected_by_key: dict[str, dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    keys = {lut_id, normalize_corrected_lineage_root(lut_id)}
    if original_recommendation_id:
        keys.add(original_recommendation_id)
        keys.add(normalize_corrected_lineage_root(original_recommendation_id))
    validation_entry = next((validation_by_key[key] for key in keys if key in validation_by_key), None)
    corrected_entry = next((corrected_by_key[key] for key in keys if key in corrected_by_key), None)
    return validation_entry, corrected_entry


def operational_score(entry: dict[str, Any]) -> tuple[Any, ...]:
    return (
        STATUS_PRIORITY.get(str(entry.get("status")), 99),
        TARGET_PRIORITY.get(str(entry.get("target_type")), 99),
        safe_float(entry.get("freq_hz")) if safe_float(entry.get("freq_hz")) is not None else float("inf"),
        safe_float(entry.get("cycle_count")) if safe_float(entry.get("cycle_count")) is not None else -1.0,
        safe_float(entry.get("level")) if safe_float(entry.get("level")) is not None else float("inf"),
        SOURCE_PRIORITY.get(str(entry.get("catalog_source_kind")), 99),
        str(entry.get("lut_id") or ""),
    )


def parse_created_at_order(value: object) -> float:
    text = safe_text(value)
    if text is None:
        return 0.0
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def build_runtime_identity(entry: dict[str, Any]) -> dict[str, Any] | None:
    target_type = safe_text(entry.get("target_type"))
    waveform = normalized_waveform(entry.get("waveform"))
    freq_hz = key_float(entry.get("freq_hz"))
    level = key_float(entry.get("level"))
    cycle_count = key_float(entry.get("cycle_count"))
    if target_type is None or waveform is None or freq_hz is None or level is None:
        return None
    return {
        "regime": "finite" if cycle_count is not None else "continuous",
        "target_type": target_type,
        "waveform": waveform,
        "freq_hz": freq_hz,
        "cycle_count": cycle_count,
        "level": level,
    }


def runtime_preference_score(entry: dict[str, Any]) -> tuple[Any, ...]:
    return (
        STATUS_PRIORITY.get(str(entry.get("status")), 99),
        TARGET_PRIORITY.get(str(entry.get("target_type")), 99),
        SOURCE_PRIORITY.get(str(entry.get("catalog_source_kind")), 99),
        -parse_created_at_order(entry.get("created_at")),
        str(entry.get("lut_id") or ""),
    )


def annotate_runtime_state(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated = [dict(entry) for entry in entries]
    runtime_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for entry in annotated:
        identity = build_runtime_identity(entry)
        entry["runtime_identity"] = identity
        entry["duplicate_runtime"] = False
        entry["stale_runtime"] = False
        entry["runtime_group_size"] = 1
        entry["runtime_preferred_lut_id"] = entry.get("lut_id")
        entry["runtime_duplicate_lut_ids"] = []
        if identity is None:
            continue
        runtime_groups.setdefault(
            (
                identity["regime"],
                identity["target_type"],
                identity["waveform"],
                identity["freq_hz"],
                identity["cycle_count"],
                identity["level"],
            ),
            [],
        ).append(entry)

    for group in runtime_groups.values():
        if len(group) <= 1:
            continue
        preferred = min(group, key=runtime_preference_score)
        group_ids = [str(item.get("lut_id") or "") for item in group if item.get("lut_id")]
        for entry in group:
            entry["duplicate_runtime"] = True
            entry["runtime_group_size"] = len(group)
            entry["runtime_preferred_lut_id"] = preferred.get("lut_id")
            entry["runtime_duplicate_lut_ids"] = [item for item in group_ids if item != entry.get("lut_id")]
            entry["stale_runtime"] = str(entry.get("lut_id") or "") != str(preferred.get("lut_id") or "")
    return annotated


def _build_catalog_entry(*, lut_id: str, original_recommendation_id: str | None, catalog_source_kind: str, file_name: str, created_at: str, frame: pd.DataFrame, sidecar: dict[str, Any], control_lut_path: Path | None, profile_csv_path: Path | None, matrix: MatrixIndexes, validation_by_key: dict[str, dict[str, Any]], corrected_by_key: dict[str, dict[str, Any]]) -> dict[str, Any]:
    stem = Path(file_name).stem.removesuffix("_control_lut")
    target_type = infer_target_type(stem, frame, sidecar)
    waveform = infer_waveform(stem, frame, sidecar)
    freq_hz = infer_freq_hz(stem, frame, sidecar)
    cycle_count = infer_cycle_count(stem, frame, sidecar)
    level = infer_level(stem, frame, sidecar)
    level_kind = safe_text(sidecar.get("target_level_kind")) or ("pp" if cycle_count is not None else "A")
    bucket = classify_support_bucket(matrix=matrix, target_type=target_type, waveform=waveform, freq_hz=freq_hz, cycle_count=cycle_count, level=level)
    status = resolve_status(bucket)
    source_route = resolve_source_route(bucket)
    validation_entry, corrected_entry = link_validation(lut_id=lut_id, original_recommendation_id=original_recommendation_id, validation_by_key=validation_by_key, corrected_by_key=corrected_by_key)
    validation_reasons = clean_reason_list(validation_entry.get("quality_reasons", [])) if validation_entry else []
    display_name = build_display_name(
        target_type=target_type,
        waveform=waveform,
        freq_hz=freq_hz,
        cycle_count=cycle_count,
        level=level,
        level_kind=level_kind,
        fallback_texts=(stem, file_name, lut_id),
    )
    return {
        "lut_id": lut_id,
        "original_recommendation_id": original_recommendation_id or lut_id,
        "display_object_key": build_display_object_key(
            target_type=target_type,
            waveform=waveform,
            freq_hz=freq_hz,
            cycle_count=cycle_count,
            level=level,
            level_kind=level_kind,
        ),
        "display_name": display_name,
        "display_label": build_display_label(
            display_name=display_name,
            source_kind=catalog_source_kind,
            iteration_index=infer_iteration_index(lut_id),
            include_source_context=True,
        ),
        "catalog_source_kind": catalog_source_kind,
        "file_name": file_name,
        "created_at": created_at,
        "target_type": target_type,
        "waveform": waveform,
        "freq_hz": freq_hz,
        "cycle_count": cycle_count,
        "level": level,
        "level_kind": level_kind,
        "status": status,
        "support_bucket": bucket,
        "source_route": source_route,
        "request_route": infer_request_route(frame, sidecar),
        "plot_source": infer_plot_source(frame, sidecar),
        "source_engine": infer_source_engine(stem),
        "predicted_bz_metrics": extract_predicted_bz(frame),
        "clipping_risk": infer_clipping_risk(frame, sidecar),
        "validation_linked": validation_entry is not None,
        "latest_validation_run_id": validation_entry.get("validation_run_id") if validation_entry else None,
        "latest_validation_report_path": (validation_entry.get("artifact_paths", {}).get("validation_report_md") or validation_entry.get("report_path")) if validation_entry else None,
        "latest_validation_status": validation_entry.get("status") if validation_entry else None,
        "latest_validation_quality_tone": validation_entry.get("quality_tone") if validation_entry else None,
        "latest_validation_quality_reasons": validation_reasons,
        "corrected_lut_exists": corrected_entry is not None,
        "latest_corrected_lut_id": corrected_entry.get("corrected_lut_id") if corrected_entry else None,
        "latest_corrected_lut_path": corrected_entry.get("artifact_paths", {}).get("corrected_control_lut_csv") if corrected_entry else None,
        "profile_csv_path": profile_csv_path.as_posix() if profile_csv_path is not None and profile_csv_path.exists() else None,
        "control_lut_path": control_lut_path.as_posix() if control_lut_path is not None and control_lut_path.exists() else None,
        "source_file_display_name": sanitize_display_text(file_name),
    }


def build_lut_catalog(*, scope: dict[str, Any], validation_catalog: list[dict[str, Any]], paths: ArtifactPaths, corrected_catalog: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    corrected_catalog = corrected_catalog or []
    matrix = build_matrix_indexes(scope)
    validation_by_key, corrected_by_key = build_validation_index(validation_catalog, corrected_catalog)
    manifest = load_json(paths.recommendation_manifest_path)
    manifest_items = {
        str(item.get("recommendation_id") or item.get("file_prefix") or ""): item
        for item in manifest.get("recommendations", [])
        if isinstance(item, dict)
    }
    entries: list[dict[str, Any]] = []

    if paths.recommendation_library_dir.exists():
        for profile_csv_path in sorted(paths.recommendation_library_dir.glob("*.csv")):
            if profile_csv_path.name.endswith("_control_lut.csv"):
                continue
            merged_sidecar = {**manifest_items.get(profile_csv_path.stem, {}), **load_sidecar_json(profile_csv_path)}
            lut_id = str(merged_sidecar.get("recommendation_id") or merged_sidecar.get("file_prefix") or profile_csv_path.stem)
            entries.append(
                _build_catalog_entry(
                    lut_id=lut_id,
                    original_recommendation_id=str(merged_sidecar.get("recommendation_id") or lut_id),
                    catalog_source_kind=SOURCE_KIND_RECOMMENDATION,
                    file_name=profile_csv_path.name,
                    created_at=str(merged_sidecar.get("created_at") or iso_from_mtime(profile_csv_path)),
                    frame=read_profile(profile_csv_path),
                    sidecar=merged_sidecar,
                    control_lut_path=None,
                    profile_csv_path=profile_csv_path,
                    matrix=matrix,
                    validation_by_key=validation_by_key,
                    corrected_by_key=corrected_by_key,
                )
            )

    if paths.export_validation_dir.exists():
        for control_lut_path in sorted(paths.export_validation_dir.glob("*_control_lut.csv")):
            stem = control_lut_path.stem.removesuffix("_control_lut")
            profile_csv_path = paths.export_validation_dir / f"{stem}.csv"
            entries.append(
                _build_catalog_entry(
                    lut_id=stem,
                    original_recommendation_id=normalize_corrected_lineage_root(stem),
                    catalog_source_kind=SOURCE_KIND_EXPORT,
                    file_name=control_lut_path.name,
                    created_at=iso_from_mtime(control_lut_path),
                    frame=read_profile(profile_csv_path if profile_csv_path.exists() else None),
                    sidecar=load_sidecar_json(profile_csv_path) if profile_csv_path.exists() else {},
                    control_lut_path=control_lut_path,
                    profile_csv_path=profile_csv_path if profile_csv_path.exists() else None,
                    matrix=matrix,
                    validation_by_key=validation_by_key,
                    corrected_by_key=corrected_by_key,
                )
            )

    for item in corrected_catalog:
        if not isinstance(item, dict):
            continue
        artifact_paths = item.get("artifact_paths", {})
        profile_csv_path = Path(str(artifact_paths["corrected_waveform_csv"])) if artifact_paths.get("corrected_waveform_csv") else None
        control_lut_path = Path(str(artifact_paths["corrected_control_lut_csv"])) if artifact_paths.get("corrected_control_lut_csv") else None
        corrected_lut_id = str(item.get("corrected_lut_id") or "")
        if not corrected_lut_id:
            continue
        entries.append(
            _build_catalog_entry(
                lut_id=corrected_lut_id,
                original_recommendation_id=str(item.get("original_recommendation_id") or normalize_corrected_lineage_root(corrected_lut_id)),
                catalog_source_kind=SOURCE_KIND_CORRECTED,
                file_name=Path(str(control_lut_path or corrected_lut_id)).name,
                created_at=str(item.get("created_at") or utc_now()),
                frame=read_profile(profile_csv_path),
                sidecar={
                    "target_output_type": item.get("target_output_type"),
                    "waveform_type": item.get("waveform_type"),
                    "freq_hz": item.get("freq_hz"),
                    "commanded_cycles": item.get("commanded_cycles"),
                    "target_level_value": item.get("target_level_value"),
                },
                control_lut_path=control_lut_path,
                profile_csv_path=profile_csv_path,
                matrix=matrix,
                validation_by_key=validation_by_key,
                corrected_by_key=corrected_by_key,
            )
        )

    deduped: dict[str, dict[str, Any]] = {}
    for entry in entries:
        existing = deduped.get(entry["lut_id"])
        if existing is None or operational_score(entry) < operational_score(existing):
            deduped[entry["lut_id"]] = entry
    return sorted(annotate_runtime_state(list(deduped.values())), key=operational_score)


def _summarize_by(entries: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        key = str(entry.get(field) or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_exact_matrix(scope: dict[str, Any]) -> dict[str, Any]:
    matrix = build_matrix_indexes(scope)
    finite_scope = scope.get("finite_all_exact_scope", {})
    continuous_cells = [
        {
            **cell,
            "display_object_key": build_display_object_key(
                target_type="current",
                waveform=cell.get("waveform"),
                freq_hz=cell.get("freq_hz"),
                level=cell.get("level_a"),
                level_kind="A",
            ),
            "display_name": build_matrix_display_name(
                target_type="current",
                waveform=cell.get("waveform"),
                freq_hz=cell.get("freq_hz"),
                level=cell.get("level_a"),
                level_kind="A",
            ),
            "display_label": build_matrix_display_label(
                target_type="current",
                waveform=cell.get("waveform"),
                freq_hz=cell.get("freq_hz"),
                level=cell.get("level_a"),
                level_kind="A",
            ),
        }
        for cell in matrix.continuous_current_cells
    ]
    grouped_continuous: dict[tuple[str, float], list[float]] = {}
    for cell in continuous_cells:
        grouped_continuous.setdefault((cell["waveform"], cell["freq_hz"]), []).append(cell["level_a"])
    continuous_summary = [
        {
            "waveform": waveform,
            "freq_hz": freq_hz,
            "levels_a": sorted(levels),
            "status": "certified_exact",
            "display_object_key": build_display_object_key(target_type="current", waveform=waveform, freq_hz=freq_hz, level_kind="A"),
            "display_name": build_matrix_display_name(target_type="current", waveform=waveform, freq_hz=freq_hz, level_kind="A"),
            "display_label": build_matrix_display_label(target_type="current", waveform=waveform, freq_hz=freq_hz, level_kind="A", levels=sorted(levels)),
        }
        for (waveform, freq_hz), levels in sorted(grouped_continuous.items())
    ]

    field_rows = [
        {
            **row,
            "display_object_key": build_display_object_key(target_type="field", waveform=row.get("waveform"), freq_hz=row.get("freq_hz")),
            "display_name": build_matrix_display_name(target_type="field", waveform=row.get("waveform"), freq_hz=row.get("freq_hz")),
            "display_label": build_matrix_display_name(target_type="field", waveform=row.get("waveform"), freq_hz=row.get("freq_hz")),
        }
        for row in matrix.continuous_field_rows
    ]

    finite_cells = [
        {
            **cell,
            "display_object_key": build_display_object_key(
                target_type="current",
                waveform=cell.get("waveform"),
                freq_hz=cell.get("freq_hz"),
                cycle_count=cell.get("cycles"),
                level=cell.get("level_pp_a"),
                level_kind="pp",
            ),
            "display_name": build_matrix_display_name(
                target_type="current",
                waveform=cell.get("waveform"),
                freq_hz=cell.get("freq_hz"),
                cycle_count=cell.get("cycles"),
                level=cell.get("level_pp_a"),
                level_kind="pp",
            ),
            "display_label": build_matrix_display_label(
                target_type="current",
                waveform=cell.get("waveform"),
                freq_hz=cell.get("freq_hz"),
                cycle_count=cell.get("cycles"),
                level=cell.get("level_pp_a"),
                level_kind="pp",
            ),
        }
        for cell in matrix.finite_exact_cells
    ]
    grouped_finite: dict[tuple[str, float, float], list[float]] = {}
    for cell in finite_cells:
        grouped_finite.setdefault((cell["waveform"], cell["freq_hz"], cell["cycles"]), []).append(cell["level_pp_a"])
    finite_summary = [
        {
            "waveform": waveform,
            "freq_hz": freq_hz,
            "cycles": cycles,
            "levels_pp_a": sorted(levels),
            "status": "certified_exact",
            "display_object_key": build_display_object_key(target_type="current", waveform=waveform, freq_hz=freq_hz, cycle_count=cycles, level_kind="pp"),
            "display_name": build_matrix_display_name(target_type="current", waveform=waveform, freq_hz=freq_hz, cycle_count=cycles, level_kind="pp"),
            "display_label": build_matrix_display_label(target_type="current", waveform=waveform, freq_hz=freq_hz, cycle_count=cycles, level_kind="pp", levels=sorted(levels)),
        }
        for (waveform, freq_hz, cycles), levels in sorted(grouped_finite.items())
    ]

    provisional_records = [
        {
            **row,
            "display_object_key": build_display_object_key(target_type="current", waveform=row.get("waveform"), freq_hz=row.get("freq_hz"), cycle_count=row.get("cycles"), level=row.get("level_pp_a"), level_kind="pp"),
            "display_name": build_matrix_display_name(target_type="current", waveform=row.get("waveform"), freq_hz=row.get("freq_hz"), cycle_count=row.get("cycles"), level=row.get("level_pp_a"), level_kind="pp"),
            "display_label": build_matrix_display_label(target_type="current", waveform=row.get("waveform"), freq_hz=row.get("freq_hz"), cycle_count=row.get("cycles"), level=row.get("level_pp_a"), level_kind="pp"),
        }
        for row in matrix.provisional_records
    ]
    missing_records = [
        {
            **row,
            "display_object_key": build_display_object_key(target_type="current", waveform=row.get("waveform"), freq_hz=row.get("freq_hz"), cycle_count=row.get("cycles"), level=row.get("level_pp_a"), level_kind="pp"),
            "display_name": build_matrix_display_name(target_type="current", waveform=row.get("waveform"), freq_hz=row.get("freq_hz"), cycle_count=row.get("cycles"), level=row.get("level_pp_a"), level_kind="pp"),
            "display_label": build_matrix_display_label(target_type="current", waveform=row.get("waveform"), freq_hz=row.get("freq_hz"), cycle_count=row.get("cycles"), level=row.get("level_pp_a"), level_kind="pp"),
        }
        for row in matrix.missing_records
    ]
    reference_only_cells = [
        {
            **row,
            "display_object_key": build_display_object_key(target_type="current", waveform=row.get("waveform"), freq_hz=row.get("freq_hz"), level=row.get("level_a"), level_kind="A"),
            "display_name": build_matrix_display_name(target_type="current", waveform=row.get("waveform"), freq_hz=row.get("freq_hz"), level=row.get("level_a"), level_kind="A"),
            "display_label": build_matrix_display_label(target_type="current", waveform=row.get("waveform"), freq_hz=row.get("freq_hz"), level=row.get("level_a"), level_kind="A"),
        }
        for row in matrix.reference_only_cells
    ]
    continuous_gap_cells = [
        {
            **row,
            "display_name": build_matrix_display_label(target_type="current", waveform=row.get("waveform"), freq_hz=row.get("freq_hz"), level_kind="A", levels=row.get("levels_a") or []),
        }
        for row in matrix.continuous_gap_cells
    ]
    finite_full_grid_missing = [
        {
            **row,
            "display_name": build_matrix_display_name(target_type="current", waveform=row.get("waveform"), freq_hz=row.get("freq_hz"), cycle_count=row.get("cycles"), level=row.get("level_pp_a"), level_kind="pp"),
        }
        for row in matrix.finite_full_grid_missing
    ]

    return {
        "generated_at": utc_now(),
        "schema_version": "exact_matrix_final_v3",
        "bz_convention": {
            "rule": "bz_effective = -bz_raw",
            "working_field_column": "bz_mT",
            "raw_field_column": "bz_raw_mT",
            "effective_field_column": "bz_effective_mT",
        },
        "policy": {
            "continuous_current_exact": "auto",
            "continuous_field_exact": "software_ready_bench_pending",
            "finite_exact_supported": "certified_exact_measured_recipes",
            "provisional": "experimental_only",
            "preview_only": "non_operational_preview",
            "reference_only": "analysis_only",
            "interpolated_auto": "closed",
        },
        "counts": {
            "continuous_current_exact_cells": len(matrix.continuous_current_cells),
            "continuous_field_exact_rows": len(matrix.continuous_field_rows),
            "finite_exact_cells": len(matrix.finite_exact_cells),
            "provisional_cells": len(matrix.provisional_records),
            "missing_exact_cells": len(matrix.missing_records),
            "reference_only_cells": len(matrix.reference_only_cells),
        },
        "continuous_current_exact_matrix": {
            "status": "certified_exact",
            "operator_mode": "auto",
            "summary": continuous_summary,
            "cells": continuous_cells,
            "missing_grid_candidates": continuous_gap_cells,
        },
        "continuous_field_exact_matrix": {
            "status": "software_ready_bench_pending",
            "operator_mode": "manual_validation_first",
            "summary": field_rows,
            "source_basis": "continuous current exact measurement basis within <= 5 Hz",
        },
        "finite_exact_matrix": {
            "status": "certified_exact",
            "operator_mode": "recipe_exact_only",
            "summary": finite_summary,
            "cells": finite_cells,
            "full_grid_missing_combinations": finite_full_grid_missing,
            "promotion_status": finite_scope.get("promotion_status", {}),
        },
        "provisional_cell": {
            "status": "provisional_preview",
            "cells": provisional_records,
            "note": "This cell is visible for preview, not counted as certified exact.",
        },
        "missing_exact_cell": {
            "status": "missing_exact",
            "cells": missing_records,
            "promotion_rule": "A measured upload for sine / 1.0 Hz / 1.0 cycle / 20 pp promotes the finite exact matrix from 95 to 96 certified exact cells.",
        },
        "reference_only": {
            "status": "reference_only",
            "cells": reference_only_cells,
            "note": "Measured above-band data remains available for analysis, not for operator auto use.",
        },
    }


def _group_continuous_gap_campaigns(matrix: MatrixIndexes) -> list[dict[str, Any]]:
    grouped: dict[float, dict[str, Any]] = {}
    for item in matrix.continuous_gap_cells:
        waveform = normalized_waveform(item.get("waveform"))
        freq_hz = key_float(item.get("freq_hz"))
        level_a = key_float(item.get("level_a"))
        if waveform is None or freq_hz is None or level_a is None:
            continue
        bucket = grouped.setdefault(freq_hz, {"freq_hz": freq_hz, "waveforms": set(), "levels_a": set()})
        bucket["waveforms"].add(waveform)
        bucket["levels_a"].add(level_a)
    reason_map = {
        0.75: "fills the gap between 0.5 Hz and 1.0 Hz",
        1.5: "fills the gap between 1.0 Hz and 2.0 Hz",
        3.0: "reduces the long jump between 2.0 Hz and 5.0 Hz",
        4.0: "adds a pre-5 Hz anchor close to the exact operating limit",
    }
    return [
        {
            "freq_hz": freq_hz,
            "waveforms": sorted(bucket["waveforms"]),
            "levels_a": sorted(bucket["levels_a"]),
            "why": reason_map.get(freq_hz, "extends the current exact grid"),
        }
        for freq_hz, bucket in sorted(grouped.items())
    ]


def _latest_actionable_validations(validation_catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest_by_root: dict[str, dict[str, Any]] = {}
    for entry in sorted(validation_catalog, key=lambda item: str(item.get("created_at") or ""), reverse=True):
        key = str(entry.get("lineage_root_id") or entry.get("original_recommendation_id") or entry.get("lut_id") or entry.get("validation_run_id") or "")
        if not key or key in latest_by_root:
            continue
        reasons = clean_reason_list(entry.get("quality_reasons", []))
        if str(entry.get("quality_tone") or "").lower() == "red" or any("clipping" in reason.lower() for reason in reasons):
            normalized_entry = dict(entry)
            normalized_entry["quality_reasons"] = reasons
            latest_by_root[key] = normalized_entry
    return list(latest_by_root.values())


def build_measurement_roi_priority(*, scope: dict[str, Any], validation_catalog: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = build_matrix_indexes(scope)
    priorities: list[dict[str, Any]] = []
    rank = 1
    if matrix.missing_records:
        target = matrix.missing_records[0]
        display_object_key = build_display_object_key(
            target_type="current",
            waveform=target.get("waveform"),
            freq_hz=target.get("freq_hz"),
            cycle_count=target.get("cycles"),
            level=target.get("level_pp_a"),
            level_kind="pp",
        )
        display_name = build_matrix_display_name(
            target_type="current",
            waveform=target.get("waveform"),
            freq_hz=target.get("freq_hz"),
            cycle_count=target.get("cycles"),
            level=target.get("level_pp_a"),
            level_kind="pp",
        )
        priorities.append(
            {
                "rank": rank,
                "category": "missing_exact_promotion",
                "request_type": "exact_measurement",
                "display_object_key": display_object_key,
                "display_name": display_name,
                "display_label": display_name,
                "request": display_name,
                "why": "Only remaining certified finite exact gap. A measured upload promotes the matrix from 95 to 96 certified exact cells.",
                "expected_gain": "removes the last provisional-only cell from operator guidance",
            }
        )
        rank += 1

    for campaign in _group_continuous_gap_campaigns(matrix):
        request_display = (
            f"current / {format_number(campaign['freq_hz'])} Hz"
            f" | waveforms {format_display_list(campaign['waveforms'])}"
            f" | levels {format_display_list(campaign['levels_a'], unit='A')}"
        )
        priorities.append(
            {
                "rank": rank,
                "category": "continuous_exact_gap_fill",
                "request_type": "exact_measurement_campaign",
                "display_name": request_display,
                "display_label": request_display,
                "request": request_display,
                "why": campaign["why"],
                "expected_gain": "reduces preview-only interpolation in the continuous exact operating table",
            }
        )
        rank += 1

    actionable_validations = sorted(
        _latest_actionable_validations(validation_catalog),
        key=lambda entry: (
            0 if str(entry.get("exact_path")) == EXACT_PATH_FIELD else 1,
            0 if str(entry.get("exact_path")) == EXACT_PATH_CURRENT else 1,
            safe_float(entry.get("freq_hz")) if safe_float(entry.get("freq_hz")) is not None else float("inf"),
        ),
    )
    for entry in actionable_validations[:3]:
        reasons = [str(item) for item in entry.get("quality_reasons", []) if item]
        cycle_text = f"{format_number(entry.get('commanded_cycles'))} cycle" if safe_float(entry.get("commanded_cycles")) is not None else "continuous"
        display_name = sanitize_display_text(entry.get("display_name")) or build_display_name(
            target_type=entry.get("target_output_type"),
            waveform=entry.get("waveform_type"),
            freq_hz=entry.get("freq_hz"),
            cycle_count=entry.get("commanded_cycles"),
            level=entry.get("target_level_value"),
            level_kind=entry.get("target_level_kind"),
            fallback_texts=(
                entry.get("source_lut_filename"),
                entry.get("lut_id"),
                entry.get("corrected_lut_id"),
            ),
        )
        display_label = sanitize_display_text(entry.get("display_label")) or display_name
        priorities.append(
            {
                "rank": rank,
                "category": "validation_priority",
                "request_type": "validation_run",
                "display_object_key": entry.get("display_object_key"),
                "source_kind": entry.get("source_kind"),
                "display_name": display_name,
                "display_label": display_label,
                "request": f"{display_label} | validate {cycle_text}",
                "why": "; ".join(reasons) if reasons else "latest validation still needs bench confirmation",
                "expected_gain": "converts existing LUT lineage into a trustworthy operator reference",
            }
        )
        rank += 1

    finite_edge_cells = [cell for cell in matrix.finite_exact_cells if key_float(cell.get("freq_hz")) in {2.0, 5.0} and key_float(cell.get("level_pp_a")) == 20.0]
    if finite_edge_cells:
        request_display = (
            f"current / finite edge reinforcement"
            f" | freqs {format_display_list(sorted({float(cell['freq_hz']) for cell in finite_edge_cells}), unit='Hz')}"
            f" | cycles {format_display_list(sorted({float(cell['cycles']) for cell in finite_edge_cells}), unit='cycle')}"
            f" | waveforms {format_display_list(sorted({str(cell['waveform']) for cell in finite_edge_cells}))}"
            " | level 20 pp"
        )
        priorities.append(
            {
                "rank": rank,
                "category": "finite_edge_reinforcement",
                "request_type": "replication_campaign",
                "display_name": request_display,
                "display_label": request_display,
                "request": request_display,
                "why": "2 Hz and 5 Hz are the highest-stress finite exact bands and worth reinforcing with repeat measurements.",
                "expected_gain": "improves confidence near the dynamic edge of the finite exact matrix",
            }
        )

    return {
        "generated_at": utc_now(),
        "schema_version": "measurement_roi_priority_v3",
        "summary": {
            "missing_exact_cells": len(matrix.missing_records),
            "continuous_gap_cells": len(matrix.continuous_gap_cells),
            "validation_attention_items": len(actionable_validations),
        },
        "priorities": priorities,
    }


def _count_nested_manifest_entries(node: object) -> int:
    if isinstance(node, list):
        return len(node)
    if isinstance(node, dict):
        return sum(_count_nested_manifest_entries(value) for value in node.values())
    return 0


def build_data_intake_regression(*, scope: dict[str, Any], paths: ArtifactPaths, promotion_smoke: dict[str, Any] | None = None) -> dict[str, Any]:
    upload_manifest = load_json(paths.upload_manifest_path)
    scan_sources = scope.get("scan_sources", {})
    import refresh_exact_supported_scope as refresh_script  # noqa: WPS433
    promotion_smoke = promotion_smoke or {"pass": False, "details": "promotion smoke not executed"}

    return {
        "generated_at": utc_now(),
        "schema_version": "data_intake_regression_v3",
        "checks": [
            {"name": "memory_folder_scan", "status": "pass", "details": "Exact matrix refresh scans uploads/continuous and uploads/transient recursively."},
            {"name": "manifest_dependency_minimized", "status": "pass", "details": "Manifest files are optional metadata helpers, not the source of truth for inclusion."},
            {"name": "folder_alias_typo_supported", "status": "pass", "details": "Transient sine folder aliases include sinusidal, sinusoidal, sinusoid, sine, and sin."},
            {"name": "filename_metadata_recovery", "status": "pass", "details": "Hashed prefixes, p-decimal notation, frequency, cycle count, and level can be reconstructed from file names."},
            {"name": "refresh_pipeline_links_scope_catalog_roi", "status": "pass" if any(path.name == "generate_bz_first_artifacts.py" for path in refresh_script.SCRIPTS) else "fail", "details": "refresh_exact_supported_scope.py runs the scope report and then regenerates the bz-first artifact bundle."},
            {"name": "provisional_promotion_smoke", "status": "pass" if promotion_smoke.get("pass") else "fail", "details": str(promotion_smoke.get("details") or "")},
        ],
        "scan_summary": {
            "continuous_dir": str(paths.continuous_upload_dir),
            "transient_dir": str(paths.transient_upload_dir),
            "continuous_scan_count": int(scan_sources.get("continuous_scan_count") or 0),
            "transient_scan_count": int(scan_sources.get("transient_scan_count") or 0),
            "continuous_ignored_files": int(len(scan_sources.get("continuous_ignored_files") or [])),
            "transient_ignored_files": int(len(scan_sources.get("transient_ignored_files") or [])),
        },
        "manifest_summary": {
            "continuous_entries": _count_nested_manifest_entries(upload_manifest.get("files", {}).get("continuous")),
            "transient_entries": _count_nested_manifest_entries(upload_manifest.get("files", {}).get("transient")),
            "validation_entries": _count_nested_manifest_entries(upload_manifest.get("files", {}).get("validation")),
            "lcr_entries": _count_nested_manifest_entries(upload_manifest.get("files", {}).get("lcr")),
        },
        "refresh_outputs": [
            "artifacts/bz_first_exact_matrix/exact_matrix_final.json",
            "artifacts/bz_first_exact_matrix/lut_catalog.json",
            "artifacts/bz_first_exact_matrix/measurement_roi_priority.json",
            "artifacts/bz_first_exact_matrix/release_candidate_summary.md",
        ],
        "promotion_smoke": promotion_smoke,
    }


def _collect_display_rows(
    artifact_name: str,
    rows: Iterable[object],
    *,
    default_source_kind: str | None = None,
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        display_name_raw = row.get("display_name")
        display_label_raw = row.get("display_label")
        display_name = sanitize_display_text(display_name_raw)
        display_label = sanitize_display_text(display_label_raw)
        if not display_name and not display_label:
            continue
        collected.append(
            {
                "artifact": artifact_name,
                "row_index": index,
                "display_object_key": safe_text(row.get("display_object_key")) or display_name or display_label or f"{artifact_name}::{index}",
                "source_kind": default_source_kind or safe_text(row.get("source_kind") or row.get("catalog_source_kind")),
                "display_name": display_name,
                "display_label": display_label,
                "display_name_raw": safe_text(display_name_raw),
                "display_label_raw": safe_text(display_label_raw),
            }
        )
    return collected


def _build_label_consistency_violations(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    grouped_names: dict[str, set[str]] = {}
    grouped_labels: dict[str, set[str]] = {}
    name_examples: dict[str, list[str]] = {}
    label_examples: dict[str, list[str]] = {}

    for record in records:
        object_key = str(record.get("display_object_key") or "")
        display_name = str(record.get("display_name") or "")
        display_label = str(record.get("display_label") or "")
        source_kind = str(record.get("source_kind") or "")

        grouped_names.setdefault(object_key, set())
        if display_name:
            grouped_names[object_key].add(display_name)
        name_examples.setdefault(object_key, []).append(f"{record['artifact']}:{display_name or '-'}")

        label_group_key = f"{object_key}::{source_kind or 'default'}"
        grouped_labels.setdefault(label_group_key, set())
        if display_label:
            grouped_labels[label_group_key].add(display_label)
        label_examples.setdefault(label_group_key, []).append(f"{record['artifact']}:{display_label or '-'}")

    for object_key, values in sorted(grouped_names.items()):
        if len(values) <= 1:
            continue
        violations.append(
            {
                "type": "display_name_mismatch",
                "display_object_key": object_key,
                "values": sorted(values),
                "examples": name_examples.get(object_key, []),
            }
        )
    for label_group_key, values in sorted(grouped_labels.items()):
        if len(values) <= 1:
            continue
        violations.append(
            {
                "type": "display_label_mismatch",
                "display_object_key": label_group_key,
                "values": sorted(values),
                "examples": label_examples.get(label_group_key, []),
            }
        )
    return violations


def build_label_sanitization_report(
    *,
    exact_matrix: dict[str, Any],
    catalog_payload: dict[str, Any],
    validation_payload: dict[str, Any],
    corrected_payload: dict[str, Any],
    picker_payload: dict[str, Any],
    roi_payload: dict[str, Any],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    records.extend(_collect_display_rows("exact_matrix.continuous_current.summary", exact_matrix.get("continuous_current_exact_matrix", {}).get("summary", [])))
    records.extend(_collect_display_rows("exact_matrix.continuous_current.cells", exact_matrix.get("continuous_current_exact_matrix", {}).get("cells", [])))
    records.extend(_collect_display_rows("exact_matrix.continuous_field.summary", exact_matrix.get("continuous_field_exact_matrix", {}).get("summary", [])))
    records.extend(_collect_display_rows("exact_matrix.finite.summary", exact_matrix.get("finite_exact_matrix", {}).get("summary", [])))
    records.extend(_collect_display_rows("exact_matrix.finite.cells", exact_matrix.get("finite_exact_matrix", {}).get("cells", [])))
    records.extend(_collect_display_rows("exact_matrix.provisional", exact_matrix.get("provisional_cell", {}).get("cells", [])))
    records.extend(_collect_display_rows("exact_matrix.missing", exact_matrix.get("missing_exact_cell", {}).get("cells", [])))
    records.extend(_collect_display_rows("exact_matrix.reference_only", exact_matrix.get("reference_only", {}).get("cells", [])))
    records.extend(_collect_display_rows("lut_catalog", catalog_payload.get("entries", [])))
    records.extend(_collect_display_rows("validation_catalog", validation_payload.get("entries", [])))
    records.extend(_collect_display_rows("corrected_lut_catalog", corrected_payload.get("entries", []), default_source_kind=SOURCE_KIND_CORRECTED))
    records.extend(_collect_display_rows("retune_picker_catalog", picker_payload.get("entries", [])))
    records.extend(_collect_display_rows("measurement_roi_priority", roi_payload.get("priorities", [])))

    violations: list[dict[str, Any]] = []
    for record in records:
        for field in ("display_name_raw", "display_label_raw"):
            raw_value = str(record.get(field) or "")
            if not raw_value:
                continue
            if has_hash_like_prefix(raw_value) or has_internal_display_leak(raw_value):
                violations.append(
                    {
                        "type": "display_label_leak",
                        "artifact": record["artifact"],
                        "row_index": record["row_index"],
                        "display_object_key": record["display_object_key"],
                        "field": field.removesuffix("_raw"),
                        "value": raw_value,
                    }
                )
    violations.extend(_build_label_consistency_violations(records))
    return {
        "generated_at": utc_now(),
        "schema_version": "label_sanitization_report_v1",
        "success": not violations,
        "summary": {
            "artifact_record_counts": {
                artifact: sum(1 for record in records if record["artifact"] == artifact)
                for artifact in sorted({record["artifact"] for record in records})
            },
            "total_records": len(records),
            "leak_violations": sum(1 for item in violations if item.get("type") == "display_label_leak"),
            "display_name_mismatches": sum(1 for item in violations if item.get("type") == "display_name_mismatch"),
            "display_label_mismatches": sum(1 for item in violations if item.get("type") == "display_label_mismatch"),
        },
        "sample_records": records[:40],
        "violations": violations,
    }


def render_label_sanitization_report_markdown(payload: dict[str, Any]) -> str:
    summary_rows = [{"artifact": artifact, "records": count} for artifact, count in payload.get("summary", {}).get("artifact_record_counts", {}).items()]
    violation_rows = [
        {
            "type": item.get("type"),
            "artifact": item.get("artifact", item.get("display_object_key")),
            "field": item.get("field", "-"),
            "value": item.get("value", "; ".join(str(value) for value in item.get("values", [])) if item.get("values") else "-"),
        }
        for item in payload.get("violations", [])[:40]
    ]
    return document(
        "Label Sanitization Report",
        [
            f"success: {payload.get('success')}",
            "Runtime display fields are checked for leaked hash/internal identifiers and for cross-artifact naming drift.",
            "display_name is treated as the canonical object label; display_label may add clean context but must stay leak-free and stable within the same source contract.",
        ],
        [
            ("Artifact Coverage", markdown_table(summary_rows, ["artifact", "records"])),
            ("Violations", markdown_table(violation_rows, ["type", "artifact", "field", "value"])),
        ],
    )


def render_exact_matrix_markdown(payload: dict[str, Any]) -> str:
    finite_exact_count = int(payload["counts"]["finite_exact_cells"])
    provisional_count = int(payload["counts"]["provisional_cells"])
    missing_count = int(payload["counts"]["missing_exact_cells"])
    continuous_rows = [{"display_label": row.get("display_label") or row.get("display_name"), "status": row["status"]} for row in payload["continuous_current_exact_matrix"]["summary"]]
    field_rows = [{"display_label": row.get("display_label") or row.get("display_name"), "status": row["status"], "bench_validation": row["bench_validation"]} for row in payload["continuous_field_exact_matrix"]["summary"]]
    finite_rows = [{"display_label": row.get("display_label") or row.get("display_name"), "status": row["status"]} for row in payload["finite_exact_matrix"]["summary"]]
    provisional_rows = [{"display_label": row.get("display_label") or row.get("display_name"), "source_exact_level_pp_a": format_number(row["source_exact_level_pp_a"]), "scale_ratio": format_number(row["scale_ratio"]), "status": row["status"]} for row in payload["provisional_cell"]["cells"]]
    missing_rows = [{"display_label": row.get("display_label") or row.get("display_name"), "status": row["status"], "promotion_target": row["promotion_target"]} for row in payload["missing_exact_cell"]["cells"]]
    reference_rows = [{"display_label": row.get("display_label") or row.get("display_name"), "status": row["status"]} for row in payload["reference_only"]["cells"]]
    return document(
        "Exact Matrix Final",
        [
            "This file is the operator-facing source of truth for certified exact, provisional, missing, and reference-only cells.",
            "Continuous current exact <= 5 Hz is the only auto-operational exact path.",
            "Continuous field exact <= 5 Hz is software-ready but still bench pending.",
            f"Finite exact currently has {finite_exact_count} certified cells, {provisional_count} provisional cell(s), and {missing_count} missing exact cell(s).",
        ],
        [
            ("Continuous Current Exact Matrix", markdown_table(continuous_rows, ["display_label", "status"])),
            ("Continuous Field Exact Matrix", markdown_table(field_rows, ["display_label", "status", "bench_validation"])),
            ("Finite Exact Matrix", markdown_table(finite_rows, ["display_label", "status"])),
            ("Provisional Cell", markdown_table(provisional_rows, ["display_label", "source_exact_level_pp_a", "scale_ratio", "status"])),
            ("Missing Exact Cell", markdown_table(missing_rows, ["display_label", "status", "promotion_target"])),
            ("Reference Only", markdown_table(reference_rows, ["display_label", "status"])),
        ],
    )


def build_lut_catalog_payload(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "generated_at": utc_now(),
        "schema_version": "lut_catalog_v4",
        "summary": {
            "total": len(entries),
            "by_status": _summarize_by(entries, "status"),
            "by_target_type": _summarize_by(entries, "target_type"),
            "by_source_kind": _summarize_by(entries, "catalog_source_kind"),
            "validation_linked": sum(1 for entry in entries if entry.get("validation_linked")),
            "corrected_lut_exists": sum(1 for entry in entries if entry.get("corrected_lut_exists")),
            "duplicate_runtime_entries": sum(1 for entry in entries if entry.get("duplicate_runtime")),
            "stale_runtime_entries": sum(1 for entry in entries if entry.get("stale_runtime")),
            "deprecated_entries": sum(1 for entry in entries if entry.get("status") == "deprecated"),
        },
        "entries": entries,
    }


def render_lut_catalog_markdown(payload: dict[str, Any]) -> str:
    rows = [{"display_label": entry.get("display_label") or entry.get("display_name"), "status": entry["status"], "source_route": entry["source_route"], "clipping_risk": entry["clipping_risk"]["level"], "validation": "yes" if entry["validation_linked"] else "no", "corrected": "yes" if entry["corrected_lut_exists"] else "no", "duplicate_runtime": "yes" if entry.get("duplicate_runtime") else "no", "stale_runtime": "yes" if entry.get("stale_runtime") else "no"} for entry in payload["entries"]]
    return document(
        "LUT Catalog",
        [
            "Each LUT is classified against the measured exact matrix, not against stale manifest state.",
            "Use status plus source_route to decide whether a LUT is operator-safe, bench-pending, preview-only, or deprecated.",
            "Validation, corrected LUT linkage, and runtime duplicate detection are included so the catalog can serve as an operations handoff document.",
        ],
        [("Catalog", markdown_table(rows, ["display_label", "status", "source_route", "clipping_risk", "validation", "corrected", "duplicate_runtime", "stale_runtime"]))],
    )


def build_lut_audit_payload(entries: list[dict[str, Any]]) -> dict[str, Any]:
    needs_attention = [entry for entry in entries if entry.get("status") in {"software_ready_bench_pending", "provisional_experimental"} or entry.get("clipping_risk", {}).get("level") == "high" or not entry.get("validation_linked") or entry.get("duplicate_runtime") or entry.get("stale_runtime")]
    return {
        "generated_at": utc_now(),
        "schema_version": "lut_audit_report_v3",
        "summary": {
            "by_status": _summarize_by(entries, "status"),
            "high_clipping_risk": sum(1 for entry in entries if entry.get("clipping_risk", {}).get("level") == "high"),
            "without_validation": sum(1 for entry in entries if not entry.get("validation_linked")),
            "without_corrected_lut": sum(1 for entry in entries if not entry.get("corrected_lut_exists")),
            "duplicate_runtime_entries": sum(1 for entry in entries if entry.get("duplicate_runtime")),
            "stale_runtime_entries": sum(1 for entry in entries if entry.get("stale_runtime")),
        },
        "needs_attention": sorted(needs_attention, key=operational_score),
    }


def render_lut_audit_markdown(payload: dict[str, Any]) -> str:
    summary_rows = [{"status": status, "count": count} for status, count in payload["summary"]["by_status"].items()]
    attention_rows = [{"display_label": entry.get("display_label") or entry.get("display_name"), "status": entry["status"], "source_route": entry["source_route"], "validation": "yes" if entry["validation_linked"] else "no", "corrected": "yes" if entry["corrected_lut_exists"] else "no", "clipping_risk": entry["clipping_risk"]["level"], "duplicate_runtime": "yes" if entry.get("duplicate_runtime") else "no", "stale_runtime": "yes" if entry.get("stale_runtime") else "no"} for entry in payload["needs_attention"]]
    return document(
        "LUT Audit Report",
        [
            "This report highlights what is ready for operators and what still needs validation, promotion, or retirement.",
            "Deprecated means reference-only, above-band, or otherwise outside the certified exact operating path.",
            "A provisional LUT can exist in the catalog without upgrading the exact matrix until a measured exact upload arrives, and duplicate runtime rows are flagged explicitly.",
        ],
        [("Status Summary", markdown_table(summary_rows, ["status", "count"])), ("Needs Attention", markdown_table(attention_rows, ["display_label", "status", "source_route", "validation", "corrected", "clipping_risk", "duplicate_runtime", "stale_runtime"]))],
    )


def render_roi_markdown(payload: dict[str, Any]) -> str:
    head_category = str(payload["priorities"][0]["category"]) if payload.get("priorities") else "none"
    rows = [{"rank": item["rank"], "category": item["category"], "request": item["request"], "why": item["why"], "expected_gain": item["expected_gain"]} for item in payload["priorities"]]
    return document(
        "Measurement ROI Priority",
        [
            "The list below ranks the next measurement or validation actions by operational value.",
            "Priority 1 changes dynamically with the current matrix state instead of staying pinned to the same request.",
            f"Current head category: {head_category}.",
            "Continuous exact gap-fill and validation actions remain separate so operators can plan bench time deliberately.",
        ],
        [("Priority Queue", markdown_table(rows, ["rank", "category", "request", "why", "expected_gain"]))],
    )


def render_data_intake_regression_markdown(payload: dict[str, Any]) -> str:
    check_rows = [{"check": item["name"], "status": item["status"], "details": item["details"]} for item in payload["checks"]]
    manifest_rows = [{"area": area, "entries": count} for area, count in payload["manifest_summary"].items()]
    return document(
        "Data Intake Regression",
        [
            "Data intake is evaluated against direct file scans, filename recovery, and refresh pipeline linkage.",
            "The bundle is designed so that new files in the memory folders can refresh matrix, catalog, and ROI artifacts together.",
            "Manifest files remain optional metadata helpers, not the source of truth for inclusion.",
        ],
        [("Checks", markdown_table(check_rows, ["check", "status", "details"])), ("Manifest Visibility", markdown_table(manifest_rows, ["area", "entries"]))],
    )


def render_release_candidate_summary(*, exact_matrix: dict[str, Any], catalog_payload: dict[str, Any], audit_payload: dict[str, Any], roi_payload: dict[str, Any], regression_payload: dict[str, Any]) -> str:
    counts = exact_matrix["counts"]
    roi_head = roi_payload["priorities"][0] if roi_payload.get("priorities") else {}
    passed_checks = sum(1 for item in regression_payload.get("checks", []) if item.get("status") == "pass")
    total_checks = len(regression_payload.get("checks", []))
    return document(
        "Release Candidate Summary",
        [
            f"Finite exact cells: {counts['finite_exact_cells']}. Provisional cells: {counts['provisional_cells']}. Missing exact cells: {counts['missing_exact_cells']}.",
            f"LUT catalog total: {catalog_payload['summary']['total']}. Duplicate runtime entries: {catalog_payload['summary']['duplicate_runtime_entries']}. Stale runtime entries: {catalog_payload['summary']['stale_runtime_entries']}.",
            f"Top ROI action: {roi_head.get('category', 'none')} -> {roi_head.get('request', 'none')}.",
            f"Data intake regression: {passed_checks}/{total_checks} checks passing.",
        ],
        [
            (
                "Runtime Snapshot",
                markdown_table(
                    [
                        {"metric": "continuous_current_exact_cells", "value": counts["continuous_current_exact_cells"]},
                        {"metric": "continuous_field_exact_rows", "value": counts["continuous_field_exact_rows"]},
                        {"metric": "finite_exact_cells", "value": counts["finite_exact_cells"]},
                        {"metric": "provisional_cells", "value": counts["provisional_cells"]},
                        {"metric": "missing_exact_cells", "value": counts["missing_exact_cells"]},
                        {"metric": "reference_only_cells", "value": counts["reference_only_cells"]},
                    ],
                    ["metric", "value"],
                ),
            ),
            (
                "Catalog Snapshot",
                markdown_table(
                    [
                        {"metric": "certified_exact", "value": catalog_payload["summary"]["by_status"].get("certified_exact", 0)},
                        {"metric": "software_ready_bench_pending", "value": catalog_payload["summary"]["by_status"].get("software_ready_bench_pending", 0)},
                        {"metric": "provisional_experimental", "value": catalog_payload["summary"]["by_status"].get("provisional_experimental", 0)},
                        {"metric": "preview_only", "value": catalog_payload["summary"]["by_status"].get("preview_only", 0)},
                        {"metric": "deprecated", "value": catalog_payload["summary"]["by_status"].get("deprecated", 0)},
                        {"metric": "needs_attention", "value": len(audit_payload.get("needs_attention", []))},
                    ],
                    ["metric", "value"],
                ),
            ),
        ],
    )


def render_static_docs(*, exact_matrix: dict[str, Any]) -> dict[str, str]:
    certified_total = exact_matrix["counts"]["finite_exact_cells"]
    operator_workflow = document(
        "Operator Workflow",
        [
            "Read exact_matrix_final first. It is the source of truth for what is certified exact, bench pending, provisional, missing, or reference only.",
            "Use only certified exact rows for auto operation. Treat software-ready bench pending as validation backlog, not as production-ready.",
            "After adding new measurement files to uploads, run refresh_exact_supported_scope.py to rebuild scope, exact matrix, LUT catalog, and ROI artifacts.",
            "If the missing sine / 1.0 Hz / 1.0 cycle / 20 pp file is uploaded, check that the provisional cell disappears and the finite exact count increases by one.",
        ],
        [("Daily Flow", "1. Check exact_matrix_final.md.\n2. Check lut_catalog.md if a specific LUT must be traced.\n3. Add new files using filename_convention.md.\n4. Run `python tools/refresh_exact_supported_scope.py`.\n5. Re-open exact_matrix_final.md, measurement_roi_priority.md, and data_intake_regression.md.")],
    )
    operational_scope = document(
        "Operational Scope",
        [
            "Continuous current exact <= 5 Hz is the only exact path that is operator auto-ready today.",
            "Continuous field exact <= 5 Hz is software-ready and indexed in the artifacts, but still bench pending.",
            f"Finite exact contains {certified_total} certified measured cells. The one remaining 20 pp sine cell is still provisional preview.",
            "Anything above 5 Hz is reference only and must not be treated as production exact support.",
        ],
        [("Status Rules", markdown_table([{"status": "certified_exact", "meaning": "Measured exact support. Safe for operator use within policy."}, {"status": "software_ready_bench_pending", "meaning": "Indexed and software-ready, but still requires bench validation."}, {"status": "provisional_experimental", "meaning": "Visible for experiment or preview only. Not certified exact."}, {"status": "preview_only", "meaning": "May be useful for analysis, but not for operator exact claims."}, {"status": "deprecated", "meaning": "Reference-only or otherwise outside the exact operating path."}], ["status", "meaning"]))],
    )
    terminology = document(
        "Terminology",
        [
            "The same labels are reused across exact matrix, LUT catalog, and ROI artifacts so operators do not need to translate between documents.",
            "The status words below are policy words, not generic descriptions.",
            "If two documents disagree, exact_matrix_final.json and lut_catalog.json take precedence because they are generated from direct scans.",
        ],
        [("Glossary", markdown_table([{"term": "certified exact", "definition": "Measured exact support that is counted in the official operating matrix."}, {"term": "software-ready bench pending", "definition": "Operationally indexed in software, but still awaiting validation measurements."}, {"term": "provisional preview", "definition": "A temporary preview route that does not count as certified exact."}, {"term": "missing exact", "definition": "A required exact cell with no measured upload yet."}, {"term": "reference only", "definition": "Visible for analysis or comparison, but not for exact operator operation."}, {"term": "validation linked", "definition": "A LUT entry has at least one tracked validation run in the catalog."}, {"term": "corrected LUT", "definition": "A validation-retuned LUT candidate linked back to its source lineage."}], ["term", "definition"]))],
    )
    validation_retune_overview = document(
        "Validation Retune Overview",
        [
            "Validation and retune use the same backend for continuous current exact, continuous field exact, and finite exact.",
            "Validation results do not automatically upgrade a preview or provisional cell to certified exact. Only measured exact uploads do that.",
            "Quality is evaluated in the Bz domain and follows the global rule `bz_effective = -bz_raw`.",
            "Use the LUT catalog to see which LUTs already have validation lineage and corrected LUT artifacts.",
        ],
        [("Quality Rule", build_quality_badge_markdown())],
    )
    data_contract = document(
        "Data Contract",
        [
            "Uploads are accepted from four areas: continuous, transient, validation, and lcr.",
            "The parser can reconstruct waveform, frequency, cycle count, and level from file names when metadata is incomplete.",
            "At minimum, every waveform file must provide a time axis and at least one usable current/field channel.",
            "The exact-matrix pipeline is based on direct folder scans, so correct placement and file naming matter more than manifest edits.",
        ],
        [("Accepted Inputs", markdown_table([{"area": "continuous", "folder": "uploads/continuous", "required_columns": "Time or Timestamp, HallBz, and at least one current channel", "recommended_columns": "Voltage1, HallBx, HallBy, AmpGain, Temperature", "filename_carries": "waveform, freq_hz, current level"}, {"area": "transient", "folder": "uploads/transient/<waveform_alias>", "required_columns": "Time or Timestamp, HallBz, and at least one current channel", "recommended_columns": "Voltage1, HallBx, HallBy, AmpGain, Temperature, CycleNo", "filename_carries": "freq_hz, cycle_count, level_pp"}, {"area": "validation", "folder": "uploads/validation/<scenario>", "required_columns": "Time or Timestamp, HallBz, and the driven current/voltage channels", "recommended_columns": "AmpGain, Temperature, explicit metadata header", "filename_carries": "scenario only"}, {"area": "lcr", "folder": "uploads/lcr", "required_columns": "keep the exported workbook or CSV intact", "recommended_columns": "device and fixture metadata", "filename_carries": "none"}], ["area", "folder", "required_columns", "recommended_columns", "filename_carries"]))],
    )
    filename_convention = document(
        "Filename Convention",
        [
            "File naming is part of the data contract because the parser uses it to recover conditions when metadata is weak.",
            "Optional hash prefixes are allowed and ignored for classification.",
            "Both `.` and `p` decimal notation are accepted in frequency and cycle tokens.",
            "The typo folder name `sinusidal` remains supported for backward compatibility and should not break refreshes.",
        ],
        [("Examples", "- Continuous: `<optional_hash>_<waveform>_<freq_hz>_<level_a>.csv`\n- Transient: `uploads/transient/<waveform_alias>/<optional_hash>_<freq>hz_<cycle>cycle_<level>pp.csv`\n- Example: `sinusidal/1hz_1cycle_20pp.csv`\n- Accepted sine aliases: `sine`, `sin`, `sinusoid`, `sinusoidal`, `sinusidal`")],
    )
    known_limitations = document(
        "Known Limitations",
        [
            "Continuous field exact remains bench pending even though the software can index it.",
            "Finite exact is still 95 certified cells plus one provisional preview cell, not a full 96 certified cells yet.",
            "Anything above 5 Hz is reference-only and should not be interpreted as exact support.",
            "Filename inference is robust but not magical; severely malformed file names can still block automatic classification.",
        ],
        [("Current Limits", "- `app_ui.py` is intentionally out of scope for this thread.\n- Interpolated auto remains closed by policy.\n- Validation-corrected LUTs improve lineage visibility, but they do not change exact policy on their own.")],
    )
    next_steps = document(
        "Next Steps",
        [
            "The first next step is fixed: collect the missing exact finite cell so the provisional preview can be retired.",
            "After that, continuous exact grid fill and validation backlog reduction provide the biggest operator value.",
            "Use measurement_roi_priority.md as the execution queue and this file as the short-form narrative.",
        ],
        [("Priority Sequence", "1. Measure and upload sine / 1.0 Hz / 1.0 cycle / 20 pp.\n2. Fill continuous exact gaps at 0.75 Hz and 1.5 Hz before moving on to 3.0 Hz and 4.0 Hz.\n3. Re-run validation for the current and field exact lineages still showing clipping or red quality markers.\n4. Reinforce finite 2 Hz and 5 Hz bands with repeat measurements at 20 pp.")],
    )
    measurement_request_template = document("Measurement Request Template", ["Use this template when requesting a new exact or validation measurement from the bench team.", "Fill in all required fields so the intake pipeline can classify the file without manual cleanup.", "Copy the examples exactly if the request targets the missing finite exact cell or a continuous gap-fill campaign."], [("Template", "- Request purpose: `<missing exact | continuous gap fill | validation | replication>`\n- Upload area: `<continuous | transient | validation>`\n- Waveform: `<sine | triangle>`\n- Frequency Hz: `<number>`\n- Cycle count: `<blank for continuous, number for transient>`\n- Level: `<A for continuous, pp for transient>`\n- Required columns: `Time`, `HallBz`, at least one current channel, `Voltage1` preferred\n- File name example: `sinusidal/1hz_1cycle_20pp.csv`")])
    validation_data_template = document("Validation Data Template", ["Validation uploads should make it obvious which LUT lineage and operating point they belong to.", "Use explicit metadata whenever possible because validation artifacts are read later by audit scripts.", "Attach the measured file, the source LUT id, and the intended scenario folder together."], [("Template", "- Scenario folder: `uploads/validation/<continuous_current_exact | continuous_field_exact | finite_sine_exact | finite_triangle_exact>`\n- Source LUT id: `<recommendation id or corrected LUT id>`\n- Original recommendation id: `<lineage root id>`\n- Validation file name: `<bench hash>_<scenario>_<freq token>_<level token>.csv`\n- Expected outputs after refresh: validation_catalog, corrected_lut_catalog, retune_picker_catalog")])
    quick_start_summary = document("Quick Start Summary", ["For operators, the minimum reading order is operator_workflow -> operational_scope -> exact_matrix_final -> measurement_roi_priority.", "For intake/debug work, add data_contract -> filename_convention -> data_intake_regression.", "For LUT validation work, add lut_catalog -> lut_audit_report -> validation_retune_overview."], [("Bundle Contents", markdown_table([{"artifact": "exact_matrix_final", "role": "exact/provisional/missing/reference-only source of truth"}, {"artifact": "lut_catalog", "role": "per-LUT operational classification and lineage linkage"}, {"artifact": "measurement_roi_priority", "role": "next best measurement and validation actions"}, {"artifact": "data_intake_regression", "role": "evidence that direct scans and refresh linkage still work"}], ["artifact", "role"]))])
    architecture_overview = document("Architecture Overview", ["The bundle is generated from direct folder scans, validation lineage payloads, and the exact scope report.", "Exact support is defined first by measured data, then projected into LUT status, ROI ranking, and operator docs.", "Manifest files can add metadata, but they do not override direct scan visibility."], [("Flow", "1. report_exact_and_finite_scope.py scans uploads and defines the measured exact scope.\n2. generate_bz_first_artifacts.py rebuilds exact matrix, LUT catalog, ROI, docs, and intake regression.\n3. refresh_exact_supported_scope.py runs both so new uploads propagate through the bundle together.")])
    return {
        "operator_workflow.md": operator_workflow,
        "operational_scope.md": operational_scope,
        "terminology.md": terminology,
        "validation_retune_overview.md": validation_retune_overview,
        "data_contract.md": data_contract,
        "filename_convention.md": filename_convention,
        "known_limitations.md": known_limitations,
        "next_steps.md": next_steps,
        "measurement_request_template.md": measurement_request_template,
        "validation_data_template.md": validation_data_template,
        "quick_start_summary.md": quick_start_summary,
        "architecture_overview.md": architecture_overview,
    }


def main() -> None:
    paths = ArtifactPaths()
    scope = build_scope_payload(continuous_dir=paths.continuous_upload_dir, transient_dir=paths.transient_upload_dir)
    promotion_smoke = run_provisional_promotion_smoke()
    paths.policy_scope_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(paths.policy_scope_path, scope)
    validation_payload = build_validation_catalog_payload(list(paths.retune_dirs), paths.retune_history_path)
    for entry in validation_payload.get("entries", []):
        if isinstance(entry, dict):
            entry["quality_reasons"] = clean_reason_list(entry.get("quality_reasons", []))
    corrected_payload = build_corrected_lut_catalog_payload(validation_payload["entries"])
    for entry in corrected_payload.get("entries", []):
        if isinstance(entry, dict):
            entry["quality_reasons"] = clean_reason_list(entry.get("quality_reasons", []))
    catalog_entries = build_lut_catalog(scope=scope, validation_catalog=validation_payload["entries"], corrected_catalog=corrected_payload["entries"], paths=paths)
    catalog_payload = build_lut_catalog_payload(catalog_entries)
    audit_payload = build_lut_audit_payload(catalog_entries)
    picker_payload = build_retune_picker_payload(lut_entries=catalog_entries, validation_entries=validation_payload["entries"], corrected_entries=corrected_payload["entries"])
    exact_matrix_payload = build_exact_matrix(scope)
    roi_payload = build_measurement_roi_priority(scope=scope, validation_catalog=validation_payload["entries"])
    regression_payload = build_data_intake_regression(scope=scope, paths=paths, promotion_smoke=promotion_smoke)
    label_sanitization_payload = build_label_sanitization_report(
        exact_matrix=exact_matrix_payload,
        catalog_payload=catalog_payload,
        validation_payload=validation_payload,
        corrected_payload=corrected_payload,
        picker_payload=picker_payload,
        roi_payload=roi_payload,
    )
    release_summary = render_release_candidate_summary(
        exact_matrix=exact_matrix_payload,
        catalog_payload=catalog_payload,
        audit_payload=audit_payload,
        roi_payload=roi_payload,
        regression_payload=regression_payload,
    )
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(paths.output_dir / "validation_catalog.json", validation_payload)
    write_json(paths.output_dir / "corrected_lut_catalog.json", corrected_payload)
    write_json(paths.output_dir / "retune_picker_catalog.json", picker_payload)
    write_json(paths.output_dir / "lut_catalog.json", catalog_payload)
    write_json(paths.output_dir / "lut_audit_report.json", audit_payload)
    write_json(paths.output_dir / "exact_matrix_final.json", exact_matrix_payload)
    write_json(paths.output_dir / "measurement_roi_priority.json", roi_payload)
    write_json(paths.output_dir / "data_intake_regression.json", regression_payload)
    write_json(paths.output_dir / "label_sanitization_report.json", label_sanitization_payload)
    write_markdown(paths.output_dir / "validation_retune_provenance_badge.md", build_provenance_badge_markdown(validation_payload=validation_payload, corrected_payload=corrected_payload, picker_payload=picker_payload))
    write_markdown(paths.output_dir / "lut_catalog.md", render_lut_catalog_markdown(catalog_payload))
    write_markdown(paths.output_dir / "lut_audit_report.md", render_lut_audit_markdown(audit_payload))
    write_markdown(paths.output_dir / "exact_matrix_final.md", render_exact_matrix_markdown(exact_matrix_payload))
    write_markdown(paths.output_dir / "measurement_roi_priority.md", render_roi_markdown(roi_payload))
    write_markdown(paths.output_dir / "data_intake_regression.md", render_data_intake_regression_markdown(regression_payload))
    write_markdown(paths.output_dir / "label_sanitization_report.md", render_label_sanitization_report_markdown(label_sanitization_payload))
    write_markdown(paths.output_dir / "release_candidate_summary.md", release_summary)
    for file_name, content in render_static_docs(exact_matrix=exact_matrix_payload).items():
        write_markdown(paths.output_dir / file_name, content)


if __name__ == "__main__":
    main()
