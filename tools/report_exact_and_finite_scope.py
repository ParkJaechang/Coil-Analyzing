from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_MAX_FREQ_HZ = 5.0
CONTINUOUS_TARGET_GRID_FREQS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
CONTINUOUS_TARGET_GRID_LEVELS = [5.0, 10.0, 20.0]
CONTINUOUS_TARGET_GRID_WAVEFORMS = ["sine", "triangle"]
CONTINUOUS_DIR = ROOT.parent / "outputs" / "field_analysis_app_state" / "uploads" / "continuous"
TRANSIENT_DIR = ROOT.parent / "outputs" / "field_analysis_app_state" / "uploads" / "transient"
OUTPUT_JSON = ROOT / "artifacts" / "policy_eval" / "exact_and_finite_scope.json"
OUTPUT_MD = ROOT / "artifacts" / "policy_eval" / "exact_and_finite_scope.md"

CONTINUOUS_PATTERN = re.compile(
    r"^(?:[0-9a-f]+[_-])?(?P<waveform>sine|sin|triangle|tri)[_\-\s]+(?P<freq>\d+(?:[.p]\d+)?)(?:hz)?[_\-\s]+(?P<level>\d+(?:[.p]\d+)?)"
    r"(?:a|app)?(?:[_\-\s].*)?\.csv$",
    re.IGNORECASE,
)
TRANSIENT_PATTERN = re.compile(
    r"^(?:[0-9a-f]+[_-])?(?P<freq>\d+(?:[.p]\d+)?)\s*hz[_\-\s]+(?P<cycle>\d+(?:[.p]\d+)?)\s*(?:cycle|cy)"
    r"[_\-\s]+(?P<level>\d+(?:[.p]\d+)?)\s*(?P<level_kind>pp|app|a)?(?:[_\-\s].*)?\.csv$",
    re.IGNORECASE,
)
SINE_TOKENS = ("sine", "sin", "sinusoid", "sinusoidal", "sinusidal")
TRIANGLE_TOKENS = ("triangle", "tri")
PROVISIONAL_FINITE_CELL = {
    "waveform": "sine",
    "freq_hz": 1.0,
    "cycles": 1.0,
    "level_pp_a": 20.0,
    "source_exact_level_pp_a": 10.0,
    "scale_ratio": 2.0,
}
PROVISIONAL_PLACEHOLDER_RELATIVE_PATH = "sinusidal/1hz_1cycle_20pp.csv"


def _format_decimal(value: float | int | None) -> str:
    if value is None:
        return ""
    numeric = float(value)
    if not np.isfinite(numeric):
        return ""
    return f"{numeric:g}"


def _markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    if not rows:
        return "_항목 없음_"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _to_float(value: str | float | int) -> float:
    return float(str(value).strip().lower().replace("p", "."))


def _iter_csv_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob("*.csv") if path.is_file())


def _normalized_relative_file(path_like: object) -> str:
    return str(path_like or "").replace("\\", "/").strip().lower()


def _is_provisional_placeholder_file(path_like: object) -> bool:
    return _normalized_relative_file(path_like).endswith(PROVISIONAL_PLACEHOLDER_RELATIVE_PATH)


def _infer_transient_waveform(path: Path) -> str | None:
    haystack_parts = []
    for part in path.parts:
        lowered = part.lower()
        haystack_parts.append(lowered)
        try:
            haystack_parts.append(Path(lowered).stem)
        except OSError:
            continue
    haystack = " ".join(haystack_parts)
    if any(token in haystack for token in TRIANGLE_TOKENS):
        return "triangle"
    if any(token in haystack for token in SINE_TOKENS):
        return "sine"
    return None


def _build_continuous_frame(directory: Path) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, object]] = []
    ignored_files: list[str] = []
    for path in _iter_csv_files(directory):
        match = CONTINUOUS_PATTERN.match(path.name)
        if match is None:
            ignored_files.append(path.relative_to(directory).as_posix())
            continue
        waveform_token = match.group("waveform").lower()
        waveform = "triangle" if waveform_token in {"triangle", "tri"} else "sine"
        rows.append(
            {
                "file": path.relative_to(directory).as_posix(),
                "waveform": waveform,
                "freq_hz": _to_float(match.group("freq")),
                "target_level_a": _to_float(match.group("level")),
            }
        )
    return pd.DataFrame(rows, columns=["file", "waveform", "freq_hz", "target_level_a"]), ignored_files


def _build_transient_frame(directory: Path) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, object]] = []
    ignored_files: list[str] = []
    for path in _iter_csv_files(directory):
        match = TRANSIENT_PATTERN.match(path.name)
        if match is None:
            ignored_files.append(path.relative_to(directory).as_posix())
            continue
        waveform = _infer_transient_waveform(path.relative_to(directory))
        if waveform is None:
            ignored_files.append(path.relative_to(directory).as_posix())
            continue
        rows.append(
            {
                "file": path.relative_to(directory).as_posix(),
                "waveform": waveform,
                "freq_hz": _to_float(match.group("freq")),
                "cycles": _to_float(match.group("cycle")),
                "level_pp_a": _to_float(match.group("level")),
                "level_kind": (match.group("level_kind") or "pp").lower(),
            }
        )
    return pd.DataFrame(rows, columns=["file", "waveform", "freq_hz", "cycles", "level_pp_a", "level_kind"]), ignored_files


def _summarize_continuous(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["waveform", "freq_hz", "levels_a", "cell_count"])
    return (
        frame.groupby(["waveform", "freq_hz"])["target_level_a"]
        .apply(lambda series: sorted({float(value) for value in series.tolist()}))
        .reset_index(name="levels_a")
        .assign(cell_count=lambda item: item["levels_a"].map(len))
        .sort_values(["waveform", "freq_hz"])
        .reset_index(drop=True)
    )


def _summarize_transient(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["waveform", "freq_hz", "cycles", "levels_pp_a", "cell_count"])
    return (
        frame.groupby(["waveform", "freq_hz", "cycles"])["level_pp_a"]
        .apply(lambda series: sorted({float(value) for value in series.tolist()}))
        .reset_index(name="levels_pp_a")
        .assign(cell_count=lambda item: item["levels_pp_a"].map(len))
        .sort_values(["waveform", "freq_hz", "cycles"])
        .reset_index(drop=True)
    )


def _build_waveform_scope(frame: pd.DataFrame, *, waveform: str, note: str) -> dict[str, object]:
    subset = frame[frame["waveform"] == waveform].copy()
    return {
        "waveform": waveform,
        "total_files": int(len(subset)),
        "freqs_hz": sorted(float(value) for value in subset["freq_hz"].dropna().unique().tolist()),
        "cycles": sorted(float(value) for value in subset["cycles"].dropna().unique().tolist()),
        "levels_pp_a": sorted(float(value) for value in subset["level_pp_a"].dropna().unique().tolist()),
        "summary": _summarize_transient(subset).to_dict(orient="records"),
        "note": note,
    }


def _is_provisional_finite_cell(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    return (
        frame["waveform"].astype(str).str.lower().eq(str(PROVISIONAL_FINITE_CELL["waveform"]))
        & np.isclose(pd.to_numeric(frame["freq_hz"], errors="coerce"), float(PROVISIONAL_FINITE_CELL["freq_hz"]), rtol=0.0, atol=1e-6)
        & np.isclose(pd.to_numeric(frame["cycles"], errors="coerce"), float(PROVISIONAL_FINITE_CELL["cycles"]), rtol=0.0, atol=1e-6)
        & np.isclose(pd.to_numeric(frame["level_pp_a"], errors="coerce"), float(PROVISIONAL_FINITE_CELL["level_pp_a"]), rtol=0.0, atol=1e-6)
    )


def _split_provisional_exact_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        empty = frame.iloc[0:0].copy()
        return empty, empty, empty, empty

    special_mask = _is_provisional_finite_cell(frame)
    special_frame = frame.loc[special_mask].copy()
    exact_frame = frame.loc[~special_mask].copy()
    if special_frame.empty:
        empty = special_frame.iloc[0:0].copy()
        return exact_frame, empty, empty, special_frame

    placeholder_mask = special_frame["file"].map(_is_provisional_placeholder_file)
    promoted_exact_frame = special_frame.loc[~placeholder_mask].copy()
    if promoted_exact_frame.empty:
        provisional_frame = special_frame.copy()
        promoted_exact_frame = special_frame.iloc[0:0].copy()
    else:
        provisional_frame = special_frame.iloc[0:0].copy()
        exact_frame = pd.concat([exact_frame, promoted_exact_frame], ignore_index=True)
    return exact_frame, provisional_frame, promoted_exact_frame, special_frame


def _count_unique_finite_recipes(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    recipe_keys = {
        (
            str(row["waveform"]),
            round(float(row["freq_hz"]), 6),
            round(float(row["cycles"]), 6),
            round(float(row["level_pp_a"]), 6),
        )
        for _, row in frame.iterrows()
    }
    return len(recipe_keys)


def _build_continuous_grid_gaps(frame: pd.DataFrame) -> list[dict[str, object]]:
    existing: set[tuple[str, float, float]] = set()
    if not frame.empty:
        existing = {
            (str(row["waveform"]), float(row["freq_hz"]), float(row["target_level_a"]))
            for _, row in frame.iterrows()
        }
    missing: list[dict[str, object]] = []
    for waveform, freq_hz, level_a in product(
        CONTINUOUS_TARGET_GRID_WAVEFORMS,
        CONTINUOUS_TARGET_GRID_FREQS,
        CONTINUOUS_TARGET_GRID_LEVELS,
    ):
        if (waveform, float(freq_hz), float(level_a)) in existing:
            continue
        missing.append(
            {
                "waveform": waveform,
                "freq_hz": float(freq_hz),
                "level_a": float(level_a),
            }
        )
    return missing


def _build_finite_missing_cells(frame: pd.DataFrame) -> list[dict[str, object]]:
    if frame.empty:
        return []
    freqs = sorted(float(value) for value in frame["freq_hz"].dropna().unique().tolist())
    cycles = sorted(float(value) for value in frame["cycles"].dropna().unique().tolist())
    levels = sorted(float(value) for value in frame["level_pp_a"].dropna().unique().tolist())
    existing = {
        (str(row["waveform"]), float(row["freq_hz"]), float(row["cycles"]), float(row["level_pp_a"]))
        for _, row in frame.iterrows()
    }
    missing: list[dict[str, object]] = []
    for waveform, freq_hz, cycle_count, level_pp in product(("sine", "triangle"), freqs, cycles, levels):
        if (waveform, float(freq_hz), float(cycle_count), float(level_pp)) in existing:
            continue
        missing.append(
            {
                "waveform": waveform,
                "freq_hz": float(freq_hz),
                "cycles": float(cycle_count),
                "level_pp_a": float(level_pp),
            }
        )
    return missing


def build_scope_payload(
    *,
    continuous_dir: Path = CONTINUOUS_DIR,
    transient_dir: Path = TRANSIENT_DIR,
) -> dict[str, object]:
    continuous_frame, continuous_ignored = _build_continuous_frame(continuous_dir)
    transient_frame, transient_ignored = _build_transient_frame(transient_dir)

    official_continuous = continuous_frame[
        pd.to_numeric(continuous_frame["freq_hz"], errors="coerce") <= OFFICIAL_MAX_FREQ_HZ
    ].copy()
    reference_continuous = continuous_frame[
        pd.to_numeric(continuous_frame["freq_hz"], errors="coerce") > OFFICIAL_MAX_FREQ_HZ
    ].copy()
    official_transient = transient_frame[
        pd.to_numeric(transient_frame["freq_hz"], errors="coerce") <= OFFICIAL_MAX_FREQ_HZ
    ].copy()

    exact_transient, provisional_frame, promoted_exact_frame, special_frame = _split_provisional_exact_rows(official_transient)

    official_continuous_summary = _summarize_continuous(official_continuous)
    reference_continuous_summary = _summarize_continuous(reference_continuous)
    exact_transient_summary = _summarize_transient(exact_transient)
    continuous_grid_gaps = _build_continuous_grid_gaps(official_continuous)
    finite_missing = _build_finite_missing_cells(exact_transient)
    promotion_state = "promoted_to_exact" if not promoted_exact_frame.empty else ("provisional_only" if not provisional_frame.empty else "missing")
    official_recipe_total = _count_unique_finite_recipes(exact_transient)

    provisional_preview_combinations: list[dict[str, object]] = []
    if provisional_frame.empty:
        provisional_preview_combinations = []
    else:
        provisional_preview_combinations = [
            {
                **PROVISIONAL_FINITE_CELL,
                "status": "provisional_preview",
                "measured_file_present": True,
                "source_files": provisional_frame["file"].tolist(),
                "promotion_rule": "Add a measured exact upload for sine / 1.0 Hz / 1.0 cycle / 20 pp with a non-placeholder file name to retire this preview cell.",
            }
        ]

    missing_exact_combinations: list[dict[str, object]] = []
    if promoted_exact_frame.empty:
        missing_exact_combinations = [
            {
                "waveform": "sine",
                "freq_hz": 1.0,
                "cycles": 1.0,
                "level_pp_a": 20.0,
                "status": "missing_exact",
                "current_route": "provisional_preview" if not provisional_frame.empty else "missing",
                "promotion_target": "96 exact recipes after measured upload arrives.",
            }
        ]

    sine_scope = _build_waveform_scope(
        exact_transient,
        waveform="sine",
        note=(
            "5 Hz 이하 finite sine은 exact recipe 기반으로 운영한다. "
            "단, 1.0 Hz / 1.0 cycle / 20 pp는 measured exact가 아니라 provisional preview로만 제공한다."
        ),
    )
    triangle_scope = _build_waveform_scope(
        exact_transient,
        waveform="triangle",
        note="5 Hz 이하 finite triangle은 exact recipe 기반으로 운영한다.",
    )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_support_band_hz": {"min": 0.25, "max": OFFICIAL_MAX_FREQ_HZ},
        "scan_sources": {
            "continuous_dir": continuous_dir.as_posix(),
            "transient_dir": transient_dir.as_posix(),
            "continuous_scan_count": int(len(continuous_frame)),
            "transient_scan_count": int(len(transient_frame)),
            "continuous_ignored_files": continuous_ignored,
            "transient_ignored_files": transient_ignored,
            "recognized_transient_waveform_aliases": {
                "sine": list(SINE_TOKENS),
                "triangle": list(TRIANGLE_TOKENS),
            },
        },
        "continuous_official_exact_scope": {
            "total_files": int(len(official_continuous)),
            "waveforms": sorted(official_continuous["waveform"].dropna().unique().tolist()),
            "summary": official_continuous_summary.to_dict(orient="records"),
            "operational_status": {
                "usable_auto": f"continuous + current target + exact support only, <= {OFFICIAL_MAX_FREQ_HZ:g} Hz",
                "preview_only": [
                    "continuous + current target + interpolated_in_hull",
                    "continuous + field target + interpolated_in_hull",
                    f"continuous exact support above {OFFICIAL_MAX_FREQ_HZ:g} Hz",
                ],
                "blocked": [
                    "interpolated_edge",
                    "out_of_hull",
                    "field target auto",
                ],
            },
        },
        "continuous_field_exact_scope": {
            "status": "software_ready_bench_pending",
            "source_basis": "continuous_official_exact_scope",
            "waveforms": sorted(official_continuous["waveform"].dropna().unique().tolist()),
            "freqs_hz": sorted(float(value) for value in official_continuous["freq_hz"].dropna().unique().tolist()),
            "summary": [
                {
                    "waveform": row["waveform"],
                    "freq_hz": row["freq_hz"],
                    "target_levels": "variable field target within hardware limits",
                    "status": "software_ready_bench_pending",
                    "bench_validation": "pending",
                }
                for row in official_continuous_summary.to_dict(orient="records")
            ],
        },
        "continuous_reference_above_band": {
            "total_files": int(len(reference_continuous)),
            "waveforms": sorted(reference_continuous["waveform"].dropna().unique().tolist()),
            "summary": reference_continuous_summary.to_dict(orient="records"),
            "status": "reference_only",
        },
        "continuous_exact_grid_candidates": {
            "waveforms": list(CONTINUOUS_TARGET_GRID_WAVEFORMS),
            "freqs_hz": list(CONTINUOUS_TARGET_GRID_FREQS),
            "levels_a": list(CONTINUOUS_TARGET_GRID_LEVELS),
            "missing_combinations": continuous_grid_gaps,
            "recommended_next_measurements": [
                {"waveform": "sine", "freq_hz": 0.75, "levels_a": [5.0, 10.0, 20.0]},
                {"waveform": "triangle", "freq_hz": 0.75, "levels_a": [5.0, 10.0, 20.0]},
                {"waveform": "sine", "freq_hz": 1.5, "levels_a": [5.0, 10.0, 20.0]},
                {"waveform": "triangle", "freq_hz": 1.5, "levels_a": [5.0, 10.0, 20.0]},
                {"waveform": "sine", "freq_hz": 3.0, "levels_a": [5.0, 10.0, 20.0]},
                {"waveform": "sine", "freq_hz": 4.0, "levels_a": [5.0, 10.0, 20.0]},
            ],
        },
        "finite_exact_scope": sine_scope,
        "finite_triangle_exact_scope": triangle_scope,
        "finite_all_exact_scope": {
            "total_files": int(len(exact_transient)),
            "waveforms": sorted(exact_transient["waveform"].dropna().unique().tolist()),
            "summary": exact_transient_summary.to_dict(orient="records"),
            "official_recipe_total": int(official_recipe_total),
            "missing_exact_combinations": missing_exact_combinations,
            "provisional_preview_combinations": provisional_preview_combinations,
            "full_grid_missing_combinations": finite_missing,
            "promotion_status": {
                "waveform": "sine",
                "freq_hz": 1.0,
                "cycles": 1.0,
                "level_pp_a": 20.0,
                "state": promotion_state,
                "placeholder_source_files": [path for path in special_frame["file"].tolist() if _is_provisional_placeholder_file(path)],
                "promoted_exact_source_files": promoted_exact_frame["file"].tolist(),
                "measured_exact_available": bool(not promoted_exact_frame.empty),
            },
        },
    }
    return payload


def run_provisional_promotion_smoke() -> dict[str, object]:
    with TemporaryDirectory(prefix="matrix_runtime_smoke_") as temp_dir:
        root = Path(temp_dir)
        transient_dir = root / "uploads" / "transient"
        canonical_dir = transient_dir / "sinusidal"
        canonical_dir.mkdir(parents=True, exist_ok=True)
        (canonical_dir / "1hz_1cycle_10pp.csv").touch()
        (canonical_dir / "1hz_1cycle_20pp.csv").touch()

        baseline = build_scope_payload(
            continuous_dir=root / "uploads" / "continuous",
            transient_dir=transient_dir,
        )

        promoted_path = canonical_dir / "abcd1234_1hz_1cycle_20pp.csv"
        promoted_path.touch()
        promoted = build_scope_payload(
            continuous_dir=root / "uploads" / "continuous",
            transient_dir=transient_dir,
        )

    baseline_scope = baseline["finite_all_exact_scope"]
    promoted_scope = promoted["finite_all_exact_scope"]
    baseline_count = int(baseline_scope["official_recipe_total"])
    promoted_count = int(promoted_scope["official_recipe_total"])
    promoted_pass = (
        len(baseline_scope["provisional_preview_combinations"]) == 1
        and len(baseline_scope["missing_exact_combinations"]) == 1
        and len(promoted_scope["provisional_preview_combinations"]) == 0
        and len(promoted_scope["missing_exact_combinations"]) == 0
        and promoted_scope["promotion_status"]["state"] == "promoted_to_exact"
        and promoted_count == baseline_count + 1
    )
    return {
        "pass": bool(promoted_pass),
        "baseline": {
            "official_recipe_total": baseline_count,
            "provisional_count": len(baseline_scope["provisional_preview_combinations"]),
            "missing_count": len(baseline_scope["missing_exact_combinations"]),
            "promotion_state": baseline_scope["promotion_status"]["state"],
        },
        "promoted": {
            "official_recipe_total": promoted_count,
            "provisional_count": len(promoted_scope["provisional_preview_combinations"]),
            "missing_count": len(promoted_scope["missing_exact_combinations"]),
            "promotion_state": promoted_scope["promotion_status"]["state"],
        },
        "details": (
            "canonical placeholder keeps the target cell provisional until a non-placeholder "
            "1hz_1cycle_20pp upload arrives, then the cell is promoted to exact."
        ),
    }


def _scope_markdown(payload: dict[str, object]) -> str:
    continuous_summary = payload["continuous_official_exact_scope"]["summary"]
    field_summary = payload["continuous_field_exact_scope"]["summary"]
    finite_sine_summary = payload["finite_exact_scope"]["summary"]
    finite_triangle_summary = payload["finite_triangle_exact_scope"]["summary"]
    missing_continuous = payload["continuous_exact_grid_candidates"]["missing_combinations"]
    missing_exact = payload["finite_all_exact_scope"]["missing_exact_combinations"]
    provisional = payload["finite_all_exact_scope"]["provisional_preview_combinations"]
    reference_only = payload["continuous_reference_above_band"]["summary"]

    continuous_rows = [
        {
            "waveform": row["waveform"],
            "freq_hz": row["freq_hz"],
            "levels_a": ", ".join(_format_decimal(value) for value in row["levels_a"]),
            "cells": row["cell_count"],
            "status": "certified_exact",
        }
        for row in continuous_summary
    ]
    field_rows = [
        {
            "waveform": row["waveform"],
            "freq_hz": row["freq_hz"],
            "target_levels": row["target_levels"],
            "status": row["status"],
            "bench_validation": row["bench_validation"],
        }
        for row in field_summary
    ]
    finite_rows = [
        {
            "waveform": row["waveform"],
            "freq_hz": row["freq_hz"],
            "cycles": row["cycles"],
            "levels_pp_a": ", ".join(_format_decimal(value) for value in row["levels_pp_a"]),
            "cells": row["cell_count"],
            "status": "exact",
        }
        for row in [*finite_sine_summary, *finite_triangle_summary]
    ]
    continuous_missing_rows = [
        {
            "waveform": item["waveform"],
            "freq_hz": item["freq_hz"],
            "level_a": item["level_a"],
        }
        for item in missing_continuous
    ]
    reference_rows = [
        {
            "waveform": row["waveform"],
            "freq_hz": row["freq_hz"],
            "levels_a": ", ".join(_format_decimal(value) for value in row["levels_a"]),
        }
        for row in reference_only
    ]

    lines = [
        "# Exact And Finite Scope",
        "",
        "- 운영 기준은 Bz-first exact-matrix이다.",
        "- continuous/current exact <= 5 Hz는 auto 운영, continuous/field exact <= 5 Hz는 software-ready bench pending이다.",
        "- finite exact <= 5 Hz는 measured exact recipe만 exact로 간주한다.",
        "- sine / 1.0 Hz / 1.0 cycle / 20 pp는 exact가 아니라 provisional preview이며, measured exact 업로드가 들어오면 승격한다.",
        "",
        "## Continuous Current Exact Matrix",
        "",
        _markdown_table(continuous_rows, ["waveform", "freq_hz", "levels_a", "cells", "status"]),
        "",
        "## Continuous Field Exact Matrix",
        "",
        _markdown_table(field_rows, ["waveform", "freq_hz", "target_levels", "status", "bench_validation"]),
        "",
        "## Continuous Exact Expansion Candidates",
        "",
        _markdown_table(continuous_missing_rows, ["waveform", "freq_hz", "level_a"]),
        "",
        "## Finite Exact Matrix",
        "",
        _markdown_table(finite_rows, ["waveform", "freq_hz", "cycles", "levels_pp_a", "cells", "status"]),
        "",
        "## Missing Exact Cell",
        "",
        _markdown_table(missing_exact, ["waveform", "freq_hz", "cycles", "level_pp_a", "status", "current_route", "promotion_target"]),
        "",
        "## Provisional Preview Cell",
        "",
        _markdown_table(
            provisional,
            ["waveform", "freq_hz", "cycles", "level_pp_a", "source_exact_level_pp_a", "scale_ratio", "status", "measured_file_present"],
        ),
        "",
        "## Reference-only Continuous Cells (> 5 Hz)",
        "",
        _markdown_table(reference_rows, ["waveform", "freq_hz", "levels_a"]),
        "",
        "## Intake Robustness Notes",
        "",
        f"- continuous scan count: `{payload['scan_sources']['continuous_scan_count']}`",
        f"- transient scan count: `{payload['scan_sources']['transient_scan_count']}`",
        "- transient waveform aliases accepted: `sine`, `sin`, `sinusoid`, `sinusoidal`, `sinusidal`, `triangle`, `tri`",
        f"- continuous ignored files: `{len(payload['scan_sources']['continuous_ignored_files'])}`",
        f"- transient ignored files: `{len(payload['scan_sources']['transient_ignored_files'])}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    payload = build_scope_payload()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_MD.write_text(_scope_markdown(payload), encoding="utf-8")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
