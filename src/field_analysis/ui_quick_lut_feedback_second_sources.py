from __future__ import annotations

from pathlib import Path

from .ui_quick_lut_feedback_selection import classify_feedback_csv_candidate

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ACTUAL_DRIVE_UPLOAD_DIR = REPO_ROOT / "outputs" / "field_analysis_app_state" / "uploads"
DEFAULT_SECOND_ACTUAL_DRIVE_UPLOAD_DIR = DEFAULT_ACTUAL_DRIVE_UPLOAD_DIR
PRIMARY_ACTUAL_DRIVE_RESULT_DIRS = (
    DEFAULT_ACTUAL_DRIVE_UPLOAD_DIR / "Continuous_1st_Result",
    DEFAULT_ACTUAL_DRIVE_UPLOAD_DIR / "Transient_1st_Result",
)
LEGACY_ACTUAL_DRIVE_RESULT_DIRS = (DEFAULT_ACTUAL_DRIVE_UPLOAD_DIR / "2nd",)


def _default_scan_roots() -> list[Path]:
    roots = [path for path in PRIMARY_ACTUAL_DRIVE_RESULT_DIRS if path.exists()]
    roots.extend(path for path in LEGACY_ACTUAL_DRIVE_RESULT_DIRS if path.exists())
    return roots or [DEFAULT_ACTUAL_DRIVE_UPLOAD_DIR]


def _source_label_for_path(path: Path) -> str:
    parts = set(path.parts)
    if "Continuous_1st_Result" in parts:
        return "Continuous 1차 구동 결과 폴더"
    if "Transient_1st_Result" in parts:
        return "Finite 1차 구동 결과 폴더"
    if "2nd" in parts:
        return "Legacy 2차 입력 폴더"
    return "1차 구동 결과 폴더"


def scan_second_actual_drive_upload_folder(
    folder: str | Path | None = None,
    *,
    run_label: str = "first_run",
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Find first-run measured result CSVs for finite second modeling.

    The current workflow uses the upload root as the configured drop folder; legacy
    `uploads/2nd` files are still discovered because the scan is recursive.
    """
    roots = [Path(folder)] if folder is not None else _default_scan_roots()
    candidates: list[dict[str, object]] = []
    actual_count = 0
    final_lut_count = 0
    unsupported_count = 0
    existing_roots = [root for root in roots if root.exists()]
    if not existing_roots:
        return candidates, {
            "folder_path": str(roots[0] if roots else DEFAULT_ACTUAL_DRIVE_UPLOAD_DIR),
            "folder_paths": [str(root) for root in roots],
            "folder_exists": False,
            "file_count": 0,
            "actual_drive_candidate_count": 0,
            "final_voltage_lut_count": 0,
            "unsupported_schema_count": 0,
        }

    files: list[Path] = []
    for root in existing_roots:
        root_files = sorted(path for path in root.rglob("*.csv") if path.is_file())
        files.extend(root_files)
        for path in root_files:
            data = path.read_bytes()
            info = classify_feedback_csv_candidate(path.name, data)
            rel_path = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name
            display_rel_path = f"{root.name}/{rel_path}" if folder is None else rel_path
            file_type = str(info.get("file_type") or "unknown")
            if file_type == "actual_drive_result":
                actual_count += 1
            elif file_type == "final_voltage_lut":
                final_lut_count += 1
            else:
                unsupported_count += 1
            candidates.append(
                {
                    "candidate_id": f"actual_drive_folder:{display_rel_path}",
                    "source_kind": "actual_drive_folder",
                    "source_label": _source_label_for_path(path),
                    "filename": path.name,
                    "original_filename": path.name,
                    "relative_path": display_rel_path,
                    "source_path": str(path),
                    "csv_bytes": data,
                    "run_label": run_label,
                    **info,
                }
            )
    return candidates, {
        "folder_path": str(existing_roots[0]),
        "folder_paths": [str(root) for root in existing_roots],
        "folder_exists": True,
        "file_count": len(files),
        "actual_drive_candidate_count": actual_count,
        "final_voltage_lut_count": final_lut_count,
        "unsupported_schema_count": unsupported_count,
    }


def count_exact_matches(
    candidates: list[dict[str, object]],
    *,
    waveform_type: str | None,
    freq_hz: float | None,
    cycle_count: float | None,
) -> int:
    from .ui_quick_lut_feedback_selection import candidate_matches

    return sum(
        1
        for candidate in candidates
        if candidate.get("file_type") == "actual_drive_result"
        and candidate_matches(candidate, waveform_type=waveform_type, freq_hz=freq_hz, cycle_count=cycle_count)
    )
