from __future__ import annotations

from pathlib import Path
from typing import Any

from .ui_quick_lut_feedback_selection import classify_feedback_csv_candidate

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SECOND_ACTUAL_DRIVE_UPLOAD_DIR = REPO_ROOT / "outputs" / "field_analysis_app_state" / "uploads" / "2nd"


def scan_second_actual_drive_upload_folder(
    folder: str | Path | None = None,
    *,
    run_label: str = "first_run",
) -> tuple[list[dict[str, object]], dict[str, object]]:
    root = Path(folder) if folder is not None else DEFAULT_SECOND_ACTUAL_DRIVE_UPLOAD_DIR
    candidates: list[dict[str, object]] = []
    actual_count = 0
    final_lut_count = 0
    unsupported_count = 0
    if not root.exists():
        return candidates, {
            "folder_path": str(root),
            "folder_exists": False,
            "file_count": 0,
            "actual_drive_candidate_count": 0,
            "final_voltage_lut_count": 0,
            "unsupported_schema_count": 0,
        }
    files = sorted(path for path in root.glob("*.csv") if path.is_file())
    for path in files:
        data = path.read_bytes()
        info = classify_feedback_csv_candidate(path.name, data)
        file_type = str(info.get("file_type") or "unknown")
        if file_type == "actual_drive_result":
            actual_count += 1
        elif file_type == "final_voltage_lut":
            final_lut_count += 1
        else:
            unsupported_count += 1
        candidates.append(
            {
                "candidate_id": f"second_folder:{path.name}",
                "source_kind": "second_folder",
                "source_label": "2차 모델링용 실구동 결과 폴더",
                "filename": path.name,
                "original_filename": path.name,
                "source_path": str(path),
                "csv_bytes": data,
                "run_label": run_label,
                **info,
            }
        )
    return candidates, {
        "folder_path": str(root),
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
