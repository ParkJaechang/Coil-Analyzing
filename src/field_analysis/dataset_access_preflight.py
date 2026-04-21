from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .dataset_library import _resolve_dataset_entry_path, load_dataset_manifest


MAX_PROBLEM_SAMPLES = 5


def check_selected_paths_access(
    dataset_root: str | Path | None,
    selected_relative_paths: Iterable[str | Path],
) -> dict[str, Any]:
    checks = [_check_relative_path(dataset_root, relative_path) for relative_path in _normalize_paths(selected_relative_paths)]
    return _summarize_checks(checks)


def build_manifest_entry_access_summary(
    dataset_root: str | Path | None,
    manifest_entries: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    entries = manifest_entries
    if entries is None and dataset_root:
        entries = load_dataset_manifest(dataset_root).get("files", [])
    entries = entries or []
    checks = [
        _check_relative_path(dataset_root, str(entry.get("path") or ""))
        for entry in entries
        if str(entry.get("path") or "").strip()
    ]
    return _summarize_checks(checks)


def build_dataset_access_preflight(
    *,
    dataset_root: str | Path | None,
    selected_paths_by_mode: Mapping[str, Sequence[str | Path]] | None = None,
    manifest_entries: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized_root = str(dataset_root or "").strip()
    root_path = Path(normalized_root).expanduser() if normalized_root else None
    dataset_root_exists = bool(root_path and root_path.is_dir())

    selected_by_mode = {
        "continuous": check_selected_paths_access(
            normalized_root,
            (selected_paths_by_mode or {}).get("continuous", []),
        ),
        "finite_cycle": check_selected_paths_access(
            normalized_root,
            (selected_paths_by_mode or {}).get("finite_cycle", []),
        ),
    }
    selected_combined = _summarize_checks(
        selected_by_mode["continuous"]["checks"] + selected_by_mode["finite_cycle"]["checks"]
    )
    manifest_summary = build_manifest_entry_access_summary(
        normalized_root,
        manifest_entries=manifest_entries,
    )

    return {
        "dataset_root": normalized_root,
        "dataset_root_exists": dataset_root_exists,
        "selected_by_mode": selected_by_mode,
        "selected": selected_combined,
        "manifest": manifest_summary,
    }


def _check_relative_path(
    dataset_root: str | Path | None,
    relative_path: str | Path,
) -> dict[str, str]:
    normalized_path = str(relative_path or "").strip().replace("\\", "/")
    if not normalized_path:
        return {
            "path": "",
            "status": "blocked",
            "message": "Dataset relative path is required",
        }

    try:
        resolved_path = _resolve_dataset_entry_path(str(dataset_root or ""), normalized_path)
    except (FileNotFoundError, NotADirectoryError) as exc:
        return {
            "path": normalized_path,
            "status": "missing",
            "message": str(exc),
        }
    except ValueError as exc:
        return {
            "path": normalized_path,
            "status": "blocked",
            "message": str(exc),
        }

    try:
        with resolved_path.open("rb") as handle:
            handle.read(1)
    except PermissionError as exc:
        return {
            "path": normalized_path,
            "status": "unreadable",
            "message": str(exc),
        }
    except OSError as exc:
        return {
            "path": normalized_path,
            "status": "unreadable",
            "message": str(exc),
        }

    return {
        "path": normalized_path,
        "status": "ok",
        "message": str(resolved_path),
    }


def _normalize_paths(paths: Iterable[str | Path]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for path in paths:
        candidate = str(path or "").strip().replace("\\", "/")
        if not candidate or candidate in seen:
            continue
        normalized.append(candidate)
        seen.add(candidate)
    return normalized


def _summarize_checks(checks: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    status_counts = {
        "ok": 0,
        "missing": 0,
        "unreadable": 0,
        "blocked": 0,
    }
    problem_samples: list[dict[str, str]] = []
    for check in checks:
        status = str(check.get("status") or "blocked")
        if status not in status_counts:
            status = "blocked"
        status_counts[status] += 1
        if status != "ok" and len(problem_samples) < MAX_PROBLEM_SAMPLES:
            problem_samples.append(
                {
                    "path": str(check.get("path") or ""),
                    "status": status,
                    "message": str(check.get("message") or ""),
                }
            )

    return {
        "checked_count": len(checks),
        "ok_count": status_counts["ok"],
        "missing_count": status_counts["missing"],
        "unreadable_count": status_counts["unreadable"],
        "blocked_count": status_counts["blocked"],
        "unavailable_count": status_counts["missing"] + status_counts["unreadable"] + status_counts["blocked"],
        "problem_samples": problem_samples,
        "checks": [dict(check) for check in checks],
    }


__all__ = [
    "build_dataset_access_preflight",
    "build_manifest_entry_access_summary",
    "check_selected_paths_access",
]
