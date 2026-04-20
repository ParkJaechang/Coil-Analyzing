from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_DIR = REPO_ROOT / ".coil_analyzer"
DEFAULT_SETTINGS_FILE = "settings.json"
DATASET_MANIFEST_FILE = "coil_analyzing_manifest.json"
SUPPORTED_DATASET_SUFFIXES = {".csv", ".txt", ".xlsx", ".xlsm", ".xls"}


def get_default_settings_path(repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root).resolve() if repo_root is not None else REPO_ROOT
    return root / ".coil_analyzer" / DEFAULT_SETTINGS_FILE


def load_dataset_library_settings(
    settings_path: str | Path | None = None,
) -> dict[str, Any]:
    resolved_path = Path(settings_path) if settings_path is not None else get_default_settings_path()
    default_payload = {
        "dataset_root": "",
    }
    if not resolved_path.exists():
        return default_payload
    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_payload
    if not isinstance(payload, dict):
        return default_payload
    dataset_root = str(payload.get("dataset_root") or "").strip()
    return {
        "dataset_root": dataset_root,
    }


def save_dataset_library_settings(
    settings: dict[str, Any],
    settings_path: str | Path | None = None,
) -> dict[str, Any]:
    resolved_path = Path(settings_path) if settings_path is not None else get_default_settings_path()
    payload = {
        "dataset_root": str(settings.get("dataset_root") or "").strip(),
    }
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def get_dataset_manifest_path(dataset_root: str | Path) -> Path:
    return Path(dataset_root).resolve() / DATASET_MANIFEST_FILE


def load_dataset_manifest(dataset_root: str | Path) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    manifest_path = get_dataset_manifest_path(root)
    default_payload = {
        "dataset_root": str(root),
        "manifest_path": str(manifest_path),
        "generated_at": None,
        "file_count": 0,
        "counts": {
            "continuous": 0,
            "finite_cycle": 0,
            "unknown": 0,
        },
        "files": [],
    }
    if not manifest_path.exists():
        return default_payload
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_payload
    if not isinstance(payload, dict):
        return default_payload
    files = payload.get("files")
    if not isinstance(files, list):
        files = []
    counts = payload.get("counts")
    if not isinstance(counts, dict):
        counts = _summarize_dataset_entries(files)
    return {
        "dataset_root": str(payload.get("dataset_root") or root),
        "manifest_path": str(payload.get("manifest_path") or manifest_path),
        "generated_at": payload.get("generated_at"),
        "file_count": int(payload.get("file_count") or len(files)),
        "counts": {
            "continuous": int(counts.get("continuous") or 0),
            "finite_cycle": int(counts.get("finite_cycle") or 0),
            "unknown": int(counts.get("unknown") or 0),
        },
        "files": [entry for entry in files if isinstance(entry, dict)],
    }


def build_dataset_manifest(dataset_root: str | Path) -> dict[str, Any]:
    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")

    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name == DATASET_MANIFEST_FILE:
            continue
        if path.suffix.lower() not in SUPPORTED_DATASET_SUFFIXES:
            continue
        relative_path = path.relative_to(root)
        stat_result = path.stat()
        entries.append(
            {
                "path": relative_path.as_posix(),
                "name": path.name,
                "size_bytes": int(stat_result.st_size),
                "modified_time": datetime.fromtimestamp(stat_result.st_mtime, tz=timezone.utc).isoformat(),
                "content_hash": _hash_file(path),
                "dataset_mode": _infer_dataset_mode(relative_path),
            }
        )

    counts = _summarize_dataset_entries(entries)
    manifest_path = get_dataset_manifest_path(root)
    payload = {
        "dataset_root": str(root),
        "manifest_path": str(manifest_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(entries),
        "counts": counts,
        "files": entries,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _infer_dataset_mode(relative_path: Path) -> str:
    tokens: set[str] = set()
    for part in relative_path.parts[:-1]:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(part).strip().lower()).strip("_")
        if normalized:
            tokens.add(normalized)

    if tokens & {"continuous", "continuous_cycle", "steady", "steady_state"}:
        return "continuous"
    if tokens & {"finite", "finite_cycle", "transient"}:
        return "finite_cycle"
    return "unknown"


def _summarize_dataset_entries(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "continuous": 0,
        "finite_cycle": 0,
        "unknown": 0,
    }
    for entry in entries:
        dataset_mode = str(entry.get("dataset_mode") or "unknown")
        if dataset_mode not in counts:
            dataset_mode = "unknown"
        counts[dataset_mode] += 1
    return counts


__all__ = [
    "DATASET_MANIFEST_FILE",
    "SUPPORTED_DATASET_SUFFIXES",
    "build_dataset_manifest",
    "get_dataset_manifest_path",
    "get_default_settings_path",
    "load_dataset_library_settings",
    "load_dataset_manifest",
    "save_dataset_library_settings",
]
