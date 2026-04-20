from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_PYTHON_FILE_LINES = 600
IGNORED_DIR_NAMES = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "artifacts",
    "outputs",
    "venv",
}

# Legacy oversized modules are temporarily allowed while the codebase is
# decomposed into smaller feature-focused modules.
LEGACY_OVERSIZED_ALLOWLIST: dict[str, str] = {
    "app.py": "Legacy standalone Streamlit app entrypoint.",
    "src/field_analysis/app_ui_snapshot.py": (
        "Legacy oversized UI shell. Do not add new feature bodies here; "
        "extract new UI logic into ui_*.py modules."
    ),
    "src/field_analysis/compensation.py": "Legacy compensation runtime with multiple code paths.",
    "src/field_analysis/lut.py": "Legacy LUT runtime that still needs follow-up module extraction.",
    "src/field_analysis/parser.py": "Legacy parser module with multiple import flows.",
    "src/field_analysis/plotting.py": "Legacy plotting bundle awaiting feature splits.",
    "src/field_analysis/recommendation_surface_runtime.py": "Legacy recommendation surface runtime bundle.",
    "src/field_analysis/validation_retune.py": "Legacy retune runtime bundle.",
    "src/field_analysis/validation_retune_catalog.py": "Legacy catalog helper bundle.",
    "src/field_analysis/plant_model/harmonic_surface.py": "Legacy harmonic surface model bundle.",
}


def _iter_python_files() -> list[Path]:
    paths: list[Path] = []
    for path in REPO_ROOT.rglob("*.py"):
        if any(part in IGNORED_DIR_NAMES for part in path.parts):
            continue
        paths.append(path)
    return sorted(paths)


def _line_count(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="ignore")
    return len(text.splitlines())


def test_allowlist_entries_still_exist() -> None:
    missing = [
        relative_path
        for relative_path in LEGACY_OVERSIZED_ALLOWLIST
        if not (REPO_ROOT / relative_path).is_file()
    ]
    assert not missing, f"Stale oversized-file allowlist entries: {missing}"


def test_python_files_stay_within_size_guardrail_or_allowlist() -> None:
    oversized_paths: list[str] = []

    for path in _iter_python_files():
        relative_path = path.relative_to(REPO_ROOT).as_posix()
        if _line_count(path) <= MAX_PYTHON_FILE_LINES:
            continue
        if relative_path in LEGACY_OVERSIZED_ALLOWLIST:
            continue
        oversized_paths.append(relative_path)

    assert not oversized_paths, (
        f"Python files over {MAX_PYTHON_FILE_LINES} lines must be split by feature: "
        f"{oversized_paths}"
    )
