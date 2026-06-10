from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_CANDIDATES = [
    PROJECT_ROOT / "src",
    PROJECT_ROOT.parent / "src",
]


def _prepend_sys_path(src_path: Path) -> None:
    resolved = str(src_path.resolve())
    filtered_paths: list[str] = []
    for existing in sys.path:
        try:
            if Path(existing).resolve() == src_path.resolve():
                continue
        except OSError:
            pass
        filtered_paths.append(existing)
    sys.path[:] = [resolved, *filtered_paths]


def _install_src_path_guardrails() -> None:
    for src_path in reversed(SRC_CANDIDATES):
        if src_path.exists():
            _prepend_sys_path(src_path)


_install_src_path_guardrails()

LAST_ERROR: Exception | None = None
try:
    run_quick_lut_v2_app = import_module("field_analysis.app_ui_v2").run_quick_lut_v2_app
except ModuleNotFoundError as exc:  # pragma: no cover - launcher guard
    LAST_ERROR = exc
    searched = ", ".join(str(path) for path in SRC_CANDIDATES)
    raise ModuleNotFoundError(
        "field_analysis Quick LUT v2 app could not be imported. "
        "Checked module: field_analysis.app_ui_v2. "
        f"Checked src candidates: {searched}"
    ) from LAST_ERROR


if __name__ == "__main__":
    run_quick_lut_v2_app()
