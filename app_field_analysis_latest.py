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
for module_name in ("field_analysis.app_ui", "field_analysis.app_ui_snapshot"):
    try:
        run_app = import_module(module_name).run_app
        break
    except ModuleNotFoundError as exc:  # pragma: no cover - launcher guard
        LAST_ERROR = exc
else:  # pragma: no cover - launcher guard
    searched = ", ".join(str(path) for path in SRC_CANDIDATES)
    raise ModuleNotFoundError(
        "field_analysis latest app could not be imported. "
        f"Checked modules: field_analysis.app_ui, field_analysis.app_ui_snapshot. "
        f"Checked src candidates: {searched}"
    ) from LAST_ERROR


if __name__ == "__main__":
    run_app()
