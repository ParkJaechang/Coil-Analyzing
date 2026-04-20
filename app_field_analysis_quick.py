from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_CANDIDATES = [
    PROJECT_ROOT / "src",
    PROJECT_ROOT.parent / "src",
]

for src_path in SRC_CANDIDATES:
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

LAST_ERROR: Exception | None = None
for module_name in ("field_analysis.app_ui", "field_analysis.app_ui_snapshot"):
    try:
        run_quick_lut_app = import_module(module_name).run_quick_lut_app
        break
    except ModuleNotFoundError as exc:  # pragma: no cover - launcher guard
        LAST_ERROR = exc
else:  # pragma: no cover - launcher guard
    searched = ", ".join(str(path) for path in SRC_CANDIDATES)
    raise ModuleNotFoundError(
        "field_analysis quick app could not be imported. "
        f"Checked modules: field_analysis.app_ui, field_analysis.app_ui_snapshot. "
        f"Checked src candidates: {searched}"
    ) from LAST_ERROR


if __name__ == "__main__":
    run_quick_lut_app()
