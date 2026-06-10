from __future__ import annotations

import ast
import py_compile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _function_names(path: Path) -> set[str]:
    module = ast.parse(_read(path), filename=str(path))
    return {
        node.name
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_quick_lut_legacy_entrypoint_preserves_existing_app_contract() -> None:
    path = REPO_ROOT / "app_field_analysis_quick_legacy.py"

    assert path.is_file()
    py_compile.compile(str(path), doraise=True)
    source = _read(path)
    assert "run_quick_lut_app" in source
    assert "field_analysis.app_ui" in source
    assert "field_analysis.app_ui_snapshot" in source


def test_quick_lut_v2_entrypoint_targets_new_streamlit_shell() -> None:
    path = REPO_ROOT / "app_field_analysis_quick_v2.py"

    assert path.is_file()
    py_compile.compile(str(path), doraise=True)
    source = _read(path)
    assert "run_quick_lut_v2_app" in source
    assert "field_analysis.app_ui_v2" in source
    assert "field_analysis.app_ui_snapshot" not in source


def test_quick_lut_v2_app_shell_exposes_ordered_workflow_contract() -> None:
    path = REPO_ROOT / "src" / "field_analysis" / "app_ui_v2.py"

    assert path.is_file()
    py_compile.compile(str(path), doraise=True)
    source = _read(path)
    assert "Quick LUT v2" in source
    assert "legacy" in source
    assert "peak-lobe" in source
    assert "run_quick_lut_v2_app" in _function_names(path)


def test_quick_lut_v2_and_legacy_local_launchers_are_explicit() -> None:
    v2_launcher = REPO_ROOT / "launch_quick_lut_v2_local.cmd"
    legacy_launcher = REPO_ROOT / "launch_quick_lut_legacy_local.cmd"

    assert v2_launcher.is_file()
    assert legacy_launcher.is_file()
    assert '"app_field_analysis_quick_v2.py"' in _read(v2_launcher)
    assert '"app_field_analysis_quick_legacy.py"' in _read(legacy_launcher)
