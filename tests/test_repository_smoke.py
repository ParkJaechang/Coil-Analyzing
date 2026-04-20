from __future__ import annotations

import ast
import py_compile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = [
    REPO_ROOT / "app_field_analysis_latest.py",
    REPO_ROOT / "app_field_analysis_quick.py",
    REPO_ROOT / "app.py",
    REPO_ROOT / "src" / "field_analysis" / "app_ui.py",
    REPO_ROOT / "src" / "field_analysis" / "app_ui_snapshot.py",
    REPO_ROOT / "src" / "field_analysis" / "ui_validation_retune.py",
    REPO_ROOT / "src" / "field_analysis" / "validation_retune.py",
]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse_module(path: Path) -> ast.AST:
    return ast.parse(_read_text(path), filename=str(path))


def test_core_directories_exist() -> None:
    assert (REPO_ROOT / "src" / "field_analysis").is_dir()
    assert (REPO_ROOT / "src" / "coil_analyzer").is_dir()


def test_entrypoints_exist_and_compile() -> None:
    for path in ENTRYPOINTS:
        assert path.is_file(), f"Missing expected file: {path}"
        py_compile.compile(str(path), doraise=True)


def test_app_ui_snapshot_exposes_expected_entrypoints() -> None:
    module = _parse_module(REPO_ROOT / "src" / "field_analysis" / "app_ui_snapshot.py")
    function_names = {
        node.name
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "run_app" in function_names
    assert "run_quick_lut_app" in function_names


def test_app_ui_remains_snapshot_loader_wrapper() -> None:
    wrapper_path = REPO_ROOT / "src" / "field_analysis" / "app_ui.py"
    wrapper_source = _read_text(wrapper_path)
    module = _parse_module(wrapper_path)
    function_names = {
        node.name
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "_load_source_module" in function_names
    assert "_load_ui_module" in function_names
    assert 'app_ui_snapshot.py' in wrapper_source
    assert "app_ui_snapshot" in wrapper_source
