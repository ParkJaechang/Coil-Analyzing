from __future__ import annotations

import ast
import py_compile
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

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


def test_lut_metric_prioritization_prefers_field_metrics() -> None:
    from field_analysis.lut import prioritize_lut_target_metrics

    metrics = [
        "achieved_current_pp_a_mean",
        "achieved_bz_mT_pp_mean",
        "achieved_bmag_mT_pp_mean",
        "achieved_bx_mT_pp_mean",
    ]

    prioritized = prioritize_lut_target_metrics(metrics, main_field_axis="bx_mT")

    assert prioritized == [
        "achieved_bx_mT_pp_mean",
        "achieved_bz_mT_pp_mean",
        "achieved_bmag_mT_pp_mean",
    ]


def test_lut_metric_prioritization_keeps_current_as_debug_fallback() -> None:
    from field_analysis.lut import prioritize_lut_target_metrics

    assert prioritize_lut_target_metrics(
        ["achieved_current_pp_a_mean"],
        main_field_axis="bz_mT",
    ) == ["achieved_current_pp_a_mean"]

    assert prioritize_lut_target_metrics(
        ["achieved_current_pp_a_mean", "achieved_bz_mT_pp_mean"],
        main_field_axis="bz_mT",
        include_current_debug=True,
    ) == ["achieved_bz_mT_pp_mean", "achieved_current_pp_a_mean"]


def test_lut_display_context_prefers_field_output_for_current_debug_target() -> None:
    from field_analysis.lut import build_lut_recommendation_display_context

    display = build_lut_recommendation_display_context(
        target_metric="achieved_current_pp_a_mean",
        used_target_value=12.0,
        estimated_current_pp=12.0,
        estimated_bz_pp=34.5,
        estimated_bmag_pp=35.0,
        finite_cycle_mode=True,
    )

    assert display["recommendation_scope"] == "finite_cycle"
    assert display["recommendation_scope_label"] == "finite-cycle"
    assert display["target_output_unit"] == "A"
    assert display["primary_output_unit"] == "mT"
    assert display["primary_output_pp"] == 34.5
