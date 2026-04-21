from __future__ import annotations

import ast
import pandas as pd
from pathlib import Path

from src.field_analysis.ui_field_waveform_diagnostics_focus import (
    build_diagnostics_focus_summary,
    extract_continuous_problem_combos,
    extract_finite_missing_support_combos,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_focus_helper_extracts_only_problem_rows_in_fixed_risk_order() -> None:
    continuous_support = pd.DataFrame(
        [
            {"waveform_type": "triangle", "freq_hz": 20.0, "risk_level": "Weak", "continuous_test_count": 1, "field_ready_test_count": 1, "voltage_ready_test_count": 1, "shape_comparison_possible": False},
            {"waveform_type": "sine", "freq_hz": 10.0, "risk_level": "OK", "continuous_test_count": 2, "field_ready_test_count": 2, "voltage_ready_test_count": 2, "shape_comparison_possible": True},
            {"waveform_type": "square", "freq_hz": 30.0, "risk_level": "Field Missing", "continuous_test_count": 1, "field_ready_test_count": 0, "voltage_ready_test_count": 1, "shape_comparison_possible": False},
            {"waveform_type": "sawtooth", "freq_hz": 40.0, "risk_level": "Missing", "continuous_test_count": 0, "field_ready_test_count": 0, "voltage_ready_test_count": 0, "shape_comparison_possible": False},
            {"waveform_type": "pulse", "freq_hz": 50.0, "risk_level": "Voltage Missing", "continuous_test_count": 1, "field_ready_test_count": 1, "voltage_ready_test_count": 0, "shape_comparison_possible": False},
        ]
    )
    finite_support = pd.DataFrame(
        [
            {"waveform_type": "sine", "freq_hz": 10.0, "has_support": True, "finite_test_count": 1, "field_ready_test_count": 1, "voltage_ready_test_count": 1, "risk_level": "OK"},
            {"waveform_type": "square", "freq_hz": 30.0, "has_support": False, "finite_test_count": 0, "field_ready_test_count": 0, "voltage_ready_test_count": 0, "risk_level": "Missing"},
        ]
    )

    continuous_problem_rows = extract_continuous_problem_combos(continuous_support)
    finite_missing_rows = extract_finite_missing_support_combos(finite_support)
    summary = build_diagnostics_focus_summary(
        continuous_support=continuous_support,
        finite_support=finite_support,
    )

    assert list(continuous_problem_rows["risk_level"].astype(str)) == [
        "Field Missing",
        "Voltage Missing",
        "Missing",
        "Weak",
    ]
    assert list(finite_missing_rows["waveform_type"]) == ["square"]
    assert summary["blocking_combo_count"] == 3
    assert summary["weak_combo_count"] == 1
    assert summary["finite_missing_combo_count"] == 1


def test_focus_helper_returns_empty_frames_and_zero_summary_for_empty_inputs() -> None:
    summary = build_diagnostics_focus_summary(
        continuous_support=pd.DataFrame(),
        finite_support=pd.DataFrame(),
    )

    assert summary["blocking_combo_count"] == 0
    assert summary["weak_combo_count"] == 0
    assert summary["finite_missing_combo_count"] == 0
    assert summary["continuous_problem_rows"].empty
    assert summary["finite_missing_rows"].empty


def test_diagnostics_ui_keeps_both_focus_and_export_helpers_connected() -> None:
    source = (REPO_ROOT / "src" / "field_analysis" / "ui_field_waveform_diagnostics.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    called_names = {
        node.func.id
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "render_field_waveform_diagnostics_focus_block" in source
    assert "render_field_waveform_diagnostics_export_panel" in source
    assert "render_field_waveform_diagnostics_focus_block" in called_names
    assert "render_field_waveform_diagnostics_export_panel" in called_names
