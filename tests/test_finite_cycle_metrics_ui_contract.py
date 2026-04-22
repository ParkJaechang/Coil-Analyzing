from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
UI_SNAPSHOT_PATH = REPO_ROOT / "src" / "field_analysis" / "app_ui_snapshot.py"


def _ui_source() -> str:
    return UI_SNAPSHOT_PATH.read_text(encoding="utf-8")


def test_finite_cycle_summary_consumes_expected_backend_fields() -> None:
    source = _ui_source()
    required_markers = [
        "Finite-Cycle Correction Summary",
        "finite_terminal_correction_applied",
        "finite_terminal_correction_reason",
        "finite_terminal_correction_gain",
        "finite_active_nrmse_before",
        "finite_active_nrmse_after",
        "finite_tail_residual_ratio_before",
        "finite_tail_residual_ratio_after",
        "finite_terminal_peak_error_mT_before",
        "finite_terminal_peak_error_mT_after",
        "finite_terminal_direction_match_before",
        "finite_terminal_direction_match_after",
        "estimated_output_lag_seconds",
        "finite_metric_improvement_summary",
        "_render_finite_cycle_correction_summary(compensation, command_profile)",
    ]
    missing = [marker for marker in required_markers if marker not in source]
    assert not missing, f"Missing finite-cycle UI markers: {missing}"


def test_finite_cycle_summary_covers_expected_status_paths() -> None:
    source = _ui_source()
    required_markers = [
        "no_material_improvement",
        "finite terminal/tail correction applied",
        "candidate correction did not clear improvement guardrails; original command retained",
        "Active NRMSE improved",
        "Tail residual ratio improved",
        "Terminal direction still mismatches the requested stop direction after correction.",
    ]
    missing = [marker for marker in required_markers if marker not in source]
    assert not missing, f"Missing finite-cycle status markers: {missing}"
