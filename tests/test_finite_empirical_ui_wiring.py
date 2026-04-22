from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
UI_SNAPSHOT_PATH = REPO_ROOT / "src" / "field_analysis" / "app_ui_snapshot.py"


def _ui_source() -> str:
    return UI_SNAPSHOT_PATH.read_text(encoding="utf-8")


def test_finite_empirical_support_is_wired_into_main_quick_lut_solver_path() -> None:
    source = _ui_source()
    required_markers = [
        "build_backend_finite_support_entries",
        "finite_support_entries = (",
        "_build_finite_support_entries(",
        "finite_support_entries=finite_support_entries,",
        "_render_finite_route_marker(compensation)",
        "finite_support_used",
        "finite_route_mode",
        "finite_route_reason",
        "support_tests_used",
        "support_count_used",
        "selected_support_id",
    ]
    missing = [marker for marker in required_markers if marker not in source]
    assert not missing, f"Missing finite empirical wiring markers: {missing}"


def test_finite_route_warning_and_plot_identity_contract_are_present() -> None:
    source = _ui_source()
    required_markers = [
        '"support_profile_preview"',
        '"Support-Blended Output"',
        '"nearest_profile_preview"',
        '"Nearest Support Preview"',
        '"Nearest Support Output"',
        "support_blended_output_nonzero",
        "Finite empirical support route used. Using uploaded transient finite-cycle support data.",
        "Steady-state fallback: finite transient support was unavailable or unusable.",
        "finite empirical support route",
        'if compensation.get("finite_support_used"):',
    ]
    missing = [marker for marker in required_markers if marker not in source]
    assert not missing, f"Missing finite route/plot identity markers: {missing}"
