from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
UI_SNAPSHOT_PATH = REPO_ROOT / "src" / "field_analysis" / "app_ui_snapshot.py"


def _ui_source() -> str:
    return UI_SNAPSHOT_PATH.read_text(encoding="utf-8")


def test_plot_reference_prefers_support_profile_preview_before_nearest_fallback() -> None:
    source = _ui_source()
    required_markers = [
        'def _resolve_compensation_plot_reference(',
        'compensation.get("support_profile_preview")',
        '"Support-Blended Output"',
        'compensation.get("nearest_profile_preview")',
        '"Nearest Support Preview"',
        'compensation.get("nearest_profile")',
        '"Nearest Support Output"',
        'reference_profile, reference_label = _resolve_compensation_plot_reference(compensation)',
        "reference_label=reference_label",
    ]
    missing = [marker for marker in required_markers if marker not in source]
    assert not missing, f"Missing plot identity markers: {missing}"


def test_finite_route_marker_contract_is_present() -> None:
    source = _ui_source()
    required_markers = [
        'def _resolve_finite_route_marker(',
        'def _render_finite_route_marker(',
        "steady_state_harmonic_expanded",
        "finite_empirical_support",
        "finite_empirical_preview",
        "fallback_harmonic_inverse",
        "Route:",
        "backend mode=",
        "finite support used=",
        "selected_support_id=",
        "support_count_used=",
        "support_tests_used=",
        "finite transient uploads are available",
        "but are not used by this main solver path.",
        "_render_finite_route_marker(compensation, finite_support_entries)",
    ]
    missing = [marker for marker in required_markers if marker not in source]
    assert not missing, f"Missing finite route markers: {missing}"
