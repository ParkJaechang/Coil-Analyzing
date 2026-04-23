from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_UI_SNAPSHOT = REPO_ROOT / "src" / "field_analysis" / "app_ui_snapshot.py"


def _source() -> str:
    return APP_UI_SNAPSHOT.read_text(encoding="utf-8")


def test_finite_cycle_ui_supported_set_promotes_0p75_and_removes_1p75_from_selector() -> None:
    source = _source()
    selector_block = source.split('key="target_cycle_count_v2"', 1)[0].rsplit("st.selectbox(", 1)[1]

    assert "UI_SUPPORTED_FINITE_CYCLE_COUNTS = (0.75, 1.0, 1.25, 1.5)" in source
    assert "UI_UNAVAILABLE_FINITE_CYCLE_COUNTS = (1.75,)" in source
    assert "options=[float(value) for value in UI_SUPPORTED_FINITE_CYCLE_COUNTS]" in selector_block
    assert "FIELD_ONLY_ALLOWED_FINITE_CYCLE_COUNTS" not in selector_block


def test_finite_cycle_ui_explains_1p75_unavailable_without_aliasing_to_0p75() -> None:
    source = _source()

    assert "1.75 is currently unavailable" in source
    assert "no safe finite support/decomposition" in source
    assert "not treated as 0.75" in source
    assert "_sanitize_finite_cycle_session_state" in source
    assert "no_safe_1_75_support" in source


def test_finite_prediction_unavailable_is_visible_for_unsafe_1p75_results() -> None:
    source = _source()

    assert "finite_prediction_available" in source
    assert "finite_prediction_unavailable_reason" in source
    assert "unsafe predicted/support traces" in source
    assert "_render_finite_prediction_availability(compensation)" in source
