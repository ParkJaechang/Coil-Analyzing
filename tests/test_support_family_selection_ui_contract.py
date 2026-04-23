from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_UI_SNAPSHOT = REPO_ROOT / "src" / "field_analysis" / "app_ui_snapshot.py"


def _source() -> str:
    return APP_UI_SNAPSHOT.read_text(encoding="utf-8")


def test_target_contract_is_separated_from_support_family_choice() -> None:
    source = _source()

    assert "Target field shape = rounded triangle fixed" in source
    assert "Target field pp = 100 fixed" in source
    assert "support/input waveform family does not change the physical target" in source
    assert "selected support family" in source
    assert "may differ from the requested family" in source


def test_support_family_selection_payload_fields_are_rendered() -> None:
    source = _source()

    assert "Support Family Selection" in source
    assert "support_family_requested" in source
    assert "user_requested_support_family" in source
    assert "selected_support_family" in source
    assert "support_family_override_applied" in source
    assert "support_family_override_reason" in source
    assert "family_sensitivity_level" in source
    assert "support_family_sensitivity_level" in source
    assert "Requested support family:" in source
    assert "Selected support family:" in source
    assert "Override applied:" in source
    assert "Reason:" in source
    assert "Family sensitivity level:" in source


def test_support_family_selection_summary_is_called_for_finite_results() -> None:
    source = _source()

    assert "_render_support_family_selection_marker(compensation, requested_support_family=target_waveform)" in source
