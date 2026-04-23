from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_UI_SNAPSHOT = REPO_ROOT / "src" / "field_analysis" / "app_ui_snapshot.py"


def _source() -> str:
    return APP_UI_SNAPSHOT.read_text(encoding="utf-8")


def test_support_family_override_markers_exist() -> None:
    source = _source()

    assert "Support Family Selection" in source
    assert "Requested support family:" in source
    assert "Selected support family:" in source
    assert "Override applied:" in source
    assert "Reason:" in source
    assert "Family sensitivity:" in source


def test_support_family_override_payload_keys_are_used() -> None:
    source = _source()

    assert "support_family_requested" in source
    assert "user_requested_support_family" in source
    assert "selected_support_family" in source
    assert "support_family_override_applied" in source
    assert "support_family_override_reason" in source
    assert "support_family_sensitivity_level" in source


def test_support_family_override_summary_explains_requested_vs_selected_split() -> None:
    source = _source()

    assert "The support/input waveform family does not change the physical target." in source
    assert "selected support family differs from the requested support/input" in source
