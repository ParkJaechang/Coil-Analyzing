from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_UI_SNAPSHOT = REPO_ROOT / "src" / "field_analysis" / "app_ui_snapshot.py"


def _source() -> str:
    return APP_UI_SNAPSHOT.read_text(encoding="utf-8")


def test_support_family_override_markers_exist() -> None:
    source = _source()

    assert "데이터 선택 기준 / Debug" in source
    assert "요청 support family:" in source
    assert "선택 support family:" in source
    assert "override 적용:" in source
    assert "사유:" in source
    assert "family sensitivity:" in source
    assert "Support Family Selection" not in source


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

    assert "support/input waveform family는 목표 자기장 개형을 바꾸지 않습니다." in source
    assert "support family override가 적용되었습니다." in source
