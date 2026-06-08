from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_UI_SNAPSHOT = REPO_ROOT / "src" / "field_analysis" / "app_ui_snapshot.py"


def test_finite_cycle_selector_policy_source_contract() -> None:
    source = APP_UI_SNAPSHOT.read_text(encoding="utf-8")

    assert "UI_SUPPORTED_FINITE_CYCLE_COUNTS = (1.0, 1.25, 1.5, 1.75)" in source
    assert "UI_UNAVAILABLE_FINITE_CYCLE_COUNTS = (0.75,)" in source
    assert "Production finite 보정은 1.0 / 1.5 cycle을 지원합니다." in source
    assert "1.25 / 1.75 / 2.0 cycle은 검토용이며 production 보정/내보내기 대상이 아닙니다." in source
    assert "2-cycle production 정책은 폐기되었습니다." in source
    assert "Previous finite cycle value `0.75` is not supported by the primary finite-cycle selector" in source
    assert "0.75 is not treated as 1.75" in source
    assert "최종 command voltage는 ±10V 기준으로 제한" in source
    assert "target field shape는 fixed rounded triangle" in source
    assert "target field remains rounded-triangle / 100pp fixed" not in source
    assert "0.75 / 1.0 / 1.25 / 1.5 are supported" not in source
    assert "0.75 is supported." not in source
    assert "0.75 is legacy and not treated as 1.75" not in source
