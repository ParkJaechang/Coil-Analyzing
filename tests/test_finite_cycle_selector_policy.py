from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_UI_SNAPSHOT = REPO_ROOT / "src" / "field_analysis" / "app_ui_snapshot.py"


def test_finite_cycle_selector_policy_source_contract() -> None:
    source = APP_UI_SNAPSHOT.read_text(encoding="utf-8")

    assert "UI_SUPPORTED_FINITE_CYCLE_COUNTS = (1.0, 1.25, 1.5, 1.75)" in source
    assert "UI_UNAVAILABLE_FINITE_CYCLE_COUNTS = (0.75,)" in source
    assert "1.75 cycle is supported when exact finite-cycle support data exists." in source
    assert "If exact 1.75 support is absent, 1.75 is unavailable rather than substituted." in source
    assert "Previous finite cycle value `0.75` is not supported by the primary finite-cycle selector" in source
    assert "0.75 is not treated as 1.75" in source
    assert "DAQ output fixed: ±5V" in source
    assert "DCAMP Gain fixed: 100%" in source
    assert "target field remains rounded-triangle / 100pp fixed" in source
    assert "0.75 / 1.0 / 1.25 / 1.5 are supported" not in source
    assert "0.75 is supported." not in source
    assert "0.75 is legacy and not treated as 1.75" not in source
