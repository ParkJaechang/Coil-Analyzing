from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_UI_SNAPSHOT = REPO_ROOT / "src" / "field_analysis" / "app_ui_snapshot.py"


def _source() -> str:
    return APP_UI_SNAPSHOT.read_text(encoding="utf-8")


def test_end_marker_labels_exist() -> None:
    source = _source()

    assert "#### End Markers" in source
    assert "target_end" in source
    assert "command_end" in source
    assert "predicted_settle_end" in source
    assert "target_end is the requested physical target end" in source
    assert "predicted_settle_end is when the predicted field settles" in source


def test_end_marker_backend_sources_are_used() -> None:
    source = _source()

    assert "target_active_end_s" in source
    assert "command_nonzero_end_s" in source
    assert "predicted_nonzero_end_s" in source
    assert "_add_finite_end_markers" in source
    assert "_render_end_marker_summary" in source


def test_fractional_cycle_policy_wording_exists() -> None:
    source = _source()

    assert "Production finite 보정은 1.0 / 1.5 cycle을 지원합니다." in source
    assert "1.25 / 1.75 / 2.0 cycle은 검토용이며 production 보정/내보내기 대상이 아닙니다." in source
    assert "2-cycle production 정책은 폐기되었습니다." in source
    assert "Previous finite cycle value `0.75` is not supported by the primary finite-cycle selector" in source
    assert "0.75 is not treated as 1.75" in source
