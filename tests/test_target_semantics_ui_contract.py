from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_UI_SNAPSHOT = REPO_ROOT / "src" / "field_analysis" / "app_ui_snapshot.py"


def _source() -> str:
    return APP_UI_SNAPSHOT.read_text(encoding="utf-8")


def test_target_support_predicted_semantics_markers_exist() -> None:
    source = _source()

    assert "목표 자기장 개형 / fixed rounded triangle" in source
    assert "목표 피크 자기장 (mT)" in source
    assert "내부 정규화 기준" in source
    assert "데이터/예측 의미 상세" in source
    assert "1차 모델링 command" in source
    assert "Internal Reference (debug, hidden by default)" in source


def test_plot_semantics_explanation_separates_target_from_support() -> None:
    source = _source()

    assert "목표 자기장 개형은 canonical fixed rounded triangle입니다." in source
    assert "목표 자기장, support preview, forward prediction은 서로 다른 진단 정보입니다." in source
    assert "fixed rounded triangle at 100pp" not in source
    assert "100mT pp fixed" not in source
    assert "100pp fixed" not in source
    assert "목표 bz_mT PP" not in source
    assert "Target metric fixed" not in source
    assert "Advanced / Debug plot references" in source
    assert "이것은 physical target이 아닙니다." in source


def test_plot_profile_uses_physical_target_backend_column() -> None:
    source = _source()

    assert "_prepare_semantic_compensation_plot_profile" in source
    assert "physical_target_output_mT" in source
    assert "support_reference_output_mT" in source
    assert "predicted_field_mT" in source
