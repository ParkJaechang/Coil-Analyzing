from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_UI_SNAPSHOT = REPO_ROOT / "src" / "field_analysis" / "app_ui_snapshot.py"


def _source() -> str:
    return APP_UI_SNAPSHOT.read_text(encoding="utf-8")


def test_target_support_predicted_semantics_markers_exist() -> None:
    source = _source()

    assert "Physical Target" in source
    assert "Predicted Output" in source
    assert "Support Reference" in source
    assert "Support-Blended Preview" in source
    assert "1차 모델링 command" in source
    assert "Internal Reference (debug, hidden by default)" in source


def test_plot_semantics_explanation_separates_target_from_support() -> None:
    source = _source()

    assert "Finite target semantics: Physical Target = fixed rounded triangle at 100pp." in source
    assert "Support Reference is a support-conditioned preview, not the physical target." in source
    assert "Plot semantics: `Physical Target`은 요청한 field waveform이고" in source
    assert "`Support Reference`는 target이 아닙니다." in source
    assert "`Predicted Output`은 1차 모델링 command의 model response" in source
    assert "Advanced / Debug plot references" in source
    assert "이것은 physical target이 아닙니다." in source


def test_plot_profile_uses_physical_target_backend_column() -> None:
    source = _source()

    assert "_prepare_semantic_compensation_plot_profile" in source
    assert "physical_target_output_mT" in source
    assert "support_reference_output_mT" in source
    assert "predicted_field_mT" in source
