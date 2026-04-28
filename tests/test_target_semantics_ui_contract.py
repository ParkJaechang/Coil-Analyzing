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
    assert "Command Waveform" in source
    assert "Internal Reference (debug, hidden by default)" in source


def test_plot_semantics_explanation_separates_target_from_support() -> None:
    source = _source()

    assert "Finite target semantics: Physical Target = fixed rounded triangle at 100pp." in source
    assert "Support Reference is a support-conditioned preview, not the physical target." in source
    assert "Plot semantics: `Physical Target` is the requested field waveform;" in source
    assert "`Support Reference` is not " in source
    assert "the target;" in source
    assert "`Predicted Output` is the model response;" in source
    assert "Advanced / Debug plot references" in source
    assert "It is not the physical target." in source


def test_plot_profile_uses_physical_target_backend_column() -> None:
    source = _source()

    assert "_prepare_semantic_compensation_plot_profile" in source
    assert "physical_target_output_mT" in source
    assert "support_reference_output_mT" in source
    assert "predicted_field_mT" in source
