from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_UI_SNAPSHOT = REPO_ROOT / "src" / "field_analysis" / "app_ui_snapshot.py"


def _source() -> str:
    return APP_UI_SNAPSHOT.read_text(encoding="utf-8")


def test_finite_signal_consistency_status_markers_are_rendered() -> None:
    source = _source()

    assert "Finite Signal Consistency" in source
    assert "finite_signal_consistency_status" in source
    assert "command_covers_target_end" in source
    assert "predicted_covers_target_end" in source
    assert "support_covers_target_end" in source
    assert "command_early_stop_s" in source
    assert "predicted_early_stop_s" in source
    assert "support_early_stop_s" in source


def test_finite_signal_consistency_status_severity_is_visible() -> None:
    source = _source()

    assert "time_axis_mismatch" in source
    assert "command_metadata_mismatch" in source
    assert "command_early_stop" in source
    assert "predicted_early_zero" in source
    assert "support_early_zero" in source
    assert 'status == "ok"' in source or 'status_tokens == {"ok"}' in source
    assert "st.error" in source
    assert "st.warning" in source
    assert "st.success" in source


def test_finite_signal_consistency_summary_is_called_for_finite_results() -> None:
    source = _source()
    finite_block = source.split("if finite_cycle_mode:", 1)[1]

    assert "_render_finite_route_marker(compensation)" in finite_block
    assert "_render_finite_signal_consistency_summary(compensation, command_profile)" in finite_block
    assert "_render_finite_cycle_correction_summary(compensation, command_profile)" in finite_block
