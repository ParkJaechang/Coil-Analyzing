from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_UI_SNAPSHOT = REPO_ROOT / "src" / "field_analysis" / "app_ui_snapshot.py"


def _source() -> str:
    return APP_UI_SNAPSHOT.read_text(encoding="utf-8")


def test_support_blended_trace_uses_finite_support_payload_column_first() -> None:
    source = _source()
    resolver = source.split("def _resolve_compensation_plot_reference", 1)[1].split(
        "def _finite_signal_value", 1
    )[0]

    assert "support_profile_preview" in resolver
    assert "support_scaled_field_mT" in resolver
    assert "support_blended_field_mT" in resolver
    assert '"Support-Blended Output"' in resolver
    assert "finite support payload" in resolver
    assert resolver.index("support_scaled_field_mT") < resolver.index("nearest_profile_preview")


def test_nearest_preview_fallback_is_not_labeled_support_blended_output() -> None:
    source = _source()
    resolver = source.split("def _resolve_compensation_plot_reference", 1)[1].split(
        "def _finite_signal_value", 1
    )[0]

    assert '"Nearest Support Preview"' in resolver
    assert '"Nearest Support Output"' in resolver
    assert "nearest support preview" in resolver
    assert "nearest support output" in resolver


def test_plot_call_uses_resolved_reference_column_and_trace_marker() -> None:
    source = _source()

    assert "nearest_column=reference_column" in source
    assert "reference_source" in source
    assert "Support trace source:" in source
    assert "support_payload_pp" in source
    assert "plotted_support_trace_pp" in source
