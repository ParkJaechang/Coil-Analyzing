from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_UI_SNAPSHOT = REPO_ROOT / "src" / "field_analysis" / "app_ui_snapshot.py"
STARTUP_UI = REPO_ROOT / "src" / "field_analysis" / "ui_startup_compensation_review.py"


def _source() -> str:
    return STARTUP_UI.read_text(encoding="utf-8")


def test_startup_compensation_review_section_markers_exist() -> None:
    source = _source()

    assert "Startup Compensation Review" in source
    assert "startup-aware compensation의 실사용자 검토용" in source
    assert "모델링 품질은 사용자 그래프 검수로 판단합니다" in source
    assert "Physical Target은 변경되지 않습니다" in source
    assert "startup compensation data unavailable" in source


def test_startup_side_by_side_plot_labels_exist() -> None:
    source = _source()

    assert "Physical Target" in source
    assert "Open-loop Predicted Field" in source
    assert "Startup Transient Component" in source
    assert "Compensated Predicted Field" in source
    assert "Baseline Recommended Voltage" in source
    assert "Compensated Recommended Voltage" in source
    assert "Startup Compensation Command Delta" in source
    assert "Startup Field Comparison" in source
    assert "Startup Command Comparison" in source


def test_startup_before_after_metrics_and_status_markers_exist() -> None:
    source = _source()

    expected_markers = [
        "startup_residual_before_mT",
        "startup_residual_after_mT",
        "early_cycle_residual_before",
        "early_cycle_residual_after",
        "active_nrmse_before",
        "active_nrmse_after",
        "active_shape_corr_before",
        "active_shape_corr_after",
        "terminal_peak_error_before_mT",
        "terminal_peak_error_after_mT",
        "tail_residual_before",
        "tail_residual_after",
        "startup_transient_applied",
        "startup_status",
        "startup_source_type",
        "startup_source_file",
        "startup_source_support_id",
        "startup_data_quality_ok",
        "startup_rejected_reason",
    ]
    missing = [marker for marker in expected_markers if marker not in source]

    assert not missing, f"Missing startup UI markers: {missing}"
    assert "tail_residual_ratio_before" not in source
    assert "tail_residual_ratio_after" not in source


def test_startup_review_is_connected_without_default_success_claim() -> None:
    source = _source()
    app_source = APP_UI_SNAPSHOT.read_text(encoding="utf-8")

    assert "from .ui_startup_compensation_review import render_startup_compensation_review" in app_source
    assert "render_startup_compensation_review(compensation, command_profile)" in app_source
    assert "startup compensation candidate is applied in this payload; inspect plots and before/after metrics." in source
    assert "startup compensation status is unavailable; this is not a quality failure by itself." in source
    assert "st.success(\"startup compensation" not in source
