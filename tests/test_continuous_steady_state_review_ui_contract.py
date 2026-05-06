from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_UI_SNAPSHOT = REPO_ROOT / "src" / "field_analysis" / "app_ui_snapshot.py"


def _source() -> str:
    return APP_UI_SNAPSHOT.read_text(encoding="utf-8")


def test_continuous_steady_state_review_markers_exist() -> None:
    source = _source()

    expected_markers = [
        "Continuous Steady-State Review",
        "steady-state evaluation window",
        "Steady-state window start:",
        "Steady-state window end:",
        "Startup excluded:",
        "Predicted from plotted command:",
        "Consistency status:",
        "Steady-state metrics",
        "Whole-window metrics are debug/secondary only.",
        "사용자가 steady-state alignment를 검토하십시오.",
        "Support Reference is diagnostic only and is not the command target.",
        "Continuous steady-state review metadata unavailable",
        "_render_continuous_steady_state_review(compensation, command_profile)",
    ]
    missing = [marker for marker in expected_markers if marker not in source]

    assert not missing, f"Missing continuous steady-state review UI markers: {missing}"


def test_continuous_steady_state_review_payload_keys_are_used() -> None:
    source = _source()

    expected_keys = [
        "steady_state_start_s",
        "steady_state_end_s",
        "startup_excluded",
        "continuous_evaluation_window",
        "startup_window_end_s",
        "steady_state_duration_s",
        "steady_state_nrmse",
        "steady_state_shape_corr",
        "steady_state_peak_error_mT",
        "predicted_from_plotted_command",
        "command_prediction_consistency_status",
        "support_reference_used_for_command",
        "whole_window_metrics_debug_only",
        "whole_window_nrmse_debug",
        "whole_window_shape_corr_debug",
        "whole_window_peak_error_debug",
    ]
    missing = [key for key in expected_keys if key not in source]

    assert not missing, f"Missing continuous steady-state payload keys: {missing}"


def test_continuous_steady_state_review_avoids_quality_overclaim() -> None:
    source = _source()

    forbidden_claims = [
        "continuous model is correct",
        "steady-state alignment passed",
        "model quality passed",
    ]

    assert not any(claim in source.lower() for claim in forbidden_claims)
