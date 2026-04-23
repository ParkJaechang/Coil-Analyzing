from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.compensation import build_finite_signal_consistency_summary


def _profile() -> pd.DataFrame:
    time_s = np.linspace(0.0, 1.2, 25)
    active = time_s <= 1.0 + 1e-12
    phase = np.clip(time_s / 1.0, 0.0, 1.0)
    target = np.where(active, 50.0 * np.sin(np.pi * phase), 0.0)
    command = np.where(active, 5.0 * np.sin(np.pi * phase), 0.0)
    return pd.DataFrame(
        {
            "time_s": time_s,
            "is_active_target": active,
            "target_field_mT": target,
            "predicted_field_mT": target.copy(),
            "support_scaled_field_mT": target.copy(),
            "recommended_voltage_v": command,
        }
    )


def test_command_nonzero_end_matches_actual_recommended_voltage() -> None:
    profile = _profile()
    summary = build_finite_signal_consistency_summary(profile, finite_support_used=True, support_input_field_pp=100.0)

    command = profile["recommended_voltage_v"].to_numpy(dtype=float)
    time_s = profile["time_s"].to_numpy(dtype=float)
    threshold = max((float(np.nanmax(command) - np.nanmin(command))) * 0.01, 1e-6)
    expected = float(np.nanmax(time_s[np.abs(command) > threshold]))
    assert abs(float(summary["command_nonzero_end_s"]) - expected) <= 1e-12
    assert summary["finite_signal_consistency_status"] == "ok"


def test_stale_command_metadata_does_not_override_final_array_end() -> None:
    profile = _profile()
    summary = build_finite_signal_consistency_summary(
        profile,
        finite_support_used=True,
        support_input_field_pp=100.0,
        command_nonzero_end_s=0.25,
    )

    assert float(summary["command_metadata_input_end_s"]) == 0.25
    assert float(summary["command_nonzero_end_s"]) > 0.9
    assert "command_metadata_mismatch" not in str(summary["finite_signal_consistency_status"])
    assert summary["plot_payload_consistency_status"] == "ok"
    assert summary["command_covers_target_end"] is True


def test_command_early_stop_violation_is_detected() -> None:
    profile = _profile()
    profile.loc[profile["time_s"] > 0.55, "recommended_voltage_v"] = 0.0

    summary = build_finite_signal_consistency_summary(profile)

    assert summary["command_covers_target_end"] is False
    assert "command_early_stop" in str(summary["finite_signal_consistency_status"])


def test_predicted_early_zero_violation_is_detected_without_command_failure() -> None:
    profile = _profile()
    profile.loc[profile["time_s"] > 0.55, "predicted_field_mT"] = 0.0

    summary = build_finite_signal_consistency_summary(profile)

    assert summary["command_covers_target_end"] is True
    assert summary["predicted_covers_target_end"] is False
    assert "predicted_early_zero" in str(summary["finite_signal_consistency_status"])


def test_support_zero_bug_is_detected_for_nonzero_support_input() -> None:
    profile = _profile()
    profile["support_scaled_field_mT"] = 0.0

    summary = build_finite_signal_consistency_summary(profile, finite_support_used=True, support_input_field_pp=80.0)

    assert summary["support_scaled_pp"] == 0.0
    assert "support_zero_bug" in str(summary["finite_signal_consistency_status"])


def test_time_axis_mismatch_is_detected_from_missing_predicted_horizon() -> None:
    profile = _profile()
    profile.loc[profile["time_s"] > 0.6, "predicted_field_mT"] = np.nan

    summary = build_finite_signal_consistency_summary(profile)

    assert summary["time_axis_consistent"] is False
    assert "time_axis_mismatch" in str(summary["finite_signal_consistency_status"])


def test_healthy_finite_empirical_profile_returns_ok() -> None:
    summary = build_finite_signal_consistency_summary(_profile(), finite_support_used=True, support_input_field_pp=100.0)

    assert summary["finite_signal_consistency_status"] == "ok"
    assert summary["plot_payload_consistency_status"] == "ok"
    assert summary["command_covers_target_end"] is True
    assert summary["predicted_covers_target_end"] is True
    assert summary["support_covers_target_end"] is True


def test_resampled_support_metadata_is_reported_without_early_zero() -> None:
    profile = _profile()
    profile["support_resampled_to_target_window"] = True
    profile["support_observed_end_s"] = 0.55
    profile["support_observed_coverage_ratio"] = 0.55
    profile["support_padding_gap_s"] = 0.45
    profile["finite_prediction_source"] = "empirical_resampled"
    profile["predicted_cover_reason"] = "active_progress_resampled"
    profile["support_cover_reason"] = "active_progress_resampled"

    summary = build_finite_signal_consistency_summary(profile, finite_support_used=True, support_input_field_pp=100.0)

    assert summary["finite_signal_consistency_status"] == "ok"
    assert summary["support_resampled_to_target_window"] is True
    assert summary["finite_prediction_source"] == "empirical_resampled"
    assert summary["predicted_cover_reason"] == "active_progress_resampled"
    assert summary["support_cover_reason"] == "active_progress_resampled"
    assert summary["predicted_covers_target_end"] is True
    assert summary["support_covers_target_end"] is True
