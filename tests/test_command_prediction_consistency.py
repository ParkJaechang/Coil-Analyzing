from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TEST_ROOT = REPO_ROOT / "tests"
for path in (SRC_ROOT, TEST_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from field_analysis.compensation import (
    _build_command_prediction_consistency_contract,
    _ensure_plotted_command_covers_target_window,
)
import test_finite_empirical_field_route as finite_fixture


def test_support_reference_is_diagnostic_not_command_target() -> None:
    result = finite_fixture._run_field_compensation(
        finite_support_entries=[
            finite_fixture._build_finite_entry(
                test_id="finite_exact_consistency",
                waveform_type="sine",
                freq_hz=3.0,
                cycle_count=1.25,
                field_pp=95.0,
            )
        ],
        waveform_type="sine",
        freq_hz=3.0,
        target_cycle_count=1.25,
    )
    profile = result["command_profile"]

    assert result["command_generation_target"] == "physical_target"
    assert result["support_reference_role"] == "diagnostic_reference"
    assert result["support_reference_used_for_command"] is False
    assert str(profile["support_reference_used_for_command"].iloc[0]) == "False"
    assert result["support_reference_plotted_column"] == "support_reference_output_mT"
    assert np.allclose(
        profile["support_reference_output_mT"],
        profile["target_aligned_support_reference_mT"],
        equal_nan=True,
    )
    assert not np.allclose(
        profile["support_reference_output_mT"],
        profile["predicted_field_mT"],
        equal_nan=True,
    )


def test_predicted_output_is_marked_as_from_plotted_command() -> None:
    result = finite_fixture._run_field_compensation(
        finite_support_entries=[
            finite_fixture._build_finite_entry(
                test_id="finite_forward_consistency",
                waveform_type="sine",
                freq_hz=5.0,
                cycle_count=1.5,
                field_pp=100.0,
            )
        ],
        waveform_type="sine",
        freq_hz=5.0,
        target_cycle_count=1.5,
    )
    profile = result["command_profile"]

    assert result["forward_prediction_source"] == "recommended_voltage_v"
    assert result["plotted_command_source"] == "recommended_voltage_v"
    assert result["plotted_predicted_source"] == "predicted_field_mT"
    assert result["forward_prediction_available"] is True
    assert result["predicted_from_plotted_command"] is True
    assert result["displayed_predicted_valid"] is True
    assert result["command_prediction_consistency_status"] == "ok"
    assert str(profile["forward_prediction_source"].iloc[0]) == "recommended_voltage_v"
    assert bool(profile["predicted_from_plotted_command"].iloc[0]) is True
    assert "displayed_predicted_field_mT" in profile.columns
    assert np.allclose(profile["displayed_predicted_field_mT"], profile["predicted_field_mT"], equal_nan=True)


def test_late_command_start_is_marked_inconsistent() -> None:
    time_s = np.linspace(0.0, 1.0, 101)
    target = np.sin(np.pi * time_s)
    command = np.where(time_s < 0.35, 0.0, np.sin(np.pi * time_s))
    profile = pd.DataFrame(
        {
            "time_s": time_s,
            "physical_target_output_mT": target * 100.0,
            "recommended_voltage_v": command,
            "predicted_field_mT": target * 95.0,
            "support_reference_output_mT": target * 80.0,
        }
    )

    contract = _build_command_prediction_consistency_contract(profile)

    assert contract["command_covers_target_active_start"] is False
    assert contract["command_target_start_delta_s"] > 0.0
    assert "command_coverage_insufficient" in contract["command_prediction_consistency_status"]
    assert contract["predicted_from_plotted_command"] is False
    assert contract["displayed_predicted_valid"] is False
    assert contract["command_nonzero_end_s"] >= contract["command_nonzero_start_s"]
    assert np.isfinite(float(contract["command_target_end_delta_s"]))


def test_forward_prediction_unavailable_is_not_marked_ok() -> None:
    time_s = np.linspace(0.0, 1.0, 64)
    target = np.sin(np.pi * time_s)
    profile = pd.DataFrame(
        {
            "time_s": time_s,
            "physical_target_output_mT": target * 100.0,
            "recommended_voltage_v": np.sin(np.pi * time_s),
            "support_reference_output_mT": target * 80.0,
        }
    )

    contract = _build_command_prediction_consistency_contract(profile)

    assert contract["forward_prediction_source"] == "recommended_voltage_v"
    assert contract["forward_prediction_available"] is False
    assert contract["predicted_from_plotted_command"] is False
    assert "forward_prediction_unavailable" in contract["command_prediction_consistency_status"]


def test_support_reference_shape_mismatch_is_diagnostic_only() -> None:
    time_s = np.linspace(0.0, 1.0, 128)
    target = np.sin(np.pi * time_s)
    bad_support = np.cos(8.0 * np.pi * time_s) * 100.0
    profile = pd.DataFrame(
        {
            "time_s": time_s,
            "physical_target_output_mT": target * 100.0,
            "recommended_voltage_v": np.sin(np.pi * time_s),
            "predicted_field_mT": target * 98.0,
            "support_reference_output_mT": bad_support,
        }
    )

    contract = _build_command_prediction_consistency_contract(profile)

    assert contract["support_reference_shape_mismatch"] is True
    assert contract["support_reference_used_for_command"] is False
    assert contract["command_generation_target"] == "physical_target"
    assert np.isfinite(float(contract["support_reference_target_corr"]))
    assert np.isfinite(float(contract["support_reference_target_nrmse"]))


def test_command_coverage_extension_removes_target_window_hard_zero_gap() -> None:
    result = finite_fixture._run_field_compensation(
        finite_support_entries=[
            finite_fixture._build_finite_entry(
                test_id="finite_coverage_fix",
                waveform_type="sine",
                freq_hz=1.0,
                cycle_count=1.0,
                field_pp=90.0,
                zero_after_fraction=None,
            )
        ],
        waveform_type="sine",
        freq_hz=1.0,
        target_cycle_count=1.0,
    )
    profile = result["command_profile"].copy()
    time_s = pd.to_numeric(profile["time_s"], errors="coerce")
    target_start = float(result["target_nonzero_start_s"])
    profile.loc[time_s < target_start + 0.08, "recommended_voltage_v"] = 0.0

    before = _build_command_prediction_consistency_contract(profile)
    adjusted = _ensure_plotted_command_covers_target_window(profile)
    after = _build_command_prediction_consistency_contract(adjusted)

    assert before["command_covers_target_active_start"] is False
    assert after["command_covers_target_active_start"] is True
    assert bool(adjusted["command_coverage_extension_applied"].iloc[0]) is True
    assert after["predicted_from_plotted_command"] is True


def test_command_coverage_extension_handles_read_only_numpy_views(monkeypatch) -> None:
    result = finite_fixture._run_field_compensation(
        finite_support_entries=[
            finite_fixture._build_finite_entry(
                test_id="finite_read_only_coverage_fix",
                waveform_type="sine",
                freq_hz=1.0,
                cycle_count=1.0,
                field_pp=90.0,
                zero_after_fraction=None,
            )
        ],
        waveform_type="sine",
        freq_hz=1.0,
        target_cycle_count=1.0,
    )
    profile = result["command_profile"].copy()
    time_s = pd.to_numeric(profile["time_s"], errors="coerce")
    target_start = float(result["target_nonzero_start_s"])
    profile.loc[time_s < target_start + 0.08, "recommended_voltage_v"] = 0.0

    original_to_numpy = pd.Series.to_numpy

    def read_only_to_numpy(self, *args, **kwargs):
        values = original_to_numpy(self, *args, **kwargs)
        if isinstance(values, np.ndarray):
            values.setflags(write=False)
        return values

    monkeypatch.setattr(pd.Series, "to_numpy", read_only_to_numpy)

    adjusted = _ensure_plotted_command_covers_target_window(profile)
    after = _build_command_prediction_consistency_contract(adjusted)

    assert after["command_covers_target_active_start"] is True
    assert bool(adjusted["command_coverage_extension_applied"].iloc[0]) is True
