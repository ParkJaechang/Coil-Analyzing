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

from field_analysis.compensation import _build_command_prediction_consistency_contract
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
    assert result["predicted_from_plotted_command"] is True
    assert result["command_prediction_consistency_status"] == "ok"
    assert str(profile["forward_prediction_source"].iloc[0]) == "recommended_voltage_v"
    assert bool(profile["predicted_from_plotted_command"].iloc[0]) is True


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
