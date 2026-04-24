from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


TEST_ROOT = Path(__file__).resolve().parent
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

import test_finite_empirical_field_route as finite_fixture


def test_zero_point_seven_five_cycle_is_not_primary_supported() -> None:
    result = finite_fixture._run_field_compensation(
        finite_support_entries=[
            finite_fixture._build_finite_entry(
                test_id="zero_point_seven_five",
                waveform_type="triangle",
                freq_hz=1.0,
                cycle_count=0.75,
                field_pp=100.0,
            )
        ],
        target_cycle_count=0.75,
        waveform_type="triangle",
        freq_hz=1.0,
    )

    assert result["finite_support_used"] is False
    assert result["finite_prediction_available"] is False
    assert result["finite_route_mode"] == "finite_unavailable_unsupported_cycle_count"
    assert result["finite_route_reason"] == "unsupported_cycle_count"
    assert result["finite_prediction_unavailable_reason"] == "unsupported_cycle_count"
    assert result["finite_cycle_decomposition_mode"] == "unsupported_fractional_cycle_request"
    assert float(result["target_cycle_count"]) == 0.75
    assert int(result["target_integer_cycle_count"]) == 0
    assert float(result["target_terminal_fraction"]) == 0.75
    assert 0.75 not in result["supported_cycle_counts"]
    assert 1.75 in result["supported_cycle_counts"]
    profile = result["command_profile"]
    assert pd.to_numeric(profile["predicted_field_mT"], errors="coerce").isna().all()
    assert pd.to_numeric(profile["support_scaled_field_mT"], errors="coerce").isna().all()


def test_one_point_seven_five_policy_uses_exact_support() -> None:
    result = finite_fixture._run_field_compensation(
        finite_support_entries=[
            finite_fixture._build_finite_entry(
                test_id="one_point_seven_five",
                waveform_type="sine",
                freq_hz=1.0,
                cycle_count=1.75,
                field_pp=100.0,
            )
        ],
        target_cycle_count=1.75,
        waveform_type="sine",
        freq_hz=1.0,
    )

    assert result["finite_support_used"] is True
    assert result["finite_route_mode"] == "finite_empirical_field_support"
    assert result["finite_route_reason"] == "exact_finite_support_match"
    assert result["exact_cycle_support_used"] is True
    assert result["selected_support_cycle_count"] == 1.75
    assert result["finite_prediction_available"] is True
    assert float(result["target_terminal_fraction"]) == 0.75
    assert result["finite_cycle_policy_version"] == "field_route_cycles_v3"
    assert 1.75 in result["supported_cycle_counts"]
    assert 0.75 not in result["supported_cycle_counts"]
