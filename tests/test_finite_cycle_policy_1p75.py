from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


TEST_ROOT = Path(__file__).resolve().parent
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

import test_finite_empirical_field_route as finite_fixture


def test_exact_one_point_seven_five_support_is_selected_without_substitution() -> None:
    result = finite_fixture._run_field_compensation(
        finite_support_entries=[
            finite_fixture._build_finite_entry(
                test_id="support_1p5",
                waveform_type="sine",
                freq_hz=1.0,
                cycle_count=1.5,
                field_pp=100.0,
            ),
            finite_fixture._build_finite_entry(
                test_id="support_1p75",
                waveform_type="sine",
                freq_hz=1.0,
                cycle_count=1.75,
                field_pp=100.0,
            ),
        ],
        target_cycle_count=1.75,
        waveform_type="sine",
        freq_hz=1.0,
    )

    assert result["finite_support_used"] is True
    assert result["finite_route_mode"] == "finite_empirical_field_support"
    assert result["finite_route_reason"] == "exact_finite_support_match"
    assert result["selected_support_id"] == "support_1p75"
    assert result["selected_support_cycle_count"] == 1.75
    assert result["exact_cycle_support_used"] is True
    assert result["finite_prediction_available"] is True
    assert result["finite_cycle_decomposition_mode"] == "whole_exact_1_75_support"


def test_one_point_seven_five_without_exact_support_does_not_substitute_zero_point_seven_five_or_one_point_five() -> None:
    result = finite_fixture._run_field_compensation(
        finite_support_entries=[
            finite_fixture._build_finite_entry(
                test_id="support_0p75",
                waveform_type="triangle",
                freq_hz=1.0,
                cycle_count=0.75,
                field_pp=100.0,
            ),
            finite_fixture._build_finite_entry(
                test_id="support_1p5",
                waveform_type="triangle",
                freq_hz=1.0,
                cycle_count=1.5,
                field_pp=100.0,
            ),
        ],
        target_cycle_count=1.75,
        waveform_type="triangle",
        freq_hz=1.0,
    )

    assert result["finite_support_used"] is False
    assert result["finite_route_mode"] == "finite_unavailable_no_exact_1_75_support"
    assert result["finite_route_reason"] == "no_exact_1_75_support"
    assert result["finite_prediction_unavailable_reason"] == "no_exact_1_75_support"
    assert result["exact_cycle_support_used"] is False
    assert result["support_tests_used"] == []
    profile = result["command_profile"]
    assert pd.to_numeric(profile["predicted_field_mT"], errors="coerce").isna().all()
    assert pd.to_numeric(profile["support_scaled_field_mT"], errors="coerce").isna().all()


def test_zero_point_seven_five_is_not_primary_supported_cycle() -> None:
    result = finite_fixture._run_field_compensation(
        finite_support_entries=[
            finite_fixture._build_finite_entry(
                test_id="support_0p75",
                waveform_type="sine",
                freq_hz=1.0,
                cycle_count=0.75,
                field_pp=100.0,
            )
        ],
        target_cycle_count=0.75,
        waveform_type="sine",
        freq_hz=1.0,
    )

    assert result["finite_support_used"] is False
    assert result["finite_route_mode"] == "finite_unavailable_unsupported_cycle_count"
    assert result["finite_route_reason"] == "unsupported_cycle_count"
    assert result["finite_prediction_unavailable_reason"] == "unsupported_cycle_count"
    assert result["exact_cycle_support_used"] is False
    assert result["supported_cycle_counts"] == [1.0, 1.25, 1.5, 1.75]
