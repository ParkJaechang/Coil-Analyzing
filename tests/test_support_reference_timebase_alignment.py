from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = REPO_ROOT / "tests"
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

import test_finite_empirical_field_route as finite_fixture
from test_support_reference_integrity import _support_entry_with_prebaseline_and_tail


def test_support_reference_active_segment_contract_metadata_is_stable() -> None:
    result = finite_fixture._run_field_compensation(
        finite_support_entries=[_support_entry_with_prebaseline_and_tail()],
        waveform_type="sine",
        freq_hz=3.0,
        target_cycle_count=1.25,
    )
    profile = result["command_profile"]
    expected_duration = 1.25 / 3.0

    required_keys = [
        "support_reference_anchor_mode",
        "support_reference_timebase_mapping_mode",
        "support_reference_source_window_start_s",
        "support_reference_source_window_end_s",
        "support_reference_source_window_duration_s",
        "support_reference_expected_duration_s",
        "source_pre_baseline_excluded_from_reference",
        "source_tail_excluded_from_reference",
    ]
    missing_result = [key for key in required_keys if key not in result]
    missing_profile = [key for key in required_keys if key not in profile.columns]

    assert not missing_result
    assert not missing_profile
    assert result["support_reference_timebase_mapping_mode"] == "active_segment_to_target_window"
    assert result["support_reference_alignment_status"] == "ok"
    assert result["source_pre_baseline_excluded_from_reference"] is True
    assert result["source_tail_excluded_from_reference"] is True
    assert np.isclose(result["support_reference_source_window_duration_s"], expected_duration, atol=0.01)
    assert np.isclose(result["support_reference_expected_duration_s"], expected_duration, atol=1e-12)


def test_support_reference_is_diagnostic_only_and_not_full_record_mapping() -> None:
    result = finite_fixture._run_field_compensation(
        finite_support_entries=[_support_entry_with_prebaseline_and_tail()],
        waveform_type="sine",
        freq_hz=3.0,
        target_cycle_count=1.25,
    )
    profile = result["command_profile"]

    assert result["support_reference_used_for_command"] is False
    assert result["support_reference_role"] == "diagnostic_reference"
    assert result["support_reference_plotted_column"] == "support_reference_output_mT"
    assert result["support_reference_plotted_source"] == "target_aligned_support_reference"
    assert result["support_reference_timebase"] == "target_aligned"
    assert "full_record_to_target_window" not in set(profile["support_reference_timebase_mapping_mode"].astype(str))
    assert np.allclose(
        profile["support_reference_output_mT"],
        profile["target_aligned_support_reference_mT"],
        equal_nan=True,
    )
