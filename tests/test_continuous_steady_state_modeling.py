from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = REPO_ROOT / "tests"
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

import test_compensation_field_route as field_fixture


def test_continuous_payload_exposes_steady_state_window_and_metrics() -> None:
    summary, analyses = field_fixture._build_support_context()
    result = field_fixture._run_field_compensation(summary, analyses, freq_hz=3.0, target_output_pp=45.0)
    profile = result["command_profile"]

    assert result["startup_excluded"] is True
    assert result["continuous_evaluation_window"] == "steady_state"
    assert result["continuous_command_objective"] == "steady_state"
    assert result["startup_used_for_command_generation"] is False
    assert result["steady_state_used_for_command_generation"] is True
    assert result["steady_state_window_source"] == "representative_cycle_selection"
    assert result["steady_state_window_reason"] == "startup_cycles_excluded_from_continuous_objective"
    assert result["continuous_steady_state_selection_mode"] == "last_n_cycles"
    assert result["continuous_source_total_duration_s"] > result["steady_state_duration_s"]
    assert result["continuous_startup_excluded_until_s"] == result["startup_window_end_s"]
    assert result["source_absolute_steady_state_start_s"] >= result["startup_window_end_s"]
    assert result["source_absolute_steady_state_end_s"] > result["source_absolute_steady_state_start_s"]
    assert np.isclose(result["steady_state_local_start_s"], float(profile["time_s"].min()))
    assert np.isclose(result["steady_state_local_end_s"], float(profile["time_s"].max()))
    assert np.isclose(result["steady_state_start_s"], float(profile["time_s"].min()))
    assert np.isclose(result["steady_state_end_s"], float(profile["time_s"].max()))
    assert result["steady_state_duration_s"] > 0.0
    assert np.isfinite(result["steady_state_nrmse"])
    assert np.isfinite(result["steady_state_shape_corr"])
    assert np.isfinite(result["steady_state_peak_error"])
    assert result["whole_window_metrics_debug_only"] is True
    for column in (
        "steady_state_start_s",
        "steady_state_end_s",
        "startup_excluded",
        "continuous_evaluation_window",
        "continuous_command_objective",
        "startup_used_for_command_generation",
        "steady_state_used_for_command_generation",
        "steady_state_window_source",
        "steady_state_window_reason",
        "steady_state_nrmse",
        "steady_state_shape_corr",
        "steady_state_peak_error",
    ):
        assert column in profile.columns


def test_continuous_startup_contamination_is_debug_only_for_whole_window_metrics() -> None:
    import test_startup_transient_compensation as startup_fixture

    analysis = startup_fixture._build_startup_offset_analysis(first_cycle_field_offset_mT=18.0, freq_hz=5.0)
    result = startup_fixture._run_compensation(analysis=analysis, freq_hz=5.0)

    assert result["startup_excluded"] is True
    assert result["continuous_evaluation_window"] == "steady_state"
    assert result["whole_window_metrics_debug_only"] is True
    assert result["whole_window_nrmse_debug"] != result["steady_state_nrmse"]
    assert result["whole_window_shape_corr_debug"] != result["steady_state_shape_corr"]
    assert result["startup_used_for_command_generation"] is False
    assert result["steady_state_used_for_command_generation"] is True


def test_continuous_steady_state_contract_preserves_target_and_forward_causality() -> None:
    summary, analyses = field_fixture._build_support_context()
    result = field_fixture._run_field_compensation(summary, analyses, freq_hz=4.0, target_output_pp=45.0)
    profile = result["command_profile"]

    assert result["target_shape_family"] == "rounded_triangle"
    assert result["target_pp_fixed"] == field_fixture.FIELD_ROUTE_NORMALIZED_TARGET_PP
    assert profile["physical_target_output_mT"].equals(profile["target_field_mT"])
    assert result["predicted_from_plotted_command"] is True
    assert result["command_prediction_consistency_status"] == "ok"
    assert result["support_reference_used_for_command"] is False
    assert result["support_reference_role"] == "diagnostic_reference"
