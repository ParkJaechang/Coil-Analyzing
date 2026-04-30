from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = REPO_ROOT / "tests"
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

import test_finite_empirical_field_route as finite_fixture


def _finite_entry_with_rest_windows() -> dict[str, object]:
    time_s = np.linspace(0.0, 2.0, 401)
    active_start_s = 0.5
    active_end_s = 1.5
    active = (time_s >= active_start_s) & (time_s <= active_end_s)
    active_progress = np.clip((time_s - active_start_s) / (active_end_s - active_start_s), 0.0, 1.0)
    field = np.zeros_like(time_s)
    current = np.zeros_like(time_s)
    voltage = np.zeros_like(time_s)
    waveform = np.sin(np.pi * active_progress)
    field[active] = finite_fixture._scaled_waveform(waveform[active], 90.0)
    current[active] = finite_fixture._scaled_waveform(waveform[active], 8.0)
    voltage[active] = finite_fixture._scaled_waveform(waveform[active], 6.0)
    return {
        "test_id": "finite_sine_1Hz_1cycle_with_rest_windows",
        "waveform_type": "sine",
        "freq_hz": 1.0,
        "source_file": "finite_sine_1Hz_1cycle.csv",
        "approx_cycle_span": 1.0,
        "field_pp": 90.0,
        "current_pp": 8.0,
        "daq_voltage_pp": 6.0,
        "frame": pd.DataFrame(
            {
                "time_s": time_s,
                "daq_input_v": voltage,
                "i_sum_signed": current,
                "bz_mT": field,
            }
        ),
    }


def test_support_reference_uses_active_window_not_full_record_compression() -> None:
    result = finite_fixture._run_field_compensation(
        finite_support_entries=[_finite_entry_with_rest_windows()],
        waveform_type="sine",
        freq_hz=1.0,
        target_cycle_count=1.0,
    )

    assert result["source_total_duration_s"] == 2.0
    assert result["source_pre_baseline_s"] == 0.5
    assert result["source_command_active_start_s"] == 0.5
    assert result["source_command_active_end_s"] == 1.5
    assert result["source_post_tail_s"] == 0.5
    assert result["source_active_duration_s"] == 1.0
    assert result["support_reference_alignment_window"] == "command_active_window"
    assert result["support_reference_timebase_mapping_mode"] == "active_window_to_target_window"
    assert result["support_reference_alignment_status"] == "ok"
    assert result["support_reference_timebase"] == "target_aligned"
    assert result["support_reference_plotted_column"] == "support_reference_output_mT"
    assert result["support_reference_plotted_source"] == "target_aligned_support_reference"


def test_support_reference_trace_is_separate_from_raw_source_record() -> None:
    result = finite_fixture._run_field_compensation(
        finite_support_entries=[_finite_entry_with_rest_windows()],
        waveform_type="sine",
        freq_hz=1.0,
        target_cycle_count=1.0,
    )
    profile = result["command_profile"]
    raw_time = np.asarray(result["selected_support_source_time_s"], dtype=float)
    reference_time = np.asarray(result["support_reference_time_s"], dtype=float)
    reference = pd.to_numeric(profile["support_reference_output_mT"], errors="coerce").to_numpy(dtype=float)
    target_time = pd.to_numeric(profile["time_s"], errors="coerce").to_numpy(dtype=float)

    assert np.isclose(float(np.nanmax(raw_time) - np.nanmin(raw_time)), 2.0)
    assert float(np.nanmax(raw_time) - np.nanmin(raw_time)) > float(np.nanmax(reference_time) - np.nanmin(reference_time))
    assert np.allclose(reference_time, target_time)
    assert np.isfinite(reference[target_time <= float(result["target_active_end_s"]) + 1e-9]).any()
    assert np.isnan(reference[target_time > float(result["target_active_end_s"]) + 1e-9]).any()
    assert result["support_reference_used_for_command"] is False
    assert result["support_reference_role"] == "diagnostic_reference"
    assert result["target_shape_family"] == "rounded_triangle"
    assert float(result["target_pp_fixed"]) == 100.0
