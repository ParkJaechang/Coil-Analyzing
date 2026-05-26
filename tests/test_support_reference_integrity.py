from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = REPO_ROOT / "tests"
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

import test_finite_empirical_field_route as finite_fixture
from field_analysis.app_ui_snapshot import _resolve_compensation_plot_reference


def _support_entries() -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for waveform_type in ("sine", "triangle"):
        for freq_hz in (1.0, 3.0, 5.0):
            for cycle_count in (1.0, 1.25, 1.5):
                entries.append(
                    finite_fixture._build_finite_entry(
                        test_id=f"{waveform_type}_{freq_hz:g}hz_{cycle_count:g}cy",
                        waveform_type=waveform_type,
                        freq_hz=freq_hz,
                        cycle_count=cycle_count,
                        field_pp=80.0 + freq_hz * 4.0 + cycle_count * 3.0,
                    )
                )
    return entries


def _support_entry_with_prebaseline_and_tail() -> dict[str, object]:
    freq_hz = 3.0
    cycle_count = 1.25
    active_duration_s = cycle_count / freq_hz
    motion_start_s = 0.4
    total_duration_s = 1.4
    time_s = np.linspace(0.0, total_duration_s, 420)
    active_rel = time_s - motion_start_s
    active_mask = (active_rel >= 0.0) & (active_rel <= active_duration_s)
    tail_mask = active_rel > active_duration_s
    phase = np.clip(active_rel / active_duration_s, 0.0, 1.0)
    active_wave = np.sin(np.pi * phase)
    tail = np.exp(-(active_rel - active_duration_s) * 2.0)
    field = np.where(active_mask, active_wave * 90.0, 0.0)
    field = np.where(tail_mask, 45.0 * tail, field)
    current = np.where(active_mask, active_wave * 8.0, 0.0)
    current = np.where(tail_mask, 4.0 * tail, current)
    voltage = np.where(active_mask, active_wave * 5.0, 0.0)
    voltage = np.where(tail_mask, 2.5 * tail, voltage)
    return {
        "test_id": "finite_prebaseline_tail_source",
        "waveform_type": "sine",
        "freq_hz": freq_hz,
        "approx_cycle_span": cycle_count,
        "field_pp": 90.0,
        "current_pp": 8.0,
        "daq_voltage_pp": 10.0,
        "frame": pd.DataFrame(
            {
                "time_s": time_s,
                "daq_input_v": voltage,
                "i_sum_signed": current,
                "bz_mT": field,
            }
        ),
    }


def _trace_summary(result: dict[str, object]) -> tuple[str, str, str, float, float]:
    profile = result["command_profile"]
    column = str(result["support_reference_plotted_column"])
    values = pd.to_numeric(profile[column], errors="coerce").to_numpy(dtype=float)
    time_s = pd.to_numeric(profile["time_s"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values)
    peak_to_peak = float(np.nanmax(values[finite]) - np.nanmin(values[finite]))
    duration_s = float(np.nanmax(time_s) - np.nanmin(time_s))
    return (
        str(result["selected_support_id"]),
        str(result["finite_route_mode"]),
        column,
        round(peak_to_peak, 3),
        round(duration_s, 3),
    )


def test_support_reference_contract_matches_plotted_selected_support_trace() -> None:
    result = finite_fixture._run_field_compensation(
        finite_support_entries=_support_entries(),
        waveform_type="sine",
        freq_hz=3.0,
        target_cycle_count=1.25,
    )
    profile = result["command_profile"]

    assert result["support_reference_trace_status"] == "ok"
    assert result["support_reference_source_label"] == "selected_support_trace"
    assert result["support_reference_timebase"] == "target_aligned"
    assert result["support_reference_plotted_source"] == "target_aligned_support_reference"
    assert result["support_reference_alignment_status"] == "ok"
    assert result["support_reference_role"] == "diagnostic_reference"
    assert result["support_reference_used_for_command"] is False
    assert result["support_reference_plotted_column"] == "support_reference_output_mT"
    assert result["support_reference_selected_support_id"] == result["selected_support_id"]
    assert "support_reference_trace_status" in profile.columns
    assert str(profile["support_reference_plotted_column"].iloc[0]) == "support_reference_output_mT"
    assert "target_aligned_support_reference_mT" in profile.columns
    assert np.allclose(profile["support_reference_output_mT"], profile["support_scaled_field_mT"], equal_nan=True)
    assert np.allclose(profile["support_reference_output_mT"], profile["target_aligned_support_reference_mT"], equal_nan=True)
    assert not np.allclose(
        profile["support_reference_output_mT"],
        profile["predicted_field_mT"],
        equal_nan=True,
    )


def test_support_reference_uses_motion_start_plus_requested_duration_not_full_record_or_tail() -> None:
    result = finite_fixture._run_field_compensation(
        finite_support_entries=[_support_entry_with_prebaseline_and_tail()],
        waveform_type="sine",
        freq_hz=3.0,
        target_cycle_count=1.25,
    )
    profile = result["command_profile"]
    expected_duration = 1.25 / 3.0
    expected_start = 0.4
    expected_end = expected_start + expected_duration

    assert result["support_reference_timebase_mapping_mode"] == "active_segment_to_target_window"
    assert result["support_reference_anchor_mode"] in {
        "command_start_plus_declared_duration",
        "motion_start_plus_declared_duration",
    }
    assert result["support_reference_alignment_status"] == "ok"
    assert np.isclose(result["support_reference_source_window_start_s"], expected_start, atol=0.01)
    assert np.isclose(result["support_reference_source_window_end_s"], expected_end, atol=0.01)
    assert np.isclose(result["support_reference_source_window_duration_s"], expected_duration, atol=0.01)
    assert np.isclose(result["support_reference_expected_duration_s"], expected_duration, atol=1e-9)
    assert result["source_pre_baseline_excluded_from_reference"] is True
    assert result["source_tail_excluded_from_reference"] is True
    assert "full_record_to_target_window" not in str(result["support_reference_timebase_mapping_mode"])
    assert str(profile["support_reference_timebase_mapping_mode"].iloc[0]) == "active_segment_to_target_window"
    assert bool(profile["source_pre_baseline_excluded_from_reference"].iloc[0]) is True
    assert bool(profile["source_tail_excluded_from_reference"].iloc[0]) is True


def test_ui_support_reference_preview_uses_native_source_beyond_target_end() -> None:
    result = finite_fixture._run_field_compensation(
        finite_support_entries=[_support_entry_with_prebaseline_and_tail()],
        waveform_type="sine",
        freq_hz=3.0,
        target_cycle_count=1.25,
    )
    reference_profile, reference_column, _label, reference_source, _pp = _resolve_compensation_plot_reference(result)

    assert reference_profile is not None
    assert reference_column == "support_reference_native_mT"
    assert reference_source == "native measured support source"
    original_start = float(result["selected_support_original_nonzero_start_s"])
    original_end = float(result["selected_support_original_nonzero_end_s"])
    assert float(reference_profile["time_s"].min()) == pytest.approx(0.0)
    assert float(reference_profile["time_s"].max()) > float(result["target_active_end_s"])
    assert float(reference_profile["time_s"].max()) >= original_end - original_start - 1e-9
    assert reference_profile.attrs["support_reference_native_normalization_mode"] == "scale_only_abs_peak_to_50mT_after_motion_start"
    assert reference_profile.attrs["support_reference_native_offset_removed_mT"] == pytest.approx(0.0)
    assert float(reference_profile["support_reference_native_mT"].iloc[0]) >= -1e-9


def test_support_reference_trace_changes_across_frequency_and_cycle_conditions() -> None:
    cases = [
        ("sine", 1.0, 1.0),
        ("triangle", 1.0, 1.0),
        ("sine", 3.0, 1.25),
        ("triangle", 5.0, 1.5),
    ]

    summaries = [
        _trace_summary(
            finite_fixture._run_field_compensation(
                finite_support_entries=_support_entries(),
                waveform_type=waveform_type,
                freq_hz=freq_hz,
                target_cycle_count=cycle_count,
            )
        )
        for waveform_type, freq_hz, cycle_count in cases
    ]

    assert all(summary[2] == "support_reference_output_mT" for summary in summaries)
    assert len({summary[0] for summary in summaries}) >= 3
    assert len({(summary[3], summary[4]) for summary in summaries}) >= 3
