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
    assert result["support_reference_plotted_column"] == "support_reference_output_mT"
    assert result["support_reference_selected_support_id"] == result["selected_support_id"]
    assert "support_reference_trace_status" in profile.columns
    assert str(profile["support_reference_plotted_column"].iloc[0]) == "support_reference_output_mT"
    assert np.allclose(profile["support_reference_output_mT"], profile["support_scaled_field_mT"], equal_nan=True)
    assert not np.allclose(
        profile["support_reference_output_mT"],
        profile["predicted_field_mT"],
        equal_nan=True,
    )


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
