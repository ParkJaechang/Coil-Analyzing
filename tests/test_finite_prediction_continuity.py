from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


TEST_ROOT = Path(__file__).resolve().parent
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

import test_finite_empirical_field_route as finite_fixture


def _adjacent_jump_ratio(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size < 2:
        return 0.0
    peak_to_peak = float(np.nanmax(finite_values) - np.nanmin(finite_values))
    if not np.isfinite(peak_to_peak) or peak_to_peak <= 1e-9:
        return 0.0
    return float(np.nanmax(np.abs(np.diff(values))) / peak_to_peak)


def _build_spiky_finite_entry(
    *,
    test_id: str = "triangle_spike",
    waveform_type: str = "triangle",
    freq_hz: float = 1.0,
    cycle_count: float = 1.25,
    spike_time_s: float = 1.138,
    spike_value_mT: float = 382.76,
) -> dict[str, object]:
    entry = finite_fixture._build_finite_entry(
        test_id=test_id,
        waveform_type=waveform_type,
        freq_hz=freq_hz,
        cycle_count=cycle_count,
        field_pp=100.0,
    )
    frame = entry["frame"].copy()
    spike_index = int(np.abs(frame["time_s"].to_numpy(dtype=float) - float(spike_time_s)).argmin())
    frame.loc[spike_index, "bz_mT"] = float(spike_value_mT)
    entry["frame"] = frame
    return entry


def test_runtime_like_finite_prediction_has_no_impulse_spikes() -> None:
    for waveform_type in ("sine", "triangle"):
        for cycle_count in (1.0, 1.25, 1.5, 1.75):
            result = finite_fixture._run_field_compensation(
                finite_support_entries=[
                    finite_fixture._build_finite_entry(
                        test_id=f"{waveform_type}_{cycle_count}",
                        waveform_type=waveform_type,
                        freq_hz=1.0,
                        cycle_count=cycle_count,
                        field_pp=100.0,
                    )
                ],
                target_cycle_count=cycle_count,
                waveform_type=waveform_type,
                freq_hz=1.0,
            )
            profile = result["command_profile"]

            assert _adjacent_jump_ratio(profile, "predicted_field_mT") <= 0.20
            assert _adjacent_jump_ratio(profile, "support_scaled_field_mT") <= 0.20
            assert float(result["predicted_jump_ratio"]) <= 0.20
            assert float(result["support_jump_ratio"]) <= 0.20
            assert result["support_continuity_status"] == "ok"
            if bool(result["terminal_trim_applied"]):
                assert float(result["terminal_trim_window_fraction"]) <= 0.20 + 1e-9


def test_triangle_one_hz_one_point_two_five_source_spike_is_not_returned_as_prediction() -> None:
    result = finite_fixture._run_field_compensation(
        finite_support_entries=[_build_spiky_finite_entry()],
        target_cycle_count=1.25,
        waveform_type="triangle",
        freq_hz=1.0,
    )

    profile = result["command_profile"]
    spike_window = profile.loc[
        (profile["time_s"] >= 1.05) & (profile["time_s"] <= 1.20),
        ["time_s", "target_field_mT", "predicted_field_mT", "support_scaled_field_mT", "recommended_voltage_v"],
    ]
    assert not spike_window.empty
    assert result["support_source_spike_detected"] is True
    assert int(result["support_spike_filtered_count"]) >= 1
    assert float(result["predicted_jump_ratio"]) <= 0.20
    assert float(result["support_jump_ratio"]) <= 0.20
    assert result["support_continuity_status"] == "ok"
    assert float(spike_window["predicted_field_mT"].abs().max()) < 160.0
    assert float(spike_window["support_scaled_field_mT"].abs().max()) < 160.0


def test_weighted_support_blend_boundary_keeps_value_continuity() -> None:
    result = finite_fixture._run_field_compensation(
        finite_support_entries=[
            finite_fixture._build_finite_entry(
                test_id="triangle_low",
                waveform_type="triangle",
                freq_hz=1.0,
                cycle_count=1.25,
                field_pp=72.0,
            ),
            finite_fixture._build_finite_entry(
                test_id="triangle_high",
                waveform_type="triangle",
                freq_hz=1.0,
                cycle_count=1.25,
                field_pp=138.0,
            ),
        ],
        target_cycle_count=1.25,
        waveform_type="triangle",
        freq_hz=1.0,
    )

    assert result["finite_support_used"] is True
    assert result["support_count_used"] == 2
    assert result["support_blend_boundary_count"] == 1
    assert float(result["predicted_jump_ratio"]) <= 0.20
    assert float(result["support_jump_ratio"]) <= 0.20
    assert result["support_continuity_status"] == "ok"


def test_one_point_seven_five_cycle_rejects_short_whole_substitutions() -> None:
    for support_cycle_count in (0.75, 1.5):
        result = finite_fixture._run_field_compensation(
            finite_support_entries=[
                finite_fixture._build_finite_entry(
                    test_id=f"whole_{support_cycle_count}",
                    waveform_type="triangle",
                    freq_hz=1.0,
                    cycle_count=support_cycle_count,
                    field_pp=100.0,
                )
            ],
            target_cycle_count=1.75,
            waveform_type="triangle",
            freq_hz=1.0,
        )

        assert result["finite_support_used"] is False
        assert result["finite_route_mode"] == "steady_state_harmonic_expanded"
        assert result["finite_cycle_decomposition_mode"] == "fallback_no_safe_1_75_decomposition"
        assert result["cycle_semantics_warning"] == "1.75_requires_1_full_cycle_plus_0.75_terminal_tail_or_exact_support"
        assert result["whole_support_substitution_used"] is False
        assert result["whole_support_substitution_valid"] is True


def test_exact_one_point_seven_five_support_records_cycle_semantics_without_spikes() -> None:
    result = finite_fixture._run_field_compensation(
        finite_support_entries=[
            finite_fixture._build_finite_entry(
                test_id="exact_one_point_seven_five",
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
    assert result["finite_cycle_decomposition_mode"] == "whole_exact_1_75_support"
    assert float(result["target_terminal_fraction"]) == 0.75
    assert result["whole_support_substitution_used"] is False
    assert result["whole_support_substitution_valid"] is True
    assert float(result["predicted_jump_ratio"]) <= 0.20
    assert float(result["support_jump_ratio"]) <= 0.20
