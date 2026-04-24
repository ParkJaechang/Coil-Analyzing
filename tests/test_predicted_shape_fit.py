from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


TEST_ROOT = Path(__file__).resolve().parent
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

import test_finite_empirical_field_route as finite_fixture


def _rippled_entry(*, waveform_type: str, cycle_count: float, ripple_mt: float) -> dict[str, object]:
    entry = finite_fixture._build_finite_entry(
        test_id=f"{waveform_type}_{cycle_count}_ripple",
        waveform_type=waveform_type,
        freq_hz=1.0,
        cycle_count=cycle_count,
        field_pp=100.0,
    )
    frame = entry["frame"].copy()
    time_s = frame["time_s"].to_numpy(dtype=float)
    active_mask = time_s <= float(cycle_count) + 1e-12
    ripple = float(ripple_mt) * np.sin(2.0 * np.pi * 10.0 * time_s[active_mask])
    frame.loc[active_mask, "bz_mT"] = frame.loc[active_mask, "bz_mT"].to_numpy(dtype=float) + ripple
    entry["frame"] = frame
    return entry


def test_frequency_proxy_mismatch_triggers_shape_lock() -> None:
    result = finite_fixture._run_field_compensation(
        finite_support_entries=[_rippled_entry(waveform_type="triangle", cycle_count=1.25, ripple_mt=4.0)],
        target_cycle_count=1.25,
        waveform_type="triangle",
        freq_hz=1.0,
    )

    assert result["target_predicted_frequency_proxy_mismatch"] is False
    assert result["predicted_spike_detected"] is False
    assert result["predicted_kink_detected"] is False
    assert result["active_shape_fit_applied"] is True
    assert result["active_shape_fit_strength"] == 1.0
    assert result["active_shape_fit_reason"] == "active_target_shape_fit_frequency_lock"
    assert float(result["active_shape_corr"]) >= 0.99
    assert float(result["active_shape_nrmse"]) <= 0.05


def test_sine_one_point_two_five_shape_fit() -> None:
    for cycle_count in (1.25,):
        result = finite_fixture._run_field_compensation(
            finite_support_entries=[
                finite_fixture._build_finite_entry(
                    test_id=f"sine_{cycle_count}",
                    waveform_type="sine",
                    freq_hz=1.0,
                    cycle_count=cycle_count,
                    field_pp=100.0,
                )
            ],
            target_cycle_count=cycle_count,
            waveform_type="sine",
            freq_hz=1.0,
        )

        assert result["target_predicted_frequency_proxy_mismatch"] is False
        assert result["predicted_spike_detected"] is False
        assert result["predicted_kink_detected"] is False
        assert float(result["active_shape_corr"]) >= 0.92
        assert float(result["active_shape_nrmse"]) <= 0.24
        assert float(result["predicted_jump_ratio"]) <= 0.20


def test_zero_point_seven_five_is_unsupported() -> None:
    result = finite_fixture._run_field_compensation(
        finite_support_entries=[
            finite_fixture._build_finite_entry(
                test_id="sine_0p75",
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

    assert result["finite_route_mode"] == "finite_unavailable_unsupported_cycle_count"
    assert result["finite_prediction_available"] is False
    profile = result["command_profile"]
    assert pd.to_numeric(profile["predicted_field_mT"], errors="coerce").isna().all()
    assert pd.to_numeric(profile["support_scaled_field_mT"], errors="coerce").isna().all()


def test_one_point_seven_five_uses_exact_support() -> None:
    result = finite_fixture._run_field_compensation(
        finite_support_entries=[
            finite_fixture._build_finite_entry(
                test_id="sine_1p75",
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

    assert result["finite_route_mode"] == "finite_empirical_field_support"
    assert result["finite_prediction_available"] is True
    assert result["exact_cycle_support_used"] is True
    assert result["selected_support_cycle_count"] == 1.75
    assert result["target_predicted_frequency_proxy_mismatch"] is False
