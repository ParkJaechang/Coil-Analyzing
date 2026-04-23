from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


TEST_ROOT = Path(__file__).resolve().parent
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

import test_finite_empirical_field_route as finite_fixture


def _normalized(values: pd.Series) -> np.ndarray:
    raw = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    raw = raw[np.isfinite(raw)]
    centered = raw - float(np.nanmean(raw))
    peak = float(np.nanmax(np.abs(centered))) if centered.size else 0.0
    return centered / max(peak, 1e-9)


def test_physical_target_is_independent_of_requested_support_family() -> None:
    sine_result = finite_fixture._run_field_compensation(
        finite_support_entries=[
            finite_fixture._build_finite_entry(
                test_id="sine_support",
                waveform_type="sine",
                freq_hz=1.0,
                cycle_count=1.25,
                field_pp=100.0,
            )
        ],
        target_cycle_count=1.25,
        waveform_type="sine",
        freq_hz=1.0,
    )
    triangle_result = finite_fixture._run_field_compensation(
        finite_support_entries=[
            finite_fixture._build_finite_entry(
                test_id="triangle_support",
                waveform_type="triangle",
                freq_hz=1.0,
                cycle_count=1.25,
                field_pp=100.0,
            )
        ],
        target_cycle_count=1.25,
        waveform_type="triangle",
        freq_hz=1.0,
    )

    sine_profile = sine_result["command_profile"]
    triangle_profile = triangle_result["command_profile"]
    np.testing.assert_allclose(
        _normalized(sine_profile["physical_target_output_mT"]),
        _normalized(triangle_profile["physical_target_output_mT"]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _normalized(sine_profile["target_output"]),
        _normalized(sine_profile["physical_target_output_mT"]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _normalized(triangle_profile["target_output"]),
        _normalized(triangle_profile["physical_target_output_mT"]),
        atol=1e-12,
    )
    assert sine_result["target_shape_family"] == "rounded_triangle"
    assert triangle_result["target_shape_family"] == "rounded_triangle"
    assert float(sine_result["target_pp_fixed"]) == 100.0
    assert float(triangle_result["target_pp_fixed"]) == 100.0
    assert sine_result["support_family_requested"] == "sine"
    assert triangle_result["support_family_requested"] == "triangle"


def test_support_reference_is_separate_from_physical_target() -> None:
    result = finite_fixture._run_field_compensation(
        finite_support_entries=[
            finite_fixture._build_finite_entry(
                test_id="sine_support",
                waveform_type="sine",
                freq_hz=1.0,
                cycle_count=1.25,
                field_pp=100.0,
            )
        ],
        target_cycle_count=1.25,
        waveform_type="sine",
        freq_hz=1.0,
    )

    profile = result["command_profile"]
    for column in ("physical_target_output_mT", "support_reference_output_mT", "predicted_field_mT"):
        assert column in profile.columns
    assert result["physical_target_output_column"] == "physical_target_output_mT"
    assert result["support_reference_output_column"] == "support_reference_output_mT"
    assert result["predicted_output_column"] == "predicted_field_mT"
    assert not np.allclose(
        _normalized(profile["support_reference_output_mT"]),
        _normalized(profile["physical_target_output_mT"]),
        atol=1e-6,
    )
