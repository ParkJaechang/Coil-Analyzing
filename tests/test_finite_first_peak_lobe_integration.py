from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.final_modeled_lut import build_final_modeled_voltage_lut_export
from field_analysis.finite_first_phase_sync import apply_finite_first_phase_sync_modeling


def _segmented_peak_profile(*, cycle_count: float = 1.5, samples: int = 1201) -> pd.DataFrame:
    time_s = np.linspace(0.0, cycle_count + 0.25, samples, endpoint=False)
    active = time_s <= cycle_count + 1e-12
    shape = np.sin(2.0 * np.pi * time_s)
    target = np.where(active, shape * 50.0, 0.0)
    base_voltage = np.where(active, shape, 0.0)
    measured = np.zeros_like(time_s)
    measured = np.where(active & (time_s < 0.5), shape * 40.0, measured)
    measured = np.where(active & (time_s >= 0.5) & (time_s < 1.0), shape * 20.0, measured)
    measured = np.where(active & (time_s >= 1.0) & (time_s <= 1.5), shape * 30.0, measured)
    return pd.DataFrame(
        {
            "time_s": time_s,
            "limited_voltage_v": base_voltage,
            "recommended_voltage_v": base_voltage,
            "physical_target_output_mT": target,
            "finite_first_actual_measured_field_mT": measured,
            "waveform_type": "triangle",
        }
    )


def _lobe_values(values: pd.Series, time_s: pd.Series, start: float, end: float) -> np.ndarray:
    mask = (time_s >= start - 1e-12) & (time_s <= end + 1e-12)
    return pd.to_numeric(values.loc[mask], errors="coerce").to_numpy(dtype=float)


def test_finite_first_peak_lobe_integration_scales_1p5_lobes_before_residual_trim() -> None:
    result, metadata = apply_finite_first_phase_sync_modeling(
        _segmented_peak_profile(cycle_count=1.5),
        freq_hz=1.0,
        cycle_count=1.5,
        voltage_limit_v=10.0,
        target_peak_field_mT=50.0,
    )

    assert metadata["finite_first_modeling_status"] == "ok"
    assert metadata["finite_first_gain_mode"] == "peak_lobe"
    assert metadata["peak_lobe_enabled"] is True
    assert metadata["peak_lobe_status"] == "ok"
    assert metadata["peak_lobe_cycle_policy"] == "1.5cycle_three_peak"
    assert metadata["peak_lobe_lobe_count"] == 3
    assert metadata["peak_lobe_expected_polarities"] == "positive,negative,positive"
    assert metadata["peak_lobe_detected_polarities"] == "positive,negative,positive"
    assert metadata["base_command_source"] == "peak_lobe_command_voltage_v"
    assert metadata["residual_for_modeling_source"] == "peak_lobe_predicted_field"
    assert metadata["residual_trim_source"] == "field_per_volt_response"
    assert metadata["residual_to_voltage_conversion_basis"] == "field_per_volt_response_on_peak_lobe_residual"
    assert metadata["final_voltage_source_column"] == "limited_voltage_v"
    assert metadata["peak_lobe_gain_envelope_smoothed"] is False
    assert metadata["peak_lobe_boundary_taper_applied"] is False

    for column in (
        "peak_lobe_gain_envelope",
        "peak_lobe_command_voltage_v",
        "peak_lobe_predicted_field_mT",
        "finite_first_base_voltage_v",
        "correction_delta_v",
        "limited_voltage_v",
    ):
        assert column in result.columns

    first = _lobe_values(result["peak_lobe_command_voltage_v"], result["time_s"], 0.0, 0.5)
    second = _lobe_values(result["peak_lobe_command_voltage_v"], result["time_s"], 0.5, 1.0)
    third = _lobe_values(result["peak_lobe_command_voltage_v"], result["time_s"], 1.0, 1.5)
    assert float(np.nanmax(first)) == pytest.approx(1.25, rel=0.03)
    assert float(np.nanmin(second)) == pytest.approx(-2.5, rel=0.03)
    assert float(np.nanmax(third)) == pytest.approx(50.0 / 30.0, rel=0.03)
    assert np.nanmax(np.abs(result["limited_voltage_v"])) <= 10.0 + 1e-12


def test_finite_first_peak_lobe_integration_scales_1cycle_lobes() -> None:
    profile = _segmented_peak_profile(cycle_count=1.0)
    profile.loc[profile["time_s"] >= 1.0, "finite_first_actual_measured_field_mT"] = 0.0

    result, metadata = apply_finite_first_phase_sync_modeling(
        profile,
        freq_hz=1.0,
        cycle_count=1.0,
        voltage_limit_v=10.0,
        target_peak_field_mT=50.0,
    )

    assert metadata["finite_first_gain_mode"] == "peak_lobe"
    assert metadata["peak_lobe_enabled"] is True
    assert metadata["peak_lobe_cycle_policy"] == "1.0cycle_two_peak"
    assert metadata["peak_lobe_lobe_count"] == 2
    assert metadata["peak_lobe_expected_polarities"] == "positive,negative"
    assert metadata["peak_lobe_detected_polarities"] == "positive,negative"
    assert "peak_lobe_command_voltage_v" in result.columns


def test_finite_first_peak_lobe_disabled_falls_back_explicitly() -> None:
    profile = _segmented_peak_profile(cycle_count=1.5)
    profile["physical_target_output_mT"] = -profile["physical_target_output_mT"]

    result, metadata = apply_finite_first_phase_sync_modeling(
        profile,
        freq_hz=1.0,
        cycle_count=1.5,
        voltage_limit_v=10.0,
        target_peak_field_mT=50.0,
    )

    assert metadata["peak_lobe_enabled"] is False
    assert metadata["peak_lobe_status"] == "unexpected_lobe_polarity_sequence"
    assert metadata["fallback_on_peak_lobe_disabled"] is True
    assert metadata["peak_lobe_fallback_reason"] == "unexpected_lobe_polarity_sequence"
    assert metadata["finite_first_gain_mode"] == "field_per_volt_response"
    assert metadata["base_command_source"] == "field_per_volt_response"
    assert metadata["residual_for_modeling_source"] == "phase_aligned_measured"
    assert np.nanmax(np.abs(result["limited_voltage_v"])) <= 10.0 + 1e-12


def test_finite_first_peak_lobe_voltage_limit_metadata_and_export_contract() -> None:
    profile = _segmented_peak_profile(cycle_count=1.5)
    profile["finite_first_actual_measured_field_mT"] *= 0.05

    result, metadata = apply_finite_first_phase_sync_modeling(
        profile,
        freq_hz=1.0,
        cycle_count=1.5,
        voltage_limit_v=10.0,
        target_peak_field_mT=50.0,
    )
    payload = build_final_modeled_voltage_lut_export(result, freq_hz=1.0, cycle_count=1.5, waveform="triangle")

    assert metadata["peak_lobe_voltage_limit_exceeded"] is True
    assert metadata["peak_lobe_command_peak_abs_v"] > 10.0
    assert np.nanmax(np.abs(result["limited_voltage_v"])) <= 10.0 + 1e-12
    assert list(payload["frame"].columns) == ["sample_index", "time_s", "voltage_v"]
    assert payload["metadata"]["voltage_source_column"] == "limited_voltage_v"
