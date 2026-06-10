from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.finite_first_peak_lobe import build_peak_lobe_model


def _finite_waveform(cycle_count: float, *, samples: int = 901) -> tuple[np.ndarray, np.ndarray]:
    time_s = np.linspace(0.0, cycle_count, samples)
    shape = np.sin(2.0 * np.pi * time_s)
    return time_s, shape


def _lobe_values(values: np.ndarray, time_s: np.ndarray, start: float, end: float) -> np.ndarray:
    mask = (time_s >= start - 1e-12) & (time_s <= end + 1e-12)
    return values[mask]


def test_peak_lobe_model_scales_three_1p5cycle_peaks_independently() -> None:
    time_s, shape = _finite_waveform(1.5)
    target = shape * 50.0
    measured = np.where(time_s < 0.5, shape * 40.0, np.where(time_s < 1.0, shape * 20.0, shape * 30.0))
    base_voltage = shape.copy()
    active_mask = np.ones_like(time_s, dtype=bool)

    result = build_peak_lobe_model(
        time_s=time_s,
        target_field_mT=target,
        aligned_measured_field_mT=measured,
        base_voltage_v=base_voltage,
        active_mask=active_mask,
        cycle_count=1.5,
    )

    assert result.enabled is True
    assert result.status == "ok"
    assert result.cycle_policy == "1.5cycle_three_peak"
    assert [lobe.polarity for lobe in result.lobes] == ["positive", "negative", "positive"]
    assert [lobe.gain for lobe in result.lobes] == pytest.approx([1.25, 2.5, 50.0 / 30.0], rel=0.02)
    assert [lobe.command_peak_v for lobe in result.lobes] == pytest.approx([1.25, -2.5, 50.0 / 30.0], rel=0.02)
    assert result.detected_lobe_polarities == ("positive", "negative", "positive")
    assert result.expected_lobe_polarities == ("positive", "negative", "positive")
    assert result.peak_lobe_gain_envelope_smoothed is False
    assert result.peak_lobe_boundary_taper_applied is False
    assert result.voltage_limit_exceeded is False
    assert result.peak_lobe_command_peak_abs_v == pytest.approx(2.5, rel=0.02)

    first = _lobe_values(result.peak_lobe_command_voltage_v, time_s, 0.0, 0.5)
    second = _lobe_values(result.peak_lobe_command_voltage_v, time_s, 0.5, 1.0)
    third = _lobe_values(result.peak_lobe_command_voltage_v, time_s, 1.0, 1.5)

    assert float(np.nanmax(first)) == pytest.approx(1.25, rel=0.02)
    assert float(np.nanmin(second)) == pytest.approx(-2.5, rel=0.02)
    assert float(np.nanmax(third)) == pytest.approx(50.0 / 30.0, rel=0.02)
    assert np.allclose(result.peak_lobe_base_voltage_v, result.peak_lobe_command_voltage_v)


def test_peak_lobe_model_uses_two_peaks_for_1cycle() -> None:
    time_s, shape = _finite_waveform(1.0)
    target = shape * 50.0
    measured = np.where(time_s < 0.5, shape * 40.0, shape * 25.0)
    base_voltage = shape.copy()
    active_mask = np.ones_like(time_s, dtype=bool)

    result = build_peak_lobe_model(
        time_s=time_s,
        target_field_mT=target,
        aligned_measured_field_mT=measured,
        base_voltage_v=base_voltage,
        active_mask=active_mask,
        cycle_count=1.0,
    )

    assert result.enabled is True
    assert result.status == "ok"
    assert result.cycle_policy == "1.0cycle_two_peak"
    assert [lobe.polarity for lobe in result.lobes] == ["positive", "negative"]
    assert [lobe.gain for lobe in result.lobes] == pytest.approx([1.25, 2.0], rel=0.02)
    assert [lobe.command_peak_v for lobe in result.lobes] == pytest.approx([1.25, -2.0], rel=0.02)
    assert result.detected_lobe_polarities == ("positive", "negative")
    assert result.expected_lobe_polarities == ("positive", "negative")


def test_peak_lobe_model_reports_voltage_limit_diagnostic_without_clipping() -> None:
    time_s, shape = _finite_waveform(1.0)
    target = shape * 50.0
    measured = np.where(time_s < 0.5, shape * 5.0, shape * 5.0)
    base_voltage = shape.copy() * 2.0
    active_mask = np.ones_like(time_s, dtype=bool)

    result = build_peak_lobe_model(
        time_s=time_s,
        target_field_mT=target,
        aligned_measured_field_mT=measured,
        base_voltage_v=base_voltage,
        active_mask=active_mask,
        cycle_count=1.0,
        voltage_limit_v=10.0,
    )

    assert result.enabled is True
    assert result.voltage_limit_v == pytest.approx(10.0)
    assert result.peak_lobe_command_peak_abs_v > 10.0
    assert result.voltage_limit_exceeded is True
    assert result.voltage_exceeded_fraction > 0.0
    assert float(np.nanmax(result.peak_lobe_command_voltage_v)) > 10.0


def test_peak_lobe_model_accepts_command_under_voltage_limit() -> None:
    time_s, shape = _finite_waveform(1.0)
    target = shape * 50.0
    measured = np.where(time_s < 0.5, shape * 40.0, shape * 25.0)
    base_voltage = shape.copy()
    active_mask = np.ones_like(time_s, dtype=bool)

    result = build_peak_lobe_model(
        time_s=time_s,
        target_field_mT=target,
        aligned_measured_field_mT=measured,
        base_voltage_v=base_voltage,
        active_mask=active_mask,
        cycle_count=1.0,
        voltage_limit_v=10.0,
    )

    assert result.peak_lobe_command_peak_abs_v < 10.0
    assert result.voltage_limit_exceeded is False
    assert result.voltage_exceeded_fraction == pytest.approx(0.0)


def test_peak_lobe_model_rejects_unexpected_polarity_sequence() -> None:
    time_s, shape = _finite_waveform(1.5)
    target = -shape * 50.0
    measured = -shape * 40.0
    base_voltage = -shape.copy()
    active_mask = np.ones_like(time_s, dtype=bool)

    result = build_peak_lobe_model(
        time_s=time_s,
        target_field_mT=target,
        aligned_measured_field_mT=measured,
        base_voltage_v=base_voltage,
        active_mask=active_mask,
        cycle_count=1.5,
    )

    assert result.enabled is False
    assert result.status == "unexpected_lobe_polarity_sequence"
    assert result.detected_lobe_polarities == ("negative", "positive", "negative")
    assert result.expected_lobe_polarities == ("positive", "negative", "positive")
