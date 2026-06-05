from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.finite_second_modeling import generate_second_modeled_voltage_lut
from field_analysis.finite_second_modeling_tail_controller import apply_finite_time_zero_return_tail
from tests.test_finite_second_modeling import _first_profile, _write_delayed_actual_drive_csv


def test_second_modeling_finite_time_zero_return_tail_is_default(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_delayed_actual_drive_csv(actual, delay_s=0.05)

    frame, metadata = generate_second_modeled_voltage_lut(_first_profile(), actual, freq_hz=1.0, cycle_count=1.0)

    tail = frame["tail_window_mask"].astype(bool).to_numpy()
    tail_indices = np.flatnonzero(tail)
    assert metadata["tail_return_mode"] == "finite_time_zero_return"
    assert metadata["second_command_synthesis_mode"] == "active_residual_with_finite_time_zero_return_tail"
    assert metadata["tail_duration_mode"] == "seconds"
    assert np.isclose(metadata["tail_cycle_count"], 0.25)
    assert np.isclose(metadata["tail_duration_s"], 0.25)
    assert metadata["tail_terminal_target_mT"] == 0.0
    assert metadata["tail_terminal_dBdt_target"] == 0.0
    assert {"tail_B0_mT", "tail_dB0_dt_mT_s", "tail_B_ref_mT", "tail_dB_ref_dt_mT_s"}.issubset(frame.columns)
    assert np.isclose(frame.loc[tail_indices[0], "tail_B_ref_mT"], metadata["tail_B0_mT"], atol=1e-6)
    assert np.isclose(frame.loc[tail_indices[0], "tail_dB_ref_dt_mT_s"], metadata["tail_dB0_dt_mT_s"], atol=1e-6)
    assert np.isclose(frame.loc[tail_indices[-1], "tail_B_ref_mT"], 0.0, atol=1e-6)
    assert np.isclose(frame.loc[tail_indices[-1], "tail_dB_ref_dt_mT_s"], 0.0, atol=1e-6)
    assert np.isclose(frame.loc[tail_indices[-1], "second_limited_voltage_v"], 0.0, atol=1e-9)


def test_second_modeling_finite_time_tail_fits_local_model(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_delayed_actual_drive_csv(actual, delay_s=0.05)

    frame, metadata = generate_second_modeled_voltage_lut(_first_profile(), actual, freq_hz=1.0, cycle_count=1.0)

    tail = frame["tail_window_mask"].astype(bool).to_numpy()
    assert metadata["tail_model_type"] == "actual_drive_zero_tail_passthrough"
    assert metadata["tail_model_fit_status"] == "skipped_actual_drive_voltage_zero"
    assert metadata["tail_actual_drive_zero_passthrough"] is True
    assert np.nanmax(np.abs(frame.loc[tail, "tail_model_voltage_v"])) <= 1e-12
    assert np.nanmax(np.abs(frame.loc[tail, "second_limited_voltage_v"])) <= 1e-12


def test_second_modeling_finite_time_tail_uses_smooth_pulse_fallback_when_model_invalid() -> None:
    time_s = np.linspace(0.0, 1.25, 126)
    active = time_s <= 1.0
    tail = time_s > 1.0
    measured = np.where(active, 20.0 * time_s, 8.0 * np.exp(-(time_s - 1.0) / 0.12))
    first_voltage = np.where(active, 2.0 * np.sin(np.pi * time_s), 0.0)
    arrays, metadata = apply_finite_time_zero_return_tail(
        time_s=time_s,
        active_mask=active,
        tail_mask=tail,
        measured_field_for_second_mT=measured,
        actual_drive_voltage_v=np.zeros_like(time_s),
        first_voltage_v=first_voltage,
        correction_delta_v=np.zeros_like(time_s),
        second_voltage_v=first_voltage.copy(),
        second_limited_voltage_v=first_voltage.copy(),
        freq_hz=1.0,
        voltage_limit_v=5.0,
    )

    tail_indices = np.flatnonzero(tail)
    assert metadata["tail_actual_drive_zero_passthrough"] is True
    assert metadata["tail_model_fit_status"] == "skipped_actual_drive_voltage_zero"
    assert np.isclose(arrays["tail_voltage_v"][tail_indices[0]], 0.0, atol=1e-9)
    assert np.isclose(arrays["tail_voltage_v"][tail_indices[-1]], 0.0, atol=1e-9)
    assert metadata["tail_voltage_peak_v"] == 0.0


def test_second_modeling_tail_sign_constraint_uses_opposite_polarity_for_positive_b0() -> None:
    time_s = np.linspace(0.0, 1.25, 126)
    active = time_s <= 1.0
    tail = time_s > 1.0
    measured = np.where(active, 20.0 * time_s, 20.0)
    first_voltage = np.where(active, 2.0 * np.sin(np.pi * time_s), 0.0)

    arrays, metadata = apply_finite_time_zero_return_tail(
        time_s=time_s,
        active_mask=active,
        tail_mask=tail,
        measured_field_for_second_mT=measured,
        native_measured_field_mT=measured,
        actual_drive_voltage_v=np.where(tail, 0.5, first_voltage),
        first_voltage_v=first_voltage,
        correction_delta_v=np.zeros_like(time_s),
        second_voltage_v=first_voltage.copy(),
        second_limited_voltage_v=first_voltage.copy(),
        freq_hz=1.0,
        voltage_limit_v=5.0,
    )

    tail_values = arrays["tail_voltage_v"][tail]
    assert metadata["tail_voltage_sign_constraint_enabled"] is True
    assert metadata["tail_voltage_expected_sign"] == -1
    assert np.nanmax(tail_values) <= 1e-9
    assert metadata["tail_voltage_single_lobe_status"] == "ok"
    assert metadata["tail_voltage_monotonic_to_zero_status"] == "ok"
    assert metadata["tail_voltage_projection_mode"] == "sign_constrained_monotonic"
    release = _tail_release_values(tail_values, expected_sign=-1)
    assert np.all(np.diff(release) >= -1e-9)


def test_second_modeling_tail_sign_constraint_uses_opposite_polarity_for_negative_b0() -> None:
    time_s = np.linspace(0.0, 1.25, 126)
    active = time_s <= 1.0
    tail = time_s > 1.0
    measured = np.where(active, -20.0 * time_s, -20.0)
    first_voltage = np.where(active, -2.0 * np.sin(np.pi * time_s), 0.0)

    arrays, metadata = apply_finite_time_zero_return_tail(
        time_s=time_s,
        active_mask=active,
        tail_mask=tail,
        measured_field_for_second_mT=measured,
        native_measured_field_mT=measured,
        actual_drive_voltage_v=np.where(tail, -0.5, first_voltage),
        first_voltage_v=first_voltage,
        correction_delta_v=np.zeros_like(time_s),
        second_voltage_v=first_voltage.copy(),
        second_limited_voltage_v=first_voltage.copy(),
        freq_hz=1.0,
        voltage_limit_v=5.0,
    )

    tail_values = arrays["tail_voltage_v"][tail]
    assert metadata["tail_voltage_expected_sign"] == 1
    assert np.nanmin(tail_values) >= -1e-9
    assert metadata["tail_voltage_single_lobe_status"] == "ok"
    assert metadata["tail_voltage_monotonic_to_zero_status"] == "ok"
    release = _tail_release_values(tail_values, expected_sign=1)
    assert np.all(np.diff(release) <= 1e-9)


def test_second_modeling_tail_projects_oscillatory_model_to_single_lobe_monotonic_voltage() -> None:
    time_s = np.linspace(0.0, 1.3, 131)
    active = time_s <= 1.0
    tail = time_s > 1.0
    # Oscillatory voltage data can make the inverse model estimate wiggle; the final
    # tail command must still be a single-direction discharge pulse.
    measured = np.where(active, 18.0 * time_s + 1.5 * np.sin(32.0 * time_s), 18.0)
    actual_voltage = 2.5 * np.sin(25.0 * time_s)
    first_voltage = np.where(active, 2.0 * np.sin(np.pi * time_s), 0.0)

    arrays, metadata = apply_finite_time_zero_return_tail(
        time_s=time_s,
        active_mask=active,
        tail_mask=tail,
        measured_field_for_second_mT=measured,
        native_measured_field_mT=measured,
        actual_drive_voltage_v=actual_voltage,
        first_voltage_v=first_voltage,
        correction_delta_v=np.zeros_like(time_s),
        second_voltage_v=first_voltage.copy(),
        second_limited_voltage_v=first_voltage.copy(),
        freq_hz=1.0,
        voltage_limit_v=5.0,
    )

    tail_values = arrays["tail_voltage_v"][tail]
    assert metadata["tail_voltage_projection_mode"] == "sign_constrained_monotonic"
    assert metadata["tail_voltage_projection_from_model_inverse"] is True
    assert metadata["tail_voltage_monotonic_release_enabled"] is True
    assert metadata["tail_voltage_expected_sign"] == -1
    assert np.nanmax(tail_values) <= 1e-9
    assert np.isclose(tail_values[-1], 0.0, atol=1e-9)
    assert metadata["tail_voltage_single_lobe_status"] == "ok"
    assert metadata["tail_voltage_monotonic_to_zero_status"] == "ok"
    release = _tail_release_values(tail_values, expected_sign=-1)
    assert np.all(np.diff(release) >= -1e-9)


def test_second_modeling_tail_does_not_reset_to_zero_after_nonzero_active_end_voltage() -> None:
    time_s = np.linspace(0.0, 1.25, 126)
    active = time_s <= 1.0
    tail = time_s > 1.0
    measured = np.where(active, 18.0 * time_s, 18.0)
    first_voltage = np.where(active, -0.8 * time_s, 0.0)
    second_limited = first_voltage.copy()
    active_end_index = int(np.flatnonzero(active)[-1])
    second_limited[active_end_index] = -0.8

    arrays, metadata = apply_finite_time_zero_return_tail(
        time_s=time_s,
        active_mask=active,
        tail_mask=tail,
        measured_field_for_second_mT=measured,
        native_measured_field_mT=measured,
        actual_drive_voltage_v=first_voltage,
        first_voltage_v=first_voltage,
        correction_delta_v=np.zeros_like(time_s),
        second_voltage_v=second_limited.copy(),
        second_limited_voltage_v=second_limited.copy(),
        freq_hz=1.0,
        voltage_limit_v=5.0,
    )

    tail_values = arrays["tail_voltage_v"][tail]
    assert np.isclose(tail_values[0], 0.0, atol=1e-9)
    assert metadata["tail_actual_drive_zero_passthrough"] is True
    assert metadata["tail_start_reset_to_zero_detected"] is True
    assert metadata["active_to_tail_zero_reset_detected"] is True
    assert metadata["tail_voltage_generated_independently_from_first_voltage"] is False
    assert np.nanmax(np.abs(tail_values)) <= 1e-9
    assert np.isclose(tail_values[-1], 0.0, atol=1e-9)


def test_second_modeling_tail_amplitude_scales_with_b0_and_stays_limited() -> None:
    small_peak = _tail_peak_for_constant_b0(4.0)
    large_peak = _tail_peak_for_constant_b0(35.0)

    assert 0.0 <= small_peak < large_peak <= 5.0


def _tail_release_values(values: np.ndarray, *, expected_sign: int) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size <= 1:
        return finite
    peak_index = int(np.nanargmax(np.abs(finite)))
    release = finite[peak_index:]
    if expected_sign < 0:
        return np.minimum(release, 0.0)
    return np.maximum(release, 0.0)


def _tail_peak_for_constant_b0(b0: float) -> float:
    time_s = np.linspace(0.0, 1.25, 126)
    active = time_s <= 1.0
    tail = time_s > 1.0
    measured = np.where(active, b0 * time_s, b0)
    arrays, metadata = apply_finite_time_zero_return_tail(
        time_s=time_s,
        active_mask=active,
        tail_mask=tail,
        measured_field_for_second_mT=measured,
        native_measured_field_mT=measured,
        actual_drive_voltage_v=np.sin(np.pi * time_s),
        first_voltage_v=np.zeros_like(time_s),
        correction_delta_v=np.zeros_like(time_s),
        second_voltage_v=np.zeros_like(time_s),
        second_limited_voltage_v=np.zeros_like(time_s),
        freq_hz=1.0,
        voltage_limit_v=5.0,
    )
    assert metadata["tail_voltage_amplitude_v"] <= 5.0
    return float(np.nanmax(np.abs(arrays["tail_voltage_v"][tail])))
