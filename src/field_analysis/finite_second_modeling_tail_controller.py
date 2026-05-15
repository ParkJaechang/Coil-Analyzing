from __future__ import annotations

from typing import Any

import numpy as np

from .finite_actual_drive_normalization import peak_abs


def normalize_tail_return_mode(value: str | None) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "off": "disabled",
        "none": "disabled",
        "disabled": "disabled",
        "사용 안 함": "disabled",
        "residual": "residual",
        "residual_tail": "residual",
        "residual 기반 tail": "residual",
        "finite_time": "finite_time_zero_return",
        "finite_time_zero_return": "finite_time_zero_return",
        "지정 시간 0 복귀 제어": "finite_time_zero_return",
    }
    return aliases.get(text, "finite_time_zero_return")


def normalize_tail_duration_mode(value: str | None) -> str:
    text = str(value or "").strip().lower()
    return "cycle" if text in {"cycle", "cycles", "사이클"} else "seconds"


def tail_cycle_count_from_duration(
    *,
    freq_hz: float,
    tail_duration_mode: str,
    tail_cycle_count: float,
    tail_duration_s: float,
) -> float:
    if tail_duration_mode == "cycle":
        return float(np.clip(tail_cycle_count, 0.0, 1.0))
    return float(np.clip(tail_duration_s, 0.0, 10.0) * max(float(freq_hz), 1e-12))


def missing_time_ranges(time_s: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    time = np.asarray(time_s, dtype=float)
    values = np.asarray(mask, dtype=bool)
    indices = np.flatnonzero(values & np.isfinite(time))
    if not indices.size:
        return []
    ranges: list[tuple[float, float]] = []
    start = previous = int(indices[0])
    for raw_index in indices[1:]:
        index = int(raw_index)
        if index != previous + 1:
            ranges.append((float(time[start]), float(time[previous])))
            start = index
        previous = index
    ranges.append((float(time[start]), float(time[previous])))
    return ranges


def fill_tail_measured_field(
    measured_for_second: np.ndarray,
    measured_smoothed: np.ndarray,
    active_mask: np.ndarray,
    tail_mask: np.ndarray,
) -> tuple[np.ndarray, bool]:
    values = np.asarray(measured_for_second, dtype=float).copy()
    tail_indices = np.flatnonzero(np.asarray(tail_mask, dtype=bool))
    if not tail_indices.size:
        return values, False
    finite_tail_values = np.isfinite(values[tail_indices])
    return values, bool(np.any(finite_tail_values))


def apply_finite_time_zero_return_tail(
    *,
    time_s: np.ndarray,
    active_mask: np.ndarray,
    tail_mask: np.ndarray,
    measured_field_for_second_mT: np.ndarray,
    native_measured_field_mT: np.ndarray | None = None,
    actual_drive_voltage_v: np.ndarray,
    first_voltage_v: np.ndarray,
    correction_delta_v: np.ndarray,
    second_voltage_v: np.ndarray,
    second_limited_voltage_v: np.ndarray,
    freq_hz: float,
    voltage_limit_v: float,
    tail_controller_scaling_mode: str = "auto",
    tail_controller_gain: float = 1.0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    time = np.asarray(time_s, dtype=float)
    active = np.asarray(active_mask, dtype=bool)
    tail = np.asarray(tail_mask, dtype=bool)
    n = len(time)
    arrays = _empty_tail_controller_arrays(n)
    if not tail.any() or not active.any():
        return arrays, _tail_controller_disabled_meta()

    active_indices = np.flatnonzero(active)
    tail_indices = np.flatnonzero(tail)
    active_end_index = int(active_indices[-1])
    active_end_s = float(time[active_end_index])
    tail_start_s = float(time[tail_indices[0]])
    tail_end_s = float(time[tail_indices[-1]])
    tail_duration_s = max(tail_end_s - active_end_s, _dt(time))
    tail_cycle_count = tail_duration_s * max(float(freq_hz), 1e-12)
    measured = np.asarray(measured_field_for_second_mT, dtype=float)
    native_measured = np.asarray(native_measured_field_mT, dtype=float) if native_measured_field_mT is not None else measured
    actual_voltage = np.asarray(actual_drive_voltage_v, dtype=float)
    first_voltage = np.asarray(first_voltage_v, dtype=float)
    correction = np.asarray(correction_delta_v, dtype=float).copy()
    second_before_clip = np.asarray(second_voltage_v, dtype=float).copy()
    second_limited = np.asarray(second_limited_voltage_v, dtype=float).copy()

    derivative = _safe_gradient(native_measured, time)
    b0 = float(native_measured[active_end_index]) if np.isfinite(native_measured[active_end_index]) else 0.0
    db0 = float(derivative[active_end_index]) if np.isfinite(derivative[active_end_index]) else 0.0
    elapsed = time[tail_indices] - active_end_s
    tau = np.clip(elapsed / max(tail_duration_s, 1e-12), 0.0, 1.0)
    if tau.size:
        tau[0] = 0.0
        tau[-1] = 1.0
    b_ref, db_ref = _cubic_hermite_zero_return(tau, tail_duration_s, b0, db0)
    fit = _fit_first_order_model(time, native_measured, actual_voltage, active_end_s, tail_end_s)
    scaling = _tail_scaling(tail_controller_scaling_mode, tail_controller_gain)
    fallback_reason = None
    if fit["status"] == "ok":
        tail_voltage_raw = (db_ref - fit["b"] * b_ref - fit["c"]) / fit["a"]
        tail_voltage_raw = tail_voltage_raw * scaling
        fallback_used = False
    else:
        fallback_reason = str(fit["status"])
        tail_voltage_raw = _smooth_pulse_tail_voltage(b0, tau, voltage_limit_v) * scaling
        fallback_used = True
    expected_sign = _expected_tail_voltage_sign(b0)
    model_sign_violation = _has_sign_violation(tail_voltage_raw, expected_sign)
    model_voltage_smoothed = _smooth_signal(tail_voltage_raw)
    active_end_voltage = float(second_limited[active_end_index]) if np.isfinite(second_limited[active_end_index]) else 0.0
    amplitude, amplitude_source = _tail_voltage_amplitude(
        tail_voltage_raw,
        b0=b0,
        expected_sign=expected_sign,
        voltage_limit_v=voltage_limit_v,
        model_valid=fit["status"] == "ok",
    )
    tail_voltage_before_clip = _single_lobe_tail_voltage(
        tau,
        expected_sign=expected_sign,
        amplitude_v=amplitude,
        active_end_voltage_v=active_end_voltage,
    )
    tail_voltage = np.clip(tail_voltage_before_clip, -abs(float(voltage_limit_v)), abs(float(voltage_limit_v)))
    projection_applied = bool(
        _has_sign_violation(model_voltage_smoothed, expected_sign)
        or not _same_finite_values(model_voltage_smoothed, tail_voltage)
    )
    monotonic_violation_count = _tail_monotonic_violation_count(tail_voltage, expected_sign)
    monotonic_status = "ok" if monotonic_violation_count == 0 else "warning_non_monotonic"
    correction[tail_indices] = tail_voltage - first_voltage[tail_indices]
    second_before_clip[tail_indices] = tail_voltage
    second_limited[tail_indices] = np.clip(tail_voltage, -abs(float(voltage_limit_v)), abs(float(voltage_limit_v)))

    arrays.update(
        {
            "tail_B0_mT": _fill_scalar(n, tail, b0),
            "tail_dB0_dt_mT_s": _fill_scalar(n, tail, db0),
            "tail_B_ref_mT": _fill_tail(n, tail_indices, b_ref),
            "tail_dB_ref_dt_mT_s": _fill_tail(n, tail_indices, db_ref),
            "tail_model_voltage_v": _fill_tail(n, tail_indices, tail_voltage_raw if fit["status"] == "ok" else np.full_like(tau, np.nan)),
            "tail_voltage_before_clip_v": _fill_tail(n, tail_indices, tail_voltage_before_clip),
            "tail_voltage_v": _fill_tail(n, tail_indices, tail_voltage),
            "correction_delta_v": correction,
            "second_voltage_before_clip_v": second_before_clip,
            "second_limited_voltage_v": second_limited,
        }
    )
    tail_start_voltage = float(second_limited[tail_indices[0]]) if tail_indices.size else active_end_voltage
    jump = float(abs(tail_start_voltage - active_end_voltage)) if tail_indices.size else 0.0
    zero_reset_detected = bool(
        tail_indices.size
        and abs(active_end_voltage) > 1e-6
        and abs(tail_start_voltage) <= 1e-9
    )
    meta = {
        "tail_return_mode": "finite_time_zero_return",
        "tail_duration_mode": "seconds",
        "tail_cycle_count": float(tail_cycle_count),
        "tail_duration_s": float(tail_duration_s),
        "tail_duration_cycle_count": float(tail_cycle_count),
        "tail_terminal_target_mT": 0.0,
        "tail_terminal_dBdt_target": 0.0,
        "tail_B0_mT": b0,
        "tail_B0_source": "native_smoothed_measured_at_active_end",
        "tail_dB0_dt_mT_s": db0,
        "tail_model_type": "first_order_local",
        "tail_model_a": fit["a"],
        "tail_model_b": fit["b"],
        "tail_model_c": fit["c"],
        "tail_model_fit_status": fit["status"],
        "tail_model_fit_r2": fit["r2"],
        "tail_model_fit_residual_rms": fit["residual_rms"],
        "tail_fallback_used": fallback_used,
        "tail_fallback_reason": fallback_reason,
        "tail_pulse_area_v_s": float(np.trapezoid(tail_voltage, time[tail_indices])) if tail_indices.size else 0.0,
        "tail_voltage_peak_v": peak_abs(tail_voltage),
        "tail_voltage_amplitude_source": amplitude_source,
        "tail_voltage_amplitude_v": float(amplitude),
        "tail_voltage_projection_mode": "sign_constrained_monotonic",
        "tail_voltage_attack_enabled": True,
        "tail_voltage_attack_duration_s": float(0.15 * tail_duration_s),
        "tail_voltage_release_shape": "smoothstep_decay",
        "tail_voltage_projection_from_model_inverse": bool(fit["status"] == "ok"),
        "tail_voltage_sign_constraint_enabled": True,
        "tail_voltage_expected_sign": int(expected_sign),
        "tail_voltage_sign_violation_detected": bool(model_sign_violation or projection_applied),
        "tail_voltage_sign_projection_applied": bool(projection_applied),
        "tail_voltage_single_lobe_status": "ok" if not _has_sign_violation(tail_voltage, expected_sign) else "warning_sign_reversal",
        "tail_voltage_monotonic_to_zero_status": monotonic_status,
        "tail_voltage_monotonic_release_enabled": True,
        "tail_voltage_monotonic_projection_applied": bool(projection_applied),
        "tail_voltage_monotonic_violation_count": int(monotonic_violation_count),
        "tail_voltage_final_v": float(tail_voltage[-1]) if tail_voltage.size else 0.0,
        "tail_voltage_generated_independently_from_first_voltage": True,
        "tail_controller_gain_used": float(scaling),
        "tail_controller_scaling_mode": str(tail_controller_scaling_mode or "auto"),
        "tail_end_voltage_zero_status": "ok" if abs(float(second_limited[-1])) <= 1e-9 else "warning_nonzero_tail_end",
        "second_command_final_voltage_v": float(second_limited[-1]) if n else 0.0,
        "active_end_voltage_v": active_end_voltage,
        "tail_start_voltage_v": tail_start_voltage,
        "active_to_tail_voltage_jump_v": jump,
        "tail_start_reset_to_zero_detected": zero_reset_detected,
        "active_to_tail_zero_reset_detected": zero_reset_detected,
        "active_to_tail_continuity_status": "ok" if jump <= 0.5 else "warning_jump_detected",
        "tail_continuity_blend_enabled": bool(tail_indices.size),
    }
    return arrays, meta


def unified_tail_diagnostics(
    time_s: np.ndarray,
    target_for_second: np.ndarray,
    measured_for_second: np.ndarray,
    raw_delta: np.ndarray,
    correction_delta: np.ndarray,
    second_limited: np.ndarray,
    active_mask: np.ndarray,
    tail_mask: np.ndarray,
    *,
    freq_hz: float,
    tail_cycle_count: float,
    post_active_measured_available: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    n = len(time_s)
    tail = np.asarray(tail_mask, dtype=bool)
    active_indices = np.flatnonzero(np.asarray(active_mask, dtype=bool))
    tail_indices = np.flatnonzero(tail)
    tail_target = np.full(n, np.nan)
    tail_residual = np.zeros(n, dtype=float)
    tail_start_voltage = np.full(n, np.nan)
    if tail_indices.size:
        tail_target[tail] = target_for_second[tail]
        tail_residual[tail] = target_for_second[tail] - measured_for_second[tail]
    active_end_index = int(active_indices[-1]) if active_indices.size else 0
    tail_start_index = int(tail_indices[0]) if tail_indices.size else active_end_index
    active_end_voltage = float(second_limited[active_end_index]) if active_end_index < n and np.isfinite(second_limited[active_end_index]) else 0.0
    tail_start_voltage_value = float(second_limited[tail_start_index]) if tail_indices.size and np.isfinite(second_limited[tail_start_index]) else active_end_voltage
    if tail_indices.size:
        tail_start_voltage[tail_start_index] = active_end_voltage
    jump = float(abs(tail_start_voltage_value - active_end_voltage)) if tail_indices.size else 0.0
    tail_duration = float(np.clip(tail_cycle_count, 0.0, 1.0)) / max(float(freq_hz), 1e-12)
    post_active_available = bool(tail_indices.size and post_active_measured_available)
    arrays = {
        "tail_target_field_mT": tail_target,
        "tail_residual_mT": tail_residual,
        "raw_tail_correction_delta_v": np.where(tail, raw_delta, 0.0),
        "tail_correction_delta_v": np.where(tail, correction_delta, 0.0),
        "stabilized_tail_voltage_v": np.where(tail, correction_delta, 0.0),
        "tail_voltage_v": np.where(tail, correction_delta, 0.0),
        "tail_start_voltage_v": tail_start_voltage,
    }
    meta = {
        "post_cycle_zero_tail_enabled": bool(tail_indices.size),
        "post_cycle_zero_tail_cycle_count": float(np.clip(tail_cycle_count, 0.0, 1.0)) if tail_indices.size else 0.0,
        "post_cycle_zero_tail_duration_s": tail_duration if tail_indices.size else 0.0,
        "post_cycle_zero_tail_target_field_mT": 0.0,
        "tail_voltage_taper_to_zero": bool(tail_indices.size),
        "tail_end_taper_out_enabled": bool(tail_indices.size),
        "tail_end_taper_duration_s": tail_duration * 0.10 if tail_indices.size else 0.0,
        "tail_end_taper_cycle_fraction": 0.10 if tail_indices.size else 0.0,
        "tail_field_source": "measured_post_active" if post_active_available else "measured_last_value_hold",
        "tail_field_source_status": "ok" if post_active_available else "warning_insufficient_tail_support_no_fake_zero_return",
        "measured_tail_fake_line_to_zero_used": False,
        "fake_line_to_zero_fallback_used": False,
        "tail_residual_start_mT": float(tail_residual[tail_start_index]) if tail_indices.size else 0.0,
        "tail_residual_end_mT": float(tail_residual[tail_indices[-1]]) if tail_indices.size else 0.0,
        "tail_unit_delta_peak_v": peak_abs(raw_delta[tail]) if tail_indices.size else 0.0,
        "tail_voltage_peak_v": peak_abs(correction_delta[tail]) if tail_indices.size else 0.0,
        "tail_taper_out_applied": bool(tail_indices.size),
        "tail_extrapolation_used": False,
        "tail_end_voltage_zero_status": "ok" if abs(float(second_limited[-1])) <= 1e-9 else "warning_nonzero_tail_end",
        "second_command_final_voltage_v": float(second_limited[-1]) if n else 0.0,
        "active_end_voltage_v": active_end_voltage,
        "tail_start_voltage_v": tail_start_voltage_value,
        "active_to_tail_voltage_jump_v": jump,
        "active_to_tail_continuity_status": "ok" if jump <= 0.5 else "warning_jump_detected",
        "active_tail_boundary_blend_enabled": False,
        "active_tail_boundary_blend_duration_s": 0.0,
        "tail_continuity_blend_enabled": bool(tail_indices.size),
    }
    return arrays, meta


def _tail_controller_disabled_meta() -> dict[str, Any]:
    return {
        "tail_return_mode": "disabled",
        "tail_model_type": "none",
        "tail_model_fit_status": "not_run",
        "tail_fallback_used": False,
        "tail_controller_gain_used": 0.0,
        "tail_controller_scaling_mode": "disabled",
    }


def _empty_tail_controller_arrays(n: int) -> dict[str, np.ndarray]:
    nan = np.full(n, np.nan)
    return {
        "tail_B0_mT": nan.copy(),
        "tail_dB0_dt_mT_s": nan.copy(),
        "tail_B_ref_mT": nan.copy(),
        "tail_dB_ref_dt_mT_s": nan.copy(),
        "tail_model_voltage_v": nan.copy(),
        "tail_voltage_before_clip_v": nan.copy(),
        "tail_voltage_v": nan.copy(),
    }


def _fit_first_order_model(time: np.ndarray, field: np.ndarray, voltage: np.ndarray, active_end_s: float, tail_end_s: float) -> dict[str, Any]:
    derivative = _safe_gradient(field, time)
    fit_start = active_end_s - max(0.35 * (tail_end_s - active_end_s), _dt(time) * 12.0)
    mask = np.isfinite(time) & np.isfinite(field) & np.isfinite(voltage) & np.isfinite(derivative)
    mask &= (time >= fit_start) & (time <= tail_end_s)
    if int(mask.sum()) < 8:
        return _invalid_model("model_fit_insufficient_data")
    x = np.column_stack([voltage[mask], field[mask], np.ones(int(mask.sum()))])
    y = derivative[mask]
    if float(np.nanstd(x[:, 0])) <= 1e-9:
        return _invalid_model("model_a_too_small")
    ridge = 1e-6 * np.eye(3)
    try:
        coeff = np.linalg.solve(x.T @ x + ridge, x.T @ y)
    except np.linalg.LinAlgError:
        return _invalid_model("model_fit_unstable")
    pred = x @ coeff
    resid = y - pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    a, b, c = (float(coeff[0]), float(coeff[1]), float(coeff[2]))
    if not np.isfinite(a) or abs(a) < 1e-6:
        return _invalid_model("model_a_too_small")
    if not np.isfinite(r2):
        return _invalid_model("model_fit_unstable")
    return {"status": "ok", "a": a, "b": b, "c": c, "r2": float(r2), "residual_rms": float(np.sqrt(np.mean(resid**2)))}


def _invalid_model(status: str) -> dict[str, Any]:
    return {"status": status, "a": np.nan, "b": np.nan, "c": np.nan, "r2": np.nan, "residual_rms": np.nan}


def _cubic_hermite_zero_return(tau: np.ndarray, duration_s: float, b0: float, db0: float) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(tau, dtype=float)
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    b_ref = h00 * b0 + h10 * duration_s * db0 + h01 * 0.0 + h11 * duration_s * 0.0
    dh00 = 6.0 * t**2 - 6.0 * t
    dh10 = 3.0 * t**2 - 4.0 * t + 1.0
    dh01 = -6.0 * t**2 + 6.0 * t
    dh11 = 3.0 * t**2 - 2.0 * t
    db_ref = (dh00 * b0 + dh10 * duration_s * db0 + dh01 * 0.0 + dh11 * duration_s * 0.0) / max(duration_s, 1e-12)
    if b_ref.size:
        b_ref[0] = b0
        db_ref[0] = db0
        b_ref[-1] = 0.0
        db_ref[-1] = 0.0
    return b_ref, db_ref


def _smooth_pulse_tail_voltage(b0: float, tau: np.ndarray, voltage_limit_v: float) -> np.ndarray:
    amplitude = float(np.clip(abs(b0) / 50.0 * float(voltage_limit_v), 0.1, float(voltage_limit_v)))
    direction = -1.0 if b0 >= 0.0 else 1.0
    pulse = direction * amplitude * np.sin(np.pi * np.asarray(tau, dtype=float)) ** 2
    if pulse.size:
        pulse[0] = 0.0
        pulse[-1] = 0.0
    return pulse


def _single_lobe_tail_voltage(
    tau: np.ndarray,
    *,
    expected_sign: int,
    amplitude_v: float,
    active_end_voltage_v: float,
) -> np.ndarray:
    t = np.clip(np.asarray(tau, dtype=float), 0.0, 1.0)
    amplitude = max(float(amplitude_v), 0.0)
    if expected_sign == 0 or amplitude <= 1e-12:
        return np.zeros_like(t, dtype=float)
    active_end = float(active_end_voltage_v) if np.isfinite(active_end_voltage_v) else 0.0
    start_voltage = active_end if np.sign(active_end) == expected_sign else 0.0
    if abs(start_voltage) > amplitude:
        amplitude = abs(start_voltage)
    peak_voltage = float(expected_sign) * amplitude
    attack_fraction = 0.15
    attack = _smoothstep(np.clip(t / attack_fraction, 0.0, 1.0))
    release_x = np.clip((t - attack_fraction) / max(1.0 - attack_fraction, 1e-12), 0.0, 1.0)
    release = 1.0 - _smoothstep(release_x)
    attack_values = start_voltage + (peak_voltage - start_voltage) * attack
    release_values = peak_voltage * release
    values = np.where(t <= attack_fraction, attack_values, release_values)
    if values.size:
        values[0] = start_voltage
        values[-1] = 0.0
    return values


def _tail_voltage_amplitude(
    values: np.ndarray,
    *,
    b0: float,
    expected_sign: int,
    voltage_limit_v: float,
    model_valid: bool,
) -> tuple[float, str]:
    limit = abs(float(voltage_limit_v))
    if expected_sign == 0 or abs(float(b0)) <= 0.5:
        return 0.0, "near_zero_B0"
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr) & (np.abs(arr) > 1e-9)]
    if model_valid and finite.size:
        signed = finite[np.sign(finite) == expected_sign]
        source_values = signed if signed.size else finite
        candidate = float(np.nanpercentile(np.abs(source_values), 90.0))
        if np.isfinite(candidate) and candidate > 1e-9:
            fallback = 0.75 * abs(float(b0)) / 50.0 * limit
            conservative_cap = max(fallback, 0.15 * limit)
            return float(np.clip(candidate, 0.0, min(limit, conservative_cap))), "model_inverse"
    fallback = 0.75 * abs(float(b0)) / 50.0 * limit
    return float(np.clip(fallback, 0.0, limit)), "fallback"


def _expected_tail_voltage_sign(b0: float) -> int:
    if abs(float(b0)) <= 1e-9:
        return 0
    return -1 if b0 > 0.0 else 1


def _has_sign_violation(values: np.ndarray, expected_sign: int) -> bool:
    if expected_sign == 0:
        return bool(np.nanmax(np.abs(values)) > 1e-6) if np.size(values) else False
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr) & (np.abs(arr) > 1e-9)]
    if not finite.size:
        return False
    return bool(np.any(np.sign(finite) != expected_sign))


def _tail_monotonic_violation_count(values: np.ndarray, expected_sign: int) -> int:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size < 3 or expected_sign == 0:
        return 0
    peak_index = int(np.nanargmax(np.abs(finite)))
    release = finite[peak_index:]
    if release.size < 2:
        return 0
    diffs = np.diff(release)
    if expected_sign < 0:
        return int(np.sum(diffs < -1e-9))
    return int(np.sum(diffs > 1e-9))


def _same_finite_values(left: np.ndarray, right: np.ndarray) -> bool:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape:
        return False
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.any():
        return True
    return bool(np.allclose(a[finite], b[finite], rtol=1e-6, atol=1e-9))


def _smooth_signal(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(arr)
    if arr.size < 5 or finite.sum() < 5:
        return np.nan_to_num(arr, nan=0.0)
    kernel_size = min(9, arr.size if arr.size % 2 else arr.size - 1)
    kernel_size = max(3, kernel_size)
    kernel = np.ones(kernel_size, dtype=float) / float(kernel_size)
    filled = np.interp(np.arange(arr.size), np.flatnonzero(finite), arr[finite])
    padded = np.pad(filled, kernel_size // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _safe_gradient(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    t = np.asarray(time, dtype=float)
    if vals.size < 2:
        return np.zeros_like(vals)
    finite = np.isfinite(vals)
    filled = vals.copy()
    if finite.sum() >= 2:
        filled[~finite] = np.interp(t[~finite], t[finite], vals[finite])
    else:
        filled[~finite] = 0.0
    return np.gradient(filled, t, edge_order=1)


def _tail_scaling(mode: str, gain: float) -> float:
    if str(mode).lower() == "manual":
        return float(np.clip(gain, 0.0, 1.0))
    return 1.0


def _dt(time: np.ndarray) -> float:
    diffs = np.diff(np.asarray(time, dtype=float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    return float(np.nanmedian(diffs)) if diffs.size else 1e-3


def _fill_scalar(n: int, mask: np.ndarray, value: float) -> np.ndarray:
    out = np.full(n, np.nan)
    out[np.asarray(mask, dtype=bool)] = value
    return out


def _fill_tail(n: int, tail_indices: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.full(n, np.nan)
    out[np.asarray(tail_indices, dtype=int)] = values
    return out


def _smoothstep(x: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


__all__ = [
    "apply_finite_time_zero_return_tail",
    "fill_tail_measured_field",
    "missing_time_ranges",
    "normalize_tail_duration_mode",
    "normalize_tail_return_mode",
    "tail_cycle_count_from_duration",
    "unified_tail_diagnostics",
]
