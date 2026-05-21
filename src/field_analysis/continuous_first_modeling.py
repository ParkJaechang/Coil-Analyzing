from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from field_analysis.continuous_phase_support import continuous_peak_alignment_metadata, interpolate_signal
from field_analysis.finite_second_modeling_stabilization import align_measured_field_for_residual
from field_analysis.finite_second_modeling_stabilization import smooth_measured_field_for_second_modeling
from field_analysis.finite_second_modeling_stabilization import stabilize_correction_delta
from field_analysis.finite_second_modeling_tail import compute_second_modeling_gain


def build_continuous_phase_aligned_command_profile(
    steady_state_one_cycle_frame: pd.DataFrame,
    *,
    support_frame: pd.DataFrame | None = None,
    freq_hz: float,
    waveform_type: str | None = None,
    correction_gain: float = 0.25,
    correction_gain_mode: str = "auto",
    voltage_limit_v: float = 5.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = steady_state_one_cycle_frame.copy()
    if frame.empty:
        return frame, {
            "continuous_first_modeling_available": False,
            "continuous_first_modeling_status": "empty_steady_state_window",
        }
    time_s = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
    time_origin = float(np.nanmin(time_s[np.isfinite(time_s)])) if np.isfinite(time_s).any() else 0.0
    time_s = time_s - time_origin
    support = support_frame.copy() if isinstance(support_frame, pd.DataFrame) and not support_frame.empty else frame.copy()
    measured = _first_numeric_column(frame, ("measured_field_normalized_mT", "normalized_measured_field_mT")).to_numpy(dtype=float)
    target = _first_numeric_column(frame, ("normalized_physical_target_output_mT", "physical_target_output_mT")).to_numpy(dtype=float)
    first_voltage = _continuous_first_voltage(frame).to_numpy(dtype=float)
    active_mask = np.isfinite(time_s) & np.isfinite(measured) & np.isfinite(target)
    support_time = pd.to_numeric(support["time_s"], errors="coerce").to_numpy(dtype=float) - time_origin
    support_measured = _first_numeric_column(support, ("measured_field_normalized_mT", "normalized_measured_field_mT")).to_numpy(dtype=float)
    support_voltage = _continuous_first_voltage(support).to_numpy(dtype=float)
    support_mask = np.isfinite(support_time) & np.isfinite(support_measured)
    support_smoothed, smoothing_meta = smooth_measured_field_for_second_modeling(
        support_time,
        support_measured,
        support_time,
        support_mask,
        freq_hz=float(freq_hz),
        cycle_count=1.0,
    )
    measured_smoothed = interpolate_signal(time_s, support_time, support_smoothed)
    period_s = 1.0 / max(float(freq_hz), 1e-12)
    peak_meta = continuous_peak_alignment_metadata(support_time, support_voltage, support_smoothed, period_s=period_s)
    delay_s = float(peak_meta.get("continuous_phase_delay_s") or 0.0)
    measured_aligned = interpolate_signal(time_s + delay_s, support_time, support_smoothed)
    alignment_meta = {
        "phase_alignment_status": "ok" if np.isfinite(measured_aligned).all() else "unavailable_source_range",
        "continuous_phase_alignment_method": "field_peak_to_voltage_peak",
        "continuous_first_modeling_phase_reference": "voltage_peak",
        "phase_alignment_fallback_used": not np.isfinite(measured_aligned).all(),
        **peak_meta,
    }
    if alignment_meta["phase_alignment_status"] != "ok":
        measured_aligned, legacy_alignment_meta = align_measured_field_for_residual(
            time_s,
            target,
            measured_smoothed,
            active_mask,
            freq_hz=float(freq_hz),
            residual_alignment_mode="first_peak_aligned",
        )
        alignment_meta.update(legacy_alignment_meta)
        alignment_meta["phase_alignment_fallback_used"] = True
    measured_for_modeling = measured_aligned if alignment_meta.get("phase_alignment_status") == "ok" else measured_smoothed
    residual = target - measured_for_modeling
    unit_delta = residual / 50.0 * float(voltage_limit_v)
    gain, gain_meta = compute_second_modeling_gain(
        unit_delta,
        first_voltage,
        active_mask,
        manual_gain=float(correction_gain),
        gain_mode=correction_gain_mode,
        voltage_limit_v=float(voltage_limit_v),
        tail_mask=np.zeros_like(active_mask, dtype=bool),
    )
    raw_delta = unit_delta * float(gain)
    correction_delta, stabilization_meta, arrays = stabilize_correction_delta(
        raw_delta,
        first_voltage,
        time_s,
        active_mask,
        freq_hz=float(freq_hz),
        cycle_count=1.0,
        enabled=True,
        tail_mask=np.zeros_like(active_mask, dtype=bool),
    )
    first_modeled = first_voltage + correction_delta
    limited = np.clip(first_modeled, -float(voltage_limit_v), float(voltage_limit_v))
    command = pd.DataFrame(
        {
            "time_s": time_s,
            "base_voltage_v": first_voltage,
            "first_modeled_voltage_v": first_modeled,
            "limited_voltage_v": limited,
            "correction_delta_v": correction_delta,
            "raw_correction_delta_v": raw_delta,
            "smoothed_correction_delta_v": arrays.get("smoothed_correction_delta_v", correction_delta),
            "measured_field_smoothed_mT": measured_smoothed,
            "measured_field_aligned_mT": measured_aligned,
            "target_field_1cycle_mT": target,
            "residual_for_modeling_mT": residual,
            "continuous_loop_output": True,
            "loop_endpoint_policy": "period_exclusive",
            "continuous_export_cycle_count": 1.0,
            "freq_hz": float(freq_hz),
            "waveform_type": waveform_type,
        }
    )
    metadata = {
        **smoothing_meta,
        **alignment_meta,
        **gain_meta,
        **stabilization_meta,
        "continuous_first_modeling_available": True,
        "continuous_first_modeling_uses_phase_aligned_kernel": True,
        "continuous_first_modeling_tail_disabled": True,
        "continuous_first_modeling_cycle_count": 1.0,
        "continuous_loop_output": True,
        "continuous_modeling_kernel_source": "finite_second_modeling_shared_kernel",
        "continuous_target_shape": "fixed_rounded_triangle",
        "continuous_target_cycle_count": 1.0,
        "continuous_export_cycle_count": 1.0,
        "loop_endpoint_policy": "period_exclusive",
        "fourier_resynthesis_involved": False,
        "harmonic_export_involved": False,
    }
    return command.reset_index(drop=True), metadata


def _first_numeric_column(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    raise ValueError(f"missing one of required columns: {columns}")


def _continuous_first_voltage(frame: pd.DataFrame) -> pd.Series:
    for column in ("limited_voltage_v", "voltage_normalized_v", "raw_voltage_v", "Voltage1_V"):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.zeros(len(frame), dtype=float))
