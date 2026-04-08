"""Derived metrics for electrical, magnetic, advanced, and gain analyses."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import integrate

from coil_analyzer.analysis.signal_analysis import (
    build_window_mask,
    compute_fundamental,
    compute_zero_crossing_frequency,
    estimate_frequency_from_signal,
)
from coil_analyzer.models import AnalysisWindow


def analyze_dataset(
    standardized_df: pd.DataFrame,
    analysis_window: AnalysisWindow,
    frequency_override_hz: float | None = None,
    metadata: dict[str, Any] | None = None,
    representative_b_field: str | None = None,
) -> dict[str, Any]:
    metadata = metadata or {}
    if "current_a" in standardized_df.columns:
        frequency_hz = float(
            frequency_override_hz
            or metadata.get("frequency_hz")
            or estimate_frequency_from_signal(
                standardized_df["time_s"].to_numpy(),
                standardized_df["current_a"].to_numpy(),
            )
        )
    elif "voltage_v" in standardized_df.columns:
        frequency_hz = float(
            frequency_override_hz
            or metadata.get("frequency_hz")
            or estimate_frequency_from_signal(
                standardized_df["time_s"].to_numpy(),
                standardized_df["voltage_v"].to_numpy(),
            )
        )
    else:
        raise ValueError("At least current or voltage signal is required for frequency estimation.")

    mask = build_window_mask(
        standardized_df["time_s"].to_numpy(),
        frequency_hz=frequency_hz,
        cycle_start=analysis_window.cycle_start,
        cycle_count=analysis_window.cycle_count,
    )
    window_df = standardized_df.loc[mask].copy()
    if len(window_df) < 8:
        raise ValueError("Selected analysis window is too short.")

    results: dict[str, Any] = {
        "frequency_hz": frequency_hz,
        "analysis_window": analysis_window.to_dict(),
        "window_start_s": float(window_df["time_s"].iloc[0]),
        "window_end_s": float(window_df["time_s"].iloc[-1]),
        "sample_count": int(len(window_df)),
        "signals": {},
        "warnings": [],
    }

    for column in [column for column in window_df.columns if column != "time_s"]:
        signal_result = compute_fundamental(
            window_df["time_s"].to_numpy(),
            window_df[column].to_numpy(),
            frequency_hz=frequency_hz,
            remove_offset=analysis_window.remove_offset,
            detrend=analysis_window.detrend,
            smoothing=analysis_window.zero_phase_smoothing,
            smoothing_order=analysis_window.smoothing_order,
            smoothing_cutoff_ratio=analysis_window.smoothing_cutoff_ratio,
        )
        results["signals"][column] = {
            "frequency_hz": signal_result.frequency_hz,
            "amplitude_pk": signal_result.amplitude_pk,
            "amplitude_rms": signal_result.amplitude_rms,
            "phase_deg": signal_result.phase_deg,
            "phase_delay_s": signal_result.phase_delay_s,
            "raw_rms": signal_result.raw_rms,
            "raw_pp": signal_result.raw_pp,
            "crest_factor": signal_result.crest_factor,
            "thd": signal_result.thd,
            "fitted": signal_result.fitted.tolist(),
            "detrended": signal_result.detrended.tolist(),
            "warnings": list(signal_result.warnings),
            "zero_crossing_frequency_hz": (
                compute_zero_crossing_frequency(
                    window_df["time_s"].to_numpy(),
                    window_df[column].to_numpy(),
                )
                if analysis_window.show_zero_crossing_aux
                else None
            ),
        }
        results["warnings"].extend(signal_result.warnings)

    results["electrical"] = compute_electrical_metrics(results["signals"], frequency_hz)
    results["magnetic"] = compute_magnetic_metrics(results["signals"], representative_b_field)
    if "voltage_v" in window_df.columns and "current_a" in window_df.columns:
        results["advanced"] = {
            "lambda_supported": True,
            "waveform": window_df[["time_s", "voltage_v", "current_a"]].to_dict(orient="list"),
        }
    else:
        results["advanced"] = {"lambda_supported": False}
    return results


def compute_electrical_metrics(signals: dict[str, dict[str, Any]], frequency_hz: float) -> dict[str, Any]:
    voltage = signals.get("voltage_v")
    current = signals.get("current_a")
    if not voltage or not current or current["amplitude_pk"] == 0:
        return {}
    delta_phi_deg = float(voltage["phase_deg"] - current["phase_deg"])
    z_mag = float(voltage["amplitude_pk"] / current["amplitude_pk"])
    req = float(z_mag * np.cos(np.deg2rad(delta_phi_deg)))
    xeq = float(z_mag * np.sin(np.deg2rad(delta_phi_deg)))
    apparent = float(voltage["amplitude_rms"] * current["amplitude_rms"])
    real = float(apparent * np.cos(np.deg2rad(delta_phi_deg)))
    reactive = float(apparent * np.sin(np.deg2rad(delta_phi_deg)))
    return {
        "V1_pk": voltage["amplitude_pk"],
        "V1_rms": voltage["amplitude_rms"],
        "I1_pk": current["amplitude_pk"],
        "I1_rms": current["amplitude_rms"],
        "phase_V_deg": voltage["phase_deg"],
        "phase_I_deg": current["phase_deg"],
        "delta_phi_VI_deg": delta_phi_deg,
        "delta_phi_VI_delay_s": float(delta_phi_deg / (360.0 * frequency_hz)),
        "|Z1|": z_mag,
        "Req": req,
        "Xeq": xeq,
        "Leq_H": float(xeq / (2.0 * np.pi * frequency_hz)) if frequency_hz else float("nan"),
        "apparent_power_VA": apparent,
        "real_power_W": real,
        "reactive_power_VAR": reactive,
        "current_crest_factor": current["crest_factor"],
        "voltage_crest_factor": voltage["crest_factor"],
        "current_thd": current.get("thd"),
        "voltage_thd": voltage.get("thd"),
    }


def compute_magnetic_metrics(
    signals: dict[str, dict[str, Any]],
    representative_b_field: str | None,
) -> dict[str, Any]:
    current = signals.get("current_a")
    voltage = signals.get("voltage_v")
    magnetic_keys = [key for key in signals if key.startswith("magnetic_")]
    if not magnetic_keys:
        return {}
    key = representative_b_field if representative_b_field in magnetic_keys else magnetic_keys[0]
    magnetic = signals[key]
    result = {
        "representative_channel": key,
        "B1_pk": magnetic["amplitude_pk"],
        "B1_rms": magnetic["amplitude_rms"],
        "phase_B_deg": magnetic["phase_deg"],
    }
    if current and current["amplitude_pk"] != 0:
        result["delta_phi_BI_deg"] = float(magnetic["phase_deg"] - current["phase_deg"])
        result["K_BI"] = float(magnetic["amplitude_pk"] / current["amplitude_pk"])
    if voltage and voltage["amplitude_pk"] != 0:
        result["delta_phi_BV_deg"] = float(magnetic["phase_deg"] - voltage["phase_deg"])
        result["K_BV"] = float(magnetic["amplitude_pk"] / voltage["amplitude_pk"])
    result["all_channels"] = {
        name: {"B1_pk": signals[name]["amplitude_pk"], "phase_deg": signals[name]["phase_deg"]}
        for name in magnetic_keys
    }
    return result


def compute_lambda_metrics(
    time_s: np.ndarray,
    voltage_v: np.ndarray,
    current_a: np.ndarray,
    rdc_ohm: float,
) -> dict[str, Any]:
    inferred_voltage = voltage_v - rdc_ohm * current_a
    lambda_wb_turn = integrate.cumulative_trapezoid(inferred_voltage, time_s, initial=0.0)
    diff_current = np.gradient(current_a, time_s)
    differential_inductance = np.divide(
        np.gradient(lambda_wb_turn, time_s),
        diff_current,
        out=np.full_like(lambda_wb_turn, np.nan, dtype=float),
        where=np.abs(diff_current) > 1e-9,
    )
    return {
        "label": "system-level inferred quantity",
        "lambda": lambda_wb_turn,
        "differential_inductance_h": differential_inductance,
    }


def compute_gain_requirement(
    frequency_hz: float,
    target_ipp_a: float,
    electrical_metrics: dict[str, Any],
    achieved_ipp_a: float | None,
    measured_vout_pk: float | None,
    gain_mode_v_per_v: float,
    vin_pk: float,
    notes: str = "",
) -> dict[str, Any]:
    target_ipk = target_ipp_a / 2.0
    required_vout_pk = None
    if measured_vout_pk is not None and achieved_ipp_a and achieved_ipp_a > 0:
        required_vout_pk = float(measured_vout_pk * (target_ipp_a / achieved_ipp_a))
    elif electrical_metrics.get("|Z1|") is not None:
        required_vout_pk = float(target_ipk * electrical_metrics["|Z1|"])
    alpha = (
        float(required_vout_pk / (vin_pk * gain_mode_v_per_v))
        if required_vout_pk is not None and vin_pk > 0 and gain_mode_v_per_v > 0
        else None
    )
    return {
        "frequency_hz": frequency_hz,
        "target_Ipp_A": target_ipp_a,
        "achieved_Ipp_A": achieved_ipp_a,
        "required_Vout_pk": required_vout_pk,
        "required_alpha": alpha,
        "required_alpha_pct": alpha * 100.0 if alpha is not None else None,
        "gain_mode_v_per_v": gain_mode_v_per_v,
        "overload": bool(alpha is not None and alpha > 1.0),
        "notes": notes,
    }
