from __future__ import annotations

import numpy as np
import pandas as pd


def _signal_margin_ratio(signal: np.ndarray, limit_pk: float, percentile: float | None = None) -> float:
    if not np.isfinite(limit_pk) or limit_pk <= 0:
        return float("nan")
    finite_signal = np.abs(signal[np.isfinite(signal)])
    if finite_signal.size == 0:
        return float("nan")
    usage = (
        float(np.nanpercentile(finite_signal, percentile))
        if percentile is not None
        else float(np.nanmax(finite_signal))
    )
    return float(np.clip(1.0 - usage / limit_pk, 0.0, 1.0))


def apply_command_hardware_model(
    command_waveform: pd.DataFrame,
    max_daq_voltage_pp: float,
    amp_gain_at_100_pct: float,
    support_amp_gain_pct: float,
    amp_gain_limit_pct: float = 100.0,
    amp_max_output_pk_v: float = 180.0,
    preserve_start_voltage: bool = False,
) -> pd.DataFrame:
    """Apply DAQ limiting and annotate DC AMP feasibility for a command waveform."""

    command = command_waveform.copy()
    recommended = pd.to_numeric(command["recommended_voltage_v"], errors="coerce").to_numpy(dtype=float)
    if len(recommended) == 0 or not np.isfinite(recommended).any():
        command["recommended_voltage_pp"] = float("nan")
        command["limited_voltage_v"] = recommended
        command["limited_voltage_pp"] = float("nan")
        command["required_amp_gain_multiplier"] = float("nan")
        command["support_amp_gain_pct"] = float("nan")
        command["required_amp_gain_pct"] = float("nan")
        command["available_amp_gain_pct"] = float("nan")
        command["amp_output_pp_at_required"] = float("nan")
        command["amp_output_pk_at_required"] = float("nan")
        command["within_daq_limit"] = False
        command["within_amp_gain_limit"] = False
        command["within_amp_output_limit"] = False
        command["within_hardware_limits"] = False
        return command

    if "is_lookahead_target" in command.columns:
        active_mask = command["is_lookahead_target"].to_numpy(dtype=bool)
    elif "is_active_target" in command.columns:
        active_mask = command["is_active_target"].to_numpy(dtype=bool)
    else:
        active_mask = np.isfinite(recommended)

    if preserve_start_voltage:
        scaled_values = recommended
        pp_values = recommended[active_mask] if active_mask.any() else recommended
    else:
        centered = np.zeros_like(recommended)
        if active_mask.any():
            active_values = recommended[active_mask]
            centered[active_mask] = active_values - float(np.nanmean(active_values))
            pp_values = centered[active_mask]
        else:
            centered = recommended - float(np.nanmean(recommended))
            pp_values = centered
        scaled_values = centered

    recommended_pp = float(np.nanmax(pp_values) - np.nanmin(pp_values)) if len(pp_values) else float("nan")
    daq_gain_multiplier = (
        max(recommended_pp / max_daq_voltage_pp, 1.0)
        if np.isfinite(recommended_pp) and np.isfinite(max_daq_voltage_pp) and max_daq_voltage_pp > 0
        else 1.0
    )
    limited = scaled_values / daq_gain_multiplier
    limited_active = limited[active_mask] if active_mask.any() else limited
    limited_pp = float(np.nanmax(limited_active) - np.nanmin(limited_active)) if len(limited_active) else float("nan")
    max_daq_voltage_pk = (
        float(max_daq_voltage_pp) / 2.0
        if np.isfinite(max_daq_voltage_pp) and max_daq_voltage_pp > 0
        else float("nan")
    )
    peak_input_limit_margin = _signal_margin_ratio(limited_active, max_daq_voltage_pk, percentile=None)
    p95_input_limit_margin = _signal_margin_ratio(limited_active, max_daq_voltage_pk, percentile=95.0)

    support_gain_pct = float(support_amp_gain_pct) if np.isfinite(support_amp_gain_pct) and support_amp_gain_pct > 0 else 100.0
    required_gain_pct = support_gain_pct * daq_gain_multiplier
    amp_output_pp_at_required = (
        float(limited_pp * amp_gain_at_100_pct * required_gain_pct / 100.0)
        if np.isfinite(limited_pp) and np.isfinite(amp_gain_at_100_pct) and amp_gain_at_100_pct > 0
        else float("nan")
    )
    amp_output_pk_at_required = amp_output_pp_at_required / 2.0 if np.isfinite(amp_output_pp_at_required) else float("nan")

    max_output_pp = 2.0 * float(amp_max_output_pk_v) if np.isfinite(amp_max_output_pk_v) and amp_max_output_pk_v > 0 else float("inf")
    amp_gain_limit_pct = float(amp_gain_limit_pct) if np.isfinite(amp_gain_limit_pct) and amp_gain_limit_pct > 0 else 100.0
    if (
        np.isfinite(limited_pp)
        and limited_pp > 0
        and np.isfinite(amp_gain_at_100_pct)
        and amp_gain_at_100_pct > 0
        and np.isfinite(max_output_pp)
    ):
        max_gain_pct_by_output = 100.0 * max_output_pp / (limited_pp * amp_gain_at_100_pct)
    else:
        max_gain_pct_by_output = float("inf")
    available_amp_gain_pct = min(amp_gain_limit_pct, max_gain_pct_by_output)

    within_daq_limit = bool(recommended_pp <= max_daq_voltage_pp + 1e-9) if np.isfinite(recommended_pp) else False
    within_amp_gain_limit = bool(required_gain_pct <= available_amp_gain_pct + 1e-9) if np.isfinite(required_gain_pct) else False
    within_amp_output_limit = bool(amp_output_pp_at_required <= max_output_pp + 1e-9) if np.isfinite(amp_output_pp_at_required) else False

    command["recommended_voltage_v"] = scaled_values
    command["recommended_voltage_pp"] = recommended_pp
    command["limited_voltage_v"] = limited
    command["limited_voltage_pp"] = limited_pp
    command["max_daq_voltage_pp"] = float(max_daq_voltage_pp)
    command["max_daq_voltage_pk_v"] = max_daq_voltage_pk
    command["peak_input_limit_margin"] = peak_input_limit_margin
    command["p95_input_limit_margin"] = p95_input_limit_margin
    command["required_amp_gain_multiplier"] = daq_gain_multiplier
    command["support_amp_gain_pct"] = support_gain_pct
    command["required_amp_gain_pct"] = required_gain_pct
    command["amp_gain_limit_pct"] = amp_gain_limit_pct
    command["max_gain_pct_by_output"] = max_gain_pct_by_output
    command["available_amp_gain_pct"] = available_amp_gain_pct
    command["amp_gain_at_100_pct"] = float(amp_gain_at_100_pct)
    command["amp_max_output_pk_v"] = float(amp_max_output_pk_v)
    command["amp_output_pp_at_required"] = amp_output_pp_at_required
    command["amp_output_pk_at_required"] = amp_output_pk_at_required
    command["within_daq_limit"] = within_daq_limit
    command["within_amp_gain_limit"] = within_amp_gain_limit
    command["within_amp_output_limit"] = within_amp_output_limit
    command["within_hardware_limits"] = within_amp_gain_limit and within_amp_output_limit
    return command
