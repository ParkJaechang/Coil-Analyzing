from __future__ import annotations

import numpy as np
import pandas as pd


def build_lcr_impedance_table(
    lcr_measurements: pd.DataFrame,
    freq_column: str = "freq_hz",
    resistance_column: str = "rs_ohm",
    inductance_column: str = "ls_h",
    capacitance_column: str = "cs_f",
) -> pd.DataFrame:
    """Normalize LCR data into impedance and current-per-voltage prior columns."""

    if lcr_measurements.empty:
        return pd.DataFrame()

    working = lcr_measurements.copy()
    working["freq_hz"] = pd.to_numeric(working[freq_column], errors="coerce")
    working["rs_ohm"] = pd.to_numeric(working[resistance_column], errors="coerce")
    working["ls_h"] = pd.to_numeric(working[inductance_column], errors="coerce")
    if capacitance_column in working.columns:
        working["cs_f"] = pd.to_numeric(working[capacitance_column], errors="coerce")
    else:
        working["cs_f"] = 0.0

    working = working.dropna(subset=["freq_hz", "rs_ohm", "ls_h"]).sort_values("freq_hz").reset_index(drop=True)
    if working.empty:
        return pd.DataFrame()

    omega = 2.0 * np.pi * working["freq_hz"].to_numpy(dtype=float)
    resistance = working["rs_ohm"].to_numpy(dtype=float)
    inductive_reactance = omega * working["ls_h"].to_numpy(dtype=float)
    capacitance = working["cs_f"].fillna(0.0).to_numpy(dtype=float)
    capacitive_reactance = np.where(capacitance > 0, 1.0 / np.maximum(omega * capacitance, 1e-18), 0.0)
    impedance = resistance + 1j * (inductive_reactance - capacitive_reactance)
    current_per_v = 1.0 / np.where(np.abs(impedance) > 1e-18, impedance, np.nan + 0j)

    working["impedance_real_ohm"] = np.real(impedance)
    working["impedance_imag_ohm"] = np.imag(impedance)
    working["impedance_mag_ohm"] = np.abs(impedance)
    working["impedance_phase_deg"] = np.degrees(np.angle(impedance))
    working["current_per_v_real"] = np.real(current_per_v)
    working["current_per_v_imag"] = np.imag(current_per_v)
    working["current_per_v_mag"] = np.abs(current_per_v)
    working["current_per_v_phase_deg"] = np.degrees(np.angle(current_per_v))
    return working


def build_lcr_harmonic_prior(
    lcr_impedance_table: pd.DataFrame,
    base_freq_hz: float,
    harmonics: list[int] | range,
    daq_to_amp_gain: float = 1.0,
    output_scale: float = 1.0,
) -> pd.DataFrame:
    """Interpolate LCR data onto harmonic frequencies and convert it into a complex transfer prior."""

    if lcr_impedance_table.empty or not np.isfinite(base_freq_hz) or base_freq_hz <= 0:
        return pd.DataFrame()

    working = lcr_impedance_table.copy()
    working = working.sort_values("freq_hz").dropna(subset=["freq_hz", "current_per_v_real", "current_per_v_imag"])
    if working.empty:
        return pd.DataFrame()

    freq_values = working["freq_hz"].to_numpy(dtype=float)
    current_real = working["current_per_v_real"].to_numpy(dtype=float)
    current_imag = working["current_per_v_imag"].to_numpy(dtype=float)
    rows: list[dict[str, float | int | bool]] = []

    for harmonic in harmonics:
        harmonic = int(harmonic)
        harmonic_freq_hz = float(base_freq_hz) * harmonic
        interpolated_real = float(np.interp(harmonic_freq_hz, freq_values, current_real))
        interpolated_imag = float(np.interp(harmonic_freq_hz, freq_values, current_imag))
        transfer = (interpolated_real + 1j * interpolated_imag) * float(daq_to_amp_gain) * float(output_scale)
        rows.append(
            {
                "harmonic": harmonic,
                "harmonic_freq_hz": harmonic_freq_hz,
                "transfer_real": float(np.real(transfer)),
                "transfer_imag": float(np.imag(transfer)),
                "transfer_mag": float(np.abs(transfer)),
                "transfer_phase_deg": float(np.degrees(np.angle(transfer))),
                "frequency_clamped": not (float(freq_values.min()) <= harmonic_freq_hz <= float(freq_values.max())),
            }
        )

    return pd.DataFrame(rows)
