"""Built-in example dataset for empty-state demos."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_example_waveform(
    frequency_hz: float = 1.0,
    duration_s: float = 6.0,
    sample_rate_hz: float = 400.0,
) -> pd.DataFrame:
    t = np.arange(0.0, duration_s, 1.0 / sample_rate_hz)
    omega = 2.0 * np.pi * frequency_hz
    current = 10.0 * np.sin(omega * t)
    voltage = 18.0 * np.sin(omega * t + np.deg2rad(35.0)) + 0.9 * np.sin(3.0 * omega * t)
    b_field = 32.0 * np.sin(omega * t + np.deg2rad(12.0))
    return pd.DataFrame(
        {
            "time_s": t,
            "coil_voltage_v": voltage,
            "coil_current_a": current,
            "b_field_mT": b_field,
            "frequency_hz": frequency_hz,
        }
    )
