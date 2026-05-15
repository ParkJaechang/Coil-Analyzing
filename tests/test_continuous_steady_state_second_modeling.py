from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.continuous_steady_state_extraction import (
    build_continuous_actual_drive_review_case,
)


def _actual_drive_like_frame(*, freq_hz: float = 2.0, cycles: int = 7, samples_per_cycle: int = 80) -> pd.DataFrame:
    period = 1.0 / freq_hz
    time_s = np.arange(0, cycles * samples_per_cycle, dtype=float) * period / samples_per_cycle
    phase = 2.0 * np.pi * freq_hz * time_s
    startup_scale = 1.0 - 0.4 * np.exp(-np.arange(len(time_s)) / samples_per_cycle)
    voltage = 2.5 * np.sin(phase)
    raw_hallbz = -(50.0 * startup_scale * np.sin(phase - 0.08))
    return pd.DataFrame(
        {
            "time_s": time_s,
            "raw_hallbz_mT": raw_hallbz,
            "raw_actual_drive_voltage_v": voltage,
        }
    )


def test_continuous_actual_drive_second_input_uses_steady_one_cycle_only() -> None:
    result = build_continuous_actual_drive_review_case(
        _actual_drive_like_frame(freq_hz=2.0),
        waveform_type="sine",
        freq_hz=2.0,
        purpose="second_modeling",
    )
    frame = result["steady_state_one_cycle_frame"]
    metadata = result["metadata"]

    assert metadata["second_modeling_input_mode"] == "continuous_steady_state"
    assert metadata["second_drive_startup_transient_excluded"] is True
    assert metadata["second_drive_window_cycle_count"] == 1.0
    assert metadata["second_drive_actual_data_used"] == "steady_state_one_cycle_only"
    assert frame["time_s"].min() == 0.0
    assert frame["time_s"].max() < 0.5
    assert frame["steady_state_cycle_index"].nunique() == 1


def test_continuous_validation_uses_stable_one_cycle_metadata() -> None:
    result = build_continuous_actual_drive_review_case(
        _actual_drive_like_frame(freq_hz=1.0, cycles=6),
        waveform_type="sine",
        freq_hz=1.0,
        purpose="validation",
    )
    metadata = result["metadata"]

    assert metadata["validation_input_mode"] == "continuous_steady_state"
    assert metadata["validation_startup_transient_excluded"] is True
    assert metadata["validation_window_cycle_count"] == 1.0
