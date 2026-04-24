from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.ui_raw_waveforms import build_raw_waveform_label_lookup


def _analysis_fixture(file_name: str) -> SimpleNamespace:
    normalized = pd.DataFrame(
        {
            "test_id": [f"opaque_{file_name}"] * 3,
            "source_file": [f"10d2317e131196fe_{file_name}"] * 3,
            "sheet_name": ["main"] * 3,
            "waveform_type": [np.nan] * 3,
            "freq_hz": [np.nan] * 3,
            "current_pp_target_a": [np.nan] * 3,
            "cycle_total_expected": [np.nan] * 3,
            "time_s": [0.0, 0.5, 1.0],
            "daq_input_v": [0.0, 1.0, 0.0],
            "bz_mT": [0.0, 20.0, 0.0],
        }
    )
    parsed = SimpleNamespace(
        source_file=f"10d2317e131196fe_{file_name}",
        sheet_name="main",
        metadata={},
        normalized_frame=normalized,
    )
    preprocess = SimpleNamespace(corrected_frame=normalized.copy())
    return SimpleNamespace(parsed=parsed, preprocess=preprocess)


def _label_for(file_name: str) -> str:
    test_id = f"opaque_{file_name}"
    label_by_id, _ = build_raw_waveform_label_lookup([test_id], {test_id: _analysis_fixture(file_name)})
    return label_by_id[test_id]


def test_new_continuous_filename_labels_do_not_require_current() -> None:
    assert _label_for("continuous_sine_1Hz.csv") == (
        "continuous | Sine | 1 Hz | ±5V | Gain 100% | continuous_sine_1Hz.csv"
    )
    assert _label_for("continuous_triangle_0.25Hz.csv") == (
        "continuous | Triangle | 0.25 Hz | ±5V | Gain 100% | continuous_triangle_0.25Hz.csv"
    )


def test_new_finite_filename_labels_include_cycle_and_fixed_conditions() -> None:
    assert _label_for("finite_sine_1Hz_1.25cycle.csv") == (
        "finite-cycle | Sine | 1 Hz | 1.25 cycle | ±5V | Gain 100% | finite_sine_1Hz_1.25cycle.csv"
    )
    assert _label_for("finite_triangle_2Hz_1.75cycle.csv") == (
        "finite-cycle | Triangle | 2 Hz | 1.75 cycle | ±5V | Gain 100% | finite_triangle_2Hz_1.75cycle.csv"
    )
