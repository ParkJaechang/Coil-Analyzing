from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.continuous_steady_state_extraction import evaluate_continuous_steady_state_validation


def test_continuous_validation_metrics_are_one_cycle_steady_state_only() -> None:
    period = 1.0
    time_s = np.linspace(0.0, 6.0, 600, endpoint=False)
    phase = 2.0 * np.pi * time_s / period
    scale = 1.0 - 0.3 * np.exp(-time_s / 1.0)
    frame = pd.DataFrame(
        {
            "time_s": time_s,
            "raw_hallbz_mT": -(50.0 * scale * np.sin(phase)),
            "raw_actual_drive_voltage_v": 3.0 * np.sin(phase),
        }
    )

    result = evaluate_continuous_steady_state_validation(frame, waveform_type="sine", freq_hz=1.0)

    metrics = result["metrics"]
    metadata = result["metadata"]
    assert metadata["validation_input_mode"] == "continuous_steady_state"
    assert metadata["validation_startup_transient_excluded"] is True
    assert metadata["validation_window_cycle_count"] == 1.0
    assert {"positive_peak_error_pct", "negative_trough_error_pct", "waveform_nrmse_pct", "shape_correlation"}.issubset(metrics)
    assert np.isfinite(metrics["waveform_nrmse_pct"])
