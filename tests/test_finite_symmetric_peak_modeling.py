from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.waveform_review_normalization import build_finite_symmetric_peak_review
from field_analysis.waveform_review_normalization import normalize_raw_waveform_frame


def _finite_frame(*, cycle_count: float = 1.0, freq_hz: float = 1.0) -> pd.DataFrame:
    time_s = np.linspace(0.0, 0.5 + cycle_count / freq_hz, 501)
    active = (time_s >= 0.25) & (time_s <= 0.25 + cycle_count / freq_hz)
    phase = 2.0 * np.pi * freq_hz * (time_s - 0.25)
    field = np.where(active, np.where(np.sin(phase) >= 0.0, 80.0 * np.sin(phase), 40.0 * np.sin(phase)), 0.0)
    voltage = np.where(active, 8.0 * np.sin(phase), 0.0)
    return pd.DataFrame({"time_s": time_s, "bz_mT": field, "daq_input_v": voltage})


def test_raw_finite_waveform_normalization_uses_active_window_and_preserves_raw() -> None:
    frame = _finite_frame(cycle_count=1.0)

    normalized, metadata = normalize_raw_waveform_frame(
        frame,
        source_type="finite-cycle",
        freq_hz=1.0,
        cycle_count=1.0,
    )

    assert "raw_finite_field_mT" in normalized.columns
    assert "normalized_finite_field_mT" in normalized.columns
    assert np.allclose(normalized["raw_finite_field_mT"], frame["bz_mT"])
    assert metadata["finite_normalization_enabled"] is True
    assert metadata["finite_normalization_mode"] == "active_peak_to_50mT"
    assert metadata["finite_active_window_start_s"] == pytest.approx(0.25, abs=0.01)
    assert metadata["finite_active_window_end_s"] == pytest.approx(1.25, abs=0.01)
    assert metadata["finite_positive_peak_normalized_mT"] == pytest.approx(50.0, abs=1e-6)
    assert np.nanmax(np.abs(normalized.loc[normalized["time_s"] < 0.25, "normalized_finite_field_mT"])) == 0.0


def test_raw_continuous_waveform_normalization_uses_steady_state_window_and_preserves_raw() -> None:
    time_s = np.linspace(0.0, 4.0, 401)
    startup = np.where(time_s < 1.0, 30.0, 0.0)
    field = startup + 120.0 * np.sin(2.0 * np.pi * time_s)
    frame = pd.DataFrame({"time_s": time_s, "bz_mT": field, "steady_state_start_s": 1.0, "steady_state_end_s": 4.0})

    normalized, metadata = normalize_raw_waveform_frame(
        frame,
        source_type="continuous",
        freq_hz=1.0,
        cycle_count=float("nan"),
    )

    assert "raw_continuous_field_mT" in normalized.columns
    assert "normalized_continuous_field_mT" in normalized.columns
    assert np.allclose(normalized["raw_continuous_field_mT"], frame["bz_mT"])
    assert metadata["waveform_normalization_enabled"] is True
    assert metadata["waveform_normalization_window"] == "steady_state"
    assert metadata["startup_excluded"] is True
    steady = (normalized["time_s"] >= 1.0) & (normalized["time_s"] <= 4.0)
    assert np.nanmax(np.abs(normalized.loc[steady, "normalized_continuous_field_mT"])) == pytest.approx(50.0, abs=1e-6)


def test_finite_symmetric_peak_modeling_one_cycle_computes_lobe_metrics() -> None:
    cycle_count = 1.0
    normalized, _metadata = normalize_raw_waveform_frame(
        _finite_frame(cycle_count=cycle_count),
        source_type="finite-cycle",
        freq_hz=1.0,
        cycle_count=cycle_count,
    )

    result_frame, result = build_finite_symmetric_peak_review(normalized, freq_hz=1.0, cycle_count=cycle_count)

    assert result["finite_symmetric_peak_modeling_enabled"] is True
    assert result["finite_symmetric_peak_cycle_supported"] is True
    assert result["finite_symmetric_peak_status"] == "ok"
    assert result["supported_finite_symmetric_cycles"] == [1.0, 1.5]
    assert result["production_supported_finite_symmetric_cycles"] == [1.0, 1.5]
    assert result["reference_supported_finite_symmetric_cycles"] == []
    assert result["unsupported_finite_symmetric_cycles"] == [1.25, 1.75, 2.0]
    assert result["finite_symmetric_peak_cycle_role"] == "production"
    assert result["normalized_peak_target_mT"] == 50.0
    assert np.isfinite(result["positive_peak_mT"])
    assert np.isfinite(result["negative_peak_mT"])
    assert result["peak_symmetry_ratio"] < 1.0
    assert result["lobe_balance_applied"] is True
    assert "symmetric_peak_recommended_voltage_v" in result_frame.columns
    assert "symmetric_peak_command_delta_v" in result_frame.columns
    assert np.nanmax(np.abs(result_frame["symmetric_peak_recommended_voltage_v"])) <= 5.0 + 1e-9
    assert result["command_voltage_limit_status"] == "ok"


@pytest.mark.parametrize("cycle_count", [1.25, 1.75, 2.0])
def test_finite_symmetric_peak_modeling_rejects_non_primary_cycles(cycle_count: float) -> None:
    normalized, _metadata = normalize_raw_waveform_frame(
        _finite_frame(cycle_count=cycle_count),
        source_type="finite-cycle",
        freq_hz=1.0,
        cycle_count=cycle_count,
    )

    result_frame, result = build_finite_symmetric_peak_review(normalized, freq_hz=1.0, cycle_count=cycle_count)

    assert result_frame is normalized
    assert result["finite_symmetric_peak_modeling_enabled"] is False
    assert result["finite_symmetric_peak_cycle_supported"] is False
    assert result["finite_symmetric_peak_status"] == "unsupported_cycle"
    assert result["finite_symmetric_peak_cycle_role"] == "unsupported_review_only"


def test_finite_symmetric_peak_modeling_rejects_bad_source_quality() -> None:
    frame = _finite_frame(cycle_count=1.0)
    frame.loc[10, "bz_mT"] = 1e6
    normalized, _metadata = normalize_raw_waveform_frame(frame, source_type="finite-cycle", freq_hz=1.0, cycle_count=1.0)

    _result_frame, result = build_finite_symmetric_peak_review(normalized, freq_hz=1.0, cycle_count=1.0)

    assert result["finite_symmetric_peak_status"] == "unavailable_bad_source_quality"
