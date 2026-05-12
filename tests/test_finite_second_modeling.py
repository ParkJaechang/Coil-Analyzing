from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.finite_second_modeling import generate_second_modeled_voltage_lut
from field_analysis.final_modeled_lut import build_final_modeled_voltage_lut_export
from tests.test_finite_actual_drive_response import _write_actual_drive_csv


def _first_profile() -> pd.DataFrame:
    time_s = np.linspace(0.0, 1.0, 101)
    target = 50.0 * np.sin(np.pi * time_s)
    voltage = 3.0 * np.sin(np.pi * time_s)
    return pd.DataFrame(
        {
            "time_s": time_s,
            "physical_target_output_mT": target,
            "limited_voltage_v": voltage,
            "recommended_voltage_v": voltage,
        }
    )


def test_second_modeling_generates_limited_voltage_for_one_cycle(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1.25Hz_1cycle_result.csv"
    _write_actual_drive_csv(actual)

    frame, metadata = generate_second_modeled_voltage_lut(
        _first_profile(),
        actual,
        freq_hz=1.0,
        cycle_count=1.0,
        correction_gain=0.25,
    )

    assert metadata["second_modeling_available"] is True
    assert metadata["second_modeling_status"] == "ok"
    assert metadata["hallbz_sign_applied"] is True
    assert metadata["final_export_voltage_source_column"] == "second_limited_voltage_v"
    assert np.nanmax(np.abs(frame["second_limited_voltage_v"])) <= 5.0 + 1e-12
    assert np.isclose(np.nanmax(np.abs(frame["measured_field_normalized_mT"])), 50.0)
    assert {"second_correction_delta_v", "second_modeled_voltage_v", "final_voltage_v"}.issubset(frame.columns)
    assert bool(frame["target_unchanged"].iloc[0]) is True


def test_second_modeling_rejects_non_one_cycle_without_delta(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1.25Hz_1.25cycle_result.csv"
    _write_actual_drive_csv(actual)

    frame, metadata = generate_second_modeled_voltage_lut(_first_profile(), actual, freq_hz=1.0, cycle_count=1.25)

    assert frame.equals(_first_profile())
    assert metadata["second_modeling_available"] is False
    assert metadata["second_modeling_status"] == "unsupported_cycle_policy_1cycle_only"
    assert metadata["second_correction_delta_v_generated"] is False
    assert "second_correction_delta_v" not in frame.columns
    assert "second_limited_voltage_v" not in frame.columns


def test_final_lut_export_prefers_second_limited_voltage_when_available(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1.25Hz_1cycle_result.csv"
    _write_actual_drive_csv(actual)
    frame, _metadata = generate_second_modeled_voltage_lut(_first_profile(), actual, freq_hz=1.0, cycle_count=1.0)
    frame["second_modeling_available"] = True
    frame["second_modeling_status"] = "ok"

    export = build_final_modeled_voltage_lut_export(frame, freq_hz=1.0, cycle_count=1.0, waveform="sine")

    assert export["metadata"]["voltage_source_column"] == "second_limited_voltage_v"
    assert np.allclose(export["frame"]["voltage_v"], frame["second_limited_voltage_v"])
    assert list(export["frame"].columns) == ["sample_index", "time_s", "voltage_v"]
    assert export["metadata"]["fourier_resynthesis_involved"] is False
    assert export["metadata"]["harmonic_export_involved"] is False
