from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.final_modeled_lut import (
    build_final_modeled_voltage_lut_export,
    final_modeled_voltage_lut_to_csv_bytes,
    load_final_modeled_voltage_lut,
)


def _command_profile() -> pd.DataFrame:
    time_s = np.linspace(0.0, 1.25, 6)
    limited_voltage_v = np.array([0.0, 1.5, 3.0, -2.0, -1.0, 0.0], dtype=float)
    return pd.DataFrame(
        {
            "time_s": time_s,
            "recommended_voltage_v": limited_voltage_v * 1.1,
            "limited_voltage_v": limited_voltage_v,
            "predicted_field_mT": limited_voltage_v * 10.0,
        }
    )


def test_export_final_modeled_lut_uses_limited_voltage_sample_by_sample() -> None:
    profile = _command_profile()

    payload = build_final_modeled_voltage_lut_export(
        profile,
        route="finite_empirical_weighted_support",
        mode="finite_cycle",
        freq_hz=1.0,
        cycle_count=1.25,
        waveform="sine",
        voltage_limit_v=5.0,
    )

    frame = payload["frame"]
    metadata = payload["metadata"]
    assert list(frame.columns) == ["sample_index", "time_s", "voltage_v"]
    assert frame["sample_index"].tolist() == list(range(len(profile)))
    assert np.array_equal(frame["time_s"].to_numpy(dtype=float), profile["time_s"].to_numpy(dtype=float))
    assert np.array_equal(frame["voltage_v"].to_numpy(dtype=float), profile["limited_voltage_v"].to_numpy(dtype=float))
    assert metadata["lut_export_type"] == "final_modeled_voltage_lut"
    assert metadata["voltage_source_column"] == "limited_voltage_v"
    assert metadata["time_source_column"] == "time_s"
    assert metadata["fourier_resynthesis_involved"] is False
    assert metadata["harmonic_export_involved"] is False
    assert metadata["finite_only"] is True
    assert metadata["sample_count"] == len(profile)
    assert "correction_delta_v" not in frame.columns
    assert "second_voltage_v" not in frame.columns


def test_exported_lut_csv_round_trips_with_seconds_timebase_preserved() -> None:
    payload = build_final_modeled_voltage_lut_export(_command_profile(), freq_hz=1.0, cycle_count=1.25, waveform="sine")
    csv_bytes = final_modeled_voltage_lut_to_csv_bytes(payload)

    parsed = load_final_modeled_voltage_lut(io.BytesIO(csv_bytes))

    assert parsed["parse_status"] == "ok"
    frame = parsed["frame"]
    diagnostics = parsed["diagnostics"]
    assert np.allclose(frame["time_s"], payload["frame"]["time_s"])
    assert np.allclose(frame["voltage_v"], payload["frame"]["voltage_v"])
    assert diagnostics["suspected_time_unit"] == "seconds"
    assert diagnostics["time_axis_status"] == "ok"
    assert diagnostics["time_monotonic"] is True
    assert diagnostics["duplicated_time_count"] == 0


def test_uploaded_lut_parser_detects_millisecond_axis_uploaded_as_seconds() -> None:
    uploaded = pd.DataFrame(
        {
            "sample_index": [0, 1, 2, 3],
            "time_s": [0.0, 250.0, 500.0, 750.0],
            "voltage_v": [0.0, 1.0, -1.0, 0.0],
        }
    )

    parsed = load_final_modeled_voltage_lut(uploaded)

    diagnostics = parsed["diagnostics"]
    assert diagnostics["suspected_time_unit"] == "milliseconds"
    assert diagnostics["time_axis_status"] == "suspect_ms_uploaded_as_s"


def test_uploaded_lut_parser_detects_non_monotonic_and_duplicate_time() -> None:
    non_monotonic = load_final_modeled_voltage_lut(
        pd.DataFrame({"time_s": [0.0, 0.2, 0.1], "voltage_v": [0.0, 1.0, 0.0]})
    )
    duplicate = load_final_modeled_voltage_lut(
        pd.DataFrame({"sample_index": [0, 1, 2], "time_s": [0.0, 0.1, 0.1], "voltage_v": [0.0, 1.0, 0.0]})
    )

    assert non_monotonic["diagnostics"]["time_axis_status"] == "non_monotonic"
    assert duplicate["diagnostics"]["time_axis_status"] == "duplicate_time"
    assert duplicate["diagnostics"]["duplicated_time_count"] == 1


def test_uploaded_lut_parser_reports_missing_required_schema_without_second_correction_fields() -> None:
    parsed = load_final_modeled_voltage_lut(pd.DataFrame({"time_s": [0.0, 0.1], "limited_voltage_v": [0.0, 1.0]}))

    assert parsed["parse_status"] == "error"
    assert "voltage_v" in str(parsed["parse_error"])
    assert "correction_delta_v" not in parsed["frame"].columns
    assert "second_voltage_v" not in parsed["frame"].columns
