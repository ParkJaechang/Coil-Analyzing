from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coil_analyzer.analysis.metrics import analyze_dataset, compute_lambda_metrics
from coil_analyzer.io.data_loader import infer_column_roles, load_dataframe
from coil_analyzer.models import AnalysisWindow, ChannelConfig, ChannelMapping
from coil_analyzer.preprocessing.channels import infer_time_unit, standardize_dataset
from coil_analyzer.utils.example_data import build_example_waveform


def test_standardize_dataset_handles_ms_and_scaling() -> None:
    df = pd.DataFrame({"time_ms": [0.0, 1.0, 2.0], "v": [1.0, 2.0, 3.0], "i": [2.0, 4.0, 6.0]})
    mapping = ChannelMapping(
        time=ChannelConfig(column="time_ms", unit="ms"),
        voltage=ChannelConfig(column="v", scale=2.0),
        current=ChannelConfig(column="i", invert=True),
    )
    standardized = standardize_dataset(df, mapping)
    assert np.allclose(standardized["time_s"].to_numpy(), np.array([0.0, 0.001, 0.002]))
    assert np.allclose(standardized["voltage_v"].to_numpy(), np.array([2.0, 4.0, 6.0]))
    assert np.allclose(standardized["current_a"].to_numpy(), np.array([-2.0, -4.0, -6.0]))


def test_analyze_dataset_computes_impedance() -> None:
    df = build_example_waveform(frequency_hz=1.0, duration_s=6.0, sample_rate_hz=500.0)
    standardized = df.rename(
        columns={
            "coil_voltage_v": "voltage_v",
            "coil_current_a": "current_a",
            "b_field_mT": "magnetic_b1",
        }
    )
    result = analyze_dataset(standardized, AnalysisWindow(cycle_start=0, cycle_count=3))
    assert abs(result["frequency_hz"] - 1.0) < 0.05
    assert result["electrical"]["|Z1|"] > 1.0
    assert "K_BI" in result["magnetic"]


def test_lambda_metrics_returns_waveforms() -> None:
    t = np.linspace(0.0, 1.0, 1000)
    current = np.sin(2 * np.pi * t)
    voltage = 2.0 * np.sin(2 * np.pi * t + 0.1)
    result = compute_lambda_metrics(t, voltage, current, rdc_ohm=0.2)
    assert result["label"] == "system-level inferred quantity"
    assert len(result["lambda"]) == len(t)
    assert len(result["differential_inductance_h"]) == len(t)


def test_load_dataframe_handles_metadata_preface_csv(tmp_path: Path) -> None:
    csv_content = "\n".join(
        [
            "Record Length,2048",
            "Sample Interval,0.0005",
            "Operator,Test",
            "Comment,Waveform export",
            "time_s,coil_voltage_v,coil_current_a,b_field_mT",
            "0.0,1.0,0.1,10.0",
            "0.5,1.2,0.2,10.5",
            "1.0,1.4,0.3,11.0",
        ]
    )
    path = tmp_path / "scope_export.csv"
    path.write_text(csv_content, encoding="utf-8")
    df = load_dataframe(path)
    assert list(df.columns) == ["time_s", "coil_voltage_v", "coil_current_a", "b_field_mT"]
    assert len(df) == 3


def test_standardize_dataset_supports_datetime_time_axis() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                "2026-04-07T16:45:31.5746008+09:00",
                "2026-04-07T16:45:31.5846008+09:00",
                "2026-04-07T16:45:31.5946008+09:00",
            ],
            "v": [1.0, 2.0, 3.0],
            "i": [0.1, 0.2, 0.3],
        }
    )
    mapping = ChannelMapping(
        time=ChannelConfig(column="timestamp", unit="datetime"),
        voltage=ChannelConfig(column="v"),
        current=ChannelConfig(column="i"),
    )
    standardized = standardize_dataset(df, mapping)
    assert infer_time_unit(df["timestamp"]) == "datetime"
    assert np.allclose(standardized["time_s"].to_numpy(), np.array([0.0, 0.01, 0.02]), atol=1e-9)


def test_infer_column_roles_does_not_map_timestamp_as_current() -> None:
    roles = infer_column_roles(
        ["Timestamp", "Current1_A", "Current2_A", "Voltage1", "HallBx", "HallBy"]
    )
    assert roles["time"][0] == "Timestamp"
    assert roles["current"][0] == "Current1_A"
    assert "Timestamp" not in roles["current"]
