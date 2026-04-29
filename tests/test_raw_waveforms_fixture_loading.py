from __future__ import annotations

import math
import re
import sys
from pathlib import Path


TEST_ROOT = Path(__file__).resolve().parent
SRC_ROOT = TEST_ROOT.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.ui_raw_waveforms import RawWaveformTestRecord, _format_raw_waveform_label
from field_analysis.ui_raw_waveforms_labels import infer_new_dataset_filename_metadata


def _record_for_fixture(file_name: str) -> RawWaveformTestRecord:
    inferred = infer_new_dataset_filename_metadata(file_name)
    cycle_count = inferred["cycle_count"]
    return RawWaveformTestRecord(
        test_id=file_name,
        label="",
        source_file=file_name,
        source_file_label=file_name,
        sheet_name="",
        waveform_type=str(inferred["waveform_type"]),
        freq_hz=float(inferred["freq_hz"]),
        cycle_count=float(cycle_count) if cycle_count is not None else float("nan"),
        target_current_a=float("nan"),
        source_type=str(inferred["source_type"]),
        sample_count=1000,
        duration_s=1.0,
        sampling_rate_hz=1000.0,
    )


def test_raw_waveform_labels_are_readable_for_fixture_files() -> None:
    cases = [
        ("continuous_sine_1Hz.csv", ["continuous", "Sine", "1 Hz", "continuous_sine_1Hz.csv"]),
        ("continuous_triangle_5Hz.csv", ["continuous", "Triangle", "5 Hz", "continuous_triangle_5Hz.csv"]),
        (
            "finite_sine_1Hz_1.25cycle.csv",
            ["finite-cycle", "Sine", "1 Hz", "1.25 cycle", "finite_sine_1Hz_1.25cycle.csv"],
        ),
        (
            "finite_triangle_1Hz_1.75cycle.csv",
            ["finite-cycle", "Triangle", "1 Hz", "1.75 cycle", "finite_triangle_1Hz_1.75cycle.csv"],
        ),
    ]

    for file_name, expected_parts in cases:
        label = _format_raw_waveform_label(_record_for_fixture(file_name))
        assert not re.match(r"^[0-9a-f]{12,}_", label)
        assert "±5V" in label
        assert "Gain 100%" in label
        assert "App" not in label
        for expected in expected_parts:
            assert expected in label


def test_fixture_filename_metadata_supports_continuous_and_finite_names() -> None:
    continuous = infer_new_dataset_filename_metadata("continuous_triangle_5Hz.csv")
    finite = infer_new_dataset_filename_metadata("finite_sine_1Hz_1.25cycle.csv")

    assert continuous == {
        "source_type": "continuous",
        "waveform_type": "triangle",
        "freq_hz": 5.0,
        "cycle_count": None,
    }
    assert finite["source_type"] == "finite-cycle"
    assert finite["waveform_type"] == "sine"
    assert finite["freq_hz"] == 1.0
    assert math.isclose(float(finite["cycle_count"]), 1.25)
