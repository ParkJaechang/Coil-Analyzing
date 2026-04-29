from __future__ import annotations

import sys
from pathlib import Path


TEST_ROOT = Path(__file__).resolve().parent
SRC_ROOT = TEST_ROOT.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.parser import infer_dataset_filename_metadata, parse_measurement_file
from field_analysis.schema_config import build_default_schema


def _sample_csv_bytes() -> bytes:
    return (
        "Time,DAQInput_V,Current1_A,Current2_A,Bz_mT\n"
        "2026-01-01T00:00:00,0.0,1.0,-1.0,0.0\n"
        "2026-01-01T00:00:00.100,1.0,1.1,-1.1,0.5\n"
        "2026-01-01T00:00:00.200,0.0,1.0,-1.0,0.0\n"
    ).encode("utf-8")


def _new_lut_csv_bytes() -> bytes:
    return (
        "# Frequency(Hz)=1.000\n"
        "# Amplitude(V)=5.000\n"
        "# Cycles=10.000\n"
        "# PreDelay(s)=1.000\n"
        "# PostDelay(s)=1.000\n"
        "Row,TimeMs,HallBx,HallBy,HallBz,Current1_A,Current2_A,Voltage1_V,Voltage2_V\n"
        "0,0.000,0.1,0.2,1.0,0.5,-0.5,0.0,0.0\n"
        "1,2.500,0.2,0.3,2.0,0.6,-0.6,1.0,-1.0\n"
        "2,5.000,0.3,0.4,1.0,0.5,-0.5,0.0,0.0\n"
    ).encode("utf-8")


def test_infer_continuous_filename_metadata() -> None:
    inferred = infer_dataset_filename_metadata("continuous_triangle_0.25Hz.csv")

    assert inferred["source_type"] == "continuous"
    assert inferred["waveform"] == "triangle"
    assert inferred["freq_hz"] == 0.25
    assert inferred["cycle"] is None
    assert inferred["daq_amplitude_v"] == 5.0
    assert inferred["daq_pp_v"] == 10.0
    assert inferred["dcamp_gain_percent"] == 100.0
    assert inferred["Target Current(A)"] is None
    assert inferred["filename_metadata_inferred"] is True


def test_infer_finite_filename_metadata_with_decimal_variants() -> None:
    inferred = infer_dataset_filename_metadata("finite_sine_1Hz_1.25cycle.csv")
    inferred_p = infer_dataset_filename_metadata("finite_triangle_2Hz_1p75cycle.csv")
    inferred_underscore = infer_dataset_filename_metadata("finite_triangle_2Hz_1_75cycle.csv")

    assert inferred["source_type"] == "finite_cycle"
    assert inferred["waveform"] == "sine"
    assert inferred["freq_hz"] == 1.0
    assert inferred["cycle"] == 1.25
    assert inferred_p["cycle"] == 1.75
    assert inferred_underscore["cycle"] == 1.75


def test_parse_measurement_file_applies_filename_defaults_without_current_token() -> None:
    schema = build_default_schema()
    parsed = parse_measurement_file(
        file_name="continuous_sine_1Hz.csv",
        file_bytes=_sample_csv_bytes(),
        schema=schema,
    )[0]

    assert parsed.metadata["source_type"] == "continuous"
    assert parsed.metadata["waveform"] == "sine"
    assert parsed.metadata["freq_hz"] == 1.0
    assert parsed.metadata["Target Current(A)"] is None
    assert parsed.metadata["daq_amplitude_v"] == 5.0
    assert parsed.metadata["daq_pp_v"] == 10.0
    assert parsed.metadata["dcamp_gain_percent"] == 100.0
    assert parsed.metadata["filename_metadata_inferred"] is True

    normalized = parsed.normalized_frame
    assert normalized["source_type"].iloc[0] == "continuous"
    assert float(normalized["freq_hz"].iloc[0]) == 1.0
    assert float(normalized["daq_amplitude_v"].iloc[0]) == 5.0
    assert float(normalized["daq_pp_v"].iloc[0]) == 10.0
    assert float(normalized["dcamp_gain_percent"].iloc[0]) == 100.0
    assert bool(normalized["filename_metadata_inferred"].iloc[0]) is True


def test_parse_measurement_file_reads_finite_175_cycle_metadata() -> None:
    schema = build_default_schema()
    parsed = parse_measurement_file(
        file_name="finite_triangle_2Hz_1.75cycle.csv",
        file_bytes=_sample_csv_bytes(),
        schema=schema,
    )[0]

    assert parsed.metadata["source_type"] == "finite_cycle"
    assert parsed.metadata["waveform"] == "triangle"
    assert parsed.metadata["freq_hz"] == 2.0
    assert parsed.metadata["cycle"] == 1.75
    assert parsed.metadata["Target Current(A)"] is None

    normalized = parsed.normalized_frame
    assert normalized["source_type"].iloc[0] == "finite_cycle"
    assert float(normalized["freq_hz"].iloc[0]) == 2.0
    assert float(normalized["cycle_count"].iloc[0]) == 1.75
    assert float(normalized["amp_gain_setting"].iloc[0]) == 100.0


def test_new_lut_timems_and_hall_columns_are_mapped_without_temperature_false_match() -> None:
    schema = build_default_schema()
    parsed = parse_measurement_file(
        file_name="continuous_sine_1Hz.csv",
        file_bytes=_new_lut_csv_bytes(),
        schema=schema,
    )[0]

    mapping = parsed.mapping
    normalized = parsed.normalized_frame

    assert mapping["timestamp"] == "TimeMs"
    assert mapping["daq_input_v"] == "Voltage1_V"
    assert mapping["daq_input_v_secondary"] == "Voltage2_V"
    assert mapping["coil1_current_a"] == "Current1_A"
    assert mapping["coil2_current_a"] == "Current2_A"
    assert mapping["bx_mT"] == "HallBx"
    assert mapping["by_mT"] == "HallBy"
    assert mapping["bz_mT"] == "HallBz"
    assert mapping["temperature_t1_c"] is None
    assert mapping["temperature_t2_c"] is None
    assert normalized["detected_format"].iloc[0] == "new_lut_csv"
    assert normalized["timebase_source"].iloc[0] == "explicit_time_column"
    assert normalized["time_unit"].iloc[0] == "milliseconds"
    assert float(normalized["time_s"].iloc[-1]) == 0.005
    assert float(normalized["sample_rate_hz"].iloc[0]) == 400.0
    assert normalized["parse_quality_flags"].iloc[0] == ""
    assert float(normalized["bx_mT"].iloc[0]) == 0.1
    assert float(normalized["bz_mT"].iloc[0]) == 1.0
