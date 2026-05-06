from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.finite_actual_drive import (
    build_actual_drive_review_case,
    parse_actual_drive_filename,
    read_actual_drive_result,
)


def _write_actual_drive_csv(path: Path) -> None:
    rows = []
    time_ms = np.linspace(0.0, 1400.0, 101)
    voltage = np.zeros_like(time_ms)
    active = (time_ms >= 200.0) & (time_ms <= 1200.0)
    voltage[active] = 2.0 * np.sin(np.pi * (time_ms[active] - 200.0) / 1000.0)
    hall = 1.5 + 40.0 * np.sin(np.pi * np.clip((time_ms - 200.0) / 1000.0, 0.0, 1.0))
    for index, (t_ms, v, h) in enumerate(zip(time_ms, voltage, hall, strict=False)):
        rows.append(f"{index},{t_ms:.3f},0.0,0.0,{h:.6f},0.1,0.0,{v:.6f},0.0")
    preamble = [
        "# Date,2026-05-06 16:00:02",
        "# Frequency(Hz),0.000",
        "# Amplitude(V),0.000",
        "# Cycles,0.000",
        "# Repeat,1.000",
        "# PreDelay(s),1.000",
        "# PostDelay(s),1.000",
        "# HallSamples,21286",
        "# CurrentSamples,1409919",
        "# CommonRange(ms),0.00~2819.84 (span 2819.84)",
        "# Rows,5000, GridStep(ms),0.564",
        "# AutoSyncHallLag,applied 70.00ms (r=0.815)",
        "#",
        "Row,TimeMs,HallBx,HallBy,HallBz,Current1_A,Current2_A,Voltage1_V,Voltage2_V",
    ]
    path.write_text("\n".join([*preamble, *rows]), encoding="utf-8")


def test_actual_drive_filename_and_preamble_parse(tmp_path: Path) -> None:
    path = tmp_path / "finite_recommended_voltage_lut_sine_1.25Hz_1.25cycle_result.csv"
    _write_actual_drive_csv(path)

    parsed_name = parse_actual_drive_filename(path)
    record = read_actual_drive_result(path)

    assert parsed_name == {"waveform_type": "sine", "freq_hz": 1.25, "cycle_count": 1.25}
    assert record.metadata["pre_delay_s"] == 1.0
    assert record.metadata["post_delay_s"] == 1.0
    assert record.metadata["auto_sync_hall_lag_ms"] == 70.0
    assert record.metadata["time_unit"] == "ms"
    assert record.metadata["voltage_unit"] == "V"
    assert record.metadata["field_unit"] == "mT_inferred_from_HallBz"
    assert {"time_s_abs", "first_voltage_v", "measured_field_raw", "current_a"}.issubset(record.frame.columns)
    assert np.isclose(float(record.frame["time_s_abs"].iloc[-1]), 1.4)


def test_actual_drive_review_metrics_and_alignment(tmp_path: Path) -> None:
    path = tmp_path / "finite_recommended_voltage_lut_sine_1.25Hz_1.25cycle_result.csv"
    _write_actual_drive_csv(path)
    record = read_actual_drive_result(path)

    review, metadata = build_actual_drive_review_case(record)

    assert np.isclose(metadata["target_active_end_s"], 1.0)
    assert np.isclose(metadata["command_start_s"], 0.2, atol=0.02)
    assert metadata["alignment_anchor"] == "Voltage1_V_command_nonzero_start"
    assert "Voltage1_V" not in review.columns
    assert {
        "time_s",
        "first_voltage_v",
        "physical_target_output_mT",
        "measured_field_mT",
        "measured_residual_mT",
        "current_a",
    }.issubset(review.columns)
    assert np.allclose(
        review["measured_residual_mT"],
        review["physical_target_output_mT"] - review["measured_field_mT"],
        atol=1e-12,
    )
    assert np.isfinite(metadata["measured_active_nrmse"])
    assert np.isfinite(metadata["measured_shape_corr"])
    assert np.isfinite(metadata["measured_peak_error_mT"])
    assert metadata["target_pp_mT"] == 100.0
    assert metadata["correction_delta_generated"] is False
    assert metadata["second_voltage_generated"] is False
    assert metadata["second_lut_generated"] is False
    assert metadata["continuous_touched"] is False
