from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.field_waveform_diagnostics import build_field_waveform_diagnostics


def _continuous_summary(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _frame(
    *,
    waveform_type: str,
    freq_hz: float,
    include_voltage: bool = True,
    include_field: bool = True,
) -> pd.DataFrame:
    data: dict[str, list[object]] = {
        "time_s": [0.0, 0.5, 1.0],
        "waveform_type": [waveform_type, waveform_type, waveform_type],
        "freq_hz": [freq_hz, freq_hz, freq_hz],
    }
    if include_voltage:
        data["daq_input_v"] = [0.0, 1.0, 0.0]
    if include_field:
        data["bz_mT"] = [0.0, 2.0, 0.0]
    return pd.DataFrame(data)


def test_diagnostics_marks_ok_support_and_shape_comparison() -> None:
    per_test_summary = _continuous_summary(
        {
            "test_id": "t1",
            "waveform_type": "sine",
            "freq_hz": 10.0,
            "daq_input_v_pp_mean": 4.0,
            "achieved_bz_mT_pp_mean": 8.0,
        },
        {
            "test_id": "t2",
            "waveform_type": "sine",
            "freq_hz": 10.0,
            "daq_input_v_pp_mean": 5.0,
            "achieved_bz_mT_pp_mean": 10.0,
        },
    )
    diagnostics = build_field_waveform_diagnostics(
        per_test_summary=per_test_summary,
        main_field_axis="bz_mT",
        continuous_frames_by_test_id={
            "t1": _frame(waveform_type="sine", freq_hz=10.0),
            "t2": _frame(waveform_type="sine", freq_hz=10.0),
        },
        transient_frames=[_frame(waveform_type="sine", freq_hz=10.0)],
    )

    continuous_support = diagnostics["continuous_support"]
    finite_support = diagnostics["finite_support"]
    summary = diagnostics["summary"]

    assert len(continuous_support) == 1
    assert continuous_support.loc[0, "risk_level"] == "OK"
    assert bool(continuous_support.loc[0, "shape_comparison_possible"]) is True
    assert bool(finite_support.loc[0, "has_support"]) is True
    assert summary["continuous_ok_combo_count"] == 1
    assert summary["shape_comparison_combo_count"] == 1


def test_diagnostics_work_without_current_columns() -> None:
    per_test_summary = _continuous_summary(
        {
            "test_id": "t1",
            "waveform_type": "triangle",
            "freq_hz": 5.0,
            "daq_input_v_pp_mean": 3.0,
            "achieved_bz_mT_pp_mean": 6.0,
        }
    )

    diagnostics = build_field_waveform_diagnostics(
        per_test_summary=per_test_summary,
        main_field_axis="bz_mT",
        continuous_frames_by_test_id={"t1": _frame(waveform_type="triangle", freq_hz=5.0)},
        transient_frames=[],
    )

    assert diagnostics["summary"]["continuous_test_count"] == 1
    assert diagnostics["summary"]["main_field_axis_available"] is True
    assert diagnostics["summary"]["voltage_input_available"] is True
    assert diagnostics["continuous_support"].loc[0, "risk_level"] == "Weak"


def test_diagnostics_distinguish_voltage_and_field_missing_states() -> None:
    per_test_summary = _continuous_summary(
        {
            "test_id": "voltage_missing",
            "waveform_type": "sine",
            "freq_hz": 10.0,
            "daq_input_v_pp_mean": 4.0,
            "achieved_bz_mT_pp_mean": 9.0,
        },
        {
            "test_id": "field_missing",
            "waveform_type": "square",
            "freq_hz": 20.0,
            "daq_input_v_pp_mean": 4.0,
            "achieved_bz_mT_pp_mean": 11.0,
        },
    )

    diagnostics = build_field_waveform_diagnostics(
        per_test_summary=per_test_summary,
        main_field_axis="bz_mT",
        continuous_frames_by_test_id={
            "voltage_missing": _frame(waveform_type="sine", freq_hz=10.0, include_voltage=False, include_field=True),
            "field_missing": _frame(waveform_type="square", freq_hz=20.0, include_voltage=True, include_field=False),
        },
        transient_frames=[],
    )

    continuous_support = diagnostics["continuous_support"].set_index(["waveform_type", "freq_hz"])
    finite_support = diagnostics["finite_support"]

    assert continuous_support.loc[("sine", 10.0), "risk_level"] == "Voltage Missing"
    assert continuous_support.loc[("square", 20.0), "risk_level"] == "Field Missing"
    assert set(finite_support["risk_level"]) == {"Missing"}
