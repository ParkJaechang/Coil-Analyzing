from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.recommendation_output_contract import (
    build_continuous_recommendation_payload,
    build_finite_cycle_recommendation_payload,
    build_recommendation_debug_payload,
)


def _base_recommendation(*, finite_cycle_mode: bool) -> dict[str, object]:
    command_waveform = pd.DataFrame(
        {
            "time_s": [0.0, 0.1, 0.2],
            "recommended_voltage_v": [0.0, 2.0, 0.0],
            "limited_voltage_v": [0.0, 1.8, 0.0],
            "is_active_target": [True, True, False],
            "target_cycle_count": [3.0, 3.0, 3.0],
            "preview_tail_cycles": [0.5, 0.5, 0.5],
            "required_amp_gain_pct": [75.0, 75.0, 75.0],
            "available_amp_gain_pct": [100.0, 100.0, 100.0],
        }
    )
    return {
        "waveform_type": "sine",
        "freq_hz": 10.0,
        "target_metric": "achieved_bz_mT_pp_mean",
        "requested_target_value": 12.0,
        "used_target_value": 11.5,
        "finite_cycle_mode": finite_cycle_mode,
        "estimated_current_pp": 2.5,
        "estimated_bz_pp": 11.5,
        "support_amp_gain_pct": 100.0,
        "available_amp_gain_pct": 100.0,
        "required_amp_gain_pct": 75.0,
        "amp_output_pp_at_required": 42.0,
        "amp_output_pk_at_required": 21.0,
        "max_daq_voltage_pp": 20.0,
        "amp_gain_at_100_pct": 20.0,
        "amp_gain_limit_pct": 100.0,
        "amp_max_output_pk_v": 180.0,
        "within_hardware_limits": True,
        "command_waveform": command_waveform,
        "frequency_support_table": pd.DataFrame({"freq_hz": [10.0], "support_point_count": [2]}),
    }


def test_continuous_payload_omits_current_and_hardware_reference_values() -> None:
    payload = build_continuous_recommendation_payload(_base_recommendation(finite_cycle_mode=False))

    assert payload["output_type"] == "continuous_recommended_voltage_waveform"
    assert payload["finite_cycle_mode"] is False
    assert payload["target_metric"] == "achieved_bz_mT_pp_mean"
    assert payload["target_value"] == 12.0
    assert payload["used_target_value"] == 11.5
    assert list(payload["command_waveform"][0]) == [
        "time_s",
        "recommended_voltage_v",
        "limited_voltage_v",
        "is_active_target",
    ]
    assert "estimated_current_pp" not in payload
    assert "required_amp_gain_pct" not in payload
    assert "support_amp_gain_pct" not in payload


def test_finite_cycle_payload_keeps_stop_waveform_columns() -> None:
    payload = build_finite_cycle_recommendation_payload(_base_recommendation(finite_cycle_mode=True))

    assert payload["output_type"] == "finite_cycle_stop_waveform"
    assert payload["finite_cycle_mode"] is True
    assert list(payload["command_waveform"][0]) == [
        "time_s",
        "recommended_voltage_v",
        "limited_voltage_v",
        "is_active_target",
        "target_cycle_count",
        "preview_tail_cycles",
    ]
    assert payload["command_waveform"][0]["target_cycle_count"] == 3.0
    assert payload["command_waveform"][0]["preview_tail_cycles"] == 0.5


def test_debug_payload_keeps_current_gain_and_hardware_reference_values() -> None:
    payload = build_recommendation_debug_payload(_base_recommendation(finite_cycle_mode=True))
    metrics = payload["reference_metrics"]

    assert payload["finite_cycle_mode"] is True
    assert metrics["estimated_current_pp"] == 2.5
    assert metrics["support_amp_gain_pct"] == 100.0
    assert metrics["available_amp_gain_pct"] == 100.0
    assert metrics["required_amp_gain_pct"] == 75.0
    assert metrics["amp_output_pp_at_required"] == 42.0
    assert metrics["amp_max_output_pk_v"] == 180.0
    assert payload["reference_tables"]["frequency_support_table"][0]["support_point_count"] == 2
