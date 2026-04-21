from __future__ import annotations

import sys
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.ui_recommendation_exports import (
    build_artifact_bundle_zip_bytes,
    build_recommendation_artifact_map,
    build_recommendation_export_payloads,
    build_recommendation_summary_csv_bytes,
    build_recommendation_summary_row,
    payload_to_json_text,
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
        "support_amp_gain_pct": 100.0,
        "available_amp_gain_pct": 100.0,
        "required_amp_gain_pct": 75.0,
        "amp_output_pp_at_required": 42.0,
        "amp_max_output_pk_v": 180.0,
        "command_waveform": command_waveform,
        "frequency_support_table": pd.DataFrame({"freq_hz": [10.0], "support_point_count": [2]}),
    }


def test_payload_to_json_text_serializes_primary_and_debug_payloads() -> None:
    payloads = build_recommendation_export_payloads(_base_recommendation(finite_cycle_mode=False))

    primary_json = payload_to_json_text(payloads["primary"])
    debug_json = payload_to_json_text(payloads["debug"])

    assert '"output_type": "continuous_recommended_voltage_waveform"' in primary_json
    assert '"estimated_current_pp"' not in primary_json
    assert '"estimated_current_pp": 2.5' in debug_json
    assert '"amp_max_output_pk_v": 180.0' in debug_json


def test_export_payload_builder_uses_finite_cycle_primary_type() -> None:
    payloads = build_recommendation_export_payloads(_base_recommendation(finite_cycle_mode=True))

    assert payloads["primary"]["output_type"] == "finite_cycle_stop_waveform"
    assert payloads["primary"]["command_waveform"][0]["target_cycle_count"] == 3.0
    assert payloads["primary"]["command_waveform"][0]["preview_tail_cycles"] == 0.5


def test_artifact_bundle_zip_contains_expected_files() -> None:
    artifacts = build_recommendation_artifact_map(
        _base_recommendation(finite_cycle_mode=False),
        file_stem="continuous_recommended_voltage_waveform_sine_10Hz_test",
    )
    bundle_bytes = build_artifact_bundle_zip_bytes(artifacts)

    with zipfile.ZipFile(BytesIO(bundle_bytes)) as archive:
        names = sorted(archive.namelist())

    assert names == [
        "continuous_recommended_voltage_waveform_sine_10Hz_test_debug.json",
        "continuous_recommended_voltage_waveform_sine_10Hz_test_primary.json",
        "continuous_recommended_voltage_waveform_sine_10Hz_test_summary.csv",
        "continuous_recommended_voltage_waveform_sine_10Hz_test_waveform.csv",
    ]


def test_artifact_map_keeps_continuous_and_finite_file_stems_distinct() -> None:
    continuous = build_recommendation_artifact_map(
        _base_recommendation(finite_cycle_mode=False),
        file_stem="continuous_recommended_voltage_waveform_case",
    )
    finite = build_recommendation_artifact_map(
        _base_recommendation(finite_cycle_mode=True),
        file_stem="finite_cycle_stop_waveform_case",
    )

    assert sorted(continuous) == [
        "continuous_recommended_voltage_waveform_case_debug.json",
        "continuous_recommended_voltage_waveform_case_primary.json",
        "continuous_recommended_voltage_waveform_case_summary.csv",
        "continuous_recommended_voltage_waveform_case_waveform.csv",
    ]
    assert sorted(finite) == [
        "finite_cycle_stop_waveform_case_debug.json",
        "finite_cycle_stop_waveform_case_primary.json",
        "finite_cycle_stop_waveform_case_summary.csv",
        "finite_cycle_stop_waveform_case_waveform.csv",
    ]


def test_summary_row_includes_compact_field_first_columns_only() -> None:
    row = build_recommendation_summary_row(_base_recommendation(finite_cycle_mode=True))

    assert row == {
        "waveform_type": "sine",
        "freq_hz": 10.0,
        "finite_cycle_mode": True,
        "target_metric": "achieved_bz_mT_pp_mean",
        "target_value": 12.0,
        "used_target_value": 11.5,
        "recommendation_scope": None,
        "recommendation_mode": None,
        "frequency_mode": None,
        "support_point_count": None,
        "frequency_support_count": None,
        "available_freq_min": None,
        "available_freq_max": None,
        "within_daq_limit": None,
        "within_hardware_limits": None,
        "estimated_voltage_pp": None,
        "limited_voltage_pp": None,
        "template_test_id": None,
    }
    assert "estimated_current_pp" not in row
    assert "support_amp_gain_pct" not in row
    assert "amp_max_output_pk_v" not in row


def test_summary_csv_exports_expected_columns_and_values() -> None:
    recommendation = _base_recommendation(finite_cycle_mode=False) | {
        "recommendation_scope": "continuous",
        "recommendation_mode": "frequency_interpolated",
        "frequency_mode": "frequency_interpolated",
        "support_point_count": 3,
        "frequency_support_count": 2,
        "available_freq_min": 8.0,
        "available_freq_max": 12.0,
        "within_daq_limit": True,
        "within_hardware_limits": False,
        "estimated_voltage_pp": 4.5,
        "limited_voltage_pp": 4.0,
        "template_test_id": "template_01",
    }
    csv_text = build_recommendation_summary_csv_bytes(recommendation).decode("utf-8-sig")

    assert (
        "waveform_type,freq_hz,finite_cycle_mode,target_metric,target_value,used_target_value,"
        "recommendation_scope,recommendation_mode,frequency_mode,support_point_count,"
        "frequency_support_count,available_freq_min,available_freq_max,within_daq_limit,"
        "within_hardware_limits,estimated_voltage_pp,limited_voltage_pp,template_test_id"
    ) in csv_text
    assert "sine,10.0,False,achieved_bz_mT_pp_mean,12.0,11.5,continuous,frequency_interpolated,frequency_interpolated,3,2,8.0,12.0,True,False,4.5,4.0,template_01" in csv_text
