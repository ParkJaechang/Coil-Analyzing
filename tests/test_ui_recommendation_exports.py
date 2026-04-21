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
        "continuous_recommended_voltage_waveform_case_waveform.csv",
    ]
    assert sorted(finite) == [
        "finite_cycle_stop_waveform_case_debug.json",
        "finite_cycle_stop_waveform_case_primary.json",
        "finite_cycle_stop_waveform_case_waveform.csv",
    ]
