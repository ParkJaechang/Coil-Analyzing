from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.ui_field_waveform_diagnostics_exports import (
    build_field_waveform_diagnostics_export_payloads,
    dataframe_to_csv_bytes,
    payload_to_json_bytes,
)


def _diagnostics_fixture() -> dict[str, object]:
    return {
        "summary": {
            "continuous_test_count": 2,
            "finite_test_count": 1,
            "continuous_ok_combo_count": 1,
        },
        "notes": [
            "Primary diagnostics are field-first.",
            "Current remains debug-only.",
        ],
        "target_metric_candidates": pd.DataFrame(
            [{"metric": "achieved_bz_mT_pp_mean", "available": True, "non_null_test_count": 2}]
        ),
        "waveform_counts": pd.DataFrame([{"waveform_type": "sine", "continuous_test_count": 2}]),
        "frequency_counts": pd.DataFrame([{"freq_hz": 10.0, "continuous_test_count": 2}]),
        "continuous_support": pd.DataFrame([{"waveform_type": "sine", "freq_hz": 10.0, "risk_level": "OK"}]),
        "finite_support": pd.DataFrame([{"waveform_type": "sine", "freq_hz": 10.0, "risk_level": "Weak"}]),
        "continuous_test_details": pd.DataFrame([{"test_id": "t1", "has_main_field_axis": True}]),
        "transient_test_details": pd.DataFrame([{"test_id": "ft1", "has_main_field_axis": True}]),
    }


def test_build_export_payloads_keeps_summary_and_table_exports() -> None:
    payloads = build_field_waveform_diagnostics_export_payloads(_diagnostics_fixture())

    assert payloads["summary_json"]["summary"]["continuous_test_count"] == 2
    assert "waveform_counts" in payloads["summary_json"]["available_tables"]
    assert list(payloads["tables"]["continuous_support"].columns) == ["waveform_type", "freq_hz", "risk_level"]
    assert "transient_test_details" in payloads["tables"]


def test_dataframe_to_csv_bytes_exports_header_and_rows() -> None:
    csv_bytes = dataframe_to_csv_bytes(pd.DataFrame([{"waveform_type": "sine", "risk_level": "OK"}]))
    csv_text = csv_bytes.decode("utf-8-sig")

    assert "waveform_type,risk_level" in csv_text
    assert "sine,OK" in csv_text


def test_payload_to_json_bytes_exports_readable_json() -> None:
    json_bytes = payload_to_json_bytes({"summary": {"continuous_test_count": 2}, "notes": ["ok"]})
    payload = json.loads(json_bytes.decode("utf-8"))

    assert payload["summary"]["continuous_test_count"] == 2
    assert payload["notes"] == ["ok"]
