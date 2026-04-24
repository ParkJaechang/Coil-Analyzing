from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.ui_raw_waveforms import (
    build_raw_waveform_label_lookup,
    build_raw_waveform_test_records,
    display_source_file_name,
    format_reference_test_label,
)


def _analysis_fixture() -> SimpleNamespace:
    normalized = pd.DataFrame(
        {
            "test_id": ["0123456789abcdef_internal_id"] * 3,
            "source_file": ["uploads/continuous/0123456789abcdef_tri_1hz_1.25cycle_10pp.csv"] * 3,
            "sheet_name": ["main"] * 3,
            "waveform_type": ["Triangle"] * 3,
            "freq_hz": [1.0] * 3,
            "current_pp_target_a": [10.0] * 3,
            "cycle_total_expected": [1.25] * 3,
            "time_s": [0.0, 0.5, 1.0],
            "daq_input_v": [0.0, 1.0, 0.0],
            "bz_mT": [0.0, 50.0, 0.0],
        }
    )
    corrected = normalized.copy()
    parsed = SimpleNamespace(
        source_file="uploads/continuous/0123456789abcdef_tri_1hz_1.25cycle_10pp.csv",
        sheet_name="main",
        metadata={"waveform": "Triangle", "frequency(Hz)": "1", "Target Current(A)": "10", "cycle": "1.25"},
        normalized_frame=normalized,
    )
    preprocess = SimpleNamespace(corrected_frame=corrected)
    return SimpleNamespace(parsed=parsed, preprocess=preprocess)


def test_raw_waveform_selector_label_uses_metadata_not_opaque_prefix() -> None:
    analysis = _analysis_fixture()
    lookup = {"0123456789abcdef_internal_id": analysis}

    records = build_raw_waveform_test_records(list(lookup), lookup)

    assert len(records) == 1
    assert records[0].label == "finite-cycle | Triangle | 1 Hz | 1.25 cycle | 10 App | ±5V | Gain 100% | continuous/tri_1hz_1.25cycle_10pp.csv"
    assert "0123456789abcdef" not in records[0].label
    assert format_reference_test_label("0123456789abcdef_internal_id", lookup) == records[0].label
    label_by_id, id_by_label = build_raw_waveform_label_lookup(list(lookup), lookup)
    assert label_by_id["0123456789abcdef_internal_id"] == records[0].label
    assert id_by_label[records[0].label] == "0123456789abcdef_internal_id"
    assert display_source_file_name("0123456789abcdef_sine_2hz_20app.csv") == "sine_2hz_20app.csv"


def test_raw_waveforms_ui_contract_is_audit_oriented() -> None:
    snapshot_source = (REPO_ROOT / "src" / "field_analysis" / "app_ui_snapshot.py").read_text(encoding="utf-8")
    raw_ui_source = (REPO_ROOT / "src" / "field_analysis" / "ui_raw_waveforms.py").read_text(encoding="utf-8")

    assert "render_raw_waveforms_tab(" in snapshot_source
    assert "transient_measurements=transient_measurements" in snapshot_source
    assert "raw_test_simple" not in snapshot_source
    assert "테스트 선택 (metadata label)" in raw_ui_source
    assert "Search metadata label / source file" in raw_ui_source
    assert "Waveform family" in raw_ui_source
    assert "Frequency (Hz)" in raw_ui_source
    assert "Cycle count" in raw_ui_source
    assert "Current/App" in raw_ui_source
    assert "Source type" in raw_ui_source
    assert "Selected Data Summary" in raw_ui_source
    assert "Internal ID" in raw_ui_source
    assert "Corrected/preprocessed" in raw_ui_source
    assert "Raw normalized parse" in raw_ui_source
    assert "비교 기준 테스트 (선택)" in snapshot_source
    assert "선택한 파형과 겹쳐 비교할 기준 테스트입니다" in snapshot_source
