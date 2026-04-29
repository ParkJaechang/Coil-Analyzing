from __future__ import annotations

import json
from pathlib import Path


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "field_analysis"
MANIFEST_PATH = FIXTURE_ROOT / "manifest.json"
EXPECTED_FLAGS_PATH = FIXTURE_ROOT / "expected" / "expected_quality_flags.json"


def test_quality_case_expected_flags_are_declared() -> None:
    expected_flags = json.loads(EXPECTED_FLAGS_PATH.read_text(encoding="utf-8"))

    assert expected_flags == {
        "quality_cases/finite_nonzero_start_example.csv": ["daq_nonzero_start"],
        "quality_cases/finite_spike_example.csv": ["field_spike"],
        "quality_cases/continuous_current_offset_example.csv": ["current_offset"],
        "quality_cases/finite_truncated_example.csv": ["truncated_active_window"],
    }


def test_quality_case_manifest_entries_are_not_marked_usable() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    quality_entries = [
        entry for entry in manifest["entries"] if entry["path"].startswith("quality_cases/")
    ]

    assert quality_entries
    for entry in quality_entries:
        assert entry["expected_quality"] == "retest_required"
        assert entry["fixture_role"] == "quality_flag_regression"
        assert entry["synthetic"] is True
        assert entry["expected_flags"]
        assert (FIXTURE_ROOT / entry["path"]).is_file()
