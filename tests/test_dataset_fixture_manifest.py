from __future__ import annotations

import json
from pathlib import Path


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "field_analysis"
MANIFEST_PATH = FIXTURE_ROOT / "manifest.json"


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_fixture_manifest_exists_and_uses_repo_relative_paths() -> None:
    manifest = _load_manifest()
    entries = manifest["entries"]

    assert MANIFEST_PATH.exists()
    assert entries
    for entry in entries:
        fixture_path = Path(entry["path"])
        assert not fixture_path.is_absolute()
        assert ":" not in entry["path"]
        assert ".." not in fixture_path.parts
        assert (FIXTURE_ROOT / fixture_path).is_file()


def test_fixture_manifest_required_metadata_and_policy() -> None:
    manifest = _load_manifest()
    data_policy = manifest["data_policy"]

    assert data_policy["full_lut_copied"] is False
    assert data_policy["outputs_cache_copied"] is False
    assert data_policy["source_original_path_reference_only"] is True
    assert data_policy["fixtures_downsampled"] is True

    for entry in manifest["entries"]:
        assert entry["source_type"] in {"continuous", "finite-cycle"}
        assert entry["waveform_type"] in {"sine", "triangle"}
        assert isinstance(entry["freq_hz"], float | int)
        assert entry["daq_amplitude_v"] == 5.0
        assert entry["daq_pp_v"] == 10.0
        assert entry["dcamp_gain_percent"] == 100.0
        assert entry["expected_quality"] in {"usable", "retest_required"}
        assert isinstance(entry["expected_flags"], list)
        assert entry["fixture_role"] in {"parser_ui_regression", "quality_flag_regression"}
        assert entry["source_original_path"]
        if entry["source_type"] == "finite-cycle":
            assert entry["cycle_count"] is not None


def test_fixture_manifest_contains_expected_usable_and_quality_groups() -> None:
    entries = _load_manifest()["entries"]
    usable = [entry for entry in entries if entry["expected_quality"] == "usable"]
    quality_cases = [entry for entry in entries if entry["expected_quality"] == "retest_required"]

    continuous = {entry["path"] for entry in usable if entry["source_type"] == "continuous"}
    finite = {entry["path"] for entry in usable if entry["source_type"] == "finite-cycle"}
    quality = {entry["path"] for entry in quality_cases}

    assert continuous == {
        "continuous/continuous_sine_1Hz.csv",
        "continuous/continuous_triangle_1Hz.csv",
        "continuous/continuous_sine_5Hz.csv",
        "continuous/continuous_triangle_5Hz.csv",
    }
    assert finite == {
        "finite/finite_sine_1Hz_1cycle.csv",
        "finite/finite_sine_1Hz_1.25cycle.csv",
        "finite/finite_sine_1Hz_1.5cycle.csv",
        "finite/finite_sine_1Hz_1.75cycle.csv",
        "finite/finite_triangle_1Hz_1cycle.csv",
        "finite/finite_triangle_1Hz_1.25cycle.csv",
        "finite/finite_triangle_1Hz_1.5cycle.csv",
        "finite/finite_triangle_1Hz_1.75cycle.csv",
    }
    assert quality == {
        "quality_cases/finite_nonzero_start_example.csv",
        "quality_cases/finite_spike_example.csv",
        "quality_cases/continuous_current_offset_example.csv",
        "quality_cases/finite_truncated_example.csv",
    }


def test_fixture_dataset_stays_small() -> None:
    total_size = sum(path.stat().st_size for path in FIXTURE_ROOT.rglob("*") if path.is_file())

    assert total_size < 5 * 1024 * 1024
