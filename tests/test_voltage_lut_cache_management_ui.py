from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.ui_voltage_lut_cache import add_lut_cache_bytes
from field_analysis.ui_voltage_lut_cache import build_lut_cache_records
from field_analysis.ui_voltage_lut_cache import build_lut_cache_selection_options
from field_analysis.ui_voltage_lut_cache import delete_lut_cache_item
from field_analysis.ui_voltage_lut_cache import edit_lut_cache_metadata
from field_analysis.ui_voltage_lut_cache import fallback_lut_cache_selection


def _csv_bytes(voltage: float = 1.0) -> bytes:
    text = f"sample_index,time_s,voltage_v\n0,0.0,{voltage}\n1,0.1,{voltage * 2}\n"
    return text.encode("utf-8")


def _contains_dataframe(value: object) -> bool:
    if isinstance(value, pd.DataFrame):
        return True
    if isinstance(value, dict):
        return any(_contains_dataframe(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_dataframe(item) for item in value)
    return False


def test_lut_cache_uses_stable_internal_id_separate_from_display_name() -> None:
    cache_state: dict[str, dict[str, object]] = {}
    cache_id = add_lut_cache_bytes(
        cache_state,
        "finite_recommended_voltage_lut_sine_1Hz_1cycle.csv",
        _csv_bytes(1.0),
        created_time="2026-05-08T00:00:00",
        display_name="Initial label",
    )

    assert cache_id.startswith("lut:")
    assert cache_state[cache_id]["display_name"] == "Initial label"

    assert edit_lut_cache_metadata(cache_state, cache_id, display_name="Reviewed label", user_note="operator ok")
    records = build_lut_cache_records(cache_state)

    assert records[0].id == cache_id
    assert records[0].display_name == "Reviewed label"
    assert records[0].user_note == "operator ok"
    assert records[0].metadata["original_filename"] == "finite_recommended_voltage_lut_sine_1Hz_1cycle.csv"


def test_lut_cache_metadata_is_scalar_and_dataframe_stays_out_of_session_state() -> None:
    cache_state: dict[str, dict[str, object]] = {}
    first_id = add_lut_cache_bytes(cache_state, "first.csv", _csv_bytes(1.0), created_time="2026-05-08T00:00:00")
    second_id = add_lut_cache_bytes(cache_state, "second.csv", _csv_bytes(2.0), created_time="2026-05-08T00:01:00")

    records = build_lut_cache_records(cache_state)
    options, records_by_id, labels_by_id = build_lut_cache_selection_options(records)

    assert options == [first_id, second_id]
    assert all(isinstance(option, str) for option in options)
    assert all(isinstance(label, str) for label in labels_by_id.values())
    assert records_by_id[first_id].metadata["sample_count"] == 2
    assert records_by_id[first_id].metadata["duration_s"] == 0.1
    assert records_by_id[first_id].metadata["time_start_s"] == 0.0
    assert records_by_id[second_id].metadata["voltage_max_v"] == 4.0
    assert _contains_dataframe(cache_state) is False


def test_lut_cache_delete_removes_item_and_selection_falls_back() -> None:
    cache_state: dict[str, dict[str, object]] = {}
    first_id = add_lut_cache_bytes(cache_state, "first.csv", _csv_bytes(1.0), created_time="2026-05-08T00:00:00")
    second_id = add_lut_cache_bytes(cache_state, "second.csv", _csv_bytes(2.0), created_time="2026-05-08T00:01:00")

    assert delete_lut_cache_item(cache_state, first_id) is True
    records = build_lut_cache_records(cache_state)
    options, records_by_id, _labels_by_id = build_lut_cache_selection_options(records)

    assert options == [second_id]
    assert first_id not in records_by_id
    assert fallback_lut_cache_selection(options, first_id) == second_id
    assert fallback_lut_cache_selection([], second_id) is None
    assert delete_lut_cache_item(cache_state, "missing") is False


def test_lut_cache_empty_broken_and_missing_file_entries_are_unavailable() -> None:
    cache_state: dict[str, dict[str, object]] = {}
    assert build_lut_cache_records(cache_state) == []

    broken_id = add_lut_cache_bytes(cache_state, "broken.csv", b"time_s,voltage_v\n0,1\n")
    missing_id = "lut:missing"
    cache_state[missing_id] = {
        "id": missing_id,
        "original_filename": "missing.csv",
        "display_name": "missing.csv",
        "user_note": "",
        "file_path": str(REPO_ROOT / "does_not_exist" / "missing.csv"),
    }
    records = build_lut_cache_records(cache_state)
    options, records_by_id, labels_by_id = build_lut_cache_selection_options(records)

    assert options == [broken_id, missing_id]
    assert records_by_id[broken_id].parsed.ok is False
    assert "Missing required LUT columns" in (records_by_id[broken_id].parsed.error or "")
    assert records_by_id[missing_id].parsed.ok is False
    assert records_by_id[missing_id].metadata["parse_status"] == "unavailable"
    assert "읽을 수 없음" in labels_by_id[missing_id]


def test_lut_cache_options_do_not_trigger_dataframe_truth_value_ambiguity() -> None:
    cache_state: dict[str, dict[str, object]] = {}
    cache_id = add_lut_cache_bytes(cache_state, "truth.csv", _csv_bytes(3.0))

    records = build_lut_cache_records(cache_state)
    options, records_by_id, labels_by_id = build_lut_cache_selection_options(records)

    assert bool(options[0]) is True
    assert isinstance(labels_by_id[cache_id], str)
    assert records_by_id[cache_id].parsed.frame["voltage_v"].tolist() == [3.0, 6.0]
    assert _contains_dataframe(options) is False
    assert _contains_dataframe(labels_by_id) is False


def test_lut_cache_management_source_has_user_visible_markers_and_no_mojibake() -> None:
    source = (REPO_ROOT / "src" / "field_analysis" / "ui_voltage_lut_review.py").read_text(encoding="utf-8")

    expected = [
        "업로드된 LUT 캐시",
        "display name",
        "user note",
        "선택한 LUT 캐시 삭제",
        "삭제 확인",
        "Internal ID",
        "읽을 수 없음",
    ]
    missing = [marker for marker in expected if marker not in source]
    mojibake_patterns = [chr(0xFFFD), chr(0xF9E4), chr(0xC4D2), "?" + chr(0xAFF0) + chr(0xC0AC)]
    found = [pattern for pattern in mojibake_patterns if pattern in source]

    assert not missing, f"Missing LUT cache management markers: {missing}"
    assert not found, f"Mojibake patterns found in LUT cache UI text: {found}"
