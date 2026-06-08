from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def test_lut_cache_add_edit_delete_uses_scalar_ids() -> None:
    from field_analysis.ui_voltage_lut_cache import add_lut_cache_bytes
    from field_analysis.ui_voltage_lut_cache import build_lut_cache_records
    from field_analysis.ui_voltage_lut_cache import build_lut_cache_selection_options
    from field_analysis.ui_voltage_lut_cache import delete_lut_cache_item
    from field_analysis.ui_voltage_lut_cache import edit_lut_cache_metadata
    from field_analysis.ui_voltage_lut_cache import fallback_lut_cache_selection

    cache_state: dict[str, dict[str, object]] = {}
    csv_bytes = b"sample_index,time_s,voltage_v\n0,0,0.0\n1,0.1,1.0\n"
    cache_id = add_lut_cache_bytes(cache_state, "finite_recommended_voltage_lut_sine_1Hz_1cycle.csv", csv_bytes)

    records = build_lut_cache_records(cache_state)
    options, records_by_id, labels_by_id = build_lut_cache_selection_options(records)

    assert options == [cache_id]
    assert isinstance(options[0], str)
    assert isinstance(records_by_id[cache_id].parsed.frame, pd.DataFrame)
    assert "finite_recommended_voltage_lut" in labels_by_id[cache_id]
    assert fallback_lut_cache_selection(options, "missing") == cache_id

    assert edit_lut_cache_metadata(cache_state, cache_id, display_name="사용자 확인 LUT", user_note="메모")
    edited = build_lut_cache_records(cache_state)[0]
    assert edited.display_name == "사용자 확인 LUT"
    assert edited.user_note == "메모"

    assert delete_lut_cache_item(cache_state, cache_id)
    assert build_lut_cache_records(cache_state) == []


def test_actual_drive_cache_add_edit_delete_uses_scalar_ids() -> None:
    from field_analysis.ui_upload_cache import add_upload_cache_bytes
    from field_analysis.ui_upload_cache import build_upload_cache_records
    from field_analysis.ui_upload_cache import build_upload_cache_selection_options
    from field_analysis.ui_upload_cache import delete_upload_cache_item
    from field_analysis.ui_upload_cache import edit_upload_cache_metadata
    from field_analysis.ui_upload_cache import fallback_upload_cache_selection

    cache_state: dict[str, dict[str, object]] = {}
    csv_bytes = b"TimeMs,Voltage1_V,HallBz\n0,0,0\n1,1,2\n"
    cache_id = add_upload_cache_bytes(
        cache_state,
        "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv",
        csv_bytes,
        cache_type="actual_drive_validation",
    )

    records = build_upload_cache_records(cache_state)
    options, records_by_id, labels_by_id = build_upload_cache_selection_options(records)

    assert options == [cache_id]
    assert isinstance(options[0], str)
    assert records_by_id[cache_id].cache_type == "actual_drive_validation"
    assert "finite_recommended_voltage_lut" in labels_by_id[cache_id]
    assert fallback_upload_cache_selection(options, "missing") == cache_id

    assert edit_upload_cache_metadata(cache_state, cache_id, display_name="실구동 확인", user_note="메모")
    edited = build_upload_cache_records(cache_state)[0]
    assert edited.display_name == "실구동 확인"
    assert edited.user_note == "메모"

    assert delete_upload_cache_item(cache_state, cache_id)
    assert build_upload_cache_records(cache_state) == []


def test_upload_cache_ui_source_has_management_markers_and_no_mojibake() -> None:
    sources = {
        "lut": (SRC_ROOT / "field_analysis" / "ui_voltage_lut_review.py").read_text(encoding="utf-8"),
        "lut_export": (SRC_ROOT / "field_analysis" / "ui_final_voltage_lut_export.py").read_text(encoding="utf-8"),
        "actual": (SRC_ROOT / "field_analysis" / "ui_actual_drive_cache.py").read_text(encoding="utf-8"),
        "actual_review": (SRC_ROOT / "field_analysis" / "ui_finite_actual_drive_review.py").read_text(encoding="utf-8"),
    }
    combined = "\n".join(sources.values())

    expected_markers = [
        "캐시 항목 삭제",
        "표시 이름",
        "메모",
        "원본 파일명",
        "내부 ID",
        "삭제 전 확인",
        "앱 캐시 목록에서 제거",
        "업로드된 LUT 캐시",
        "업로드된 validation 캐시",
        "캐시가 비어 있습니다",
        "읽을 수 없음",
        "최종 전압 LUT 추출",
        "LUT Review",
        "실구동 결과 리뷰",
        "2차 보정 전압은 계산하지 않습니다",
    ]
    missing = [marker for marker in expected_markers if marker not in combined]
    assert not missing, f"Missing upload cache UI markers: {missing}"

    forbidden = [
        chr(0xFFFD),
        chr(0xF9E4),
        chr(0xC4D2),
        "?" + chr(0xAFF0) + chr(0xC0AC),
        chr(0x00EC),
        chr(0x00ED),
        chr(0x00EB),
        chr(0x00EA),
    ]
    found = [pattern for pattern in forbidden if pattern in combined]
    assert not found, f"Mojibake patterns found: {found}"


def test_upload_cache_ui_source_keeps_no_second_correction_button() -> None:
    combined = "\n".join(
        [
            (SRC_ROOT / "field_analysis" / "ui_voltage_lut_review.py").read_text(encoding="utf-8"),
            (SRC_ROOT / "field_analysis" / "ui_actual_drive_cache.py").read_text(encoding="utf-8"),
            (SRC_ROOT / "field_analysis" / "ui_finite_actual_drive_review.py").read_text(encoding="utf-8"),
        ]
    )

    assert "2차 보정 계산" not in combined
    assert "2차 보정 LUT" not in combined
    assert "second_voltage_v" not in combined
