from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

APP_UI_SNAPSHOT = REPO_ROOT / "src" / "field_analysis" / "app_ui_snapshot.py"
REVIEW_UI = REPO_ROOT / "src" / "field_analysis" / "ui_finite_actual_drive_review.py"
ACTUAL_DRIVE_CACHE_UI = REPO_ROOT / "src" / "field_analysis" / "ui_actual_drive_cache.py"


def _write_actual_drive_csv(path: Path, *, include_voltage: bool = True) -> None:
    rows = []
    time_ms = np.linspace(0.0, 1400.0, 51)
    voltage = np.zeros_like(time_ms)
    active = (time_ms >= 200.0) & (time_ms <= 1200.0)
    voltage[active] = 2.0 * np.sin(np.pi * (time_ms[active] - 200.0) / 1000.0)
    hall = 1.5 + 40.0 * np.sin(np.pi * np.clip((time_ms - 200.0) / 1000.0, 0.0, 1.0))
    header = "Row,TimeMs,HallBx,HallBy,HallBz,Current1_A,Current2_A"
    if include_voltage:
        header += ",Voltage1_V,Voltage2_V"
    for index, (t_ms, v, h) in enumerate(zip(time_ms, voltage, hall, strict=False)):
        base = f"{index},{t_ms:.3f},0.0,0.0,{h:.6f},0.1,0.0"
        rows.append(f"{base},{v:.6f},0.0" if include_voltage else base)
    preamble = [
        "# PreDelay(s),1.000",
        "# PostDelay(s),1.000",
        "# AutoSyncHallLag,applied 70.00ms (r=0.815)",
        "#",
        header,
    ]
    path.write_text("\n".join([*preamble, *rows]), encoding="utf-8")


def test_ui_section_markers_are_connected_to_raw_waveforms_path() -> None:
    source = APP_UI_SNAPSHOT.read_text(encoding="utf-8")
    helper = REVIEW_UI.read_text(encoding="utf-8")

    assert "render_finite_actual_drive_review_section" in source
    assert "render_finite_actual_drive_review_section()" in source
    assert "실구동 결과 리뷰" in helper
    assert "validation/result CSV 업로드" in helper
    assert "캐시된 validation 결과 불러오기" in helper
    assert "outputs/field_analysis_app_state/uploads/validation" in helper
    assert "이 섹션은 1차 추천 전압을 실제 장비에 넣은 결과를 확인하기 위한 리뷰 화면입니다." in helper
    assert "아직 2차 보정 전압은 계산하지 않습니다." in helper
    assert "파싱 성공" in helper
    assert "파싱 실패" in helper
    assert "2차 보정 계산" not in helper


def test_cached_validation_result_paths_find_result_files(tmp_path: Path) -> None:
    from field_analysis.ui_finite_actual_drive_review import cached_validation_result_paths

    validation_dir = tmp_path / "outputs" / "field_analysis_app_state" / "uploads" / "validation"
    validation_dir.mkdir(parents=True)
    expected = validation_dir / "7d72d4709ef49600_finite_recommended_voltage_lut_sine_0.5Hz_1.25cycle_result.csv"
    ignored = validation_dir / "not_a_result.csv"
    _write_actual_drive_csv(expected)
    ignored.write_text("x,y\n1,2\n", encoding="utf-8")

    paths = cached_validation_result_paths(project_root=tmp_path)

    assert paths == [expected]


def test_hash_prefixed_upload_filename_parses_to_review_case(tmp_path: Path) -> None:
    from field_analysis.ui_finite_actual_drive_review import build_actual_drive_review_cases_from_paths

    path = tmp_path / "7d72d4709ef49600_finite_recommended_voltage_lut_sine_0.5Hz_1.25cycle_result.csv"
    _write_actual_drive_csv(path)

    result = build_actual_drive_review_cases_from_paths([path])

    assert result.parsed_count == 1
    assert result.failed_count == 0
    case = result.cases[0]
    assert case.upload_internal_id == "7d72d4709ef49600"
    assert case.canonical_source_filename == "finite_recommended_voltage_lut_sine_0.5Hz_1.25cycle_result.csv"
    assert case.label == "sine | 0.5 Hz | 1.25 cycle | finite_recommended_voltage_lut_sine_0.5Hz_1.25cycle_result.csv"
    assert {"time_s", "first_voltage_v", "command_voltage_v", "actual_drive_voltage_v", "physical_target_output_mT", "measured_field_mT", "measured_residual_mT"}.issubset(case.review_frame.columns)


def test_actual_drive_upload_cache_uses_scalar_ids_and_preserves_shape_normalized_payload(tmp_path: Path) -> None:
    from field_analysis.ui_finite_actual_drive_review import build_actual_drive_review_cases_from_cache_state
    from field_analysis.ui_upload_cache import add_upload_cache_bytes
    from field_analysis.ui_upload_cache import build_upload_cache_records
    from field_analysis.ui_upload_cache import build_upload_cache_selection_options
    from field_analysis.ui_upload_cache import delete_upload_cache_item
    from field_analysis.ui_upload_cache import edit_upload_cache_metadata
    from field_analysis.ui_upload_cache import fallback_upload_cache_selection

    path = tmp_path / "finite_recommended_voltage_lut_sine_0.5Hz_1.25cycle_result.csv"
    _write_actual_drive_csv(path)
    cache_state: dict[str, dict[str, object]] = {}
    first_id = add_upload_cache_bytes(
        cache_state,
        path.name,
        path.read_bytes(),
        cache_type="actual_drive_validation",
        display_name="First validation",
    )
    duplicate_id = add_upload_cache_bytes(
        cache_state,
        path.name,
        path.read_bytes(),
        cache_type="actual_drive_validation",
        display_name="Duplicate validation",
    )

    records = build_upload_cache_records(cache_state)
    options, records_by_id, labels_by_id = build_upload_cache_selection_options(records)
    result = build_actual_drive_review_cases_from_cache_state(cache_state)

    assert options == [first_id, duplicate_id]
    assert all(isinstance(option, str) for option in options)
    assert records_by_id[duplicate_id].duplicate_of == first_id
    assert "duplicate_of=" in labels_by_id[duplicate_id]
    assert result.parsed_count == 2
    assert "normalized_measured_field_mT" in result.cases[0].review_frame.columns
    assert "normalized_first_voltage_v" in result.cases[0].review_frame.columns
    assert result.cases[0].metadata["shape_review_only"] is True
    assert edit_upload_cache_metadata(cache_state, first_id, display_name="Renamed", user_note="keep")
    assert build_upload_cache_records(cache_state)[0].cache_item_id == first_id
    assert delete_upload_cache_item(cache_state, first_id) is True
    assert fallback_upload_cache_selection([duplicate_id], first_id) == duplicate_id


def test_missing_required_columns_are_reported(tmp_path: Path) -> None:
    from field_analysis.ui_finite_actual_drive_review import build_actual_drive_review_cases_from_paths

    path = tmp_path / "finite_recommended_voltage_lut_sine_1.25Hz_1cycle_result.csv"
    _write_actual_drive_csv(path, include_voltage=False)

    result = build_actual_drive_review_cases_from_paths([path])

    assert result.parsed_count == 0
    assert result.failed_count == 1
    assert result.failures[0]["parse_status"] == "error"
    assert "Missing actual-drive result columns" in result.failures[0]["parse_error"]


def test_review_ui_contract_labels_and_no_overclaim() -> None:
    helper = REVIEW_UI.read_text(encoding="utf-8")

    expected = [
        "이 화면은 절대 gain 평가가 아니라 파형 개형/타이밍 검토용입니다.",
        "Measured field는 peak 기준 ±50mT로 정규화되어 표시됩니다.",
        "Voltage는 peak 기준 ±5V 이내로 정규화되어 표시됩니다.",
        "Raw 값은 보존되며, 정규화는 review plot/metrics용입니다.",
        "modeled cycle과 intended drive cycle은 별도 metadata로 표시됩니다.",
        "Normalization status",
        "field_normalization_enabled",
        "field_normalization_mode",
        "peak_to_50mT",
        "field_normalization_source_peak_mT",
        "field_normalization_scale_factor",
        "voltage_normalization_enabled",
        "voltage_normalization_mode",
        "peak_to_5V",
        "voltage_normalization_source_peak_v",
        "voltage_normalization_scale_factor",
        "shape_review_only",
        "Cycle semantics",
        "modeled_cycle_count",
        "intended_drive_cycle_count",
        "source_filename_cycle_count",
        "cycle_usage_mode",
        "모델링 cycle label과 실제 구동 의도 cycle은 별도로 표시됩니다. target을 바꾼 것이 아닙니다.",
        "Normalized Target vs Normalized Measured Field",
        "Raw Measured Field",
        "Normalized First/Actual Drive Voltage",
        "Raw First/Actual Drive Voltage",
        "Normalized Residual",
        "Terminal peak zoom",
        "Physical Target",
        "Measured HallBz",
        "Voltage1_V는 실제 1차 command로 사용됩니다.",
        "HallBz는 measured field로 사용됩니다.",
        "field unit은 inferred mT로 표시됩니다.",
        "First Command Voltage",
        "Command vs Actual Drive Voltage",
        "Actual Drive Voltage",
        "Target - Measured",
        "All Actual-Drive Measured Fields",
        "리뷰 CSV 다운로드",
        "요약 CSV 다운로드",
        "measured_active_nrmse",
        "measured_shape_corr",
        "measured_peak_error_mT",
        "measured_phase_error_s",
        "measured_terminal_error_mT",
        "measured_tail_residual",
        "normalized_nrmse",
        "normalized_shape_corr",
        "terminal_peak_time_error_s",
        "raw_field_peak_mT",
        "raw_voltage_peak_v",
        "alignment_confidence",
        "possible_polarity_flip_suggested",
        "review, not acceptance",
        "사용자가 그래프를 확인한 뒤 2차 보정 방향을 결정합니다.",
        "Command target/gain quality is not automatically judged.",
    ]
    missing = [marker for marker in expected if marker not in helper]

    assert not missing
    assert "합격" not in helper
    assert "pass/fail" not in helper.lower()
    assert "2차 보정 LUT" not in helper
    assert "correction_delta_v" not in helper


def test_review_ui_text_has_no_mojibake_patterns() -> None:
    helper = REVIEW_UI.read_text(encoding="utf-8")
    mojibake_patterns = (
        chr(0xFFFD),
        chr(0xF9E4),
        chr(0xC4D2),
        "?" + chr(0xAFF0) + chr(0xC0AC),
        chr(0x00EC),
        chr(0x00ED),
        chr(0x00EB),
        chr(0x00EA),
        chr(0x00F0),
    )

    assert not any(pattern in helper for pattern in mojibake_patterns)


def test_actual_drive_cache_management_source_has_user_visible_korean_markers() -> None:
    helper = ACTUAL_DRIVE_CACHE_UI.read_text(encoding="utf-8")

    expected = [
        "업로드된 validation 캐시",
        "표시 이름",
        "메모",
        "원본 파일명",
        "내부 ID",
        "삭제 전 확인",
        "선택한 validation 캐시 항목 삭제",
        "앱 캐시 목록에서 제거합니다",
    ]
    missing = [marker for marker in expected if marker not in helper]
    forbidden = [
        "Delete selected actual-drive cache item",
        "Save actual-drive cache metadata",
        "display name",
        "user note",
    ]

    assert not missing
    assert not any(marker in helper for marker in forbidden)
