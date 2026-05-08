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
        "alignment_confidence",
        "possible_polarity_flip_suggested",
        "review, not acceptance",
        "사용자가 그래프를 확인한 뒤 2차 보정 방향을 결정합니다.",
    ]
    missing = [marker for marker in expected if marker not in helper]

    assert not missing
    assert "합격" not in helper
    assert "pass/fail" not in helper.lower()


def test_review_ui_text_has_no_mojibake_patterns() -> None:
    helper = REVIEW_UI.read_text(encoding="utf-8")
    mojibake_patterns = (
        "\ufffd",
        "\uf9e4",
        "\uc4d2",
        "?\uaff0\uc0ac",
        "ì",
        "í",
        "ë",
        "ê",
        "ð",
    )

    assert not any(pattern in helper for pattern in mojibake_patterns)
