from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.ui_voltage_lut_review import (
    build_final_voltage_lut_frame,
    build_final_voltage_lut_filename,
    build_lut_diagnostics,
    build_lut_review_options,
    build_normalized_lut_csv_bytes,
    parse_voltage_lut_upload,
)


APP_UI_SNAPSHOT = REPO_ROOT / "src" / "field_analysis" / "app_ui_snapshot.py"
LUT_REVIEW_UI = REPO_ROOT / "src" / "field_analysis" / "ui_voltage_lut_review.py"


def test_final_voltage_lut_export_uses_limited_voltage_without_fourier() -> None:
    command_profile = pd.DataFrame(
        {
            "time_s": [0.0, 0.1, 0.2],
            "limited_voltage_v": [0.0, 1.5, -0.25],
            "recommended_voltage_v": [0.0, 1.7, -0.3],
        }
    )

    frame = build_final_voltage_lut_frame(command_profile)

    assert list(frame.columns[:3]) == ["sample_index", "time_s", "voltage_v"]
    assert frame["sample_index"].tolist() == [0, 1, 2]
    assert np.allclose(frame["voltage_v"], command_profile["limited_voltage_v"])
    assert list(frame.columns) == ["sample_index", "time_s", "voltage_v"]


def test_final_voltage_lut_export_can_select_second_model_voltage() -> None:
    command_profile = pd.DataFrame(
        {
            "time_s": [0.0, 0.1],
            "limited_voltage_v": [1.0, 2.0],
            "second_limited_voltage_v": [3.0, 4.0],
            "second_modeling_status": ["ok", "ok"],
            "second_modeling_available": [True, True],
        }
    )

    frame = build_final_voltage_lut_frame(command_profile)

    assert list(frame.columns) == ["sample_index", "time_s", "voltage_v"]
    assert np.allclose(frame["voltage_v"], command_profile["second_limited_voltage_v"])


def test_final_voltage_lut_second_export_uses_only_second_limited_voltage_columns() -> None:
    command_profile = pd.DataFrame(
        {
            "time_s": [0.0, 0.1, 0.2],
            "limited_voltage_v": [1.0, 2.0, 3.0],
            "second_correction_delta_v": [0.1, 0.2, 0.3],
            "second_modeled_voltage_v": [1.1, 2.2, 3.3],
            "second_limited_voltage_v": [1.05, 2.05, 3.05],
            "second_modeling_status": ["ok", "ok", "ok"],
            "second_modeling_available": [True, True, True],
        }
    )

    frame = build_final_voltage_lut_frame(command_profile, voltage_source_column="second_limited_voltage_v")

    assert list(frame.columns) == ["sample_index", "time_s", "voltage_v"]
    assert frame["sample_index"].tolist() == [0, 1, 2]
    assert np.allclose(frame["time_s"], command_profile["time_s"])
    assert np.allclose(frame["voltage_v"], command_profile["second_limited_voltage_v"])
    assert "second_correction_delta_v" not in frame.columns
    assert "second_modeled_voltage_v" not in frame.columns
    assert "second_limited_voltage_v" not in frame.columns


def test_final_voltage_lut_filename_has_finite_case_identity() -> None:
    assert (
        build_final_voltage_lut_filename(waveform_type="sine", freq_hz=5.0, cycle_count=1.25)
        == "finite_recommended_voltage_lut_sine_5Hz_1.25cycle.csv"
    )
    assert build_final_voltage_lut_filename(waveform_type=None, freq_hz=None, cycle_count=None) == (
        "finite_recommended_voltage_lut.csv"
    )


def test_uploaded_lut_parse_and_diagnostics_cover_timebase_warnings() -> None:
    csv_bytes = b"sample_index,time_s,voltage_v\n0,0,0.0\n1,10,1.0\n2,20,-1.0\n"

    parsed = parse_voltage_lut_upload("finite_recommended_voltage_lut_sine_1Hz_1.25cycle.csv", csv_bytes)
    diagnostics = build_lut_diagnostics(parsed.frame)

    assert parsed.ok is True
    assert parsed.frame["voltage_v"].tolist() == [0.0, 1.0, -1.0]
    assert diagnostics["sample_count"] == 3
    assert diagnostics["duration_s"] == 20.0
    assert diagnostics["time_monotonic"] is True
    assert diagnostics["duplicated_time_count"] == 0
    assert diagnostics["suspected_time_unit"] == "ms_like_seconds_column"
    assert diagnostics["time_axis_status"] == "warning_time_s_may_be_ms"


def test_uploaded_lut_review_preserves_raw_voltage_and_adds_review_normalized_voltage() -> None:
    csv_bytes = b"sample_index,time_s,voltage_v\n0,0,-12.0\n1,0.1,0.0\n2,0.2,6.0\n"

    parsed = parse_voltage_lut_upload("finite_recommended_voltage_lut_sine_1Hz_1cycle.csv", csv_bytes)
    diagnostics = build_lut_diagnostics(parsed.frame)

    assert parsed.ok is True
    assert parsed.frame["raw_voltage_v"].tolist() == [-12.0, 0.0, 6.0]
    assert np.nanmax(np.abs(parsed.frame["normalized_voltage_v"])) <= 5.0 + 1e-12
    assert np.nanmax(np.abs(parsed.frame["normalized_voltage_v"])) == 5.0
    assert diagnostics["voltage_normalization_enabled"] is True
    assert diagnostics["voltage_normalization_mode"] == "peak_to_5V"
    assert diagnostics["voltage_normalization_source_peak_v"] == 12.0
    assert diagnostics["shape_review_only"] is True


def test_uploaded_lut_missing_required_columns_reports_unavailable() -> None:
    parsed = parse_voltage_lut_upload("bad.csv", b"time_s,limited_voltage_v\n0,1\n")

    assert parsed.ok is False
    assert "Missing required LUT columns" in parsed.error
    assert "voltage_v" in parsed.error


def test_normalized_lut_csv_download_uses_required_schema() -> None:
    parsed = parse_voltage_lut_upload("lut.csv", b"sample_index,time_s,voltage_v\n5,0.0,1.0\n6,0.1,2.0\n")

    normalized = pd.read_csv(BytesIO(build_normalized_lut_csv_bytes(parsed.frame)))

    assert list(normalized.columns[:3]) == ["sample_index", "time_s", "voltage_v"]
    assert normalized["sample_index"].tolist() == [0, 1]
    assert normalized["voltage_v"].tolist() == [1.0, 2.0]


def test_app_ui_contract_connects_export_and_lut_review_section() -> None:
    source = APP_UI_SNAPSHOT.read_text(encoding="utf-8")

    expected_markers = [
        "render_final_voltage_lut_export_panel",
        "render_voltage_lut_review_section",
        "LUT Review",
    ]
    missing = [marker for marker in expected_markers if marker not in source]

    assert not missing, f"Missing final voltage LUT UI markers: {missing}"


def test_lut_review_helper_source_contains_user_visible_review_markers() -> None:
    source = "\n".join(
        [
            (REPO_ROOT / "src" / "field_analysis" / "ui_voltage_lut_review.py").read_text(encoding="utf-8"),
            (REPO_ROOT / "src" / "field_analysis" / "ui_final_voltage_lut_export.py").read_text(encoding="utf-8"),
        ]
    )

    expected_markers = [
        "LUT 검수 / LUT Review",
        "최종 전압 LUT 추출",
        "Fourier 재합성이나 harmonic coefficient export가 아닙니다.",
        "저장 컬럼은 sample_index, time_s, voltage_v 세 개뿐입니다.",
        "최종 전압 LUT 추출 사용 불가",
        "사용자 시간축/전압 파형 검수용",
        "장비 구동 적합성이나 보정 품질을 자동 판정하지 않습니다",
        "추출 대상",
        "1차 모델링 command",
        "2차 보정 command",
        "현재 추출 대상: 1차 모델링 command",
        "현재 추출 대상: 2차 보정 command",
        "2차 보정 command가 아직 없습니다",
        "voltage_v = second_limited_voltage_v",
        "voltage_v = 1차 모델링 command",
        "2차 보정 후 ±5V 제한이 적용된 전압 샘플을 저장합니다.",
        "LUT Voltage vs time_s",
        "LUT Voltage vs sample_index",
        "dt_irregularity_ratio",
        "time_axis_status",
        "normalized LUT CSV 다운로드",
        "diagnostics summary CSV 다운로드",
    ]
    missing = [marker for marker in expected_markers if marker not in source]

    assert not missing, f"Missing LUT review UI markers: {missing}"


def test_lut_review_tab_lists_continuous_session_result_sources() -> None:
    source = LUT_REVIEW_UI.read_text(encoding="utf-8")

    for marker in [
        "quick_lut_first_model_result",
        "quick_lut_second_model_result",
        "quick_lut_first_model_result_continuous",
        "quick_lut_second_model_result_continuous",
        "Finite 1차",
        "Finite 2차",
        "Continuous 1차",
        "Continuous 2차",
        "finite_first_voltage_lut",
        "finite_second_voltage_lut",
        "continuous_first_voltage_lut",
        "continuous_second_voltage_lut",
    ]:
        assert marker in source


def test_lut_review_selectbox_options_are_scalar_ids_not_dataframe_objects() -> None:
    parsed = [
        parse_voltage_lut_upload("first.csv", b"sample_index,time_s,voltage_v\n0,0,1\n"),
        parse_voltage_lut_upload("second.csv", b"sample_index,time_s,voltage_v\n0,0,2\n"),
    ]

    options, records_by_id, labels_by_id = build_lut_review_options(parsed)

    assert options == ["first.csv", "second.csv"]
    assert all(isinstance(option, str) for option in options)
    assert all(hasattr(record.frame, "columns") for record in records_by_id.values())
    assert labels_by_id["first.csv"] == "first.csv"


def test_lut_review_duplicate_source_names_still_use_scalar_unique_ids() -> None:
    parsed = [
        parse_voltage_lut_upload("same.csv", b"sample_index,time_s,voltage_v\n0,0,1\n"),
        parse_voltage_lut_upload("same.csv", b"sample_index,time_s,voltage_v\n0,0,2\n"),
    ]

    options, records_by_id, labels_by_id = build_lut_review_options(parsed)

    assert options == ["same.csv", "same.csv#2"]
    assert all(isinstance(option, str) for option in options)
    assert records_by_id["same.csv"].frame["voltage_v"].tolist() == [1]
    assert records_by_id["same.csv#2"].frame["voltage_v"].tolist() == [2]
    assert labels_by_id["same.csv#2"] == "same.csv"


def test_lut_review_render_path_uses_scalar_selectbox_options() -> None:
    source = "\n".join(
        [
            (REPO_ROOT / "src" / "field_analysis" / "ui_voltage_lut_review.py").read_text(encoding="utf-8"),
            (REPO_ROOT / "src" / "field_analysis" / "ui_final_voltage_lut_export.py").read_text(encoding="utf-8"),
        ]
    )

    assert "options=successes" not in source
    assert "options=cached_files" not in source
    assert "options=cached_ids" in source
    assert "options=cache_ids" in source
    assert "cache_records_by_id[selected_cache_id]" in source


def test_voltage_lut_review_source_has_no_mojibake_patterns() -> None:
    source = (REPO_ROOT / "src" / "field_analysis" / "ui_voltage_lut_review.py").read_text(encoding="utf-8")
    mojibake_patterns = [
        chr(0xFFFD),
        chr(0xF9E4),
        chr(0xC4D2),
        "?" + chr(0xAFF0) + chr(0xC0AC),
        chr(0x00EC),
        chr(0x00ED),
        chr(0x00EB),
        chr(0x00EA),
    ]
    found = [pattern for pattern in mojibake_patterns if pattern in source]

    assert not found, f"Mojibake patterns found in LUT review UI text: {found}"
