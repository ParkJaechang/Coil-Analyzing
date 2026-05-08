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
    build_normalized_lut_csv_bytes,
    parse_voltage_lut_upload,
)


APP_UI_SNAPSHOT = REPO_ROOT / "src" / "field_analysis" / "app_ui_snapshot.py"


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
    assert "recommended_voltage_v" in frame.columns


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
    source = (REPO_ROOT / "src" / "field_analysis" / "ui_voltage_lut_review.py").read_text(encoding="utf-8")

    expected_markers = [
        "LUT 검수 / LUT Review",
        "최종 모델링 전압 LUT CSV 다운로드",
        "Fourier 재합성 없이 그대로 저장합니다",
        "voltage_v는 limited_voltage_v와 sample-by-sample 동일합니다",
        "finite compensation LUT unavailable",
        "사용자 시간축/전압 파형 검수용",
        "장비 구동 적합성이나 보정 품질을 자동 판정하지 않음",
        "LUT Voltage vs time_s",
        "LUT Voltage vs sample_index",
        "dt_irregularity_ratio",
        "time_axis_status",
        "normalized LUT CSV 다운로드",
        "diagnostics summary CSV 다운로드",
    ]
    missing = [marker for marker in expected_markers if marker not in source]

    assert not missing, f"Missing LUT review UI markers: {missing}"
