from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

APP_UI_SNAPSHOT = REPO_ROOT / "src" / "field_analysis" / "app_ui_snapshot.py"


def _source() -> str:
    return APP_UI_SNAPSHOT.read_text(encoding="utf-8")


def test_finite_voltage_lut_export_ui_markers_exist() -> None:
    source = _source()

    expected_markers = [
        "Final Recommended Voltage LUT",
        "최종 추천 전압 LUT CSV 다운로드",
        "이 CSV는 화면에 표시된 최종 추천 전압 파형을 그대로 저장합니다.",
        "Fourier 재합성 파형이 아닙니다.",
        "전압 컬럼은 현재 Command Waveform plot의 `limited_voltage_v`와 동일합니다.",
        "finite compensation LUT unavailable",
        "limited_voltage_v is missing",
        "_render_final_voltage_lut_export_panel(",
        "download_final_voltage_lut_v2",
        "download_compensation_waveform_v2",
    ]
    missing = [marker for marker in expected_markers if marker not in source]

    assert not missing, f"Missing finite voltage LUT export UI markers: {missing}"


def test_final_voltage_lut_export_frame_uses_limited_voltage_sample_by_sample() -> None:
    from field_analysis.app_ui_snapshot import _build_final_voltage_lut_export_frame

    command_profile = pd.DataFrame(
        {
            "time_s": [0.0, 0.1, 0.2],
            "limited_voltage_v": [0.0, 1.5, -2.0],
            "recommended_voltage_v": [0.0, 1.7, -2.2],
            "baseline_recommended_voltage_v": [0.1, 1.0, -1.0],
            "compensated_recommended_voltage_v": [0.0, 1.5, -2.0],
            "startup_compensation_command_delta_v": [-0.1, 0.5, -1.0],
        }
    )

    export_frame = _build_final_voltage_lut_export_frame(command_profile)

    assert list(export_frame.columns) == [
        "sample_index",
        "time_s",
        "voltage_v",
        "recommended_voltage_v",
        "baseline_recommended_voltage_v",
        "compensated_recommended_voltage_v",
        "startup_compensation_command_delta_v",
    ]
    assert export_frame["sample_index"].tolist() == [0, 1, 2]
    assert export_frame["time_s"].tolist() == command_profile["time_s"].tolist()
    assert export_frame["voltage_v"].tolist() == command_profile["limited_voltage_v"].tolist()


def test_final_voltage_lut_export_frame_requires_displayed_command_columns() -> None:
    from field_analysis.app_ui_snapshot import _build_final_voltage_lut_export_frame

    missing_voltage = pd.DataFrame({"time_s": [0.0, 0.1]})
    missing_time = pd.DataFrame({"limited_voltage_v": [0.0, 1.0]})

    assert _build_final_voltage_lut_export_frame(missing_voltage) is None
    assert _build_final_voltage_lut_export_frame(missing_time) is None


def test_final_voltage_lut_filename_has_safe_fallbacks() -> None:
    from field_analysis.app_ui_snapshot import _final_voltage_lut_file_name

    assert (
        _final_voltage_lut_file_name("sine", 5.0, 1.25)
        == "finite_recommended_voltage_lut_sine_5Hz_1.25cycle.csv"
    )
    assert _final_voltage_lut_file_name("", None, None) == "finite_recommended_voltage_lut.csv"


def test_finite_voltage_lut_export_does_not_modify_backend_files() -> None:
    source = _source()

    assert "src/field_analysis/compensation.py" not in source
    assert "src/field_analysis/parser.py" not in source
