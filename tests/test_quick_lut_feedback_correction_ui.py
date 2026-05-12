from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def test_feedback_review_panel_renders_supported_status_and_graph_markers() -> None:
    from field_analysis.ui_quick_lut_feedback import build_command_source_rows
    from field_analysis.ui_quick_lut_feedback import build_feedback_status_rows
    from field_analysis.ui_quick_lut_feedback import feedback_export_source_column

    metadata = {
        "feedback_route": "finite_feedback_symmetric_peak_correction",
        "feedback_correction_available": True,
        "feedback_correction_status": "ok",
        "feedback_source_file": "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv",
        "feedback_alignment_status": "ok",
        "hallbz_sign_applied": True,
        "field_normalization_mode": "peak_to_50mT",
        "voltage_normalization_mode": "peak_to_5V_or_limit",
        "target_unchanged": True,
    }
    rows = build_feedback_status_rows(metadata)

    assert {"field": "route", "value": "finite_actual_feedback_peak_correction"} in rows
    assert {"field": "supported cycles", "value": "1.0, 1.5"} in rows
    assert {"field": "unsupported cycles", "value": "1.25, 1.75, 2.0"} in rows
    assert {"field": "unsupported reason", "value": "unsupported_cycle_policy_1p0_1p5_only"} in rows
    assert {"field": "production cycle policy", "value": "1p0_1p5_cycles"} in rows

    profile = pd.DataFrame(
        {
            "limited_voltage_v": [0.0],
            "feedback_corrected_limited_voltage_v": [1.0],
            "feedback_correction_status": ["ok"],
            "feedback_correction_available": [True],
        }
    )
    assert feedback_export_source_column(profile) == "feedback_corrected_limited_voltage_v"
    assert {"field": "active_command_source", "value": "feedback_corrected_limited_voltage_v"} in build_command_source_rows(
        profile, metadata
    )
    assert {
        "field": "command_prediction_consistency_status",
        "value": "forward_prediction_unavailable_for_feedback_corrected_command",
    } in build_command_source_rows(profile, metadata)


def test_command_source_rows_show_baseline_when_feedback_unavailable() -> None:
    from field_analysis.ui_quick_lut_feedback import build_command_source_rows

    rows = build_command_source_rows(
        pd.DataFrame(
            {
                "limited_voltage_v": [0.0],
                "feedback_correction_status": ["feedback_source_unavailable"],
                "feedback_correction_available": [False],
            }
        )
    )

    assert {"field": "active_command_source", "value": "limited_voltage_v"} in rows
    assert {"field": "feedback_used_for_correction", "value": False} in rows


def test_feedback_export_source_falls_back_to_baseline_when_unavailable() -> None:
    from field_analysis.ui_quick_lut_feedback import feedback_export_source_column

    assert (
        feedback_export_source_column(
            pd.DataFrame(
                {
                    "limited_voltage_v": [0.0],
                    "feedback_corrected_limited_voltage_v": [1.0],
                    "feedback_correction_status": ["unsupported_cycle_policy_1p0_1p5_only"],
                    "feedback_correction_available": [False],
                }
            )
        )
        == "limited_voltage_v"
    )


def test_quick_lut_feedback_source_contract_markers_present_and_no_mojibake() -> None:
    sources = "\n".join(
        [
            (SRC_ROOT / "field_analysis" / "app_ui_snapshot.py").read_text(encoding="utf-8"),
            (SRC_ROOT / "field_analysis" / "ui_quick_lut_feedback.py").read_text(encoding="utf-8"),
            (SRC_ROOT / "field_analysis" / "ui_voltage_lut_review.py").read_text(encoding="utf-8"),
        ]
    )

    expected = [
        "Quick LUT 피드백 보정",
        "목표 자기장은 유지하고, 실제 구동 결과로 전압 명령만 보정합니다.",
        "finite_actual_feedback_peak_correction",
        "실구동 결과 CSV 업로드",
        "캐시된 실구동 결과 파일",
        "first_run",
        "second_run",
        "unknown",
        "HallBz 부호 보정 적용",
        "실측 자기장 peak를 ±50mT 기준으로 정규화",
        "전압을 ±5V 기준으로 정규화/제한",
        "보정 전압 변화량",
        "피드백 보정 후 제한 전압",
        "상세 진단",
        "active_command_source",
        "plotted_command_source",
        "run_waveform_voltage_source",
        "feedback_used_for_correction",
        "baseline_limited_voltage_v",
        "현재 표시 중인 전압 명령",
        "예측 출력 상태",
        "displayed_predicted_valid",
        "command_prediction_consistency_status",
        "forward_prediction_unavailable_for_feedback_corrected_command",
        "화면에 표시된 전압 명령과 같은 column을 저장합니다.",
        "exported_voltage_source_column",
        "Production finite 보정은 1.0 / 1.5 cycle을 지원합니다.",
        "2-cycle production 정책은 폐기되었습니다.",
        "unsupported_cycle_policy_1p0_1p5_only",
        "apply_finite_feedback_peak_correction",
    ]
    missing = [marker for marker in expected if marker not in sources]
    assert not missing, f"Missing Quick LUT feedback UI markers: {missing}"

    forbidden = [
        chr(0xFFFD),
        chr(0xF9E4),
        chr(0xC4D2),
        "?" + chr(0xAFF0) + chr(0xC0AC),
        chr(0x00EC),
        chr(0x00ED),
        chr(0x00EB),
        chr(0x00EA),
        "�",
    ]
    found = [pattern for pattern in forbidden if pattern in sources]
    assert not found, f"Mojibake patterns found: {found}"


def test_feedback_plot_dataframe_accepts_optional_prediction() -> None:
    from field_analysis.ui_quick_lut_feedback import build_feedback_plot_frame

    profile = pd.DataFrame(
        {
            "time_s": [0.0, 0.1],
            "physical_target_output_mT": [0.0, 50.0],
            "measured_field_normalized_mT": [0.0, 40.0],
            "baseline_limited_voltage_v": [0.0, 4.0],
            "limited_voltage_v": [0.0, 4.5],
            "feedback_correction_delta_v": [0.0, 0.5],
            "feedback_corrected_limited_voltage_v": [0.0, 4.5],
        }
    )
    frame = build_feedback_plot_frame(profile)

    assert list(frame["time_s"]) == [0.0, 0.1]
    assert "feedback_corrected_predicted_field_mT" not in frame.columns
    assert np.allclose(frame["현재 표시 중인 전압 명령"], [0.0, 4.5])
    assert np.allclose(frame["1차 추천/제한 전압"], [0.0, 4.0])
    assert np.allclose(frame["오차 (목표 - 실측)"], [0.0, 10.0])


def test_user_facing_quick_lut_feedback_default_copy_is_korean() -> None:
    source = (SRC_ROOT / "field_analysis" / "ui_quick_lut_feedback.py").read_text(encoding="utf-8")

    expected = [
        "목표 자기장 vs 실측 자기장",
        "명령 전압 vs 실제 구동 전압",
        "1차 전압 vs 2차 보정 전압",
        "Raw 데이터 상세 보기",
        "상세 진단",
        "1차 모델링 전압",
        "2차 모델링 전압",
        "최종 적합성은 사용자가 그래프를 보고 판단합니다.",
    ]
    missing = [marker for marker in expected if marker not in source]
    assert not missing

    forbidden_default_copy = [
        "Quick LUT feedback correction",
        "Actual-drive review",
        "Command source panel",
        "Normalization panel",
        "Field feedback review",
        "Baseline vs corrected command",
        "Raw Actual-drive Visualization",
    ]
    found = [marker for marker in forbidden_default_copy if marker in source]
    assert not found
