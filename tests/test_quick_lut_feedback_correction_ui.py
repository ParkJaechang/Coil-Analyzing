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


def test_feedback_correction_wrong_file_type_returns_unavailable_without_crash() -> None:
    from field_analysis.ui_quick_lut_feedback import apply_feedback_correction_from_selection

    command_profile = pd.DataFrame(
        {
            "time_s": [0.0, 0.1],
            "limited_voltage_v": [0.0, 1.0],
            "physical_target_output_mT": [0.0, 50.0],
        }
    )
    selection = {
        "filename": "finite_recommended_voltage_lut_sine_2Hz_1.5cycle.csv",
        "csv_bytes": b"sample_index,time_s,voltage_v\n0,0,0\n1,0.1,1\n",
        "run_label": "first_run",
    }

    returned_profile, metadata = apply_feedback_correction_from_selection(
        command_profile,
        selection,
        waveform_type="sine",
        freq_hz=2.0,
        cycle_count=1.5,
    )

    assert returned_profile.equals(command_profile)
    assert metadata["feedback_correction_available"] is False
    assert metadata["feedback_correction_status"] == "feedback_source_invalid"
    assert metadata["feedback_correction_unavailable_reason"] == "unsupported_actual_drive_result_file"
    assert metadata["feedback_used_for_correction"] is False
    assert "finite_recommended_voltage_lut_sine_2Hz_1.5cycle.csv" in str(metadata["feedback_source_file"])


def test_actual_drive_feedback_candidate_auto_selects_single_exact_match() -> None:
    from field_analysis.ui_quick_lut_feedback import choose_actual_drive_feedback_candidate

    selected, metadata = choose_actual_drive_feedback_candidate(
        [
            {
                "filename": "finite_recommended_voltage_lut_sine_2Hz_1.5cycle_result.csv",
                "csv_bytes": b"Row,TimeMs,HallBz,Voltage1_V\n0,0,1,0\n",
            }
        ],
        waveform_type="sine",
        freq_hz=2.0,
        cycle_count=1.5,
    )

    assert selected is not None
    assert selected["filename"] == "finite_recommended_voltage_lut_sine_2Hz_1.5cycle_result.csv"
    assert metadata["selection_reason"] == "exact_match"


def test_actual_drive_feedback_candidate_does_not_auto_select_multiple_exact_matches() -> None:
    from field_analysis.ui_quick_lut_feedback import choose_actual_drive_feedback_candidate

    selected, metadata = choose_actual_drive_feedback_candidate(
        [
            {"filename": "a_finite_recommended_voltage_lut_sine_2Hz_1.5cycle_result.csv", "csv_bytes": b""},
            {"filename": "b_finite_recommended_voltage_lut_sine_2Hz_1.5cycle_result.csv", "csv_bytes": b""},
        ],
        waveform_type="sine",
        freq_hz=2.0,
        cycle_count=1.5,
    )

    assert selected is None
    assert metadata["selection_reason"] == "multiple_exact_matches"


def test_actual_drive_feedback_candidate_does_not_auto_select_single_mismatch_for_production() -> None:
    from field_analysis.ui_quick_lut_feedback import choose_actual_drive_feedback_candidate

    selected, metadata = choose_actual_drive_feedback_candidate(
        [
            {
                "filename": "finite_recommended_voltage_lut_sine_0.25Hz_1.5cycle_result.csv",
                "csv_bytes": b"Row,TimeMs,HallBz,Voltage1_V\n0,0,1,0\n",
            }
        ],
        waveform_type="sine",
        freq_hz=2.0,
        cycle_count=1.5,
    )

    assert selected is None
    assert metadata["selection_status"] == "needs_manual_selection"
    assert metadata["selection_reason"] == "single_candidate_mismatch_raw_preview_only"


def test_actual_drive_feedback_candidate_accepts_schema_without_result_filename() -> None:
    from field_analysis.ui_quick_lut_feedback import classify_feedback_csv_candidate

    info = classify_feedback_csv_candidate(
        "bench_upload.csv",
        b"TimeMs,Voltage1_V,HallBz\n0,0,1\n",
    )

    assert info["file_type"] == "actual_drive_result"
    assert info["schema_status"] == "actual_drive_schema_no_filename_metadata"
    assert info["metadata_source"] == "unavailable"


def test_actual_drive_feedback_candidate_uses_preamble_metadata_without_result_filename() -> None:
    from field_analysis.ui_quick_lut_feedback import choose_actual_drive_feedback_candidate

    selected, metadata = choose_actual_drive_feedback_candidate(
        [
            {
                "filename": "bench_upload.csv",
                "csv_bytes": b"# Frequency(Hz),2\n# Cycles,1.5\n# Waveform,sine\nTimeMs,Voltage1_V,HallBz\n0,0,1\n",
            }
        ],
        waveform_type="sine",
        freq_hz=2.0,
        cycle_count=1.5,
    )

    assert selected is not None
    assert selected["metadata_source"] == "preamble"
    assert metadata["selection_reason"] == "exact_match"


def test_actual_drive_feedback_candidate_identifies_final_lut_as_wrong_file_type() -> None:
    from field_analysis.ui_quick_lut_feedback import choose_actual_drive_feedback_candidate

    selected, metadata = choose_actual_drive_feedback_candidate(
        [
            {
                "filename": "finite_recommended_voltage_lut_sine_2Hz_1.5cycle.csv",
                "csv_bytes": b"sample_index,time_s,voltage_v\n0,0,0\n",
            }
        ],
        waveform_type="sine",
        freq_hz=2.0,
        cycle_count=1.5,
    )

    assert selected is None
    assert metadata["selection_reason"] == "final_voltage_lut_not_actual_drive_result"


def test_quick_lut_feedback_user_facing_source_has_no_mojibake_patterns() -> None:
    source = (SRC_ROOT / "field_analysis" / "ui_quick_lut_feedback.py").read_text(encoding="utf-8")
    selection_source = (SRC_ROOT / "field_analysis" / "ui_quick_lut_feedback_selection.py").read_text(encoding="utf-8")
    combined = source + "\n" + selection_source

    for pattern in [chr(0xFFFD), "?? target", "?50mT", "1? ???", "??"]:
        assert pattern not in combined
    assert "현재 Quick LUT 설정의 실구동 결과로 사용" in source
    assert "이 파일은 최종 전압 LUT CSV입니다. 실구동 결과 CSV가 아닙니다." in source


def test_quick_lut_feedback_source_contract_markers_present_and_no_mojibake() -> None:
    sources = "\n".join(
        [
            (SRC_ROOT / "field_analysis" / "app_ui_snapshot.py").read_text(encoding="utf-8"),
            (SRC_ROOT / "field_analysis" / "ui_quick_lut_feedback.py").read_text(encoding="utf-8"),
            (SRC_ROOT / "field_analysis" / "ui_quick_lut_feedback_contract.py").read_text(encoding="utf-8"),
            (SRC_ROOT / "field_analysis" / "ui_voltage_lut_review.py").read_text(encoding="utf-8"),
        ]
    )

    expected = [
        "Quick LUT 피드백 보정",
        "TimeMs / Voltage1_V / HallBz 컬럼이 있으면 실구동 결과 후보로 사용할 수 있습니다.",
        "finite_actual_feedback_peak_correction",
        "실구동 결과 CSV 업로드",
        "캐시된 실구동 결과 파일",
        "first_run",
        "second_run",
        "unknown",
        "현재 Quick LUT 설정의 실구동 결과로 사용",
        "1차 실구동 데이터 원본 확인",
        "실측 자기장 (HallBz 부호 보정, ±50mT)",
        "정규화 전압 (±5V)",
        "보정 전압 변화량",
        "피드백 보정 후 제한 전압",
        "active_command_source",
        "plotted_command_source",
        "run_waveform_voltage_source",
        "feedback_used_for_correction",
        "baseline_limited_voltage_v",
        "현재 표시 중인 전압 명령",
        "displayed_predicted_valid",
        "command_prediction_consistency_status",
        "forward_prediction_unavailable_for_feedback_corrected_command",
        "exported_voltage_source_column",
        "Production finite 보정은 1.0 / 1.5 cycle을 지원합니다.",
        "unsupported_cycle_policy_1p0_1p5_only",
        "apply_finite_feedback_peak_correction",
    ]
    missing = [marker for marker in expected if marker not in sources]
    assert not missing, f"Missing Quick LUT feedback UI markers: {missing}"

    forbidden = [chr(0xFFFD), "?? target", "?50mT", "1? ???", "??"]
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
    source = "\n".join(
        [
            (SRC_ROOT / "field_analysis" / "ui_quick_lut_feedback.py").read_text(encoding="utf-8"),
            (SRC_ROOT / "field_analysis" / "ui_final_voltage_lut_export.py").read_text(encoding="utf-8"),
        ]
    )

    expected = [
        "Quick LUT 피드백 보정",
        "최종 전압 LUT 추출",
        "현재 추출 대상",
        "1차 모델링 결과",
        "2차 모델링 결과",
        "최종 LUT는 화면에 표시된 최종 전압 샘플을 그대로 저장합니다.",
        "상세 진단",
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


def test_final_lut_export_hides_internal_metadata_from_default_screen() -> None:
    source = (SRC_ROOT / "field_analysis" / "ui_final_voltage_lut_export.py").read_text(encoding="utf-8")

    default_region = source.split('with st.expander("상세 진단"', maxsplit=1)[0]
    assert "exported_voltage_source_column" not in default_region
    assert 'st.info("현재 추출 대상' in source
