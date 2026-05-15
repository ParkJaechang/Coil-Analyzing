from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.ui_final_voltage_lut_export import build_final_voltage_lut_frame
from field_analysis.ui_continuous_steady_state import (
    build_continuous_second_command_profile,
    discover_continuous_candidate_frames,
)

APP_UI = SRC_ROOT / "field_analysis" / "app_ui_snapshot.py"
CONT_UI = SRC_ROOT / "field_analysis" / "ui_continuous_steady_state.py"
RAW_ACTUAL_UI = SRC_ROOT / "field_analysis" / "ui_raw_waveforms_actual_drive.py"
EXPORT_UI = SRC_ROOT / "field_analysis" / "ui_final_voltage_lut_export.py"


def test_quick_lut_continuous_mode_visible_contract() -> None:
    source = APP_UI.read_text(encoding="utf-8") + CONT_UI.read_text(encoding="utf-8")

    for marker in [
        "모델링 입력 방식",
        "Finite startup-aware",
        "Continuous steady-state",
        "Continuous steady-state mode는 연속 구동 데이터에서 안정화된 1cycle만 추출해 모델링합니다.",
        "생성된 1cycle voltage LUT는 반복 출력용입니다.",
        "초반 들뜸/startup transient 응답은 모델링에 사용하지 않습니다.",
        "Continuous mode에서는 1.5cycle을 생성하지 않습니다.",
        "quick_lut_modeling_input_mode",
        "continuous_production_cycle_count",
        "continuous_repeating_lut",
        "zero_return_tail_enabled",
        "Steady-state 1cycle",
        "continuous_steady_state_extraction_result",
        "continuous_steady_state_window_frame",
        "continuous_steady_state_metadata",
        "quick_lut_first_model_result_continuous",
    ]:
        assert marker in source


def test_quick_lut_continuous_runtime_path_calls_extractor_and_is_button_gated() -> None:
    app_source = APP_UI.read_text(encoding="utf-8")
    source = app_source + CONT_UI.read_text(encoding="utf-8")

    assert "build_continuous_steady_state_modeling_case(" in source
    assert "st.button(\"Steady-state 1cycle" in source
    assert "render_continuous_steady_state_runtime_panel(" in app_source
    assert "Continuous 1차 모델링 실행" in source
    assert "quick_lut_modeling_input_mode" in source


def test_quick_lut_continuous_actual_drive_and_validation_runtime_markers_exist() -> None:
    app_source = APP_UI.read_text(encoding="utf-8")
    source = app_source + CONT_UI.read_text(encoding="utf-8")

    for marker in [
        "Continuous 1차 실구동 결과 업로드",
        "실구동 결과에서 안정 1cycle 추출",
        "Continuous 2차 보정 command 생성",
        "Continuous 2차 구동 결과 평가",
        "평가는 안정화된 1cycle 기준입니다.",
        "continuous_first_drive_actual_result",
        "continuous_first_drive_steady_window_frame",
        "continuous_first_drive_steady_metadata",
        "quick_lut_second_model_result_continuous",
    ]:
        assert marker in source
    assert "render_continuous_actual_drive_runtime_panel(" in app_source


def test_continuous_mode_does_not_offer_finite_tail_or_one_point_five_as_production() -> None:
    source = APP_UI.read_text(encoding="utf-8")

    assert "continuous_production_cycle_count = 1.0" in source
    assert "continuous_zero_return_tail_enabled = False" in source
    assert "preview_tail_cycles = 0.0" in source
    assert "target_cycle_count = 1.0" in source


def test_raw_waveforms_continuous_extraction_preview_markers_exist() -> None:
    source = RAW_ACTUAL_UI.read_text(encoding="utf-8")

    for marker in [
        "Continuous steady-state extraction preview",
        "Steady-state 1cycle 추출",
        "Continuous 원본: startup transient와 steady-state 구간",
        "선택된 steady-state 1cycle",
        "cycle stability metrics",
        "이 1cycle이 continuous steady-state modeling에 사용됩니다.",
    ]:
        assert marker in source


def test_continuous_export_contract_markers_exist() -> None:
    source = EXPORT_UI.read_text(encoding="utf-8")

    assert "continuous_loop_output" in source
    assert "loop_endpoint_policy" in source
    assert "sample_index, time_s, voltage_v" in source


def test_continuous_export_drops_period_endpoint_for_loop_safe_lut() -> None:
    frame = pd.DataFrame(
        {
            "time_s": [0.0, 0.25, 0.5, 0.75, 1.0],
            "limited_voltage_v": [0.0, 1.0, 0.0, -1.0, 0.0],
            "continuous_loop_output": [True] * 5,
            "loop_endpoint_policy": ["period_exclusive"] * 5,
            "freq_hz": [1.0] * 5,
        }
    )

    exported = build_final_voltage_lut_frame(frame)

    assert exported.columns.tolist() == ["sample_index", "time_s", "voltage_v"]
    assert exported["time_s"].tolist() == [0.0, 0.25, 0.5, 0.75]
    assert exported["sample_index"].tolist() == [0, 1, 2, 3]


def test_upload_memory_continuous_candidate_discovery_accepts_cached_payload() -> None:
    csv_bytes = b"TimeMs,Voltage1_V,HallBz\n0,0,0\n10,1,-1\n20,0,0\n"

    names, candidates, scan = discover_continuous_candidate_frames(
        {},
        upload_payloads=[("continuous_sine_1Hz.csv", csv_bytes)],
        dataset_library_payloads=[],
    )

    assert names == ["upload_memory:continuous_sine_1Hz.csv"]
    assert scan["continuous_candidate_source_counts"]["upload_memory_continuous"] == 1
    assert scan["continuous_candidate_rejected_count"] == 0
    assert "raw_hallbz_mT" in candidates[names[0]].columns


def test_dataset_library_continuous_candidate_discovery_accepts_library_payload() -> None:
    csv_bytes = b"time_s,command_voltage_v,raw_hallbz_mT\n0,0,0\n0.01,1,-1\n0.02,0,0\n"

    names, candidates, scan = discover_continuous_candidate_frames(
        {},
        upload_payloads=[],
        dataset_library_payloads=[("library/continuous_case.csv", csv_bytes)],
    )

    assert names == ["dataset_library:library/continuous_case.csv"]
    assert scan["continuous_candidate_source_counts"]["dataset_library"] == 1
    assert scan["continuous_candidate_rejected_count"] == 0
    assert "raw_voltage_v" in candidates[names[0]].columns


def test_continuous_candidate_discovery_rejects_final_lut_schema() -> None:
    csv_bytes = b"sample_index,time_s,voltage_v\n0,0,0\n1,0.1,1\n"

    names, _candidates, scan = discover_continuous_candidate_frames(
        {},
        upload_payloads=[("first_modeled_voltage_lut.csv", csv_bytes)],
        dataset_library_payloads=[],
    )

    assert names == []
    assert scan["continuous_candidate_rejected_count"] == 1
    assert "final_voltage_lut_not_measured_input" in scan["continuous_candidate_reject_reasons"][0]


def test_continuous_second_command_profile_is_not_placeholder_copy() -> None:
    period = 0.5
    time_s = np.linspace(0.0, period, 80, endpoint=False)
    first_command = pd.DataFrame(
        {
            "time_s": time_s,
            "limited_voltage_v": 2.0 * np.sin(2.0 * np.pi * time_s / period),
        }
    )
    steady_actual = pd.DataFrame(
        {
            "time_s": time_s,
            "measured_field_normalized_mT": 42.0 * np.sin(2.0 * np.pi * time_s / period - 0.1),
            "normalized_physical_target_output_mT": 50.0 * np.sin(2.0 * np.pi * time_s / period),
            "voltage_normalized_v": 2.0 * np.sin(2.0 * np.pi * time_s / period),
        }
    )

    second, metadata = build_continuous_second_command_profile(
        first_command,
        steady_actual,
        freq_hz=2.0,
        waveform_type="sine",
    )

    assert metadata["continuous_second_modeling_uses_phase_aligned_kernel"] is True
    assert metadata["continuous_second_modeling_tail_disabled"] is True
    assert metadata["continuous_second_modeling_input_window"] == "steady_state_one_cycle_only"
    assert "second_limited_voltage_v" in second.columns
    assert "correction_delta_v" in second.columns
    assert not np.allclose(second["second_limited_voltage_v"], first_command["limited_voltage_v"])
