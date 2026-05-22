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
from field_analysis.ui_continuous_final_lut_export import (
    build_continuous_final_lut_filename,
    build_continuous_final_lut_frame,
    continuous_result_export_record,
)
from field_analysis.ui_continuous_steady_state import (
    build_continuous_second_command_profile,
    discover_continuous_candidate_frames,
    infer_continuous_source_frequency,
    rank_continuous_candidates_for_target,
    run_continuous_first_modeling,
    run_continuous_steady_state_extraction,
)
from field_analysis.continuous_candidate_frequency import attach_continuous_frequency_attrs

APP_UI = SRC_ROOT / "field_analysis" / "app_ui_snapshot.py"
CONT_UI = SRC_ROOT / "field_analysis" / "ui_continuous_steady_state.py"
CONT_FIRST_UI = SRC_ROOT / "field_analysis" / "ui_continuous_first_modeling.py"
RAW_ACTUAL_UI = SRC_ROOT / "field_analysis" / "ui_raw_waveforms_actual_drive.py"
EXPORT_UI = SRC_ROOT / "field_analysis" / "ui_final_voltage_lut_export.py"
CONT_EXPORT_UI = SRC_ROOT / "field_analysis" / "ui_continuous_final_lut_export.py"


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
    assert "Steady-state 1cycle 추출이 유효하지 않아 Continuous 1차 모델링을 실행할 수 없습니다." in source
    assert "continuous_first_modeling_input_valid" in source
    assert "run_continuous_steady_state_extraction(" in source
    assert "run_continuous_first_modeling(" in source
    assert "Continuous 1차 모델링 command" in source
    assert "1cycle 반복 출력용 voltage LUT" in source
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
    source = EXPORT_UI.read_text(encoding="utf-8") + CONT_EXPORT_UI.read_text(encoding="utf-8")

    assert "continuous_loop_output" in source
    assert "loop_endpoint_policy" in source
    assert "sample_index, time_s, voltage_v" in source
    assert "Continuous 최종 전압 LUT 추출" in source
    assert "Continuous 1차 modeling command" in source
    assert "Continuous 2차 보정 command" in source
    assert "Continuous 2차 보정 command가 아직 생성되지 않았습니다." in source


def test_continuous_export_section_uses_namespaced_streamlit_keys() -> None:
    source = CONT_EXPORT_UI.read_text(encoding="utf-8")
    callers = APP_UI.read_text(encoding="utf-8") + CONT_UI.read_text(encoding="utf-8") + CONT_FIRST_UI.read_text(encoding="utf-8")

    assert "key_namespace" in source
    assert "continuous_final_lut_export_stage_selector_{key_namespace}" in source
    assert "download_continuous_final_lut_{key_namespace}_{selected['stage']}" in source
    assert callers.count("render_continuous_final_voltage_lut_export_section(") >= 3
    assert "key_namespace=\"quick_lut_first_modeling\"" in callers
    assert "key_namespace=\"continuous_actual_drive\"" in callers
    assert "key_namespace=\"quick_lut_compensation_result\"" in callers


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


def test_continuous_first_final_lut_export_uses_limited_voltage_and_loop_filename() -> None:
    command = pd.DataFrame(
        {
            "time_s": [0.0, 0.25, 0.5, 0.75, 1.0],
            "limited_voltage_v": [0.0, 1.0, 0.0, -1.0, 0.0],
            "continuous_loop_output": [True] * 5,
            "loop_endpoint_policy": ["period_exclusive"] * 5,
            "freq_hz": [1.0] * 5,
        }
    )
    result = {"command_profile": command, "metadata": {"continuous_result_stage": "first_model"}}

    record = continuous_result_export_record("first", result)
    exported, metadata = build_continuous_final_lut_frame(
        command,
        voltage_source_column=record["voltage_source_column"],
        freq_hz=1.0,
        stage="first",
    )
    filename = build_continuous_final_lut_filename(stage="first", waveform_type="sine", freq_hz=1.0)

    assert record["available"] is True
    assert record["voltage_source_column"] == "limited_voltage_v"
    assert exported.columns.tolist() == ["sample_index", "time_s", "voltage_v"]
    assert exported["time_s"].max() < 1.0
    assert np.allclose(exported["voltage_v"], [0.0, 1.0, 0.0, -1.0])
    assert metadata["continuous_final_lut_export_status"] == "ok"
    assert metadata["continuous_final_lut_export_selected_stage"] == "first"
    assert metadata["continuous_final_lut_export_voltage_source_column"] == "limited_voltage_v"
    assert filename == "continuous_first_voltage_lut_sine_1Hz_1cycle_loop.csv"


def test_continuous_second_final_lut_export_prefers_second_limited_voltage() -> None:
    command = pd.DataFrame(
        {
            "time_s": [0.0, 0.25, 0.5, 0.75],
            "limited_voltage_v": [0.0, 1.0, 0.0, -1.0],
            "second_limited_voltage_v": [0.1, 0.8, -0.1, -0.8],
            "continuous_loop_output": [True] * 4,
            "loop_endpoint_policy": ["period_exclusive"] * 4,
            "freq_hz": [1.0] * 4,
        }
    )
    result = {"command_profile": command, "metadata": {"continuous_result_stage": "second_model"}}

    record = continuous_result_export_record("second", result)
    exported, metadata = build_continuous_final_lut_frame(
        command,
        voltage_source_column=record["voltage_source_column"],
        freq_hz=1.0,
        stage="second",
    )
    filename = build_continuous_final_lut_filename(stage="second", waveform_type="sine", freq_hz=1.0)

    assert record["available"] is True
    assert record["voltage_source_column"] == "second_limited_voltage_v"
    assert np.allclose(exported["voltage_v"], command["second_limited_voltage_v"])
    assert metadata["continuous_final_lut_export_selected_stage"] == "second"
    assert filename == "continuous_second_voltage_lut_sine_1Hz_1cycle_loop.csv"


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


def test_upload_memory_continuous_candidate_discovery_accepts_metadata_preamble_csv() -> None:
    csv_bytes = (
        b"# Date,2026-04-28 13:46:05\n"
        b"# Frequency(Hz),1.000\n"
        b"# Amplitude(V),5.000\n"
        b"# Cycles,10.000\n"
        b"# Repeat,1.000\n"
        b"# PreDelay(s),1.000\n"
        b"# PostDelay(s),1.000\n"
        b"# HallSamples,90175\n"
        b"# CurrentSamples,6107527\n"
        b"# CommonRange(ms),0.00~12215.05 (span 12215.05)\n"
        b"Row,TimeMs,HallBx,HallBy,HallBz,Current1_A,Current2_A,Voltage1_V,Voltage2_V\n"
        b"0,0.0,0,0,1.0,0,0,0.0,0\n"
        b"1,10.0,0,0,-2.0,0,0,1.0,0\n"
        b"2,20.0,0,0,1.0,0,0,0.0,0\n"
    )

    names, candidates, scan = discover_continuous_candidate_frames(
        {},
        upload_payloads=[("continuous_sine_1Hz.csv", csv_bytes)],
        dataset_library_payloads=[],
    )

    assert names == ["upload_memory:continuous_sine_1Hz.csv"]
    assert scan["continuous_candidate_source_counts"]["upload_memory_continuous"] == 1
    assert scan["continuous_candidate_rejected_count"] == 0
    frame = candidates[names[0]]
    assert frame.attrs["continuous_source_freq_hz"] == 1.0
    assert frame.attrs["continuous_source_file"] == "continuous_sine_1Hz.csv"
    assert frame["time_s_abs"].tolist() == [0.0, 0.01, 0.02]
    assert np.allclose(frame["measured_field_effective_mT"], -frame["raw_hallbz_mT"])


def test_continuous_source_frequency_inferred_from_filename_when_preamble_missing() -> None:
    assert infer_continuous_source_frequency("continuous_sine_0.25Hz.csv")[0] == 0.25
    assert infer_continuous_source_frequency("continuous_sine_2Hz.csv")[0] == 2.0
    assert infer_continuous_source_frequency("continuous_rounded_triangle_3Hz.csv")[0] == 3.0
    assert infer_continuous_source_frequency("continuous_any_4Hz_extra.csv")[0] == 4.0

    csv_bytes = b"TimeMs,Voltage1_V,HallBz\n0,0,0\n10,1,-1\n20,0,0\n"
    names, candidates, scan = discover_continuous_candidate_frames(
        {},
        upload_payloads=[("continuous_sine_0.5Hz.csv", csv_bytes)],
        dataset_library_payloads=[],
        target_freq_hz=0.5,
    )

    assert names == ["upload_memory:continuous_sine_0.5Hz.csv"]
    frame = candidates[names[0]]
    assert frame.attrs["continuous_source_freq_hz"] == 0.5
    assert frame.attrs["continuous_source_freq_source"] == "filename"
    assert scan["continuous_candidate_details"][0]["frequency_match_status"] == "match"


def test_preamble_frequency_takes_priority_over_filename_frequency() -> None:
    csv_bytes = (
        b"# Frequency(Hz),5.000\n"
        b"TimeMs,Voltage1_V,HallBz\n0,0,0\n10,1,-1\n20,0,0\n"
    )

    names, candidates, _scan = discover_continuous_candidate_frames(
        {},
        upload_payloads=[("continuous_sine_4Hz.csv", csv_bytes)],
        dataset_library_payloads=[],
        target_freq_hz=5.0,
    )

    frame = candidates[names[0]]
    assert frame.attrs["continuous_source_freq_hz"] == 5.0
    assert frame.attrs["continuous_source_freq_source"] == "preamble"


def test_filename_frequency_takes_priority_over_generic_attrs() -> None:
    frame = pd.DataFrame({"TimeMs": [0, 10], "Voltage1_V": [0.0, 1.0], "HallBz": [0.0, -1.0]})
    frame.attrs["continuous_source_freq_hz"] = 9.0
    frame.attrs["continuous_source_freq_source"] = "frame_attrs"

    adapted = attach_continuous_frequency_attrs(frame, name="continuous_sine_1.5Hz.csv")

    assert adapted.attrs["continuous_source_freq_hz"] == 1.5
    assert adapted.attrs["continuous_source_freq_source"] == "filename"


def test_continuous_candidates_are_ranked_by_target_frequency_match() -> None:
    csv_bytes = b"TimeMs,Voltage1_V,HallBz\n0,0,0\n10,1,-1\n20,0,0\n"
    names, candidates, scan = discover_continuous_candidate_frames(
        {},
        upload_payloads=[
            ("continuous_sine_2Hz.csv", csv_bytes),
            ("continuous_sine_3Hz.csv", csv_bytes),
            ("continuous_sine_5Hz.csv", csv_bytes),
        ],
        dataset_library_payloads=[],
        target_freq_hz=3.0,
    )

    assert names[0] == "upload_memory:continuous_sine_3Hz.csv"
    assert scan["continuous_candidate_matching_count"] == 1
    assert scan["matching_candidate_count"] == 1
    assert scan["matching_candidate_names"] == ["upload_memory:continuous_sine_3Hz.csv"]
    assert scan["continuous_candidate_details"][0]["frequency_match_status"] == "match"
    ranked = rank_continuous_candidates_for_target(candidates, target_freq_hz=3.0)
    assert ranked[0]["name"] == "upload_memory:continuous_sine_3Hz.csv"
    assert ranked[0]["frequency_match_status"] == "match"
    assert ranked[1]["frequency_match_status"] == "mismatch"


def test_continuous_candidate_waveform_filter_defaults_to_triangle() -> None:
    csv_bytes = b"TimeMs,Voltage1_V,HallBz\n0,0,0\n10,1,-1\n20,0,0\n"
    names, candidates, scan = discover_continuous_candidate_frames(
        {},
        upload_payloads=[
            ("continuous_triangle_3Hz.csv", csv_bytes),
            ("continuous_sine_3Hz.csv", csv_bytes),
        ],
        dataset_library_payloads=[],
        target_freq_hz=3.0,
        source_waveform_filter="triangle",
    )

    assert names == ["upload_memory:continuous_triangle_3Hz.csv"]
    detail = scan["continuous_candidate_details"][0]
    assert detail["continuous_source_waveform_family"] == "triangle"
    assert "triangle" in detail["continuous_candidate_label"]
    assert candidates[names[0]].attrs["continuous_source_waveform_family"] == "triangle"


def test_continuous_candidate_waveform_filter_can_select_triangle() -> None:
    csv_bytes = b"TimeMs,Voltage1_V,HallBz\n0,0,0\n10,1,-1\n20,0,0\n"
    names, _candidates, scan = discover_continuous_candidate_frames(
        {},
        upload_payloads=[
            ("continuous_triangle_3Hz.csv", csv_bytes),
            ("continuous_sine_3Hz.csv", csv_bytes),
        ],
        dataset_library_payloads=[],
        target_freq_hz=3.0,
        source_waveform_filter="triangle",
    )

    assert names == ["upload_memory:continuous_triangle_3Hz.csv"]
    assert scan["continuous_source_waveform_filter"] == "triangle"
    assert scan["continuous_candidate_details"][0]["continuous_source_waveform_match_status"] == "match"


def test_mismatch_extraction_reports_source_target_error_details() -> None:
    csv_bytes = b"TimeMs,Voltage1_V,HallBz\n0,0,0\n10,1,-1\n20,0,0\n"
    names, candidates, _scan = discover_continuous_candidate_frames(
        {},
        upload_payloads=[("continuous_sine_2Hz.csv", csv_bytes)],
        dataset_library_payloads=[],
        target_freq_hz=3.0,
    )

    result = run_continuous_steady_state_extraction(
        selected_candidate_name=names[0],
        selected_frame=candidates[names[0]],
        waveform_type="sine",
        freq_hz=3.0,
    )

    metadata = result["extraction_result"]["metadata"]
    assert result["status"] == "error"
    assert result["error_reason"] == "unavailable_frequency_mismatch"
    assert metadata["frequency_match_status"] == "mismatch"
    assert metadata["source_freq_hz"] == 2.0
    assert metadata["target_freq_hz"] == 3.0
    assert metadata["frequency_error_pct"] > 30.0
    assert metadata["continuous_source_freq_source"] == "filename"


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
    assert "final_voltage_lut_not_measured_input" in scan["continuous_candidate_rejection_reasons"][0]


def test_continuous_candidate_discovery_reports_parse_and_schema_rejection_reasons() -> None:
    names, _candidates, scan = discover_continuous_candidate_frames(
        {},
        upload_payloads=[
            ("continuous_broken.csv", b"# Date,2026\n# no data header\n"),
            ("continuous_missing_hall.csv", b"time_s,Voltage1_V\n0,0\n"),
        ],
        dataset_library_payloads=[],
    )

    assert names == []
    assert scan["continuous_candidate_rejected_count"] == 2
    reasons = "\n".join(scan["continuous_candidate_rejection_reasons"])
    assert "csv_parse_error" in reasons
    assert "continuous_schema_missing" in reasons


def test_continuous_runtime_panel_has_schema_rejected_message_markers() -> None:
    source = CONT_UI.read_text(encoding="utf-8")

    for marker in [
        "schema rejected",
        "expected period_s",
        "selected duration_s",
        "duration ratio",
        "Continuous extraction summary",
        "Continuous 원본과 선택된 steady-state 구간",
        "Continuous runtime debug",
        "Continuous 파일은 찾았지만 schema 인식에 실패했습니다.",
        "Continuous source 파일은 발견되었지만 time/voltage/field 컬럼 매핑에 실패했습니다.",
        "continuous_candidate_rejection_reasons",
    ]:
        assert marker in source


def test_continuous_extraction_orchestrator_returns_renderable_result_bundle() -> None:
    period = 0.5
    time_s = np.linspace(0.0, period * 8, 640, endpoint=False)
    frame = pd.DataFrame(
        {
            "time_s": time_s,
            "Voltage1_V": 3.0 * np.sin(2.0 * np.pi * time_s / period),
            "HallBz": -(45.0 * np.sin(2.0 * np.pi * time_s / period - 0.05)),
        }
    )

    bundle = run_continuous_steady_state_extraction(
        selected_candidate_name="analysis_lookup:synthetic_2Hz",
        selected_frame=frame,
        waveform_type="sine",
        freq_hz=2.0,
    )

    assert bundle["status"] == "ok"
    case = bundle["extraction_result"]
    assert isinstance(case["steady_state_one_cycle_frame"], pd.DataFrame)
    assert not case["steady_state_one_cycle_frame"].empty
    assert isinstance(case["stability_metrics"], pd.DataFrame)
    assert not case["stability_metrics"].empty


def test_continuous_first_modeling_orchestrator_uses_phase_aligned_kernel() -> None:
    period = 0.5
    time_s = np.linspace(0.0, period * 8, 640, endpoint=False)
    frame = pd.DataFrame(
        {
            "time_s": time_s,
            "Voltage1_V": 3.0 * np.sin(2.0 * np.pi * time_s / period),
            "HallBz": -(42.0 * np.sin(2.0 * np.pi * time_s / period - 0.08)),
        }
    )
    extraction = run_continuous_steady_state_extraction(
        selected_candidate_name="analysis_lookup:synthetic_2Hz",
        selected_frame=frame,
        waveform_type="sine",
        freq_hz=2.0,
    )

    first = run_continuous_first_modeling(
        extraction_result=extraction["extraction_result"],
        waveform_type="sine",
        freq_hz=2.0,
    )

    assert first["status"] == "ok"
    command = first["command_profile"]
    metadata = first["first_model_metadata"]
    assert metadata["continuous_first_modeling_uses_phase_aligned_kernel"] is True
    assert metadata["continuous_first_modeling_tail_disabled"] is True
    assert metadata["continuous_loop_output"] is True
    assert "correction_delta_v" in command.columns
    assert "measured_field_smoothed_mT" in command.columns
    assert "measured_field_aligned_mT" in command.columns
    assert "residual_for_modeling_mT" in command.columns
    assert command["time_s"].max() < period


def test_continuous_runtime_markers_include_waveform_filter_and_modeling_plots() -> None:
    source = CONT_UI.read_text(encoding="utf-8") + CONT_FIRST_UI.read_text(encoding="utf-8")

    for marker in [
        "Continuous source waveform family",
        "continuous_source_waveform_filter",
        "[\"triangle\", \"sine\", \"rounded_triangle\", \"all\"]",
        "목표 자기장 개형: finite와 동일한 fixed rounded-triangle",
        "Continuous 1차 모델링 실행",
        "Phase alignment 확인",
        "목표 자기장 vs phase-aligned 실측 자기장",
        "Continuous 1차 modeling command",
    ]:
        assert marker in source


def test_continuous_first_modeling_orchestrator_rejects_empty_command_profile() -> None:
    result = run_continuous_first_modeling(
        extraction_result={"metadata": {"steady_state_extraction_status": "ok"}, "steady_state_one_cycle_frame": pd.DataFrame()},
        waveform_type="sine",
        freq_hz=2.0,
    )

    assert result["status"] == "error"
    assert result["error_reason"] == "extraction_result_empty"


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
            "measured_field_normalized_mT": 42.0 * np.sin(2.0 * np.pi * time_s / period),
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
