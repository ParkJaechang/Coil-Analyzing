from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_UI_SNAPSHOT = REPO_ROOT / "src" / "field_analysis" / "app_ui_snapshot.py"


def _source() -> str:
    return APP_UI_SNAPSHOT.read_text(encoding="utf-8")


def test_support_reference_provenance_panel_markers_exist() -> None:
    source = _source()

    expected_markers = [
        "참조 데이터 출처 / Debug",
        "원본 선택 support source",
        "Target timebase 정렬 support reference",
        "Override / matching 사유",
        "raw selected support는 업로드/라이브러리 원본 record입니다.",
        "target-aligned support reference는 target timebase에 맞춘 비교 trace이며",
        "_render_support_reference_provenance_panel(compensation, command_profile)",
    ]
    missing = [marker for marker in expected_markers if marker not in source]

    assert not missing, f"Missing support provenance UI markers: {missing}"


def test_support_reference_provenance_payload_keys_are_used() -> None:
    source = _source()

    expected_keys = [
        "selected_support_id",
        "selected_support_family",
        "selected_support_source_file",
        "selected_support_freq_hz",
        "selected_support_cycle_count",
        "selected_support_original_duration_s",
        "selected_support_original_pp_mT",
        "support_reference_plotted_column",
        "support_reference_alignment_status",
        "support_reference_pp",
        "support_reference_duration_s",
        "support_reference_timebase",
        "requested_support_family",
        "support_family_requested",
        "support_family_override_applied",
        "support_family_override_reason",
        "support_cycle_match_type",
        "support_cycle_match_reason",
    ]
    missing = [key for key in expected_keys if key not in source]

    assert not missing, f"Missing support provenance payload keys: {missing}"


def test_support_reference_provenance_explains_requested_vs_selected_split() -> None:
    source = _source()

    assert "요청 support family:" in source
    assert "선택 support family:" in source
    assert "요청 cycle:" in source
    assert "선택 support cycle:" in source
    assert "물리 목표 자기장이 아닙니다" in source


def test_command_prediction_consistency_status_card_markers_exist() -> None:
    source = _source()

    expected_markers = [
        "전압 예측 일관성 / Debug",
        "command 기준 target: fixed rounded triangle",
        "support reference 역할: 진단용",
        "support reference가 command에 사용됨:",
        "예측 field source:",
        "표시 command 기준 예측 여부:",
        "전압 예측 일관성:",
        "support reference shape mismatch:",
        "support/target 상관:",
        "support/target NRMSE:",
        "Command/prediction consistency metadata unavailable",
        "support reference는 명령 목표가 아니라 선택된 support의 비교/진단용 trace입니다.",
        "추천 전압은 fixed rounded triangle target 기준으로 계산됩니다.",
        "predicted field는 표시된 command 기준 forward prediction입니다.",
        "_render_command_prediction_consistency_card(compensation, command_profile)",
    ]
    missing = [marker for marker in expected_markers if marker not in source]

    assert not missing, f"Missing command prediction consistency UI markers: {missing}"


def test_command_prediction_consistency_payload_keys_are_used() -> None:
    source = _source()

    expected_keys = [
        "command_generation_target",
        "support_reference_used_for_command",
        "support_reference_role",
        "forward_prediction_source",
        "predicted_from_plotted_command",
        "command_prediction_consistency_status",
        "support_reference_shape_mismatch",
        "support_reference_target_corr",
        "support_reference_target_nrmse",
        "command_nonzero_start_s",
        "target_nonzero_start_s",
        "command_covers_target_active_start",
        "command_covers_target_active_end",
    ]
    missing = [key for key in expected_keys if key not in source]

    assert not missing, f"Missing command prediction consistency payload keys: {missing}"
