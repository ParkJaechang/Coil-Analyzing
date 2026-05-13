from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

APP_UI = SRC_ROOT / "field_analysis" / "app_ui_snapshot.py"
RAW_UI = SRC_ROOT / "field_analysis" / "ui_raw_waveforms.py"
LUT_UI = SRC_ROOT / "field_analysis" / "ui_voltage_lut_review.py"
ACTUAL_UI = SRC_ROOT / "field_analysis" / "ui_finite_actual_drive_review.py"
SECOND_UI = SRC_ROOT / "field_analysis" / "ui_second_modeling.py"


def test_quick_lut_primary_nav_is_simple_and_debug_tabs_are_hidden() -> None:
    source = APP_UI.read_text(encoding="utf-8")
    assert 'options=["Quick LUT", "Raw Waveforms", "LUT Review", "Data / Cache Status"]' in source
    assert "Advanced / Debug" in source
    assert "Show Advanced / Debug tabs" in source
    assert "Run Readiness" in source
    assert "Field Model Diagnostics" in source


def test_heavy_app_paths_have_explicit_buttons() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in [APP_UI, RAW_UI, LUT_UI, ACTUAL_UI, SECOND_UI])
    for marker in [
        "Load / Analyze LUT Data",
        "Apply Raw Waveform Selection",
        "Render Raw Waveform Plot",
        "Load LUT CSV",
        "Render LUT Plot",
        "Review Actual-drive Result",
        "2차 보정 command 생성",
    ]:
        assert marker in combined


def test_second_modeling_user_trigger_contract_is_visible() -> None:
    source = SECOND_UI.read_text(encoding="utf-8")
    assert "사용자가 버튼을 눌렀을 때만 생성합니다" in source
    assert "업로드나 옵션 변경만으로 2차 보정을 자동 생성하지 않습니다" in source
    assert "Raw peak 값은 참고용입니다" in source
    assert "자동 합격/불합격 판정은 하지 않습니다" in source


def test_quick_lut_runtime_preserves_first_model_snapshot_for_startup_review() -> None:
    source = APP_UI.read_text(encoding="utf-8")

    assert "first_command_profile = command_profile.copy(deep=True)" in source
    assert 'st.session_state["quick_lut_first_model_result"]' in source
    assert 'plot_command_waveform(first_command_profile, value_column="limited_voltage_v")' in source
    assert "render_startup_compensation_review(compensation, first_command_profile)" in source
    assert "command_profile=first_command_profile" in source
    assert "#### 2. 1차 모델링 command" in source
    assert "#### 1. LUT 데이터 준비" in source
    assert "이 전압은 실제 장비에 처음 넣는 1차 command입니다." in source


def test_loaded_lut_analysis_is_reused_until_load_analyze_is_pressed_again() -> None:
    source = APP_UI.read_text(encoding="utf-8")
    assert "quick_lut_analysis_result" in source
    assert "cached_analysis.get(\"payload_hash\") == active_payload_hash" in source
    assert "분석 결과 로드됨" in source


def test_actual_drive_feedback_review_is_button_gated_and_plotted() -> None:
    source = (SRC_ROOT / "field_analysis" / "ui_quick_lut_feedback.py").read_text(encoding="utf-8")
    assert "실구동 결과 검토" in source
    assert "quick_lut_actual_drive_review_result" in source
    assert "목표 자기장 vs 실측 자기장" in source
    assert "1차 실구동 데이터 원본 확인" in source
    assert "부호 보정 자기장 (-HallBz)" in source
