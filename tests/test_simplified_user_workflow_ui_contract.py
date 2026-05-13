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
FEEDBACK_UI = SRC_ROOT / "field_analysis" / "ui_quick_lut_feedback.py"


def test_quick_mode_primary_navigation_hides_debug_sections() -> None:
    source = APP_UI.read_text(encoding="utf-8")

    assert '"Quick LUT"' in source
    assert '"Raw Waveforms"' in source
    assert '"LUT Review"' in source
    assert '"Data Import"' in source
    assert '"Export"' in source


def test_core_policy_wording_is_visible_in_user_facing_ui() -> None:
    source = "\n".join(
        [
            APP_UI.read_text(encoding="utf-8"),
            RAW_UI.read_text(encoding="utf-8"),
            LUT_UI.read_text(encoding="utf-8"),
            FEEDBACK_UI.read_text(encoding="utf-8"),
        ]
    )

    expected = [
        "Field review/modeling은 ±50mT 기준으로 정규화합니다.",
        "Command voltage는 ±5V 기준으로 정규화/제한합니다.",
        "DCAMP gain은 앱 밖에서 조절합니다.",
        "HallBz convention: effective field = -HallBz raw.",
        "최종 LUT는 화면에 표시된 최종 전압 샘플을 그대로 저장하며 Fourier 재합성을 사용하지 않습니다.",
        "Production finite 보정은 1.0 / 1.5 cycle을 지원합니다.",
        "1.25 / 1.75 / 2.0 cycle은 검토용이며 production 보정/내보내기 대상이 아닙니다.",
        "2-cycle production 정책은 폐기되었습니다.",
    ]
    missing = [item for item in expected if item not in source]
    assert not missing, f"Missing simplified workflow policy copy: {missing}"


def test_debug_heavy_panels_are_advanced_only() -> None:
    source = APP_UI.read_text(encoding="utf-8")

    assert "render_startup_compensation_review" in source
    assert "_render_support_reference_provenance_panel" in source
    assert "render_recommendation_export_panel" in source


def test_final_lut_copy_mentions_plotted_samples_and_required_schema() -> None:
    source = "\n".join(
        [
            LUT_UI.read_text(encoding="utf-8"),
            (SRC_ROOT / "field_analysis" / "ui_final_voltage_lut_export.py").read_text(encoding="utf-8"),
        ]
    )

    assert "최종 LUT는 화면에 표시된 최종 전압 샘플을 그대로 저장합니다." in source
    assert "Fourier 재합성이나 harmonic coefficient export가 아닙니다." in source
    assert "저장 컬럼은 sample_index, time_s, voltage_v 세 개뿐입니다." in source


def test_no_mojibake_in_new_policy_sources() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in [RAW_UI, LUT_UI, FEEDBACK_UI])
    forbidden = [chr(0xFFFD), chr(0xF9E4), chr(0xC4D2), "?" + chr(0xAFF0) + chr(0xC0AC), "ì", "í", "ë", "ê"]
    found = [pattern for pattern in forbidden if pattern in combined]
    assert not found, f"Mojibake patterns found: {found}"
