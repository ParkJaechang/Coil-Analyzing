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
        "Generate 2nd Modeled Voltage LUT",
    ]:
        assert marker in combined


def test_second_modeling_user_trigger_contract_is_visible() -> None:
    source = SECOND_UI.read_text(encoding="utf-8")
    assert "User-triggered only" in source
    assert "No automatic second correction is generated" in source
    assert "Raw peak values are informational only" in source
    assert "No automatic pass/fail judgement" in source


def test_loaded_lut_analysis_is_reused_until_load_analyze_is_pressed_again() -> None:
    source = APP_UI.read_text(encoding="utf-8")
    assert "quick_lut_analysis_result" in source
    assert "cached_analysis.get(\"payload_hash\") == active_payload_hash" in source
    assert "Loaded analysis result" in source


def test_actual_drive_feedback_review_is_button_gated_and_plotted() -> None:
    source = (SRC_ROOT / "field_analysis" / "ui_quick_lut_feedback.py").read_text(encoding="utf-8")
    assert "Load / Review Actual-drive Result" in source
    assert "quick_lut_actual_drive_review_result" in source
    assert "Intended vs Actual Comparison" in source
    assert "Raw Actual-drive Visualization" in source
    assert "effective field = -HallBz raw" in source
