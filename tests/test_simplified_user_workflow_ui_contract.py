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
        "Field review/modeling is normalized to ±50mT.",
        "Command voltage is normalized/limited to ±5V.",
        "DCAMP gain is handled outside this app.",
        "HallBz sign convention applied: effective field = -HallBz raw.",
        "Final LUT uses plotted final voltage samples, not Fourier resynthesis.",
        "Production finite correction is 1.0 cycle only.",
        "1.25 / 1.5 / 1.75 / 2.0 are Raw Review only",
        "2-cycle policy discarded.",
    ]
    missing = [item for item in expected if item not in source]
    assert not missing, f"Missing simplified workflow policy copy: {missing}"


def test_debug_heavy_panels_are_advanced_only() -> None:
    source = APP_UI.read_text(encoding="utf-8")

    assert "render_startup_compensation_review" in source
    assert "_render_support_reference_provenance_panel" in source
    assert "render_recommendation_export_panel" in source


def test_final_lut_copy_mentions_plotted_samples_and_required_schema() -> None:
    source = LUT_UI.read_text(encoding="utf-8")

    assert "exported CSV uses plotted final command voltage samples" in source
    assert "not Fourier" in source
    assert "not harmonic resynthesis" in source
    assert "columns: sample_index, time_s, voltage_v" in source


def test_no_mojibake_in_new_policy_sources() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in [RAW_UI, LUT_UI, FEEDBACK_UI])
    forbidden = [chr(0xFFFD), chr(0xF9E4), chr(0xC4D2), "?" + chr(0xAFF0) + chr(0xC0AC), "ì", "í", "ë", "ê"]
    found = [pattern for pattern in forbidden if pattern in combined]
    assert not found, f"Mojibake patterns found: {found}"
