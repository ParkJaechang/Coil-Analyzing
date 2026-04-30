from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_UI_SNAPSHOT = REPO_ROOT / "src" / "field_analysis" / "app_ui_snapshot.py"


def _source() -> str:
    return APP_UI_SNAPSHOT.read_text(encoding="utf-8")


def test_support_reference_provenance_panel_markers_exist() -> None:
    source = _source()

    expected_markers = [
        "Support Reference Provenance",
        "Raw Selected Support Source",
        "Target-aligned Support Reference",
        "Override / Match Reason",
        "Raw selected support is the original uploaded/support record.",
        "Target-aligned support reference is the plotted support trace aligned to the target timebase.",
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

    assert "Requested support family:" in source
    assert "Selected support family:" in source
    assert "Requested cycle:" in source
    assert "Selected support cycle:" in source
    assert "support reference is not the physical target" in source.lower()
