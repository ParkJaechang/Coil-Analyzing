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
    assert {"field": "unsupported cycles", "value": "1.25, 1.75"} in rows
    assert {"field": "unsupported reason", "value": "unsupported_cycle_phase_delay"} in rows
    assert feedback_export_source_column(
        pd.DataFrame(
            {
                "limited_voltage_v": [0.0],
                "feedback_corrected_limited_voltage_v": [1.0],
                "feedback_correction_status": ["ok"],
                "feedback_correction_available": [True],
            }
        )
    ) == "feedback_corrected_limited_voltage_v"


def test_feedback_export_source_falls_back_to_baseline_when_unavailable() -> None:
    from field_analysis.ui_quick_lut_feedback import feedback_export_source_column

    assert (
        feedback_export_source_column(
            pd.DataFrame(
                {
                    "limited_voltage_v": [0.0],
                    "feedback_corrected_limited_voltage_v": [1.0],
                    "feedback_correction_status": ["unsupported_cycle_phase_delay"],
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
        "Quick LUT feedback correction",
        "실제 구동 결과를 이용해 전압 command만 보정합니다.",
        "finite_actual_feedback_peak_correction",
        "actual-drive result files",
        "cached feedback files",
        "first_run",
        "second_run",
        "unknown",
        "HallBz sign applied",
        "field peak normalized to ±50mT",
        "voltage normalized/limited to ±5V",
        "Feedback correction delta",
        "Feedback corrected limited voltage",
        "exported_voltage_source_column",
        "unsupported_cycle_phase_delay",
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
            "feedback_correction_delta_v": [0.0, 0.5],
            "feedback_corrected_limited_voltage_v": [0.0, 4.5],
        }
    )
    frame = build_feedback_plot_frame(profile)

    assert list(frame["time_s"]) == [0.0, 0.1]
    assert "feedback_corrected_predicted_field_mT" not in frame.columns
    assert np.allclose(frame["Residual"], [0.0, 10.0])
