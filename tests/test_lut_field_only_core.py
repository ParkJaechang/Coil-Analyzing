from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.lut import FIELD_ONLY_FIXED_TARGET_PP, recommend_voltage_waveform


def _analysis_stub(*, test_id: str, waveform_type: str, freq_hz: float, voltage_values: np.ndarray) -> SimpleNamespace:
    grid = np.linspace(0.0, 1.0, len(voltage_values))
    period_s = 1.0 / freq_hz
    cycle_frame = pd.DataFrame(
        {
            "test_id": test_id,
            "waveform_type": waveform_type,
            "freq_hz": freq_hz,
            "cycle_index": 0,
            "cycle_progress": grid,
            "cycle_time_s": grid * period_s,
            "time_s": grid * period_s,
            "daq_input_v": voltage_values,
        }
    )
    return SimpleNamespace(cycle_detection=SimpleNamespace(annotated_frame=cycle_frame))


def _support_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "test_id": "low_a",
                "waveform_type": "sine",
                "freq_hz": 1.0,
                "daq_input_v_pp_mean": 4.0,
                "achieved_bz_mT_pp_mean": 100.0,
                "achieved_current_pp_a_mean": 12.0,
            },
            {
                "test_id": "low_b",
                "waveform_type": "sine",
                "freq_hz": 1.2,
                "daq_input_v_pp_mean": 5.0,
                "achieved_bz_mT_pp_mean": 100.0,
                "achieved_current_pp_a_mean": 18.0,
            },
        ]
    )


def _analysis_lookup() -> dict[str, SimpleNamespace]:
    grid = np.linspace(0.0, 1.0, 300)
    sine = np.sin(2.0 * np.pi * grid)
    return {
        "low_a": _analysis_stub(
            test_id="low_a",
            waveform_type="sine",
            freq_hz=1.0,
            voltage_values=sine,
        ),
        "low_b": _analysis_stub(
            test_id="low_b",
            waveform_type="sine",
            freq_hz=1.2,
            voltage_values=-sine,
        ),
    }


def _normalized_waveform(frame: pd.DataFrame) -> np.ndarray:
    values = pd.to_numeric(frame["recommended_voltage_v"], errors="coerce").to_numpy(dtype=float)
    values = values - float(np.nanmean(values))
    scale = float(np.nanmax(np.abs(values)))
    return values / scale if scale > 0 else values


def test_field_only_lut_ignores_current_metric_and_uses_fixed_100pp() -> None:
    recommendation = recommend_voltage_waveform(
        per_test_summary=_support_summary(),
        analyses_by_test_id=_analysis_lookup(),
        waveform_type="sine",
        freq_hz=1.1,
        target_metric="achieved_current_pp_a_mean",
        target_value=25.0,
        frequency_mode="interpolate",
    )

    assert recommendation is not None
    assert recommendation["target_metric"] == "achieved_bz_mT_pp_mean"
    assert recommendation["requested_target_value"] == FIELD_ONLY_FIXED_TARGET_PP
    assert recommendation["field_model_route"].startswith("field_only_")


def test_low_frequency_shape_blending_keeps_positive_shape_continuity() -> None:
    recommendation = recommend_voltage_waveform(
        per_test_summary=_support_summary(),
        analyses_by_test_id=_analysis_lookup(),
        waveform_type="sine",
        freq_hz=1.1,
        target_metric="achieved_bz_mT_pp_mean",
        target_value=100.0,
        frequency_mode="interpolate",
    )

    assert recommendation is not None
    template = recommendation["template_waveform"]["voltage_normalized"].to_numpy(dtype=float)
    reference = np.sin(2.0 * np.pi * recommendation["template_waveform"]["cycle_progress"].to_numpy(dtype=float))
    corr = float(np.corrcoef(template, reference)[0, 1])

    assert recommendation["freq_regularization_applied"] is True
    assert recommendation["field_shape_corr"] > 0.9
    assert recommendation["field_shape_nrmse"] < 0.5
    assert corr > 0.8


def test_small_frequency_changes_keep_normalized_shape_stable() -> None:
    low = recommend_voltage_waveform(
        per_test_summary=_support_summary(),
        analyses_by_test_id=_analysis_lookup(),
        waveform_type="sine",
        freq_hz=1.05,
        target_metric="achieved_bz_mT_pp_mean",
        target_value=100.0,
        frequency_mode="interpolate",
    )
    high = recommend_voltage_waveform(
        per_test_summary=_support_summary(),
        analyses_by_test_id=_analysis_lookup(),
        waveform_type="sine",
        freq_hz=1.15,
        target_metric="achieved_bz_mT_pp_mean",
        target_value=100.0,
        frequency_mode="interpolate",
    )

    assert low is not None and high is not None
    corr = float(np.corrcoef(_normalized_waveform(low["command_waveform"]), _normalized_waveform(high["command_waveform"]))[0, 1])
    assert corr > 0.95


def test_target_scale_changes_do_not_change_normalized_shape() -> None:
    low_target = recommend_voltage_waveform(
        per_test_summary=_support_summary(),
        analyses_by_test_id=_analysis_lookup(),
        waveform_type="sine",
        freq_hz=1.1,
        target_metric="achieved_bz_mT_pp_mean",
        target_value=50.0,
        frequency_mode="interpolate",
    )
    high_target = recommend_voltage_waveform(
        per_test_summary=_support_summary(),
        analyses_by_test_id=_analysis_lookup(),
        waveform_type="sine",
        freq_hz=1.1,
        target_metric="achieved_bz_mT_pp_mean",
        target_value=180.0,
        frequency_mode="interpolate",
    )

    assert low_target is not None and high_target is not None
    assert low_target["requested_target_value"] == FIELD_ONLY_FIXED_TARGET_PP
    assert high_target["requested_target_value"] == FIELD_ONLY_FIXED_TARGET_PP
    assert np.allclose(
        _normalized_waveform(low_target["command_waveform"]),
        _normalized_waveform(high_target["command_waveform"]),
        atol=1e-6,
    )
