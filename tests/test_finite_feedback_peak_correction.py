from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.final_modeled_lut import build_final_modeled_voltage_lut_export
from field_analysis.finite_feedback_peak_correction import apply_finite_feedback_peak_correction


def _write_feedback_csv(path: Path, *, freq_hz: float = 1.0, cycle_count: float = 1.0) -> None:
    duration = cycle_count / freq_hz
    time_s = np.linspace(-0.2, duration + 0.2, 301)
    active = (time_s >= 0.0) & (time_s <= duration)
    phase = 2.0 * np.pi * freq_hz * time_s
    voltage = np.where(active, 6.0 * np.sin(phase), 0.0)
    # HallBz is intentionally opposite sign; feedback helper must preserve raw and use signed field.
    signed_field = np.where(active, np.where(np.sin(phase) >= 0.0, 35.0 * np.sin(phase), 75.0 * np.sin(phase)), 0.0)
    hallbz = -signed_field
    rows = ["# PreDelay(s),1.000", "# PostDelay(s),1.000", "# AutoSyncHallLag,applied 0.0ms"]
    rows.extend([f"# meta{i},x" for i in range(10)])
    rows.append("Row,TimeMs,Voltage1_V,HallBz,Current1_A")
    for index, (t_s, v, b) in enumerate(zip(time_s + 1.0, voltage, hallbz)):
        rows.append(f"{index},{t_s * 1000.0:.6f},{v:.9f},{b:.9f},{v / 2.0:.9f}")
    path.write_text("\n".join(rows), encoding="utf-8")


def _command_profile(*, freq_hz: float = 1.0, cycle_count: float = 1.0) -> pd.DataFrame:
    duration = cycle_count / freq_hz
    time_s = np.linspace(0.0, duration, 301)
    phase = 2.0 * np.pi * freq_hz * time_s
    target = 50.0 * np.sin(phase)
    baseline = 4.0 * np.sin(phase)
    return pd.DataFrame(
        {
            "time_s": time_s,
            "physical_target_output_mT": target,
            "recommended_voltage_v": baseline,
            "limited_voltage_v": baseline,
            "predicted_field_mT": target * 0.25,
        }
    )


def test_quick_lut_finite_feedback_peak_correction_supports_one_cycle_only(tmp_path: Path) -> None:
    feedback_path = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_feedback_csv(feedback_path, cycle_count=1.0)
    profile = _command_profile(cycle_count=1.0)
    original_target = profile["physical_target_output_mT"].copy()

    corrected, metadata = apply_finite_feedback_peak_correction(
        profile,
        feedback_path,
        waveform_type="sine",
        freq_hz=1.0,
        cycle_count=1.0,
        forward_model=lambda time_s, voltage_v: voltage_v * 10.0,
    )

    assert metadata["feedback_correction_available"] is True
    assert metadata["feedback_correction_status"] == "ok"
    assert metadata["feedback_used_for_correction"] is True
    assert metadata["feedback_route"] == "finite_feedback_symmetric_peak_correction"
    assert metadata["hallbz_sign_applied"] is True
    assert metadata["field_normalization_mode"] == "peak_to_50mT"
    assert metadata["voltage_normalization_mode"] == "peak_to_5V_or_limit"
    assert metadata["target_unchanged"] is True
    assert np.allclose(corrected["physical_target_output_mT"], original_target)
    assert np.nanmax(np.abs(corrected["measured_field_normalized_mT"])) == pytest.approx(50.0, abs=1e-6)
    assert np.nanmax(np.abs(corrected["feedback_corrected_limited_voltage_v"])) <= 5.0 + 1e-9
    assert np.allclose(corrected["active_limited_voltage_v"], corrected["feedback_corrected_limited_voltage_v"])
    assert np.allclose(corrected["limited_voltage_v"], corrected["active_limited_voltage_v"])
    assert np.allclose(corrected["baseline_limited_voltage_v"], profile["limited_voltage_v"])
    assert corrected["active_command_source"].iloc[0] == "feedback_corrected_limited_voltage_v"
    assert corrected["plotted_command_source"].iloc[0] == corrected["active_command_source"].iloc[0]
    assert corrected["exported_voltage_source_column"].iloc[0] == corrected["active_command_source"].iloc[0]
    assert corrected["run_waveform_voltage_source"].iloc[0] == corrected["active_command_source"].iloc[0]
    assert metadata["active_command_source"] == "feedback_corrected_limited_voltage_v"
    assert metadata["plotted_command_source"] == "feedback_corrected_limited_voltage_v"
    assert metadata["exported_voltage_source_column"] == "feedback_corrected_limited_voltage_v"
    assert metadata["run_waveform_voltage_source"] == "feedback_corrected_limited_voltage_v"
    assert metadata["correction_method"] == "residual_proportional_feedback"
    assert metadata["production_supported_cycles"] == [1.0]
    assert metadata["reference_supported_cycles"] == []
    assert metadata["unsupported_cycles"] == [1.25, 1.5, 1.75, 2.0]
    assert metadata["production_cycle_policy"] == "1cycle_only"
    assert metadata["cycle_policy"] == "production_supported"
    assert np.nanmax(np.abs(corrected["feedback_correction_delta_v"])) == pytest.approx(metadata["correction_delta_peak_v"])
    assert corrected["feedback_corrected_limited_voltage_v"].equals(corrected["feedback_corrected_recommended_voltage_v"]) or metadata[
        "voltage_limit_status"
    ] in {"ok", "clamped"}
    assert {"positive_lobe_mask", "negative_lobe_mask"}.issubset(corrected.columns)
    assert np.isfinite(metadata["positive_peak_error_before_mT"])
    assert np.isfinite(metadata["negative_peak_error_before_mT"])
    assert np.isfinite(metadata["peak_symmetry_error_before_mT"])
    assert "feedback_corrected_predicted_field_mT" in corrected.columns
    assert "displayed_predicted_field_mT" in corrected.columns
    assert np.allclose(corrected["displayed_predicted_field_mT"], corrected["feedback_corrected_predicted_field_mT"])
    assert metadata["predicted_from_plotted_command"] is True
    assert metadata["displayed_predicted_valid"] is True


@pytest.mark.parametrize("cycle_count", [1.25, 1.5, 1.75, 2.0])
def test_feedback_peak_correction_rejects_non_one_cycle_production_cycles(tmp_path: Path, cycle_count: float) -> None:
    feedback_path = tmp_path / f"finite_recommended_voltage_lut_sine_1Hz_{cycle_count:g}cycle_result.csv"
    _write_feedback_csv(feedback_path, cycle_count=cycle_count)
    profile = _command_profile(cycle_count=cycle_count)

    corrected, metadata = apply_finite_feedback_peak_correction(
        profile,
        feedback_path,
        waveform_type="sine",
        freq_hz=1.0,
        cycle_count=cycle_count,
    )

    assert corrected is not profile
    assert metadata["feedback_correction_available"] is False
    assert metadata["feedback_correction_status"] == "unsupported_cycle_policy_1cycle_only"
    assert metadata["feedback_correction_unavailable_reason"] == "non_1cycle_production_disabled"
    assert metadata["feedback_used_for_correction"] is False
    assert metadata["target_unchanged"] is True
    assert metadata["suggested_replacement_cycle"] == 1.0
    assert metadata["production_supported_cycles"] == [1.0]
    assert metadata["reference_supported_cycles"] == []
    assert metadata["unsupported_cycles"] == [1.25, 1.5, 1.75, 2.0]
    assert metadata["production_cycle_policy"] == "1cycle_only"
    assert "feedback_correction_delta_v" not in corrected.columns
    assert "feedback_corrected_limited_voltage_v" not in corrected.columns
    assert "second_voltage_v" not in corrected.columns
    assert "second_lut" not in corrected.columns
    assert np.allclose(corrected["limited_voltage_v"], profile["limited_voltage_v"])


def test_feedback_correction_preserves_hallbz_raw_signed_and_normalized_arrays(tmp_path: Path) -> None:
    feedback_path = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_feedback_csv(feedback_path)

    corrected, metadata = apply_finite_feedback_peak_correction(
        _command_profile(),
        feedback_path,
        waveform_type="sine",
        freq_hz=1.0,
        cycle_count=1.0,
    )

    assert {"measured_field_raw_mT", "measured_field_signed_mT", "measured_field_normalized_mT"}.issubset(corrected.columns)
    assert np.nanmax(np.abs(corrected["measured_field_raw_mT"])) == pytest.approx(metadata["raw_field_peak_mT"], abs=1e-6)
    assert np.allclose(corrected["measured_field_signed_mT"], -corrected["measured_field_raw_mT"], equal_nan=True)
    assert np.nanmax(np.abs(corrected["measured_field_normalized_mT"])) == pytest.approx(50.0, abs=1e-6)
    assert metadata["hallbz_effective_convention"] == "effective_field_mT=-HallBz_raw"
    assert metadata["raw_peak_values_informational_only"] is True
    assert metadata["feedback_schema_status"] == "ok"
    assert metadata["feedback_alignment_status"] == "ok"


def test_final_lut_export_uses_feedback_corrected_voltage_when_valid(tmp_path: Path) -> None:
    feedback_path = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_feedback_csv(feedback_path)
    corrected, _metadata = apply_finite_feedback_peak_correction(
        _command_profile(cycle_count=1.0),
        feedback_path,
        waveform_type="sine",
        freq_hz=1.0,
        cycle_count=1.0,
    )

    payload = build_final_modeled_voltage_lut_export(corrected, freq_hz=1.0, cycle_count=1.0, waveform="sine")

    active_source = corrected["active_command_source"].iloc[0]
    assert payload["metadata"]["voltage_source_column"] == active_source
    assert payload["metadata"]["exported_voltage_source_column"] == active_source
    assert np.allclose(payload["frame"]["voltage_v"], corrected[active_source])
    assert "correction_delta_v" not in payload["frame"].columns
    assert "second_voltage_v" not in payload["frame"].columns
    assert payload["metadata"]["fourier_resynthesis_involved"] is False
    assert payload["metadata"]["harmonic_export_involved"] is False


def test_feedback_without_forward_model_marks_displayed_prediction_invalid(tmp_path: Path) -> None:
    feedback_path = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_feedback_csv(feedback_path)
    profile = _command_profile()

    corrected, metadata = apply_finite_feedback_peak_correction(
        profile,
        feedback_path,
        waveform_type="sine",
        freq_hz=1.0,
        cycle_count=1.0,
    )

    assert metadata["feedback_correction_status"] == "ok"
    assert metadata["active_command_source"] == "feedback_corrected_limited_voltage_v"
    assert metadata["displayed_predicted_valid"] is False
    assert metadata["predicted_from_plotted_command"] is False
    assert metadata["command_prediction_consistency_status"] == "forward_prediction_unavailable_for_feedback_corrected_command"
    assert metadata["plotted_predicted_source"] == "unavailable"
    assert "displayed_predicted_field_mT" not in corrected.columns
    assert "feedback_corrected_predicted_field_mT" not in corrected.columns
    assert "predicted_field_mT" in corrected.columns


def test_startup_offset_diagnostic_metadata_for_offset_like_finite_response(tmp_path: Path) -> None:
    feedback_path = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_feedback_csv(feedback_path, cycle_count=1.0)
    lines = feedback_path.read_text(encoding="utf-8").splitlines()
    header_index = next(i for i, line in enumerate(lines) if line.startswith("Row,"))
    adjusted = lines[: header_index + 1]
    for row in lines[header_index + 1 :]:
        parts = row.split(",")
        time_s = float(parts[1]) / 1000.0 - 1.0
        if 0.0 <= time_s <= 0.35:
            parts[3] = f"{float(parts[3]) - 25.0:.9f}"
        adjusted.append(",".join(parts))
    feedback_path.write_text("\n".join(adjusted), encoding="utf-8")

    _corrected, metadata = apply_finite_feedback_peak_correction(
        _command_profile(cycle_count=1.0),
        feedback_path,
        waveform_type="sine",
        freq_hz=1.0,
        cycle_count=1.0,
    )

    assert metadata["startup_offset_status"] == "ok"
    assert metadata["startup_window_start_s"] == 0.0
    assert metadata["startup_window_end_s"] > 0.0
    assert np.isfinite(metadata["startup_offset_mT"])
    assert np.isfinite(metadata["startup_offset_normalized_mT"])
    assert np.isfinite(metadata["startup_offset_ratio"])
    assert np.isfinite(metadata["startup_decay_slope"])


def test_final_lut_export_falls_back_to_limited_voltage_without_feedback() -> None:
    profile = _command_profile()

    payload = build_final_modeled_voltage_lut_export(profile, freq_hz=1.0, cycle_count=1.0, waveform="sine")

    assert payload["metadata"]["voltage_source_column"] == "limited_voltage_v"
    assert payload["metadata"]["exported_voltage_source_column"] == "limited_voltage_v"
    assert np.allclose(payload["frame"]["voltage_v"], profile["limited_voltage_v"])
