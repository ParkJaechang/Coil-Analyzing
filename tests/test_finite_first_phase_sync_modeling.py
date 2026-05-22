from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.finite_first_phase_sync import apply_finite_first_phase_sync_modeling

APP_UI = SRC_ROOT / "field_analysis" / "app_ui_snapshot.py"
CONFIG = SRC_ROOT / "field_analysis" / "quick_lut_target_config.py"
UI_FINITE_FIRST = SRC_ROOT / "field_analysis" / "ui_finite_first_phase_sync.py"


def _finite_profile(*, freq_hz: float = 1.0, cycle_count: float = 1.0, delay_s: float = 0.08, include_support: bool = True) -> pd.DataFrame:
    duration = cycle_count / freq_hz
    support_duration = duration + (delay_s + 0.05 if include_support else 0.0)
    time_s = np.linspace(0.0, support_duration, 320, endpoint=False)
    phase = 2.0 * np.pi * freq_hz * time_s
    target = 50.0 * np.sin(phase)
    measured = 42.0 * np.sin(2.0 * np.pi * freq_hz * (time_s - delay_s))
    voltage = 2.5 * np.sin(phase)
    return pd.DataFrame(
        {
            "time_s": time_s,
            "limited_voltage_v": voltage,
            "recommended_voltage_v": voltage,
            "physical_target_output_mT": target,
            "finite_first_actual_measured_field_mT": measured,
            "finite_first_measured_source_file": "finite_tri_1Hz_1cycle.csv",
            "freq_hz": freq_hz,
            "target_cycle_count": cycle_count,
        }
    )


def test_finite_first_phase_sync_kernel_adds_aligned_residual_columns() -> None:
    result, metadata = apply_finite_first_phase_sync_modeling(_finite_profile(), freq_hz=1.0, cycle_count=1.0)

    assert metadata["finite_first_modeling_mode"] == "phase_synced"
    assert metadata["finite_first_modeling_phase_sync_enabled"] is True
    assert metadata["finite_first_modeling_kernel"] == "shared_phase_aligned"
    assert metadata["finite_first_modeling_legacy_delay_preserving"] is False
    assert metadata["finite_first_measured_source_is_actual_measured"] is True
    assert metadata["finite_first_measured_source_column"] == "finite_first_actual_measured_field_mT"
    assert metadata["finite_first_uses_support_reference_as_measured"] is False
    assert metadata["finite_first_uses_target_as_measured"] is False
    assert "phase_delay_s" in result.columns
    assert "measured_field_aligned_mT" in result.columns
    assert "residual_for_modeling_mT" in result.columns
    assert "correction_delta_v" in result.columns
    assert metadata["measured_abs_peak_effective_mT"] == pytest.approx(42.0, rel=0.01)
    assert metadata["measured_field_scale_to_50mT"] == pytest.approx(50.0 / 42.0, rel=0.01)
    assert metadata["residual_gain_field_scale_applied"] is True
    assert metadata["active_residual_finite_through_end"] is True
    assert result["measured_field_aligned_mT"].notna().all()
    assert result["residual_for_modeling_mT"].notna().all()
    assert np.nanmax(np.abs(result["measured_field_aligned_mT"])) <= 50.0 + 1e-6


def test_finite_first_phase_sync_rejects_reference_or_predicted_field_as_measured() -> None:
    profile = _finite_profile().drop(columns=["finite_first_actual_measured_field_mT"])
    profile["support_reference_output_mT"] = profile["physical_target_output_mT"] * 0.9
    profile["predicted_field_mT"] = profile["physical_target_output_mT"] * 0.8

    result, metadata = apply_finite_first_phase_sync_modeling(profile, freq_hz=1.0, cycle_count=1.0)

    assert result["measured_field_aligned_mT"].isna().all() if "measured_field_aligned_mT" in result else True
    assert metadata["finite_first_modeling_status"] == "missing_actual_measured_field"
    assert metadata["finite_first_rejected_reference_field_source"] is True
    assert metadata["finite_first_uses_support_reference_as_measured"] is False
    assert metadata["finite_first_uses_target_as_measured"] is False


def test_finite_first_phase_sync_reports_suspicious_target_like_measured() -> None:
    profile = _finite_profile()
    profile["finite_first_actual_measured_field_mT"] = profile["physical_target_output_mT"]

    _result, metadata = apply_finite_first_phase_sync_modeling(profile, freq_hz=1.0, cycle_count=1.0)

    assert metadata["measured_target_nearly_identical_detected"] is True
    assert metadata["measured_target_identity_risk"] in {"warning", "high"}


def test_finite_first_phase_sync_supports_one_and_one_point_five_cycle() -> None:
    one, one_meta = apply_finite_first_phase_sync_modeling(_finite_profile(cycle_count=1.0), freq_hz=1.0, cycle_count=1.0)
    one_half, one_half_meta = apply_finite_first_phase_sync_modeling(
        _finite_profile(cycle_count=1.5),
        freq_hz=1.0,
        cycle_count=1.5,
    )

    assert one_meta["finite_first_modeling_cycle_count"] == 1.0
    assert one_half_meta["finite_first_modeling_cycle_count"] == 1.5
    assert one["time_s"].max() < 1.0
    assert one_half["time_s"].max() < 1.5
    assert 0.0 < float(one_meta["phase_delay_s"]) < 0.2
    assert 0.0 < float(one_half_meta["phase_delay_s"]) < 0.2


def test_finite_first_phase_sync_blocks_when_active_end_support_missing() -> None:
    result, metadata = apply_finite_first_phase_sync_modeling(
        _finite_profile(include_support=False),
        freq_hz=1.0,
        cycle_count=1.0,
    )

    assert metadata["finite_first_modeling_status"] == "insufficient_phase_sync_support"
    assert metadata["phase_sync_support_status"] == "insufficient"
    assert result["residual_for_modeling_mT"].isna().any()


def test_finite_first_phase_sync_uses_dominant_negative_peak_and_requires_tail_support() -> None:
    active_duration = 1.0
    delay_s = 0.09
    time_s = np.linspace(0.0, active_duration + delay_s + 0.05, 520, endpoint=False)
    phase = 2.0 * np.pi * time_s
    voltage = 2.5 * np.sin(phase)
    measured = 35.0 * np.sin(2.0 * np.pi * (time_s - delay_s))
    measured += -22.0 * np.exp(-((time_s - (0.75 + delay_s)) / 0.035) ** 2)
    target = 50.0 * np.sin(phase)
    profile = pd.DataFrame(
        {
            "time_s": time_s,
            "limited_voltage_v": voltage,
            "recommended_voltage_v": voltage,
            "physical_target_output_mT": target,
            "finite_first_actual_measured_field_mT": measured,
        }
    )

    result, metadata = apply_finite_first_phase_sync_modeling(profile, freq_hz=1.0, cycle_count=1.0)

    assert metadata["phase_sync_peak_reference"] == "dominant_absolute_peak"
    assert metadata["phase_sync_peak_polarity"] == "negative"
    assert float(metadata["measured_first_peak_time_s"]) > 0.75
    assert float(metadata["voltage_first_peak_time_s"]) == pytest.approx(0.75, abs=0.02)
    assert float(metadata["phase_delay_s"]) == pytest.approx(delay_s, abs=0.03)
    assert metadata["active_residual_finite_through_end"] is True
    assert metadata["active_residual_finite_ratio"] == pytest.approx(1.0)
    assert result["residual_for_modeling_mT"].notna().all()


def test_finite_first_phase_sync_uses_native_support_beyond_active_output_grid() -> None:
    active_duration = 1.0
    delay_s = 0.09
    active_time = np.linspace(0.0, active_duration, 420, endpoint=False)
    support_time = np.linspace(0.0, active_duration + delay_s + 0.05, 520, endpoint=False)
    active_phase = 2.0 * np.pi * active_time
    support_measured = 35.0 * np.sin(2.0 * np.pi * (support_time - delay_s))
    support_measured += -22.0 * np.exp(-((support_time - (0.75 + delay_s)) / 0.035) ** 2)
    active_measured = np.interp(active_time, support_time, support_measured)
    profile = pd.DataFrame(
        {
            "time_s": active_time,
            "limited_voltage_v": 2.5 * np.sin(active_phase),
            "recommended_voltage_v": 2.5 * np.sin(active_phase),
            "physical_target_output_mT": 50.0 * np.sin(active_phase),
            "finite_first_actual_measured_field_mT": active_measured,
            "selected_support_source_time_s": [support_time.tolist()] * active_time.size,
            "selected_support_source_mT": [support_measured.tolist()] * active_time.size,
        }
    )

    result, metadata = apply_finite_first_phase_sync_modeling(profile, freq_hz=1.0, cycle_count=1.0)

    assert metadata["finite_first_modeling_status"] == "ok"
    assert metadata["measurement_support_grid_separate_from_output_grid"] is True
    assert metadata["measured_alignment_source"] == "native_smoothed_source"
    assert metadata["measurement_support_source"] == "selected_support_source_native"
    assert metadata["phase_sync_peak_polarity"] == "negative"
    assert metadata["active_residual_finite_ratio"] == pytest.approx(1.0)
    assert result["time_s"].max() < active_duration
    assert result["measured_field_aligned_mT"].notna().all()
    assert result["residual_for_modeling_mT"].notna().all()


def test_finite_first_phase_sync_uses_native_support_from_dataframe_attrs() -> None:
    active_duration = 1.0
    delay_s = 0.09
    active_time = np.linspace(0.0, active_duration, 420, endpoint=False)
    support_time = np.linspace(0.0, active_duration + delay_s + 0.05, 520, endpoint=False)
    active_phase = 2.0 * np.pi * active_time
    support_measured = 35.0 * np.sin(2.0 * np.pi * (support_time - delay_s))
    support_measured += -22.0 * np.exp(-((support_time - (0.75 + delay_s)) / 0.035) ** 2)
    active_measured = np.interp(active_time, support_time, support_measured)
    profile = pd.DataFrame(
        {
            "time_s": active_time,
            "limited_voltage_v": 2.5 * np.sin(active_phase),
            "recommended_voltage_v": 2.5 * np.sin(active_phase),
            "physical_target_output_mT": 50.0 * np.sin(active_phase),
            "finite_first_actual_measured_field_mT": active_measured,
        }
    )
    profile.attrs["selected_support_source_time_s"] = support_time.tolist()
    profile.attrs["selected_support_source_mT"] = support_measured.tolist()

    result, metadata = apply_finite_first_phase_sync_modeling(profile, freq_hz=1.0, cycle_count=1.0)

    assert metadata["finite_first_modeling_status"] == "ok"
    assert metadata["measurement_support_grid_separate_from_output_grid"] is True
    assert metadata["measurement_support_source"] == "selected_support_source_native_attrs"
    assert metadata["active_residual_finite_ratio"] == pytest.approx(1.0)
    assert result["measured_field_aligned_mT"].notna().all()
    assert result["residual_for_modeling_mT"].notna().all()


def test_finite_first_phase_sync_maps_target_relative_grid_to_native_source_start() -> None:
    active_duration = 1.0
    source_start = 0.4
    delay_s = 0.08
    active_time = np.linspace(0.0, active_duration, 420, endpoint=False)
    support_time = np.linspace(source_start, source_start + active_duration + delay_s + 0.05, 540, endpoint=False)
    support_rel = support_time - source_start
    active_phase = 2.0 * np.pi * active_time
    support_measured = 42.0 * np.sin(2.0 * np.pi * (support_rel - delay_s))
    active_measured = np.interp(source_start + active_time, support_time, support_measured)
    profile = pd.DataFrame(
        {
            "time_s": active_time,
            "limited_voltage_v": 2.5 * np.sin(active_phase),
            "recommended_voltage_v": 2.5 * np.sin(active_phase),
            "physical_target_output_mT": 50.0 * np.sin(active_phase),
            "finite_first_actual_measured_field_mT": active_measured,
            "support_reference_source_window_start_s": source_start,
        }
    )
    profile.attrs["selected_support_source_time_s"] = support_time.tolist()
    profile.attrs["selected_support_source_mT"] = support_measured.tolist()

    result, metadata = apply_finite_first_phase_sync_modeling(profile, freq_hz=1.0, cycle_count=1.0)

    assert metadata["finite_first_modeling_status"] == "ok"
    assert metadata["phase_sync_source_active_start_s"] == pytest.approx(source_start)
    assert metadata["required_phase_aligned_source_end_s"] > source_start + active_duration
    assert metadata["actual_source_time_end_s"] >= metadata["required_phase_aligned_source_end_s"]
    assert metadata["active_residual_finite_ratio"] == pytest.approx(1.0)
    assert result["measured_field_aligned_mT"].notna().all()
    assert result["residual_for_modeling_mT"].notna().all()


def test_finite_first_phase_sync_normalizes_after_smoothing_not_raw_spike() -> None:
    active_duration = 1.0
    delay_s = 0.09
    support_time = np.linspace(0.0, active_duration + delay_s + 0.05, 520, endpoint=False)
    phase = 2.0 * np.pi * support_time
    measured = 10.0 * np.sin(2.0 * np.pi * (support_time - delay_s))
    measured += 150.0 * np.exp(-((support_time - 0.41) / 0.004) ** 2)
    profile = pd.DataFrame(
        {
            "time_s": support_time,
            "limited_voltage_v": 2.5 * np.sin(phase),
            "recommended_voltage_v": 2.5 * np.sin(phase),
            "physical_target_output_mT": 50.0 * np.sin(phase),
            "finite_first_actual_measured_field_mT": measured,
        }
    )

    result, metadata = apply_finite_first_phase_sync_modeling(profile, freq_hz=1.0, cycle_count=1.0)

    assert metadata["finite_first_modeling_status"] == "ok"
    assert metadata["phase_peak_detection_signal"] == "smoothed_normalized_measured_field"
    assert metadata["measured_abs_peak_effective_mT"] < metadata["measured_abs_peak_raw_mT"]
    assert result["measured_field_smoothed_mT"].abs().max() > 40.0
    assert result["measured_field_aligned_mT"].abs().max() > 40.0


def test_finite_first_phase_sync_blocks_dominant_negative_peak_when_tail_support_missing() -> None:
    active_duration = 1.0
    delay_s = 0.09
    time_s = np.linspace(0.0, active_duration, 420, endpoint=False)
    phase = 2.0 * np.pi * time_s
    voltage = 2.5 * np.sin(phase)
    measured = 35.0 * np.sin(2.0 * np.pi * (time_s - delay_s))
    measured += -22.0 * np.exp(-((time_s - (0.75 + delay_s)) / 0.035) ** 2)
    profile = pd.DataFrame(
        {
            "time_s": time_s,
            "limited_voltage_v": voltage,
            "recommended_voltage_v": voltage,
            "physical_target_output_mT": 50.0 * np.sin(phase),
            "finite_first_actual_measured_field_mT": measured,
        }
    )

    result, metadata = apply_finite_first_phase_sync_modeling(profile, freq_hz=1.0, cycle_count=1.0)

    assert metadata["phase_sync_peak_reference"] == "dominant_absolute_peak"
    assert metadata["phase_sync_peak_polarity"] == "negative"
    assert metadata["finite_first_modeling_status"] == "insufficient_phase_sync_support"
    assert metadata["phase_support_status"] == "insufficient"
    assert metadata["active_residual_finite_through_end"] is False
    assert result["residual_for_modeling_mT"].isna().any()


def test_finite_first_legacy_delay_preserving_is_review_only() -> None:
    result, metadata = apply_finite_first_phase_sync_modeling(
        _finite_profile(),
        freq_hz=1.0,
        cycle_count=1.0,
        mode="legacy_delay_preserving",
    )

    assert metadata["finite_first_modeling_mode"] == "legacy_delay_preserving"
    assert metadata["finite_first_modeling_review_only"] is True
    assert metadata["finite_first_modeling_phase_sync_enabled"] is False
    assert "measured_field_aligned_mT" not in result.columns


def test_quick_lut_source_family_default_and_finite_mode_markers() -> None:
    source = (
        APP_UI.read_text(encoding="utf-8")
        + CONFIG.read_text(encoding="utf-8")
        + UI_FINITE_FIRST.read_text(encoding="utf-8")
    )

    assert "지원/input waveform family" in source
    assert "기본값은 triangle입니다." in source
    assert "source_input_waveform_family_default" in source
    assert "triangle" in source
    assert "sine" in source
    assert "Finite 1차 모델링 방식" in source
    assert "피크 싱크 기반, 기본" in source
    assert "기존 delay 포함 방식, review only" in source
    assert "finite_first_modeling_mode_default" in source
    assert "±50mT 정규화 scale" in source
    assert "1차 command diagnostic traces" in source
    assert '"1차 모델링 command"' in source
