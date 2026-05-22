from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

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
    assert result["measured_field_aligned_mT"].notna().all()
    assert result["residual_for_modeling_mT"].notna().all()


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
