from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from field_analysis.quick_lut_target_config import (
    build_quick_lut_target_config,
    modeling_metadata_from_target_config,
    target_configs_equal,
)


def test_finite_target_config_preserves_one_hz_and_one_cycle() -> None:
    config = build_quick_lut_target_config(
        modeling_input_mode="finite_startup_aware",
        target_waveform_family="sine",
        target_freq_hz=1.0,
        target_cycle_count=1.0,
        use_frequency_trend=True,
        finite_cycle_mode=True,
        preview_tail_cycles=0.25,
    )

    assert config["target_freq_hz"] == 1.0
    assert config["target_cycle_count"] == 1.0
    assert config["target_freq_hz_source"] == "ui_user_selection"
    assert config["target_cycle_count_source"] == "ui_user_selection"
    assert config["target_config_auto_overwrite_blocked"] is True


def test_finite_target_config_preserves_one_point_five_cycle_without_changing_frequency() -> None:
    config = build_quick_lut_target_config(
        modeling_input_mode="finite_startup_aware",
        target_waveform_family="sine",
        target_freq_hz=1.0,
        target_cycle_count=1.5,
        use_frequency_trend=True,
        finite_cycle_mode=True,
        preview_tail_cycles=0.25,
    )

    assert config["target_freq_hz"] == 1.0
    assert config["target_cycle_count"] == 1.5


def test_continuous_target_config_locks_cycle_to_one_but_preserves_frequency() -> None:
    config = build_quick_lut_target_config(
        modeling_input_mode="continuous_steady_state",
        target_waveform_family="sine",
        target_freq_hz=1.0,
        target_cycle_count=1.5,
        use_frequency_trend=True,
        finite_cycle_mode=False,
        preview_tail_cycles=0.25,
    )

    assert config["target_freq_hz"] == 1.0
    assert config["target_cycle_count"] == 1.0
    assert config["target_cycle_count_source"] == "mode_policy"
    assert config["continuous_loop_output"] is True
    assert config["continuous_zero_return_tail_enabled"] is False


def test_source_metadata_frequency_does_not_make_configs_equal_to_different_target() -> None:
    target_one_hz = build_quick_lut_target_config(
        modeling_input_mode="finite_startup_aware",
        target_waveform_family="sine",
        target_freq_hz=1.0,
        target_cycle_count=1.0,
        use_frequency_trend=True,
        finite_cycle_mode=True,
        preview_tail_cycles=0.25,
    )
    source_metadata_1p5_hz = dict(target_one_hz, target_freq_hz=1.5)

    assert not target_configs_equal(source_metadata_1p5_hz, target_one_hz)
    assert target_one_hz["target_freq_hz"] == 1.0


def test_first_model_metadata_includes_target_config_snapshot() -> None:
    config = build_quick_lut_target_config(
        modeling_input_mode="finite_startup_aware",
        target_waveform_family="sine",
        target_freq_hz=1.0,
        target_cycle_count=1.0,
        use_frequency_trend=False,
        finite_cycle_mode=True,
        preview_tail_cycles=0.0,
    )
    metadata = modeling_metadata_from_target_config(config, prefix="modeled")

    assert metadata["modeled_target_freq_hz"] == 1.0
    assert metadata["modeled_target_cycle_count"] == 1.0
    assert metadata["modeled_input_mode"] == "finite_startup_aware"
    assert metadata["modeled_target_config_snapshot"]["target_freq_hz"] == 1.0
