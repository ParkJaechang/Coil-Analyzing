from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


def build_quick_lut_target_config(
    *,
    modeling_input_mode: str,
    target_waveform_family: str | None,
    target_freq_hz: float,
    target_cycle_count: float | None,
    use_frequency_trend: bool,
    finite_cycle_mode: bool,
    preview_tail_cycles: float,
    finite_first_modeling_mode: str = "phase_synced",
    user_target_peak_field_mT: float = 50.0,
) -> dict[str, Any]:
    mode = "continuous_steady_state" if str(modeling_input_mode) == "continuous_steady_state" else "finite_startup_aware"
    cycle = 1.0 if mode == "continuous_steady_state" else (float(target_cycle_count) if target_cycle_count is not None else None)
    return {
        "modeling_input_mode": mode,
        "target_waveform_family": target_waveform_family,
        "target_freq_hz": float(target_freq_hz),
        "target_cycle_count": cycle,
        "target_field_shape": "fixed_rounded_triangle",
        "source_input_waveform_family": target_waveform_family,
        "source_input_waveform_family_default": "triangle",
        "target_peak_mT": float(user_target_peak_field_mT),
        "user_target_peak_field_mT": float(user_target_peak_field_mT),
        "target_peak_field_source": "ui_user_selection",
        "field_modeling_normalization_reference_mT": 50.0,
        "target_pp_fixed_removed": True,
        "use_frequency_trend": bool(use_frequency_trend),
        "finite_cycle_mode": bool(finite_cycle_mode and mode != "continuous_steady_state"),
        "finite_tail_mode": None,
        "finite_first_modeling_mode": "legacy_delay_preserving"
        if str(finite_first_modeling_mode) == "legacy_delay_preserving"
        else "phase_synced",
        "finite_first_modeling_mode_default": "phase_synced",
        "preview_tail_cycles": float(preview_tail_cycles),
        "continuous_loop_output": mode == "continuous_steady_state",
        "continuous_repeating_lut": mode == "continuous_steady_state",
        "continuous_zero_return_tail_enabled": False if mode == "continuous_steady_state" else None,
        "quick_lut_target_config_source": "ui_user_selection",
        "target_freq_hz_source": "ui_user_selection",
        "target_cycle_count_source": "mode_policy" if mode == "continuous_steady_state" else "ui_user_selection",
        "target_config_auto_overwrite_detected": False,
        "target_config_auto_overwrite_blocked": True,
        "target_field_shape_policy": "fixed_rounded_triangle",
        "target_shape": "fixed_rounded_triangle",
        "target_shape_fixed": True,
    }


def legacy_quick_lut_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "target_waveform": config.get("target_waveform_family"),
        "target_freq": config.get("target_freq_hz"),
        "use_frequency_trend": config.get("use_frequency_trend"),
        "finite_cycle_mode": config.get("finite_cycle_mode"),
        "target_cycle_count": config.get("target_cycle_count"),
        "preview_tail_cycles": config.get("preview_tail_cycles"),
        "finite_first_modeling_mode": config.get("finite_first_modeling_mode"),
        "source_input_waveform_family": config.get("source_input_waveform_family"),
        "modeling_input_mode": config.get("modeling_input_mode"),
        "continuous_production_cycle_count": 1.0,
        "continuous_repeating_lut": config.get("continuous_repeating_lut"),
        "zero_return_tail_enabled": config.get("continuous_zero_return_tail_enabled"),
    }


def target_configs_equal(left: Mapping[str, Any] | None, right: Mapping[str, Any] | None) -> bool:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return False
    return _canonical(left) == _canonical(right)


def target_config_snapshot(config: Mapping[str, Any] | None) -> dict[str, Any]:
    return deepcopy(dict(config or {}))


def modeling_metadata_from_target_config(config: Mapping[str, Any] | None, *, prefix: str = "modeled") -> dict[str, Any]:
    snapshot = target_config_snapshot(config)
    return {
        f"{prefix}_target_freq_hz": snapshot.get("target_freq_hz"),
        f"{prefix}_target_cycle_count": snapshot.get("target_cycle_count"),
        f"{prefix}_input_mode": snapshot.get("modeling_input_mode"),
        f"{prefix}_target_config_snapshot": snapshot,
        "quick_lut_target_config_source": snapshot.get("quick_lut_target_config_source", "ui_user_selection"),
        "target_freq_hz_source": snapshot.get("target_freq_hz_source", "ui_user_selection"),
        "target_cycle_count_source": snapshot.get("target_cycle_count_source", "ui_user_selection"),
        "target_config_auto_overwrite_detected": False,
        "target_config_auto_overwrite_blocked": True,
    }


def _canonical(config: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    keys = (
        "modeling_input_mode",
        "target_waveform_family",
        "target_freq_hz",
        "target_cycle_count",
        "target_field_shape",
        "use_frequency_trend",
        "finite_cycle_mode",
        "preview_tail_cycles",
        "finite_first_modeling_mode",
        "continuous_loop_output",
        "user_target_peak_field_mT",
    )
    return tuple((key, _normal(config.get(key))) for key in keys)


def _normal(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, int):
        return float(value)
    return value
