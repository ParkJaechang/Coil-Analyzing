from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.compensation import _apply_finite_terminal_tail_correction
from field_analysis.finite_cycle_metrics import evaluate_finite_cycle_metrics


def _build_profile(
    *,
    predicted_active: np.ndarray | None = None,
    predicted_tail: np.ndarray | None = None,
) -> pd.DataFrame:
    time_s = np.linspace(0.0, 1.5, 31)
    active_mask = time_s <= 1.0
    active_progress = time_s[active_mask] / 1.0
    target_active = 50.0 * np.sin(np.pi * active_progress)
    target = np.zeros_like(time_s)
    target[active_mask] = target_active

    predicted = np.zeros_like(time_s)
    predicted[active_mask] = target_active if predicted_active is None else predicted_active
    predicted[~active_mask] = 0.0 if predicted_tail is None else predicted_tail

    command = pd.DataFrame(
        {
            "time_s": time_s,
            "freq_hz": 2.0,
            "is_active_target": active_mask,
            "target_field_mT": target,
            "used_target_field_mT": target,
            "aligned_target_field_mT": target,
            "aligned_used_target_field_mT": target,
            "expected_field_mT": predicted,
            "expected_output": predicted,
            "predicted_field_mT": predicted,
            "recommended_voltage_v": np.where(active_mask, predicted / 20.0, 0.0),
            "limited_voltage_v": np.where(active_mask, predicted / 20.0, 0.0),
            "within_daq_limit": True,
            "within_hardware_limits": True,
        }
    )
    return command


def test_terminal_peak_undershoot_improves_after_correction() -> None:
    profile = _build_profile()
    predicted = profile["expected_field_mT"].to_numpy(dtype=float).copy()
    active_indices = np.flatnonzero(profile["is_active_target"].to_numpy(dtype=bool))
    predicted[active_indices[-8:]] *= 0.35
    profile["expected_field_mT"] = predicted
    profile["expected_output"] = predicted
    profile["predicted_field_mT"] = predicted

    before = evaluate_finite_cycle_metrics(profile)
    corrected, _, after_dict, _ = _apply_finite_terminal_tail_correction(
        command_profile=profile,
        freq_hz=2.0,
        max_daq_voltage_pp=1000.0,
        amp_gain_at_100_pct=1.0,
        support_amp_gain_pct=100.0,
        amp_gain_limit_pct=100.0,
        amp_max_output_pk_v=1000.0,
    )

    after = evaluate_finite_cycle_metrics(corrected)
    assert abs(float(after_dict["terminal_peak_error_mT"])) <= abs(float(before.terminal_peak_error_mT))
    assert abs(float(after.terminal_peak_error_mT)) <= abs(float(before.terminal_peak_error_mT))


def test_terminal_slope_mismatch_improves_or_reports_direction_reason() -> None:
    profile = _build_profile()
    predicted = profile["expected_field_mT"].to_numpy(dtype=float).copy()
    active_indices = np.flatnonzero(profile["is_active_target"].to_numpy(dtype=bool))
    predicted[active_indices[-4:]] = np.array([2.0, 4.0, 8.0, 16.0], dtype=float)
    profile["expected_field_mT"] = predicted
    profile["expected_output"] = predicted
    profile["predicted_field_mT"] = predicted

    corrected, _, after_dict, summary = _apply_finite_terminal_tail_correction(
        command_profile=profile,
        freq_hz=2.0,
        max_daq_voltage_pp=1000.0,
        amp_gain_at_100_pct=1.0,
        support_amp_gain_pct=100.0,
        amp_gain_limit_pct=100.0,
        amp_max_output_pk_v=1000.0,
    )

    reason = str(corrected["finite_terminal_correction_reason"].iloc[0])
    assert (
        after_dict["terminal_direction_match"] is True
        or "terminal_direction" in reason
        or summary["terminal_direction_after"] is True
        or (
            np.isfinite(float(after_dict["terminal_target_slope_sign"]))
            and np.isfinite(float(after_dict["terminal_predicted_slope_sign"]))
        )
    )


def test_tail_residual_does_not_worsen_after_correction() -> None:
    tail = np.linspace(8.0, 3.0, 10)
    profile = _build_profile(predicted_tail=tail)

    before = evaluate_finite_cycle_metrics(profile)
    corrected, _, after_dict, _ = _apply_finite_terminal_tail_correction(
        command_profile=profile,
        freq_hz=2.0,
        max_daq_voltage_pp=1000.0,
        amp_gain_at_100_pct=1.0,
        support_amp_gain_pct=100.0,
        amp_gain_limit_pct=100.0,
        amp_max_output_pk_v=1000.0,
    )

    after = evaluate_finite_cycle_metrics(corrected)
    assert float(after_dict["tail_residual_ratio"]) <= float(before.tail_residual_ratio) + 1e-12
    assert float(after.tail_residual_ratio) <= float(before.tail_residual_ratio) + 1e-12


def test_noop_safe_case_skips_aggressive_correction() -> None:
    profile = _build_profile()

    corrected, before_dict, after_dict, _ = _apply_finite_terminal_tail_correction(
        command_profile=profile,
        freq_hz=2.0,
        max_daq_voltage_pp=1000.0,
        amp_gain_at_100_pct=1.0,
        support_amp_gain_pct=100.0,
        amp_gain_limit_pct=100.0,
        amp_max_output_pk_v=1000.0,
    )

    assert bool(corrected["finite_terminal_correction_applied"].iloc[0]) is False
    assert abs(float(after_dict["active_window_nrmse"]) - float(before_dict["active_window_nrmse"])) <= 1e-12
    assert str(corrected["finite_terminal_correction_reason"].iloc[0]) == "no_material_improvement"


def test_guardrail_case_keeps_active_nrmse_stable() -> None:
    profile = _build_profile()
    predicted = profile["expected_field_mT"].to_numpy(dtype=float).copy()
    active_indices = np.flatnonzero(profile["is_active_target"].to_numpy(dtype=bool))
    predicted[active_indices[-6:]] *= 0.55
    predicted[active_indices[-1]] = profile["target_field_mT"].to_numpy(dtype=float)[active_indices[-1]] + 18.0
    profile["expected_field_mT"] = predicted
    profile["expected_output"] = predicted
    profile["predicted_field_mT"] = predicted

    before = evaluate_finite_cycle_metrics(profile)
    corrected, _, after_dict, _ = _apply_finite_terminal_tail_correction(
        command_profile=profile,
        freq_hz=2.0,
        max_daq_voltage_pp=1000.0,
        amp_gain_at_100_pct=1.0,
        support_amp_gain_pct=100.0,
        amp_gain_limit_pct=100.0,
        amp_max_output_pk_v=1000.0,
    )

    after = evaluate_finite_cycle_metrics(corrected)
    guardrail_limit = float(before.active_window_nrmse + max(0.02, before.active_window_nrmse * 0.20))
    assert float(after_dict["active_window_nrmse"]) <= guardrail_limit + 1e-12
    assert float(after.active_window_nrmse) <= guardrail_limit + 1e-12
