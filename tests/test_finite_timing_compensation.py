import numpy as np
import pandas as pd

from src.field_analysis.finite_timing_compensation import (
    evaluate_finite_timing_compensation,
)


def _profile(*, lag_s: float = 0.08, tail_scale: float = 0.08) -> pd.DataFrame:
    time_s = np.linspace(0.0, 1.8, 361)
    active = time_s <= 1.0 + 1e-12
    target = np.where(active, 50.0 * np.sin(np.pi * time_s), 0.0)
    predicted = np.where(
        active,
        50.0 * np.sin(np.pi * np.clip(time_s - lag_s, 0.0, 1.0)),
        tail_scale * 50.0 * np.exp(-(time_s - 1.0) * 7.0),
    )
    command = np.where(active, 4.0 * np.sin(np.pi * time_s), 0.0)
    return pd.DataFrame(
        {
            "time_s": time_s,
            "target_field_mT": target,
            "physical_target_output_mT": target,
            "predicted_field_mT": predicted,
            "recommended_voltage_v": command,
            "limited_voltage_v": command,
            "is_active_target": active,
            "freq_hz": 1.0,
        }
    )


def _candidate_predictor(
    candidate_profile: pd.DataFrame,
    candidate: dict,
) -> pd.DataFrame:
    predicted = candidate_profile["predicted_field_mT"].to_numpy(dtype=float).copy()
    target = candidate_profile["target_field_mT"].to_numpy(dtype=float)
    active = candidate_profile["is_active_target"].to_numpy(dtype=bool)
    time_s = candidate_profile["time_s"].to_numpy(dtype=float)
    extension = float(candidate["command_extension_s"])
    if extension > 0:
        predicted[active] = target[active]
        tail_mask = time_s > float(time_s[active].max())
        predicted[tail_mask] *= 0.5
    candidate_profile = candidate_profile.copy()
    candidate_profile["predicted_field_mT"] = predicted
    return candidate_profile


def test_target_immutability_and_metadata_keys_present():
    baseline = _profile()
    report = evaluate_finite_timing_compensation(
        baseline,
        freq_hz=1.0,
        candidate_predictor=_candidate_predictor,
    )

    assert report["physical_target_unchanged"] is True
    assert report["target_duration_changed"] is False
    assert report["target_active_end_s"] == 1.0
    for key in (
        "timing_compensation_evaluate_only",
        "timing_compensation_applied",
        "timing_route_selected",
        "timing_route_reject_reason",
        "selected_timing_candidate",
        "rejected_timing_candidates",
        "timing_candidate_count",
        "timing_score_before",
        "timing_score_after",
        "command_extension_s",
        "command_hold_end_s",
        "predicted_peak_time_s",
        "predicted_settle_end_s",
        "phase_delay_source",
        "empirical_lag_s",
        "finite_support_peak_delay_s",
        "lcr_phase_delay_s",
        "voltage_limit_respected",
    ):
        assert key in report


def test_evaluate_only_default_does_not_change_baseline_command():
    baseline = _profile()
    before_command = baseline["recommended_voltage_v"].copy()
    report = evaluate_finite_timing_compensation(
        baseline,
        freq_hz=1.0,
        candidate_predictor=_candidate_predictor,
    )

    pd.testing.assert_series_equal(before_command, baseline["recommended_voltage_v"])
    assert report["timing_compensation_evaluate_only"] is True
    assert report["timing_compensation_applied"] is False
    assert report["timing_route_selected"] is True


def test_command_extension_candidate_generated_without_hidden_voltage_increase():
    baseline = _profile()
    report = evaluate_finite_timing_compensation(
        baseline,
        freq_hz=1.0,
        max_daq_voltage_pp=10.0,
        candidate_predictor=_candidate_predictor,
    )

    assert report["timing_candidate_count"] > 1
    assert report["selected_timing_candidate"]["command_extension_s"] > 0.0
    assert report["command_nonzero_end_s"] >= report["target_active_end_s"]
    assert report["command_extension_s"] <= 0.15
    assert report["voltage_limit_respected"] is True


def test_reject_active_nrmse_degradation():
    def degrading_predictor(candidate_profile: pd.DataFrame, candidate: dict) -> pd.DataFrame:
        candidate_profile = candidate_profile.copy()
        active = candidate_profile["is_active_target"].to_numpy(dtype=bool)
        candidate_profile.loc[active, "predicted_field_mT"] *= -1.0
        return candidate_profile

    report = evaluate_finite_timing_compensation(
        _profile(),
        freq_hz=1.0,
        candidate_predictor=degrading_predictor,
    )

    assert report["timing_route_selected"] is False
    assert "active_nrmse_worsened" in report["timing_route_reject_reason"]


def test_reject_tail_residual_increase():
    def tail_bad_predictor(candidate_profile: pd.DataFrame, candidate: dict) -> pd.DataFrame:
        candidate_profile = candidate_profile.copy()
        active_end = candidate_profile.loc[candidate_profile["is_active_target"], "time_s"].max()
        tail_mask = candidate_profile["time_s"] > active_end
        candidate_profile.loc[tail_mask, "predicted_field_mT"] = 50.0
        return candidate_profile

    report = evaluate_finite_timing_compensation(
        _profile(tail_scale=0.02),
        freq_hz=1.0,
        candidate_predictor=tail_bad_predictor,
    )

    assert report["timing_route_selected"] is False
    assert "tail_residual_increased" in report["timing_route_reject_reason"]


def test_no_lcr_data_uses_empirical_lag_or_unavailable():
    report = evaluate_finite_timing_compensation(
        _profile(),
        freq_hz=1.0,
        lcr_phase_delay_s=None,
        candidate_predictor=_candidate_predictor,
    )

    assert report["lcr_phase_delay_s"] is None
    assert report["phase_delay_source"] in {"empirical_support_lag", "finite_support_peak_delay", "unavailable"}


def test_lcr_optional_source_records_when_supplied_without_empirical_lag():
    report = evaluate_finite_timing_compensation(
        _profile(lag_s=0.0),
        freq_hz=1.0,
        empirical_lag_s=None,
        finite_support_peak_delay_s=None,
        lcr_phase_delay_s=0.03,
        candidate_predictor=_candidate_predictor,
    )

    assert report["phase_delay_source"] == "lcr_phase_prior"
    assert report["lcr_phase_delay_s"] == 0.03


def test_candidate_prediction_unavailable_prevents_selection():
    report = evaluate_finite_timing_compensation(
        _profile(),
        freq_hz=1.0,
        candidate_predictor=None,
    )

    assert report["timing_route_selected"] is False
    assert report["timing_candidate_prediction_available"] is False
    assert report["timing_route_reject_reason"] == "candidate_forward_prediction_unavailable"
