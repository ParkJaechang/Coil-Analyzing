from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

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

    return pd.DataFrame(
        {
            "time_s": time_s,
            "freq_hz": 2.0,
            "is_active_target": active_mask,
            "target_field_mT": target,
            "predicted_field_mT": predicted,
        }
    )


def test_perfect_prediction_has_zero_error_and_matching_direction() -> None:
    profile = _build_profile()

    metrics = evaluate_finite_cycle_metrics(profile)

    assert metrics.evaluation_status == "ok"
    assert metrics.unavailable_reason is None
    assert metrics.active_sample_count == 21
    assert metrics.tail_sample_count == 10
    assert metrics.active_window_nrmse <= 1e-12
    assert metrics.active_window_rmse_mT <= 1e-12
    assert metrics.active_window_shape_corr > 0.999999
    assert metrics.terminal_direction_match is True
    assert abs(metrics.estimated_lag_seconds) <= 1e-12
    assert metrics.tail_residual_peak_mT == 0.0
    assert metrics.tail_residual_ratio == 0.0


def test_terminal_peak_error_is_negative_when_terminal_prediction_is_low() -> None:
    baseline = _build_profile()
    predicted = baseline["predicted_field_mT"].to_numpy(dtype=float).copy()
    active_indices = np.flatnonzero(baseline["is_active_target"].to_numpy(dtype=bool))
    predicted[active_indices[-8:]] *= 0.35
    profile = baseline.copy()
    profile["predicted_field_mT"] = predicted

    metrics = evaluate_finite_cycle_metrics(profile)

    assert metrics.evaluation_status == "ok"
    assert metrics.terminal_peak_error_mT < 0.0
    assert metrics.terminal_peak_error_ratio < 0.0


def test_terminal_direction_match_is_false_for_opposite_end_slope() -> None:
    baseline = _build_profile()
    predicted = baseline["predicted_field_mT"].to_numpy(dtype=float).copy()
    active_indices = np.flatnonzero(baseline["is_active_target"].to_numpy(dtype=bool))
    predicted[active_indices[-4:]] = np.array([2.0, 4.0, 8.0, 16.0], dtype=float)
    profile = baseline.copy()
    profile["predicted_field_mT"] = predicted

    metrics = evaluate_finite_cycle_metrics(profile)

    assert metrics.evaluation_status == "ok"
    assert metrics.terminal_target_slope_sign == -1.0
    assert metrics.terminal_predicted_slope_sign == 1.0
    assert metrics.terminal_direction_match is False


def test_tail_residual_ratio_is_positive_when_tail_energy_remains() -> None:
    tail = np.linspace(6.0, 2.0, 10)
    profile = _build_profile(predicted_tail=tail)

    metrics = evaluate_finite_cycle_metrics(profile)

    assert metrics.evaluation_status == "ok"
    assert metrics.tail_sample_count == 10
    assert metrics.tail_residual_peak_mT > 0.0
    assert metrics.tail_residual_ratio > 0.0


def test_missing_columns_return_unavailable_reason_without_crashing() -> None:
    profile = pd.DataFrame(
        {
            "time_s": np.linspace(0.0, 1.0, 5),
            "is_active_target": [True, True, True, False, False],
        }
    )

    metrics = evaluate_finite_cycle_metrics(profile)

    assert metrics.evaluation_status == "unavailable"
    assert metrics.unavailable_reason == "missing_target_column"
