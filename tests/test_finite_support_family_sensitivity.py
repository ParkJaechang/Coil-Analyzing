from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.compensation import build_support_family_sensitivity_summary


def _result(
    *,
    predicted: np.ndarray,
    command: np.ndarray | None = None,
    terminal_peak_error: float = 0.0,
    direction_match: bool = True,
    active_nrmse: float = 0.01,
) -> dict[str, object]:
    time_s = np.linspace(0.0, 1.0, len(predicted))
    command_values = predicted / 10.0 if command is None else command
    return {
        "command_profile": pd.DataFrame(
            {
                "time_s": time_s,
                "predicted_field_mT": predicted,
                "recommended_voltage_v": command_values,
            }
        ),
        "finite_cycle_metrics": {
            "terminal_peak_error_mT": terminal_peak_error,
            "terminal_direction_match": direction_match,
            "active_window_nrmse": active_nrmse,
        },
    }


def test_similar_family_outputs_have_low_sensitivity() -> None:
    base = np.sin(np.linspace(0.0, np.pi, 64))
    summary = build_support_family_sensitivity_summary(
        {
            "sine": _result(predicted=base),
            "triangle": _result(predicted=base * 1.01, terminal_peak_error=0.2, active_nrmse=0.012),
        }
    )

    assert summary["sensitivity_level"] == "low"
    assert float(summary["predicted_shape_corr"]) > 0.99


def test_terminal_peak_error_delta_can_raise_sensitivity() -> None:
    base = np.sin(np.linspace(0.0, np.pi, 64))
    summary = build_support_family_sensitivity_summary(
        {
            "sine": _result(predicted=base, terminal_peak_error=-1.0),
            "triangle": _result(predicted=base, terminal_peak_error=8.0),
        }
    )

    assert summary["sensitivity_level"] in {"medium", "excessive"}
    assert float(summary["terminal_peak_error_delta_mT"]) == 9.0


def test_predicted_shape_corr_below_threshold_is_excessive() -> None:
    base = np.sin(np.linspace(0.0, np.pi, 64))
    inverted = -base
    summary = build_support_family_sensitivity_summary(
        {
            "sine": _result(predicted=base),
            "triangle": _result(predicted=inverted),
        }
    )

    assert summary["sensitivity_level"] == "excessive"
    assert float(summary["predicted_shape_corr"]) < 0.0


def test_terminal_direction_change_is_excessive() -> None:
    base = np.sin(np.linspace(0.0, np.pi, 64))
    summary = build_support_family_sensitivity_summary(
        {
            "sine": _result(predicted=base, direction_match=True),
            "triangle": _result(predicted=base * 0.95, direction_match=False),
        }
    )

    assert summary["terminal_direction_match_changed"] is True
    assert summary["sensitivity_level"] == "excessive"


def test_family_sensitivity_summary_is_deterministic() -> None:
    base = np.sin(np.linspace(0.0, np.pi, 64))
    payload = {
        "triangle": _result(predicted=base * 0.97, terminal_peak_error=1.5),
        "sine": _result(predicted=base, terminal_peak_error=0.0),
    }

    first = build_support_family_sensitivity_summary(payload)
    second = build_support_family_sensitivity_summary(payload)

    assert first == second
