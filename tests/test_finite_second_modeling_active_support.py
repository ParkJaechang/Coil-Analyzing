from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.finite_second_modeling import generate_second_modeled_voltage_lut
from tests.test_finite_second_modeling import _first_profile, _write_delayed_actual_drive_csv


def test_tail_off_phase_aligned_active_residual_stays_finite_through_end(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_delayed_actual_drive_csv(actual, delay_s=0.12)
    profile = _first_profile()
    tail_rows = profile.tail(8).copy()
    tail_rows["time_s"] = np.linspace(1.01, 1.16, len(tail_rows))
    profile = pd.concat([profile, tail_rows], ignore_index=True)

    frame, metadata = generate_second_modeled_voltage_lut(
        profile,
        actual,
        freq_hz=1.0,
        cycle_count=1.0,
        post_cycle_zero_tail_enabled=False,
    )

    active = frame["active_window_mask"].astype(bool).to_numpy()
    active_indices = np.flatnonzero(active)
    last_active = active_indices[-5:]
    assert metadata["measurement_support_grid_separate_from_output_grid"] is True
    assert metadata["output_command_grid_tail_off_active_only"] is True
    assert metadata["phase_aligned_active_support_status"] == "ok"
    assert np.isclose(metadata["aligned_measured_active_finite_ratio"], 1.0)
    assert metadata["tail_disabled_active_correction_preserved"] is True
    assert metadata["active_correction_finite_through_end"] is True
    assert metadata["active_end_kink_detected"] is False
    assert frame["time_s"].max() <= 1.0 + 1e-12
    for column in ("measured_field_for_second_mT", "residual_for_second_mT", "unit_delta_v", "correction_delta_v"):
        assert np.isfinite(frame.loc[last_active, column].to_numpy(dtype=float)).all()


def test_tail_off_active_end_missing_phase_support_is_not_zero_filled(tmp_path: Path) -> None:
    actual = tmp_path / "finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv"
    _write_delayed_actual_drive_csv(actual, delay_s=0.12)
    lines = actual.read_text(encoding="utf-8").splitlines()
    header_index = next(index for index, line in enumerate(lines) if line.startswith("Row,TimeMs"))
    rows = [line for line in lines[header_index + 1 :] if float(line.split(",")[1]) <= 1280.0]
    actual.write_text("\n".join([*lines[: header_index + 1], *rows]), encoding="utf-8")

    frame, metadata = generate_second_modeled_voltage_lut(
        _first_profile(),
        actual,
        freq_hz=1.0,
        cycle_count=1.0,
        post_cycle_zero_tail_enabled=False,
    )

    active = frame["active_window_mask"].astype(bool).to_numpy()
    active_end_index = int(np.flatnonzero(active)[-1])
    assert metadata["phase_aligned_active_support_status"].startswith("insufficient")
    assert metadata["active_residual_invalid_detected"] is True
    assert metadata["active_unit_delta_zero_fill_used"] is False
    assert metadata["active_end_residual_support_status"] == "missing"
    assert metadata["active_end_kink_detected"] is True
    assert metadata["active_end_kink_source"] in {"residual_nan_to_zero", "phase_support_missing"}
    assert np.isnan(float(frame.loc[active_end_index, "unit_delta_v"]))
