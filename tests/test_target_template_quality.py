from __future__ import annotations

import numpy as np
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.compensation import _finite_target_template
from field_analysis.lut import build_fixed_field_target_template
from field_analysis.target_templates import (
    analytic_fixed_rounded_triangle_normalized,
    target_template_quality,
)


def test_analytic_fixed_rounded_triangle_peak_and_ripple_quality() -> None:
    phase = np.linspace(0.0, 1.0, 1001, endpoint=False)
    normalized = analytic_fixed_rounded_triangle_normalized(phase)
    target_mT = normalized * 50.0
    quality = target_template_quality(target_mT, target_peak_mT=50.0)

    assert quality["target_template_type"] == "analytic_fixed_rounded_triangle"
    assert quality["target_template_ripple_check_passed"] is True
    assert abs(float(quality["target_peak_positive_mT"]) - 50.0) < 1e-6
    assert abs(float(quality["target_peak_negative_mT"]) + 50.0) < 1e-6
    assert float(quality["target_linear_segment_deviation_max_mT"]) < 1e-3


def test_finite_and_lut_templates_use_same_analytic_shape() -> None:
    points = 501
    time_s = np.linspace(0.0, 1.0, points)
    finite = _finite_target_template(
        time_s,
        waveform_type="triangle",
        freq_hz=1.0,
        target_cycle_count=1.0,
        target_output_pp=100.0,
        force_rounded_triangle=True,
    )
    lut = build_fixed_field_target_template(freq_hz=1.0, points_per_cycle=points)

    assert np.allclose(finite / 50.0, lut["voltage_normalized"].to_numpy(dtype=float), atol=1e-9)
