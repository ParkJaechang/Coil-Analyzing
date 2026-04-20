from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIELD_ANALYSIS_SRC = ROOT.parent / "src"
if str(FIELD_ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(FIELD_ANALYSIS_SRC))

from field_analysis.utils import BZ_EFFECTIVE_COLUMN, BZ_RAW_COLUMN, apply_bz_effective_convention
from field_analysis.validation_retune import build_validation_run


def test_apply_bz_effective_convention_preserves_raw_and_flips_sign() -> None:
    frame = pd.DataFrame({"time_s": [0.0, 0.1, 0.2], "bz_mT": [1.0, -2.5, 3.0]})

    converted = apply_bz_effective_convention(frame.copy())

    assert BZ_RAW_COLUMN in converted.columns
    assert BZ_EFFECTIVE_COLUMN in converted.columns
    np.testing.assert_allclose(converted[BZ_RAW_COLUMN].to_numpy(dtype=float), [1.0, -2.5, 3.0])
    np.testing.assert_allclose(converted[BZ_EFFECTIVE_COLUMN].to_numpy(dtype=float), [-1.0, 2.5, -3.0])
    np.testing.assert_allclose(converted["bz_mT"].to_numpy(dtype=float), converted[BZ_EFFECTIVE_COLUMN].to_numpy(dtype=float))


def test_apply_bz_effective_convention_is_idempotent_for_effective_frames() -> None:
    frame = pd.DataFrame(
        {
            "time_s": [0.0, 0.1],
            "bz_mT": [-4.0, 5.0],
            BZ_RAW_COLUMN: [4.0, -5.0],
            BZ_EFFECTIVE_COLUMN: [-4.0, 5.0],
        }
    )

    converted = apply_bz_effective_convention(frame.copy())

    np.testing.assert_allclose(converted[BZ_RAW_COLUMN].to_numpy(dtype=float), [4.0, -5.0])
    np.testing.assert_allclose(converted[BZ_EFFECTIVE_COLUMN].to_numpy(dtype=float), [-4.0, 5.0])
    np.testing.assert_allclose(converted["bz_mT"].to_numpy(dtype=float), [-4.0, 5.0])


def test_build_validation_run_tracks_bz_first_provenance() -> None:
    base_profile = pd.DataFrame(
        {
            "time_s": [0.0, 0.1],
            "waveform_type": ["sine", "sine"],
            "freq_hz": [0.5, 0.5],
            "target_cycle_count": [1.0, 1.0],
            "target_field_mT": [0.0, 20.0],
        }
    )
    validation_candidate = {
        "test_id": "valid_case_01",
        "score": 0.1,
        "eligible": True,
        "freq_hz": 0.5,
        "output_pp": 20.0,
    }

    run = build_validation_run(
        base_profile=base_profile,
        validation_candidate=validation_candidate,
        export_file_prefix="field_exact_case",
        target_output_type="field",
        original_recommendation_id="recommended_case_01",
    )

    assert run.original_recommendation_id == "recommended_case_01"
    assert run.lut_id == "field_exact_case"
    assert run.corrected_lut_id == "field_exact_case_retuned_control_lut"
    assert run.validation_run_id.startswith("valid_case_01::")
    assert run.target_output_type == "field"
