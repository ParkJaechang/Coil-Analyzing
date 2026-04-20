from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIELD_ANALYSIS_SRC = ROOT.parent / "src"
if str(FIELD_ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(FIELD_ANALYSIS_SRC))

from field_analysis.canonical_runs import summarize_canonical_runs
from field_analysis.canonicalize import CanonicalizeConfig, canonicalize_batch, canonicalize_run
from field_analysis.models import ParsedMeasurement, SheetPreview


def _build_parsed_measurement(
    *,
    source_file: str = "finite_1.25cycle_10pp.csv",
    sheet_name: str = "main",
    commanded_cycles: float = 1.25,
    freq_hz: float = 1.0,
) -> ParsedMeasurement:
    time_s = np.array([0.0, 0.07, 0.16, 0.29, 0.41, 0.58, 0.73, 0.94, 1.13], dtype=float)
    phase = time_s / time_s[-1]
    drive = 2.0 * np.sin(2.0 * np.pi * phase)
    current_signed = 5.0 * np.sin(2.0 * np.pi * phase)
    unsigned_current = np.abs(current_signed) + 0.15
    normalized_frame = pd.DataFrame(
        {
            "time_s": time_s,
            "daq_input_v": drive,
            "coil1_current_a": np.zeros_like(unsigned_current),
            "coil2_current_a": unsigned_current,
            "bx_mT": 0.1 * np.sin(2.0 * np.pi * phase),
            "by_mT": 0.2 * np.sin(2.0 * np.pi * phase),
            "bz_mT": 10.0 * np.sin(2.0 * np.pi * phase),
            "waveform_type": "Sine",
            "freq_hz": freq_hz,
            "current_pp_target_a": 10.0,
            "current_pk_target_a": 5.0,
            "cycle_total_expected": commanded_cycles,
            "source_cycle_no": np.zeros(len(time_s), dtype=float),
        }
    )
    preview = SheetPreview(
        sheet_name=sheet_name,
        row_count=len(normalized_frame),
        column_count=len(normalized_frame.columns),
        columns=list(normalized_frame.columns),
        header_row_index=0,
        metadata={"cycle": str(commanded_cycles)},
        preview_rows=normalized_frame.head(3).to_dict(orient="records"),
        recommended_mapping={},
    )
    return ParsedMeasurement(
        source_file=source_file,
        file_type="csv",
        sheet_name=sheet_name,
        structure_preview=preview,
        metadata={"cycle": str(commanded_cycles), "waveform": "sine", "freq_hz": str(freq_hz)},
        mapping={},
        raw_frame=normalized_frame.copy(),
        normalized_frame=normalized_frame,
        warnings=[],
        logs=[],
    )


def test_canonicalize_run_builds_uniform_signed_contract() -> None:
    parsed = _build_parsed_measurement()

    run = canonicalize_run(parsed, regime="transient", role="train", config=CanonicalizeConfig())

    diffs = np.diff(run.time_s)
    assert len(run.time_s) > 4
    assert np.all(diffs > 0)
    assert np.allclose(diffs, diffs[0], atol=1e-9)
    assert run.command_waveform == "sine"
    assert run.commanded_cycles == 1.25
    assert run.target_type == "current"
    assert run.target_level_kind == "pp"
    assert run.primary_field_axis == "bz"
    assert np.nanmax(run.signed_current_a) > 0
    assert np.nanmin(run.signed_current_a) < 0
    assert "current_sign_inferred" in run.quality_flags
    assert "resampled_uniform" in run.quality_flags
    assert "cycle_label_untrusted" in run.quality_flags
    assert run.sample_rate_hz > 0
    assert run.source_hash
    assert run.run_id.startswith("v1__")


def test_canonicalize_batch_preserves_regime_role_and_summary() -> None:
    parsed_runs = [
        _build_parsed_measurement(source_file="continuous_0.5_10.csv", commanded_cycles=10.0, freq_hz=0.5),
        _build_parsed_measurement(source_file="validation_1.0_10.csv", commanded_cycles=1.0, freq_hz=1.0),
    ]

    runs = canonicalize_batch(
        parsed_runs,
        regime="continuous",
        role="validation",
        config=CanonicalizeConfig(preferred_field_axis="bz_mT"),
    )
    summary = summarize_canonical_runs(runs)

    assert len(runs) == 2
    assert all(run.regime == "continuous" for run in runs)
    assert all(run.role == "validation" for run in runs)
    assert summary["primary_field_axis"].tolist() == ["bz", "bz"]
    assert summary["commanded_cycles"].tolist() == [10.0, 1.0]
    assert "quality_flags" in summary.columns
    assert summary["source_file"].tolist() == ["continuous_0.5_10.csv", "validation_1.0_10.csv"]
