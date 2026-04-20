from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIELD_ANALYSIS_SRC = ROOT.parent / "src"
if str(FIELD_ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(FIELD_ANALYSIS_SRC))

from field_analysis.canonicalize import CanonicalizeConfig, canonicalize_run
from field_analysis.models import ParsedMeasurement, PreprocessResult, SheetPreview
from field_analysis.recommendation_service import build_finite_support_entries
from field_analysis.support_extraction import build_finite_support_entries_from_canonical


def _build_transient_case(cycles: float = 1.25, freq_hz: float = 1.0) -> tuple[ParsedMeasurement, PreprocessResult]:
    time_s = np.linspace(0.0, cycles / freq_hz, 160)
    phase = time_s * freq_hz
    current = 5.0 * np.sin(2.0 * np.pi * phase)
    frame = pd.DataFrame(
        {
            "time_s": time_s,
            "daq_input_v": 2.0 * np.sin(2.0 * np.pi * phase),
            "coil1_current_a": np.zeros_like(current),
            "coil2_current_a": np.abs(current) + 0.05,
            "bx_mT": 0.1 * np.sin(2.0 * np.pi * phase),
            "by_mT": 0.2 * np.sin(2.0 * np.pi * phase),
            "bz_mT": 8.0 * np.sin(2.0 * np.pi * phase),
            "waveform_type": "Sine",
            "freq_hz": freq_hz,
            "current_pp_target_a": 10.0,
            "current_pk_target_a": 5.0,
            "cycle_total_expected": cycles,
            "source_cycle_no": np.zeros(len(time_s)),
        }
    )
    preview = SheetPreview(
        sheet_name="main",
        row_count=len(frame),
        column_count=len(frame.columns),
        columns=list(frame.columns),
        header_row_index=0,
        metadata={"cycle": str(cycles)},
        preview_rows=frame.head(3).to_dict(orient="records"),
        recommended_mapping={},
    )
    parsed = ParsedMeasurement(
        source_file="finite_case.csv",
        file_type="csv",
        sheet_name="main",
        structure_preview=preview,
        metadata={"cycle": str(cycles), "waveform": "sine", "freq_hz": str(freq_hz)},
        mapping={},
        raw_frame=frame.copy(),
        normalized_frame=frame.copy(),
        warnings=[],
        logs=[],
    )
    preprocess = PreprocessResult(
        corrected_frame=frame.copy(),
        offsets={},
        lags=[],
        warnings=[],
        logs=[],
    )
    return parsed, preprocess


def test_build_finite_support_entries_from_canonical_uses_commanded_cycles() -> None:
    parsed, preprocess = _build_transient_case(cycles=1.25, freq_hz=1.0)
    canonical = canonicalize_run(parsed, regime="transient", role="train", config=CanonicalizeConfig())

    entries = build_finite_support_entries_from_canonical(
        transient_measurements=[parsed],
        transient_preprocess_results=[preprocess],
        transient_canonical_runs=[canonical],
        current_channel="i_sum_signed",
        field_channel="bz_mT",
    )

    assert len(entries) == 1
    entry = entries[0]
    assert np.isclose(entry["approx_cycle_span"], 1.25, atol=1e-6)
    assert entry["segmentation_mode"] == "canonical_transient"
    assert entry["boundary_count"] == 2
    assert entry["frame"]["cycle_index"].notna().any()
    assert np.isclose(entry["requested_current_pp"], 10.0, atol=1e-6)


def test_recommendation_service_support_builder_prefers_canonical_entries() -> None:
    parsed, preprocess = _build_transient_case(cycles=1.25, freq_hz=1.0)
    canonical = canonicalize_run(parsed, regime="transient", role="train", config=CanonicalizeConfig())

    entries = build_finite_support_entries(
        transient_measurements=[parsed],
        transient_preprocess_results=[preprocess],
        transient_canonical_runs=[canonical],
        current_channel="i_sum_signed",
        field_channel="bz_mT",
    )

    assert len(entries) == 1
    assert entries[0]["segmentation_mode"] == "canonical_transient"
    assert np.isclose(entries[0]["approx_cycle_span"], 1.25, atol=1e-6)
    assert np.isclose(entries[0]["requested_current_pp"], 10.0, atol=1e-6)
