from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIELD_ANALYSIS_SRC = ROOT.parent / "src"
if str(FIELD_ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(FIELD_ANALYSIS_SRC))

from field_analysis.analysis import analyze_measurements
from field_analysis.canonicalize import CanonicalizeConfig, canonicalize_run
from field_analysis.models import CycleDetectionConfig, ParsedMeasurement, PreprocessConfig, SheetPreview
from field_analysis.segmentation import build_analysis_frame_from_canonical, segment_transient_run


def _build_parsed_measurement(
    *,
    source_file: str = "segmentation.csv",
    cycles: float = 3.0,
    freq_hz: float = 1.0,
) -> ParsedMeasurement:
    time_s = np.array(
        [0.00, 0.04, 0.09, 0.15, 0.22, 0.31, 0.41, 0.52, 0.64, 0.77, 0.91, 1.06, 1.22, 1.39, 1.57, 1.76, 1.96, 2.17, 2.39, 2.62, 2.86, 3.11],
        dtype=float,
    )
    phase = time_s * freq_hz
    drive = 2.0 * np.sin(2.0 * np.pi * phase)
    current_signed = 5.0 * np.sin(2.0 * np.pi * phase)
    normalized_frame = pd.DataFrame(
        {
            "time_s": time_s,
            "daq_input_v": drive,
            "coil1_current_a": np.zeros_like(current_signed),
            "coil2_current_a": np.abs(current_signed) + 0.05,
            "bx_mT": 0.1 * np.sin(2.0 * np.pi * phase),
            "by_mT": 0.2 * np.sin(2.0 * np.pi * phase),
            "bz_mT": 8.0 * np.sin(2.0 * np.pi * phase),
            "temperature_c": np.full_like(current_signed, 25.0),
            "waveform_type": "Sine",
            "freq_hz": freq_hz,
            "current_pp_target_a": 10.0,
            "current_pk_target_a": 5.0,
            "cycle_total_expected": cycles,
            "source_cycle_no": np.zeros(len(time_s), dtype=float),
        }
    )
    preview = SheetPreview(
        sheet_name="main",
        row_count=len(normalized_frame),
        column_count=len(normalized_frame.columns),
        columns=list(normalized_frame.columns),
        header_row_index=0,
        metadata={"cycle": str(cycles)},
        preview_rows=normalized_frame.head(3).to_dict(orient="records"),
        recommended_mapping={},
    )
    return ParsedMeasurement(
        source_file=source_file,
        file_type="csv",
        sheet_name="main",
        structure_preview=preview,
        metadata={"cycle": str(cycles), "waveform": "sine", "freq_hz": str(freq_hz)},
        mapping={},
        raw_frame=normalized_frame.copy(),
        normalized_frame=normalized_frame,
        warnings=[],
        logs=[],
    )


def test_build_analysis_frame_from_canonical_uses_canonical_grid() -> None:
    parsed = _build_parsed_measurement(cycles=3.0, freq_hz=1.0)
    canonical = canonicalize_run(parsed, regime="continuous", role="train", config=CanonicalizeConfig())

    frame = build_analysis_frame_from_canonical(parsed, canonical)

    assert len(frame) == len(canonical.time_s)
    assert np.allclose(frame["time_s"].to_numpy(dtype=float), canonical.time_s)
    assert np.allclose(frame["daq_input_v"].to_numpy(dtype=float), canonical.input_v)
    assert np.allclose(frame["i_sum_signed"].to_numpy(dtype=float), canonical.signed_current_a)
    assert frame["test_id"].notna().all()
    assert frame["bproj_mT"].notna().all()


def test_segment_transient_run_handles_fractional_cycles() -> None:
    parsed = _build_parsed_measurement(source_file="finite.csv", cycles=1.25, freq_hz=1.0)
    canonical = canonicalize_run(parsed, regime="transient", role="train", config=CanonicalizeConfig())
    frame = build_analysis_frame_from_canonical(parsed, canonical)

    result = segment_transient_run(
        canonical,
        frame,
        CycleDetectionConfig(reference_channel="i_sum_signed", expected_cycles=2),
    )

    assert result.estimated_frequency_hz == 1.0
    assert len(result.boundaries) == 2
    assert result.annotated_frame["cycle_index"].notna().any()
    assert int(np.nanmax(result.annotated_frame["cycle_index"].to_numpy(dtype=float))) == 2


def test_analyze_measurements_uses_canonical_runs() -> None:
    parsed = _build_parsed_measurement(cycles=3.0, freq_hz=1.0)
    canonical = canonicalize_run(parsed, regime="continuous", role="train", config=CanonicalizeConfig())

    analyses = analyze_measurements(
        [parsed],
        preprocess_config=PreprocessConfig(),
        cycle_config=CycleDetectionConfig(reference_channel="i_sum_signed", expected_cycles=3),
        current_channel="i_sum_signed",
        main_field_axis="bz_mT",
        canonical_runs=[canonical],
    )

    assert len(analyses) == 1
    analysis = analyses[0]
    assert len(analysis.preprocess.corrected_frame) == len(canonical.time_s)
    assert not analysis.per_cycle_summary.empty
    assert not analysis.per_test_summary.empty
