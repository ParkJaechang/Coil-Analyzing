from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = REPO_ROOT / "tests"
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

import test_finite_empirical_field_route as finite_fixture
from field_analysis.parser import infer_dataset_filename_metadata


def test_selected_support_family_is_explicit_for_cross_family_selection() -> None:
    result = finite_fixture._run_field_compensation(
        finite_support_entries=[
            finite_fixture._build_finite_entry(
                test_id="finite_tri_only",
                waveform_type="triangle",
                freq_hz=3.0,
                cycle_count=1.25,
                field_pp=90.0,
            )
        ],
        waveform_type="sine",
        freq_hz=3.0,
        target_cycle_count=1.25,
    )
    profile = result["command_profile"]

    assert result["requested_support_family"] == "sine"
    assert result["selected_support_family"] == "triangle"
    assert result["selected_support_waveform_family"] == "triangle"
    assert result["support_family_override_applied"] is True
    assert result["support_family_override_reason"] in {
        "requested_family_unavailable",
        "cross_family_candidate_scored_better",
    }
    assert result["support_family_score_summary"]
    assert str(profile["selected_support_family"].iloc[0]) == "triangle"
    assert str(profile["selected_support_waveform_family"].iloc[0]) == "triangle"
    assert str(profile["selected_support_family"].iloc[0]).lower() not in {"", "n/a", "nan", "none"}
    assert result["requested_support_family_normalized"] == "sine"
    assert result["selected_support_family_normalized"] == "triangle"


def test_cycle_mismatch_reports_match_type_and_reason() -> None:
    result = finite_fixture._run_field_compensation(
        finite_support_entries=[
            finite_fixture._build_finite_entry(
                test_id="finite_one_cycle_representative",
                waveform_type="sine",
                freq_hz=3.0,
                cycle_count=1.0,
                field_pp=85.0,
            )
        ],
        waveform_type="sine",
        freq_hz=3.0,
        target_cycle_count=1.5,
    )
    profile = result["command_profile"]

    assert result["requested_cycle_count"] == 1.5
    assert result["selected_support_cycle_count"] == 1.0
    assert result["support_cycle_override_applied"] is True
    assert result["support_cycle_match_type"] in {"nearest", "representative", "weighted_blend", "fallback"}
    assert result["support_cycle_override_reason"]
    assert str(profile["support_cycle_match_type"].iloc[0]) == result["support_cycle_match_type"]
    assert str(profile["support_cycle_override_reason"].iloc[0]) == result["support_cycle_override_reason"]


def test_weighted_blend_reports_representative_reason_even_without_cycle_override() -> None:
    result = finite_fixture._run_field_compensation(
        finite_support_entries=[
            finite_fixture._build_finite_entry(
                test_id="finite_exact_a",
                waveform_type="sine",
                freq_hz=1.0,
                cycle_count=1.0,
                field_pp=84.0,
            ),
            finite_fixture._build_finite_entry(
                test_id="finite_exact_b",
                waveform_type="sine",
                freq_hz=1.0,
                cycle_count=1.0,
                field_pp=86.0,
            ),
        ],
        waveform_type="sine",
        freq_hz=1.0,
        target_cycle_count=1.0,
    )

    assert result["support_cycle_match_type"] == "weighted_blend"
    assert result["support_cycle_override_applied"] is False
    assert result["support_cycle_override_reason"] == "weighted_support_blend_uses_multiple_candidates"


def test_tri_alias_and_declared_cycle_are_preserved_separately_from_measured_duration() -> None:
    inferred = infer_dataset_filename_metadata("finite_tri_1Hz_1cycle.csv")
    entry = finite_fixture._build_finite_entry(
        test_id="finite_tri_1hz_1cycle",
        waveform_type="triangle",
        freq_hz=1.0,
        cycle_count=1.0,
        field_pp=88.0,
    )
    entry["source_file"] = "finite_tri_1Hz_1cycle.csv"
    entry["approx_cycle_span"] = 3.036
    entry["estimated_cycle_span"] = 3.036
    entry.pop("requested_cycle_count", None)
    entry.pop("declared_cycle_count", None)

    result = finite_fixture._run_field_compensation(
        finite_support_entries=[entry],
        waveform_type="sine",
        freq_hz=1.0,
        target_cycle_count=1.0,
    )
    profile = result["command_profile"]

    assert inferred["waveform"] == "triangle"
    assert inferred["cycle"] == 1.0
    assert result["selected_support_family"] == "triangle"
    assert result["selected_support_family_normalized"] == "triangle"
    assert result["support_family_override_applied"] is True
    assert result["support_family_override_reason"] not in {"", None, "n/a"}
    assert result["selected_support_declared_cycle_count"] == 1.0
    assert result["selected_support_measured_active_cycle_count"] == 3.036
    assert result["selected_support_target_aligned_cycle_count"] == 1.0
    assert result["selected_support_cycle_count"] == 1.0
    assert result["support_cycle_match_type"] == "exact"
    assert result["support_cycle_override_applied"] is False
    assert float(profile["selected_support_declared_cycle_count"].iloc[0]) == 1.0
    assert float(profile["selected_support_measured_active_cycle_count"].iloc[0]) == 3.036


def test_finite_tri_id_without_waveform_metadata_resolves_family_and_declared_cycle() -> None:
    entry = finite_fixture._build_finite_entry(
        test_id="finite_tri_1Hz_1cycle_runtime_alias",
        waveform_type="triangle",
        freq_hz=1.0,
        cycle_count=1.0,
        field_pp=88.0,
    )
    entry.pop("waveform_type", None)
    entry["source_file"] = "finite_tri_1Hz_1cycle_runtime_alias.csv"
    entry["approx_cycle_span"] = 1.772
    entry["estimated_cycle_span"] = 1.772
    entry.pop("declared_cycle_count", None)
    entry.pop("requested_cycle_count", None)

    result = finite_fixture._run_field_compensation(
        finite_support_entries=[entry],
        waveform_type="sine",
        freq_hz=1.0,
        target_cycle_count=1.0,
    )
    profile = result["command_profile"]

    assert result["selected_support_id"] == "finite_tri_1Hz_1cycle_runtime_alias"
    assert result["selected_support_family"] == "triangle"
    assert result["selected_support_family_normalized"] == "triangle"
    assert result["selected_support_family_source"] in {"explicit_metadata", "source_file_alias", "support_id_alias"}
    assert result["selected_support_family_unknown_reason"] in {None, ""}
    assert result["support_family_override_applied"] is True
    assert result["support_family_override_reason"] not in {"", None, "n/a"}
    assert result["selected_support_cycle_count"] == 1.0
    assert result["selected_support_declared_cycle_count"] == 1.0
    assert result["selected_support_measured_active_cycle_count"] == 1.772
    assert result["selected_support_cycle_source"] == "filename"
    assert float(profile["selected_support_cycle_count"].iloc[0]) == 1.0
    assert float(profile["selected_support_declared_cycle_count"].iloc[0]) == 1.0
    assert float(profile["selected_support_measured_active_cycle_count"].iloc[0]) == 1.772
    assert str(profile["selected_support_family"].iloc[0]) == "triangle"


def test_weighted_blend_runtime_alias_keeps_declared_cycle_user_facing() -> None:
    entry = finite_fixture._build_finite_entry(
        test_id="finite_tri_3Hz_1cycle_runtime_alias",
        waveform_type="triangle",
        freq_hz=3.0,
        cycle_count=1.0,
        field_pp=88.0,
    )
    entry.pop("waveform_type", None)
    entry["source_file"] = "finite_tri_3Hz_1cycle_runtime_alias.csv"
    entry["approx_cycle_span"] = 7.082
    entry["estimated_cycle_span"] = 7.082
    entry.pop("declared_cycle_count", None)
    entry.pop("requested_cycle_count", None)

    result = finite_fixture._run_field_compensation(
        finite_support_entries=[entry],
        waveform_type="sine",
        freq_hz=3.0,
        target_cycle_count=1.25,
    )

    assert result["selected_support_family"] == "triangle"
    assert result["support_family_override_applied"] is True
    assert result["support_family_override_reason"] not in {"", None, "n/a"}
    assert result["selected_support_cycle_count"] == 1.0
    assert result["selected_support_declared_cycle_count"] == 1.0
    assert result["selected_support_measured_active_cycle_count"] == 7.082
    assert result["support_cycle_match_type"] in {"nearest", "representative", "weighted_blend", "fallback"}
    assert result["support_cycle_override_reason"] not in {"", None, "n/a"}


def test_support_reference_is_target_aligned_and_separate_from_raw_source() -> None:
    result = finite_fixture._run_field_compensation(
        finite_support_entries=[
            finite_fixture._build_finite_entry(
                test_id="finite_exact_source",
                waveform_type="sine",
                freq_hz=3.0,
                cycle_count=1.25,
                field_pp=95.0,
            )
        ],
        waveform_type="sine",
        freq_hz=3.0,
        target_cycle_count=1.25,
    )
    profile = result["command_profile"]

    assert result["support_reference_timebase"] == "target_aligned"
    assert result["support_reference_plotted_source"] == "target_aligned_support_reference"
    assert result["support_reference_plotted_column"] == "support_reference_output_mT"
    assert result["support_reference_alignment_status"] == "ok"
    assert result["support_reference_selected_support_id"] == result["selected_support_id"]
    assert "target_aligned_support_reference_mT" in profile.columns
    assert np.allclose(
        pd.to_numeric(profile["support_reference_output_mT"], errors="coerce"),
        pd.to_numeric(profile["target_aligned_support_reference_mT"], errors="coerce"),
        equal_nan=True,
    )
    assert not np.allclose(
        pd.to_numeric(profile["support_reference_output_mT"], errors="coerce"),
        pd.to_numeric(profile["predicted_field_mT"], errors="coerce"),
        equal_nan=True,
    )

    assert result["selected_support_source_available"] is True
    assert result["selected_support_source_mT"] is not None
    assert result["selected_support_source_time_s"] is not None
    assert float(result["selected_support_original_pp_mT"]) > 0.0
    assert float(result["selected_support_original_duration_s"]) > 0.0
    assert float(result["selected_support_original_nonzero_end_s"]) > 0.0
    assert str(profile["selected_support_source_available"].iloc[0]) == "True"


def test_physical_target_is_unchanged_by_support_provenance_contract() -> None:
    result = finite_fixture._run_field_compensation(
        finite_support_entries=[
            finite_fixture._build_finite_entry(
                test_id="finite_target_invariant",
                waveform_type="triangle",
                freq_hz=5.0,
                cycle_count=1.5,
                field_pp=105.0,
            )
        ],
        waveform_type="triangle",
        freq_hz=5.0,
        target_cycle_count=1.5,
    )
    profile = result["command_profile"]

    assert result["target_shape_family"] == "rounded_triangle"
    assert float(result["target_pp_fixed"]) == 100.0
    assert np.allclose(
        pd.to_numeric(profile["physical_target_output_mT"], errors="coerce"),
        pd.to_numeric(profile["target_field_mT"], errors="coerce"),
        equal_nan=True,
    )
