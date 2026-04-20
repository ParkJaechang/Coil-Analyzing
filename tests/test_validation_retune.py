from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIELD_ANALYSIS_SRC = ROOT.parent / "src"
if str(FIELD_ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(FIELD_ANALYSIS_SRC))

from field_analysis.validation_retune import (  # noqa: E402
    SOURCE_KIND_CORRECTED,
    SOURCE_KIND_RECOMMENDATION,
    ValidationComparison,
    ValidationRun,
    build_prediction_debug_snapshot,
    build_validation_comparison,
    build_retune_acceptance_decision,
    build_retune_quality_badge_payload,
    execute_validation_retune,
    save_retune_artifacts,
)
from field_analysis.validation_retune_catalog import (  # noqa: E402
    build_corrected_lut_catalog_payload,
    build_retune_picker_payload,
    build_validation_catalog_payload,
)


def _normalized_waveform(phase: np.ndarray, waveform_type: str) -> np.ndarray:
    phase_array = np.mod(np.asarray(phase, dtype=float), 1.0)
    if waveform_type == "triangle":
        waveform = np.empty_like(phase_array)
        rising = phase_array < 0.25
        falling = (phase_array >= 0.25) & (phase_array < 0.75)
        waveform[rising] = 4.0 * phase_array[rising]
        waveform[falling] = 2.0 - 4.0 * phase_array[falling]
        waveform[~(rising | falling)] = -4.0 + 4.0 * phase_array[~(rising | falling)]
        return waveform
    return np.sin(2.0 * np.pi * phase_array)


def _build_profiles(
    *,
    target_output_type: str = "current",
    waveform_type: str = "sine",
    freq_hz: float = 1.0,
    cycle_count: float = 1.0,
    finite_cycle_mode: bool = False,
    validation_lead_silence_s: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    duration_s = float(cycle_count) / float(freq_hz)
    time_s = np.linspace(0.0, duration_s, 257)
    phase = np.linspace(0.0, float(cycle_count), len(time_s))
    drive = _normalized_waveform(phase, waveform_type)
    target_peak = 5.0 if target_output_type == "current" else 20.0
    target_signal = target_peak * drive
    measured_signal = 0.82 * target_peak * _normalized_waveform(phase - 0.06, waveform_type)
    base_profile = pd.DataFrame(
        {
            "time_s": time_s,
            "waveform_type": waveform_type,
            "freq_hz": float(freq_hz),
            "finite_cycle_mode": bool(finite_cycle_mode),
            "target_cycle_count": float(cycle_count) if finite_cycle_mode else np.nan,
            "is_active_target": True,
            "recommended_voltage_v": 2.0 * drive,
            "limited_voltage_v": 2.0 * drive,
            "within_hardware_limits": True,
            "within_daq_limit": True,
            "within_amp_gain_limit": True,
            "within_amp_output_limit": True,
            "target_output": target_signal,
            "expected_output": 0.9 * target_signal,
            "target_output_pp": float(np.nanmax(target_signal) - np.nanmin(target_signal)),
            "expected_field_mT": 1.8 * 0.9 * target_signal,
            "modeled_field_mT": 1.8 * 0.9 * target_signal,
        }
    )
    if target_output_type == "current":
        base_profile["target_current_a"] = target_signal
        base_profile["expected_current_a"] = 0.9 * target_signal
    else:
        base_profile["target_field_mT"] = target_signal
        base_profile["expected_field_mT"] = 0.9 * target_signal
        base_profile["modeled_field_mT"] = 0.9 * target_signal
        base_profile["expected_current_a"] = 0.2 * 0.9 * target_signal

    validation_time = time_s
    validation_current = measured_signal if target_output_type == "current" else 0.4 * measured_signal
    validation_bz = measured_signal if target_output_type == "field" else 2.0 * measured_signal
    validation_voltage = 2.0 * drive
    if validation_lead_silence_s > 0:
        dt = float(np.nanmedian(np.diff(time_s))) if len(time_s) > 1 else 1.0 / 256.0
        silence_time = np.arange(0.0, float(validation_lead_silence_s), dt, dtype=float)
        validation_time = np.concatenate([silence_time, validation_lead_silence_s + time_s])
        validation_current = np.concatenate([np.zeros_like(silence_time), validation_current])
        validation_bz = np.concatenate([np.zeros_like(silence_time), validation_bz])
        validation_voltage = np.concatenate([np.zeros_like(silence_time), validation_voltage])
    validation_frame = pd.DataFrame(
        {
            "time_s": validation_time,
            "daq_input_v": validation_voltage,
            "i_sum_signed": validation_current,
            "bz_mT": validation_bz,
        }
    )
    candidate = {
        "test_id": f"{waveform_type}_{freq_hz:g}Hz_{cycle_count:g}cycle",
        "source_file": f"{waveform_type}_{freq_hz:g}.csv",
        "score": 0.05,
        "eligible": True,
        "freq_hz": float(freq_hz),
        "output_pp": float(np.nanmax(measured_signal) - np.nanmin(measured_signal)),
    }
    return base_profile, validation_frame, candidate


def _build_validation_run_stub(
    *,
    exact_path: str = "exact_current",
    target_output_type: str = "current",
    metadata: dict[str, object] | None = None,
) -> ValidationRun:
    return ValidationRun(
        export_file_prefix="retune_stub",
        lut_id="baseline_lut",
        source_kind=SOURCE_KIND_RECOMMENDATION,
        source_selection_id="baseline_lut",
        source_lut_filename="baseline_lut.csv",
        source_profile_path="D:/tmp/baseline_lut.csv",
        original_recommendation_id="baseline_lut",
        validation_run_id="validation_stub",
        corrected_lut_id="baseline_lut__corrected_iter01",
        iteration_index=1,
        exact_path=exact_path,
        target_output_type=target_output_type,
        waveform_type="sine",
        freq_hz=1.0,
        commanded_cycles=1.0,
        finite_cycle_mode=exact_path == "finite_exact",
        target_level_value=20.0,
        target_level_kind="pp",
        selected_validation_test_id="validation_stub_case",
        selected_validation_source_file="D:/tmp/validation.csv",
        measured_file_name="validation.csv",
        created_at="2026-04-16T00:00:00",
        correction_rule="validation_residual_recommendation_loop[correction_gain=0.7]",
        metadata=metadata or {},
    )


def _build_comparison(
    *,
    label: str,
    nrmse: float,
    shape_corr: float,
    phase_lag_s: float,
    metrics_available: bool = True,
    unavailable_reason: str | None = None,
    reason_codes: list[str] | None = None,
    clipping_detected: bool = False,
    saturation_detected: bool = False,
    valid_sample_count: int = 256,
    metric_domain: str = "bz_effective",
    target_basis: str = "target_field_mT",
    comparison_source: str = "actual",
) -> ValidationComparison:
    return ValidationComparison(
        label=label,
        output_column="bz_mT" if metric_domain == "bz_effective" else "i_sum_signed",
        rmse=nrmse,
        nrmse=nrmse,
        shape_corr=shape_corr,
        phase_lag_s=phase_lag_s,
        pp_error=0.0,
        peak_error=0.0,
        clipping_detected=clipping_detected,
        saturation_detected=saturation_detected,
        metric_domain=metric_domain,
        target_basis=target_basis,
        comparison_source=comparison_source,
        metrics_available=metrics_available,
        unavailable_reason=unavailable_reason,
        reason_codes=list(reason_codes or []),
        valid_sample_count=valid_sample_count,
    )


def test_validation_retune_current_exact_smoke(tmp_path: Path) -> None:
    base_profile, validation_frame, candidate = _build_profiles(target_output_type="current", waveform_type="sine")
    result = execute_validation_retune(
        base_profile=base_profile,
        validation_candidate=candidate,
        validation_frame=validation_frame,
        export_file_prefix="current_exact_validation",
        target_output_type="current",
        current_channel="i_sum_signed",
        field_channel="bz_mT",
        max_daq_voltage_pp=20.0,
        amp_gain_at_100_pct=20.0,
        amp_gain_limit_pct=100.0,
        amp_max_output_pk_v=180.0,
        support_amp_gain_pct=100.0,
        correction_gain=0.75,
        max_iterations=2,
        improvement_threshold=0.0,
        original_recommendation_id="rec_current_exact",
        source_selection={
            "source_kind": SOURCE_KIND_RECOMMENDATION,
            "lut_id": "rec_current_exact",
            "profile_csv_path": "D:/tmp/rec_current_exact.csv",
            "source_lut_filename": "rec_current_exact.csv",
        },
    )

    assert result is not None
    assert not result.overlay_frame.empty
    assert "actual_measured" in result.overlay_frame.columns
    assert "actual_bz_effective" in result.overlay_frame.columns
    assert result.corrected_command_profile is not None
    paths = save_retune_artifacts(retune_result=result, output_dir=tmp_path)
    assert Path(paths["corrected_waveform_csv"]).exists()
    assert Path(paths["validation_report_json"]).exists()
    assert Path(paths["validation_report_md"]).exists()
    assert Path(paths["retune_result_json"]).exists()
    assert Path(paths["retune_result_md"]).exists()
    assert result.validation_run.exact_path == "exact_current"
    assert result.validation_run.iteration_index == 1
    assert result.validation_run.source_lut_filename == "rec_current_exact.csv"
    assert result.quality_badge["metric_domain"] == "bz_effective"
    assert result.baseline_bz_comparison.metrics_available is True
    assert result.corrected_bz_comparison.metrics_available is True
    assert result.corrected_comparison.nrmse <= result.baseline_comparison.nrmse
    assert result.acceptance_decision["decision"] == "improved_and_accepted"
    assert result.preferred_output_id == result.validation_run.corrected_lut_id
    assert result.artifact_payload["preferred_output_id"] == result.validation_run.corrected_lut_id
    assert result.artifact_payload["acceptance_decision"]["decision"] == "improved_and_accepted"
    assert result.artifact_payload["preferred_output_kind"] == "corrected_candidate"
    assert result.artifact_payload["candidate_status"] == "improved_and_accepted"
    assert result.artifact_payload["rejection_reason"] is None


def test_validation_retune_field_exact_smoke(tmp_path: Path) -> None:
    base_profile, validation_frame, candidate = _build_profiles(target_output_type="field", waveform_type="sine", freq_hz=0.5)
    result = execute_validation_retune(
        base_profile=base_profile,
        validation_candidate=candidate,
        validation_frame=validation_frame,
        export_file_prefix="field_exact_validation",
        target_output_type="field",
        current_channel="i_sum_signed",
        field_channel="bz_mT",
        max_daq_voltage_pp=20.0,
        amp_gain_at_100_pct=20.0,
        amp_gain_limit_pct=100.0,
        amp_max_output_pk_v=180.0,
        support_amp_gain_pct=100.0,
        correction_gain=0.75,
        max_iterations=2,
        improvement_threshold=0.0,
        original_recommendation_id="rec_field_exact",
        source_selection={"source_kind": SOURCE_KIND_RECOMMENDATION, "lut_id": "rec_field_exact"},
    )

    assert result is not None
    paths = save_retune_artifacts(retune_result=result, output_dir=tmp_path)
    assert "corrected_control_lut_csv" in paths
    report_payload = result.artifact_payload
    assert report_payload["validation_run"]["target_output_type"] == "field"
    assert report_payload["provenance"]["source_kind"] == SOURCE_KIND_RECOMMENDATION
    assert report_payload["provenance"]["iteration_index"] == 1
    assert "before_after_metrics" in report_payload
    assert Path(paths["corrected_control_lut_csv"]).exists()


def test_validation_retune_finite_exact_smoke(tmp_path: Path) -> None:
    base_profile, validation_frame, candidate = _build_profiles(
        target_output_type="current",
        waveform_type="triangle",
        freq_hz=2.0,
        cycle_count=1.25,
        finite_cycle_mode=True,
    )
    result = execute_validation_retune(
        base_profile=base_profile,
        validation_candidate=candidate,
        validation_frame=validation_frame,
        export_file_prefix="finite_exact_validation",
        target_output_type="current",
        current_channel="i_sum_signed",
        field_channel="bz_mT",
        max_daq_voltage_pp=20.0,
        amp_gain_at_100_pct=20.0,
        amp_gain_limit_pct=100.0,
        amp_max_output_pk_v=180.0,
        support_amp_gain_pct=100.0,
        correction_gain=0.75,
        max_iterations=2,
        improvement_threshold=0.0,
        original_recommendation_id="finite_export_root",
        source_selection={
            "source_kind": "export",
            "lut_id": "finite_export_root",
            "profile_csv_path": "D:/tmp/finite_export_root.csv",
            "source_lut_filename": "finite_export_root.csv",
        },
    )

    assert result is not None
    assert bool(result.corrected_command_profile["finite_cycle_mode"].iloc[0]) is True
    paths = save_retune_artifacts(retune_result=result, output_dir=tmp_path)
    assert Path(paths["corrected_waveform_csv"]).exists()
    assert result.validation_run.commanded_cycles == 1.25
    assert result.validation_run.exact_path == "finite_exact"
    assert result.baseline_bz_comparison.metrics_available is True
    assert result.loop_summary["effective_correction_gain"] == 0.35
    assert result.validation_run.metadata["requested_correction_gain"] == 0.75


def test_quality_badge_uses_bz_thresholds() -> None:
    comparison = ValidationComparison(
        label="after_retune_bz",
        output_column="bz_mT",
        rmse=0.1,
        nrmse=0.12,
        shape_corr=0.985,
        phase_lag_s=0.01,
        pp_error=0.0,
        peak_error=0.0,
        clipping_detected=False,
        saturation_detected=False,
        metric_domain="bz_effective",
        target_basis="expected_field_surrogate",
        comparison_source="predicted",
    )
    badge = build_retune_quality_badge_payload(comparison)
    assert badge["label"] == "재현 양호"
    assert badge["metric_domain"] == "bz_effective"


def test_validation_catalog_lineage_and_picker_payload(tmp_path: Path) -> None:
    base_profile, validation_frame, candidate = _build_profiles(target_output_type="current", waveform_type="sine")
    first = execute_validation_retune(
        base_profile=base_profile,
        validation_candidate=candidate,
        validation_frame=validation_frame,
        export_file_prefix="current_exact_validation",
        target_output_type="current",
        current_channel="i_sum_signed",
        field_channel="bz_mT",
        max_daq_voltage_pp=20.0,
        amp_gain_at_100_pct=20.0,
        amp_gain_limit_pct=100.0,
        amp_max_output_pk_v=180.0,
        support_amp_gain_pct=100.0,
        correction_gain=0.75,
        max_iterations=2,
        improvement_threshold=0.0,
        original_recommendation_id="rec_current_exact",
        source_selection={"source_kind": SOURCE_KIND_RECOMMENDATION, "lut_id": "rec_current_exact"},
    )
    assert first is not None
    save_retune_artifacts(retune_result=first, output_dir=tmp_path)

    second = execute_validation_retune(
        base_profile=base_profile,
        validation_candidate=candidate,
        validation_frame=validation_frame,
        export_file_prefix="current_exact_validation_second",
        target_output_type="current",
        current_channel="i_sum_signed",
        field_channel="bz_mT",
        max_daq_voltage_pp=20.0,
        amp_gain_at_100_pct=20.0,
        amp_gain_limit_pct=100.0,
        amp_max_output_pk_v=180.0,
        support_amp_gain_pct=100.0,
        correction_gain=0.75,
        max_iterations=2,
        improvement_threshold=0.0,
        original_recommendation_id="rec_current_exact",
        source_selection={
            "source_kind": SOURCE_KIND_CORRECTED,
            "lut_id": first.validation_run.corrected_lut_id,
            "original_recommendation_id": "rec_current_exact",
        },
    )
    assert second is not None
    save_retune_artifacts(retune_result=second, output_dir=tmp_path)

    validation_payload = build_validation_catalog_payload([tmp_path], tmp_path / "missing_history.json")
    corrected_payload = build_corrected_lut_catalog_payload(validation_payload["entries"])
    lut_entries = [
        {
            "lut_id": "rec_current_exact",
            "catalog_source_kind": SOURCE_KIND_RECOMMENDATION,
            "original_recommendation_id": "rec_current_exact",
            "file_name": "rec_current_exact.csv",
            "created_at": "2026-04-16T00:00:00",
            "target_type": "current",
            "waveform": "sine",
            "freq_hz": 1.0,
            "cycle_count": None,
            "route_origin": "exact",
            "exact_support": True,
            "exact_path": "exact_current",
            "profile_csv_path": "D:/tmp/rec_current_exact.csv",
            "control_lut_path": None,
        }
    ]
    picker_payload = build_retune_picker_payload(
        lut_entries=lut_entries,
        validation_entries=validation_payload["entries"],
        corrected_entries=corrected_payload["entries"],
    )

    corrected_entries = corrected_payload["entries"]
    assert len(corrected_entries) == 2
    assert "candidate_status" in validation_payload["filters"]
    latest = next(item for item in corrected_entries if item["latest_corrected_candidate"])
    stale = next(item for item in corrected_entries if item["stale"])
    assert latest["iteration_index"] == 2
    assert stale["iteration_index"] == 1
    assert isinstance(latest["acceptance_decision"], dict)
    assert latest["candidate_status"] == latest["acceptance_decision"]["decision"]
    assert latest["preferred_output_id"] in {latest["lut_id"], latest["corrected_lut_id"]}
    assert latest["preferred_output_kind"] in {"baseline", "corrected_candidate"}
    corrected_picker = [item for item in picker_payload["entries"] if item["source_kind"] == SOURCE_KIND_CORRECTED]
    assert corrected_picker
    assert any(item["retune_eligible"] for item in corrected_picker)
    latest_picker = next(item for item in corrected_picker if item["source_id"] == latest["corrected_lut_id"])
    assert latest_picker["display_name"] == "current / sine / 1 Hz / 10 pp"
    assert latest_picker["display_label"] == "current / sine / 1 Hz / 10 pp | corrected iter02"
    assert latest_picker["current_lut_id"] == latest["corrected_lut_id"]
    assert latest_picker["corrected_lut_id"] == latest["corrected_lut_id"]
    assert latest_picker["validation_run_id"] == latest["validation_run_id"]
    source_picker = next(item for item in picker_payload["entries"] if item["source_kind"] == SOURCE_KIND_RECOMMENDATION)
    assert source_picker["latest_corrected_candidate_id"] == latest["latest_corrected_candidate_id"]
    assert source_picker["latest_validation_run_id"] == latest["validation_run_id"]
    assert source_picker["display_label"] == "current / sine / 1 Hz | recommendation"


def test_corrected_catalog_display_label_strips_hash_prefixes() -> None:
    validation_entries = [
        {
            "corrected_lut_id": "ff132682eb37f728_1.25hz_1.25cycle_20pp__corrected_iter01",
            "original_recommendation_id": "ff132682eb37f728_1.25hz_1.25cycle_20pp",
            "lut_id": "ff132682eb37f728_1.25hz_1.25cycle_20pp",
            "validation_run_id": "validation_run_01",
            "source_kind": SOURCE_KIND_RECOMMENDATION,
            "source_lut_filename": "ff132682eb37f728_1.25hz_1.25cycle_20pp.csv",
            "iteration_index": 1,
            "created_at": "2026-04-16T00:00:00",
            "exact_path": "finite_exact",
            "target_output_type": "current",
            "waveform_type": "sine",
            "freq_hz": 1.25,
            "commanded_cycles": 1.25,
            "target_level_value": 20.0,
            "target_level_kind": "pp",
            "display_object_key": "current::sine::1.25::1.25::20::pp",
            "display_name": "current / sine / 1.25 Hz / 1.25 cycle / 20 pp",
            "display_label": "current / sine / 1.25 Hz / 1.25 cycle / 20 pp | recommendation",
            "source_lut_display_name": "current / sine / 1.25 Hz / 1.25 cycle / 20 pp",
            "validation_test_display_name": "sine / 1.25 Hz bench run",
            "quality_label": "green",
            "quality_tone": "green",
            "quality_reasons": [],
            "acceptance_decision": {"decision": "improved_and_accepted"},
            "candidate_status": "improved_and_accepted",
            "candidate_status_label": "accepted",
            "preferred_output_id": "ff132682eb37f728_1.25hz_1.25cycle_20pp__corrected_iter01",
            "preferred_output_kind": "corrected_candidate",
            "rejection_reason": None,
            "lineage_root_id": "1.25hz_1.25cycle_20pp",
            "latest_corrected_candidate": True,
            "latest_corrected_candidate_id": "ff132682eb37f728_1.25hz_1.25cycle_20pp__corrected_iter01",
            "duplicate": False,
            "stale": False,
            "status": "latest_corrected_candidate",
            "artifact_complete": True,
            "report_path": "D:/tmp/report.json",
            "artifact_paths": {},
        }
    ]

    corrected_payload = build_corrected_lut_catalog_payload(validation_entries)
    entry = corrected_payload["entries"][0]

    assert entry["display_name"] == "current / sine / 1.25 Hz / 1.25 cycle / 20 pp"
    assert entry["display_label"] == "current / sine / 1.25 Hz / 1.25 cycle / 20 pp | corrected iter01"
    assert "ff132682eb37f728" not in entry["display_label"]
    assert "ff132682eb37f728" not in entry["source_lut_display_name"]


def test_acceptance_fields_match_report_result_and_catalog(tmp_path: Path) -> None:
    base_profile, validation_frame, candidate = _build_profiles(target_output_type="current", waveform_type="sine")
    result = execute_validation_retune(
        base_profile=base_profile,
        validation_candidate=candidate,
        validation_frame=validation_frame,
        export_file_prefix="current_exact_validation_consistency",
        target_output_type="current",
        current_channel="i_sum_signed",
        field_channel="bz_mT",
        max_daq_voltage_pp=20.0,
        amp_gain_at_100_pct=20.0,
        amp_gain_limit_pct=100.0,
        amp_max_output_pk_v=180.0,
        support_amp_gain_pct=100.0,
        correction_gain=0.75,
        max_iterations=2,
        improvement_threshold=0.0,
        original_recommendation_id="rec_current_exact",
        source_selection={"source_kind": SOURCE_KIND_RECOMMENDATION, "lut_id": "rec_current_exact"},
    )
    assert result is not None
    artifact_paths = save_retune_artifacts(retune_result=result, output_dir=tmp_path)

    report_payload = json.loads(Path(artifact_paths["validation_report_json"]).read_text(encoding="utf-8"))
    result_payload = json.loads(Path(artifact_paths["retune_result_json"]).read_text(encoding="utf-8"))
    validation_payload = build_validation_catalog_payload([tmp_path], tmp_path / "missing_history.json")
    corrected_payload = build_corrected_lut_catalog_payload(validation_payload["entries"])
    catalog_entry = next(
        item for item in corrected_payload["entries"] if item["corrected_lut_id"] == result.validation_run.corrected_lut_id
    )

    expected_decision = result.acceptance_decision["decision"]
    expected_preferred_id = result.preferred_output_id
    expected_preferred_kind = result.acceptance_decision["preferred_output_kind"]
    expected_rejection_reason = result.acceptance_decision["rejection_reason"]

    assert report_payload["acceptance_decision"]["decision"] == expected_decision
    assert result_payload["acceptance_decision"]["decision"] == expected_decision
    assert catalog_entry["acceptance_decision"]["decision"] == expected_decision

    assert report_payload["preferred_output_id"] == expected_preferred_id
    assert result_payload["preferred_output_id"] == expected_preferred_id
    assert catalog_entry["preferred_output_id"] == expected_preferred_id

    assert report_payload["preferred_output_kind"] == expected_preferred_kind
    assert result_payload["preferred_output_kind"] == expected_preferred_kind
    assert catalog_entry["preferred_output_kind"] == expected_preferred_kind

    assert report_payload["rejection_reason"] == expected_rejection_reason
    assert result_payload["rejection_reason"] == expected_rejection_reason
    assert catalog_entry["rejection_reason"] == expected_rejection_reason

    assert report_payload["candidate_status"] == expected_decision
    assert result_payload["candidate_status"] == expected_decision
    assert catalog_entry["candidate_status"] == expected_decision


def test_validation_retune_canonicalizes_delayed_real_window() -> None:
    base_profile, validation_frame, candidate = _build_profiles(
        target_output_type="current",
        waveform_type="sine",
        freq_hz=0.5,
        validation_lead_silence_s=2.5,
    )
    result = execute_validation_retune(
        base_profile=base_profile,
        validation_candidate=candidate,
        validation_frame=validation_frame,
        export_file_prefix="current_exact_validation_delayed",
        target_output_type="current",
        current_channel="i_sum_signed",
        field_channel="bz_mT",
        max_daq_voltage_pp=20.0,
        amp_gain_at_100_pct=20.0,
        amp_gain_limit_pct=100.0,
        amp_max_output_pk_v=180.0,
        support_amp_gain_pct=100.0,
        correction_gain=0.75,
        max_iterations=2,
        improvement_threshold=0.0,
        original_recommendation_id="rec_current_exact",
        source_selection={"source_kind": SOURCE_KIND_RECOMMENDATION, "lut_id": "rec_current_exact"},
    )

    assert result is not None
    window = result.validation_run.metadata["validation_window"]
    assert window["applied"] is True
    assert window["start_s"] > 2.0
    assert result.baseline_bz_comparison.metrics_available is True


def test_validation_retune_reports_missing_bz_channel_reason() -> None:
    base_profile, validation_frame, candidate = _build_profiles(target_output_type="current", waveform_type="sine")
    result = execute_validation_retune(
        base_profile=base_profile,
        validation_candidate=candidate,
        validation_frame=validation_frame.drop(columns=["bz_mT"]),
        export_file_prefix="current_exact_validation_missing_bz",
        target_output_type="current",
        current_channel="i_sum_signed",
        field_channel="bz_mT",
        max_daq_voltage_pp=20.0,
        amp_gain_at_100_pct=20.0,
        amp_gain_limit_pct=100.0,
        amp_max_output_pk_v=180.0,
        support_amp_gain_pct=100.0,
        correction_gain=0.75,
        max_iterations=2,
        improvement_threshold=0.0,
        original_recommendation_id="rec_current_exact",
        source_selection={"source_kind": SOURCE_KIND_RECOMMENDATION, "lut_id": "rec_current_exact"},
    )

    assert result is not None
    assert result.baseline_bz_comparison.metrics_available is False
    assert result.baseline_bz_comparison.unavailable_reason == "missing_bz_channel"
    assert "missing_bz_channel" in result.baseline_bz_comparison.reason_codes
    assert result.quality_badge["evaluation_status"] in {"evaluated", "unevaluable"}


def test_acceptance_gate_rejects_degraded_finite_candidate() -> None:
    validation_run = _build_validation_run_stub(
        exact_path="finite_exact",
        metadata={
            "bz_target_mapping": {"available": True, "basis": "mapped_target_bz_effective_mT"},
            "corrected_bz_projection": {
                "available": True,
                "source": "reference_voltage_to_bz_transfer",
                "reason_code": None,
            },
        },
    )
    baseline_target = _build_comparison(
        label="before_retune",
        nrmse=0.20,
        shape_corr=0.96,
        phase_lag_s=0.01,
        metric_domain="target_output",
        comparison_source="actual",
    )
    corrected_target = _build_comparison(
        label="after_retune",
        nrmse=0.12,
        shape_corr=0.98,
        phase_lag_s=0.008,
        metric_domain="target_output",
        comparison_source="predicted",
    )
    baseline_bz = _build_comparison(
        label="before_retune_bz",
        nrmse=0.18,
        shape_corr=0.97,
        phase_lag_s=0.01,
        saturation_detected=True,
        reason_codes=["clipped_actual"],
    )
    corrected_bz = _build_comparison(
        label="after_retune_bz",
        nrmse=0.42,
        shape_corr=0.89,
        phase_lag_s=0.032,
        comparison_source="predicted",
    )

    decision = build_retune_acceptance_decision(
        validation_run=validation_run,
        baseline_comparison=baseline_target,
        corrected_comparison=corrected_target,
        baseline_bz_comparison=baseline_bz,
        corrected_bz_comparison=corrected_bz,
    )

    assert decision["decision"] == "degraded_and_rejected"
    assert decision["preferred_output_id"] == "baseline_lut"
    assert decision["rejection_reason"] == "clipped_actual"
    assert "finite_alignment_sensitive" in decision["reason_codes"]
    assert "unstable_transfer_estimate" in decision["reason_codes"]
    assert "weak_bz_mapping" in decision["reason_codes"]
    assert "correction_overfit" in decision["reason_codes"]


def test_acceptance_gate_reports_metrics_unavailable() -> None:
    validation_run = _build_validation_run_stub(
        metadata={
            "bz_target_mapping": {"available": False, "basis": "mapped_target_bz_effective_mT"},
            "corrected_bz_projection": {
                "available": False,
                "source": "validation_transfer",
                "reason_code": "surrogate_unstable",
            },
        },
    )
    baseline_target = _build_comparison(
        label="before_retune",
        nrmse=0.20,
        shape_corr=0.96,
        phase_lag_s=0.01,
        metric_domain="target_output",
        comparison_source="actual",
    )
    corrected_target = _build_comparison(
        label="after_retune",
        nrmse=0.18,
        shape_corr=0.97,
        phase_lag_s=0.009,
        metric_domain="target_output",
        comparison_source="predicted",
    )
    baseline_bz = _build_comparison(
        label="before_retune_bz",
        nrmse=float("nan"),
        shape_corr=float("nan"),
        phase_lag_s=float("nan"),
        metrics_available=False,
        unavailable_reason="missing_bz_channel",
        reason_codes=["missing_bz_channel"],
        valid_sample_count=0,
    )
    corrected_bz = _build_comparison(
        label="after_retune_bz",
        nrmse=float("nan"),
        shape_corr=float("nan"),
        phase_lag_s=float("nan"),
        metrics_available=False,
        unavailable_reason="surrogate_unstable",
        reason_codes=["surrogate_unstable"],
        valid_sample_count=0,
        comparison_source="predicted",
    )

    decision = build_retune_acceptance_decision(
        validation_run=validation_run,
        baseline_comparison=baseline_target,
        corrected_comparison=corrected_target,
        baseline_bz_comparison=baseline_bz,
        corrected_bz_comparison=corrected_bz,
    )

    assert decision["decision"] == "metrics_unavailable"
    assert decision["preferred_output_id"] == "baseline_lut"
    assert decision["rejection_reason"] == "surrogate_unstable"
    assert "missing_bz_channel" in decision["reason_codes"]
    assert "unstable_transfer_estimate" in decision["reason_codes"]


def test_acceptance_gate_prefers_correction_overfit_for_field_degradation() -> None:
    validation_run = _build_validation_run_stub(
        exact_path="exact_field",
        target_output_type="field",
        metadata={
            "bz_target_mapping": {"available": True, "basis": "target_field_mT", "reason_code": None},
            "corrected_bz_projection": {"available": True, "source": "validation_voltage_to_bz_transfer", "reason_code": None},
        },
    )
    baseline_target = _build_comparison(
        label="before_retune",
        nrmse=0.13,
        shape_corr=0.99,
        phase_lag_s=-0.05,
        metric_domain="target_output",
        comparison_source="actual",
    )
    corrected_target = _build_comparison(
        label="after_retune",
        nrmse=0.21,
        shape_corr=0.95,
        phase_lag_s=0.0,
        metric_domain="target_output",
        comparison_source="predicted",
    )
    baseline_bz = _build_comparison(label="before_retune_bz", nrmse=0.13, shape_corr=0.99, phase_lag_s=-0.05)
    corrected_bz = _build_comparison(
        label="after_retune_bz",
        nrmse=0.21,
        shape_corr=0.95,
        phase_lag_s=0.0,
        comparison_source="predicted",
    )

    decision = build_retune_acceptance_decision(
        validation_run=validation_run,
        baseline_comparison=baseline_target,
        corrected_comparison=corrected_target,
        baseline_bz_comparison=baseline_bz,
        corrected_bz_comparison=corrected_bz,
    )

    assert decision["decision"] == "degraded_and_rejected"
    assert decision["rejection_reason"] == "correction_overfit"
    assert decision["reason_codes"] == ["correction_overfit"]


def test_build_validation_comparison_softens_near_limit_hardware_gate() -> None:
    time_s = np.linspace(0.0, 1.0, 129)
    target = np.sin(2.0 * np.pi * time_s)
    command_profile = pd.DataFrame(
        {
            "time_s": time_s,
            "target_output": target,
            "expected_output": 0.98 * target,
            "target_current_a": target,
            "expected_current_a": 0.98 * target,
            "within_hardware_limits": False,
            "within_daq_limit": True,
            "required_amp_gain_pct": 100.0,
            "available_amp_gain_pct": 99.82,
            "peak_input_limit_margin": 0.10,
            "p95_input_limit_margin": 0.11,
        }
    )
    validation_frame = pd.DataFrame(
        {
            "time_s": time_s,
            "i_sum_signed": 0.98 * target,
            "bz_mT": 1.8 * 0.98 * target,
            "daq_input_v": 2.0 * target,
        }
    )

    comparison = build_validation_comparison(
        command_profile=command_profile,
        validation_frame=validation_frame,
        label="after_retune",
        comparison_source="predicted",
        target_output_type="current",
        current_channel="i_sum_signed",
        field_channel="bz_mT",
        metric_domain="target_output",
    )

    assert comparison.clipping_detected is False


def test_build_validation_comparison_accepts_exact_tolerance_boundary() -> None:
    time_s = np.linspace(0.0, 1.0, 129)
    target = np.sin(2.0 * np.pi * time_s)
    command_profile = pd.DataFrame(
        {
            "time_s": time_s,
            "target_output": target,
            "expected_output": 0.98 * target,
            "target_current_a": target,
            "expected_current_a": 0.98 * target,
            "within_hardware_limits": False,
            "within_daq_limit": True,
            "required_amp_gain_pct": 100.0,
            "available_amp_gain_pct": 99.75,
            "peak_input_limit_margin": 0.10,
            "p95_input_limit_margin": 0.11,
        }
    )
    validation_frame = pd.DataFrame(
        {
            "time_s": time_s,
            "i_sum_signed": 0.98 * target,
            "bz_mT": 1.8 * 0.98 * target,
            "daq_input_v": 2.0 * target,
        }
    )

    comparison = build_validation_comparison(
        command_profile=command_profile,
        validation_frame=validation_frame,
        label="after_retune",
        comparison_source="predicted",
        target_output_type="current",
        current_channel="i_sum_signed",
        field_channel="bz_mT",
        metric_domain="target_output",
    )

    assert comparison.clipping_detected is False


def test_build_validation_comparison_rejects_tolerance_boundary_plus_margin() -> None:
    time_s = np.linspace(0.0, 1.0, 129)
    target = np.sin(2.0 * np.pi * time_s)
    command_profile = pd.DataFrame(
        {
            "time_s": time_s,
            "target_output": target,
            "expected_output": 0.98 * target,
            "target_current_a": target,
            "expected_current_a": 0.98 * target,
            "within_hardware_limits": False,
            "within_daq_limit": True,
            "required_amp_gain_pct": 100.0,
            "available_amp_gain_pct": 99.74,
            "peak_input_limit_margin": 0.10,
            "p95_input_limit_margin": 0.11,
        }
    )
    validation_frame = pd.DataFrame(
        {
            "time_s": time_s,
            "i_sum_signed": 0.98 * target,
            "bz_mT": 1.8 * 0.98 * target,
            "daq_input_v": 2.0 * target,
        }
    )

    comparison = build_validation_comparison(
        command_profile=command_profile,
        validation_frame=validation_frame,
        label="after_retune",
        comparison_source="predicted",
        target_output_type="current",
        current_channel="i_sum_signed",
        field_channel="bz_mT",
        metric_domain="target_output",
    )

    assert comparison.clipping_detected is True


def test_build_validation_comparison_keeps_negative_input_margin_clipped() -> None:
    time_s = np.linspace(0.0, 1.0, 129)
    target = np.sin(2.0 * np.pi * time_s)
    command_profile = pd.DataFrame(
        {
            "time_s": time_s,
            "target_output": target,
            "expected_output": 0.98 * target,
            "target_current_a": target,
            "expected_current_a": 0.98 * target,
            "within_hardware_limits": False,
            "within_daq_limit": True,
            "required_amp_gain_pct": 100.0,
            "available_amp_gain_pct": 99.90,
            "peak_input_limit_margin": -0.01,
            "p95_input_limit_margin": 0.05,
        }
    )
    validation_frame = pd.DataFrame(
        {
            "time_s": time_s,
            "i_sum_signed": 0.98 * target,
            "bz_mT": 1.8 * 0.98 * target,
            "daq_input_v": 2.0 * target,
        }
    )

    comparison = build_validation_comparison(
        command_profile=command_profile,
        validation_frame=validation_frame,
        label="after_retune",
        comparison_source="predicted",
        target_output_type="current",
        current_channel="i_sum_signed",
        field_channel="bz_mT",
        metric_domain="target_output",
    )

    assert comparison.clipping_detected is True


def test_build_validation_comparison_keeps_real_hardware_shortfall_clipped() -> None:
    time_s = np.linspace(0.0, 1.0, 129)
    target = np.sin(2.0 * np.pi * time_s)
    command_profile = pd.DataFrame(
        {
            "time_s": time_s,
            "target_output": target,
            "expected_output": 0.98 * target,
            "target_current_a": target,
            "expected_current_a": 0.98 * target,
            "within_hardware_limits": False,
            "within_daq_limit": True,
            "required_amp_gain_pct": 100.0,
            "available_amp_gain_pct": 99.0,
            "peak_input_limit_margin": 0.10,
            "p95_input_limit_margin": 0.11,
        }
    )
    validation_frame = pd.DataFrame(
        {
            "time_s": time_s,
            "i_sum_signed": 0.98 * target,
            "bz_mT": 1.8 * 0.98 * target,
            "daq_input_v": 2.0 * target,
        }
    )

    comparison = build_validation_comparison(
        command_profile=command_profile,
        validation_frame=validation_frame,
        label="after_retune",
        comparison_source="predicted",
        target_output_type="current",
        current_channel="i_sum_signed",
        field_channel="bz_mT",
        metric_domain="target_output",
    )

    assert comparison.clipping_detected is True


def test_prediction_debug_marks_exact_field_near_zero_as_unavailable() -> None:
    time_s = np.linspace(0.0, 1.0, 129)
    target = 20.0 * np.sin(2.0 * np.pi * time_s)
    command_profile = pd.DataFrame(
        {
            "time_s": time_s,
            "target_output": target,
            "target_field_mT": target,
            "expected_output": np.zeros_like(target),
            "expected_field_mT": np.zeros_like(target),
            "request_route": "exact",
            "plot_source": "exact_prediction",
        }
    )
    command_profile.attrs["engine_summary"] = {
        "request_route": "exact",
        "solver_route": "harmonic_surface_inverse_exact",
        "plot_source": "exact_prediction",
    }
    validation_frame = pd.DataFrame(
        {
            "time_s": time_s,
            "bz_mT": 0.9 * target,
            "i_sum_signed": 0.2 * target,
            "daq_input_v": np.sin(2.0 * np.pi * time_s),
        }
    )

    debug = build_prediction_debug_snapshot(
        command_profile=command_profile,
        validation_frame=validation_frame,
        target_output_type="field",
        current_channel="i_sum_signed",
        field_channel="bz_mT",
    )

    assert debug["field_prediction_source"] == "exact_field_direct"
    assert debug["field_prediction_status"] == "unavailable"
    assert debug["zero_field_reason"] in {"expected_field_near_zero", "expected_field_collapse"}
    assert debug["field_prediction_unavailable_reason"] in {"expected_field_near_zero", "expected_field_collapse"}
    assert debug["field_prediction_available"] is False
    assert debug["loss_target_type"] == "field"


def test_prediction_debug_marks_same_recipe_surrogate_for_exact_field_fallback() -> None:
    time_s = np.linspace(0.0, 1.0, 129)
    target = 20.0 * np.sin(2.0 * np.pi * time_s)
    surrogate = 16.0 * np.sin(2.0 * np.pi * time_s)
    command_profile = pd.DataFrame(
        {
            "time_s": time_s,
            "target_output": target,
            "target_field_mT": target,
            "expected_output": surrogate,
            "expected_field_mT": surrogate,
            "expected_current_a": 8.0 * np.sin(2.0 * np.pi * time_s),
            "request_route": "exact",
            "plot_source": "exact_prediction",
        }
    )
    command_profile.attrs["engine_summary"] = {
        "request_route": "exact",
        "solver_route": "harmonic_surface_inverse_exact",
        "plot_source": "exact_prediction",
        "selected_support_id": "support_exact_5app",
        "selected_support_family": "current_pp_target_a:5",
        "support_selection_reason": "exact_family_level_lock",
    }
    command_profile.attrs["prediction_debug"] = {
        "field_prediction_source_hint": "current_to_bz_surrogate",
        "field_prediction_status": "available",
        "field_prediction_fallback_reason": "expected_field_collapse",
        "same_recipe_surrogate_candidate_available": True,
        "same_recipe_surrogate_applied": True,
        "same_recipe_surrogate_ratio": 2.0,
        "surrogate_scope": "same_recipe_validated_exact_support",
    }

    debug = build_prediction_debug_snapshot(
        command_profile=command_profile,
        validation_frame=None,
        target_output_type="field",
        current_channel="i_sum_signed",
        field_channel="bz_mT",
    )

    assert debug["field_prediction_source"] == "current_to_bz_surrogate"
    assert debug["field_prediction_status"] == "available"
    assert debug["field_prediction_fallback_reason"] == "expected_field_collapse"
    assert debug["same_recipe_surrogate_applied"] is True
    assert debug["surrogate_scope"] == "same_recipe_validated_exact_support"


def test_prediction_debug_marks_validation_transfer_for_corrected_field() -> None:
    time_s = np.linspace(0.0, 1.0, 129)
    target = 20.0 * np.sin(2.0 * np.pi * time_s)
    predicted = 18.0 * np.sin(2.0 * np.pi * (time_s - 0.01))
    command_profile = pd.DataFrame(
        {
            "time_s": time_s,
            "target_output": target,
            "target_field_mT": target,
            "expected_output": predicted,
            "expected_field_mT": predicted,
            "request_route": "exact",
            "plot_source": "exact_prediction",
        }
    )
    command_profile.attrs["engine_summary"] = {
        "request_route": "exact",
        "solver_route": "validation_residual_second_stage",
        "plot_source": "exact_prediction",
    }
    command_profile.attrs["bz_projection"] = {
        "available": True,
        "reason_code": None,
        "source": "validation_voltage_to_bz_transfer",
    }

    debug = build_prediction_debug_snapshot(
        command_profile=command_profile,
        validation_frame=None,
        target_output_type="field",
        current_channel="i_sum_signed",
        field_channel="bz_mT",
    )

    assert debug["field_prediction_source"] == "validation_transfer"
    assert debug["request_route"] == "exact"
    assert debug["solver_route"] == "validation_residual_second_stage"


def test_prediction_debug_flags_finite_target_template_leak() -> None:
    time_s = np.linspace(0.0, 1.0, 129)
    target = 10.0 * np.sign(np.sin(2.0 * np.pi * time_s))
    command_profile = pd.DataFrame(
        {
            "time_s": time_s,
            "target_output": target,
            "target_current_a": target,
            "expected_output": target.copy(),
            "expected_current_a": target.copy(),
            "expected_field_mT": target.copy(),
            "request_route": "exact",
            "plot_source": "exact_prediction",
            "finite_cycle_mode": True,
        }
    )
    command_profile.attrs["engine_summary"] = {
        "request_route": "exact",
        "solver_route": "finite_exact_direct",
        "plot_source": "exact_prediction",
    }

    debug = build_prediction_debug_snapshot(
        command_profile=command_profile,
        validation_frame=None,
        target_output_type="current",
        current_channel="i_sum_signed",
        field_channel="bz_mT",
    )

    assert debug["target_leak_suspect"] is True
    assert debug["target_leak_reason"] == "expected_output_matches_target_template"
    assert debug["field_prediction_source"] == "target_leak_suspect"


def test_prediction_debug_marks_exact_preview_source_as_unavailable_bug() -> None:
    time_s = np.linspace(0.0, 1.0, 129)
    target = 20.0 * np.sin(2.0 * np.pi * time_s)
    preview = 18.0 * np.sin(2.0 * np.pi * time_s)
    command_profile = pd.DataFrame(
        {
            "time_s": time_s,
            "target_output": target,
            "target_field_mT": target,
            "expected_output": preview,
            "expected_field_mT": preview,
            "support_scaled_field_mT": preview,
            "request_route": "exact",
            "plot_source": "support_blended_preview",
        }
    )
    command_profile.attrs["engine_summary"] = {
        "request_route": "exact",
        "solver_route": "harmonic_surface_inverse_exact",
        "plot_source": "support_blended_preview",
    }

    debug = build_prediction_debug_snapshot(
        command_profile=command_profile,
        validation_frame=None,
        target_output_type="field",
        current_channel="i_sum_signed",
        field_channel="bz_mT",
    )

    assert debug["field_prediction_source"] == "support_blended_preview"
    assert debug["field_prediction_status"] == "unavailable"
    assert debug["field_prediction_unavailable_reason"] == "exact_route_support_blended_preview_bug"


def test_prediction_debug_maps_zero_fill_fallback_without_silence() -> None:
    time_s = np.linspace(0.0, 1.0, 129)
    target = 20.0 * np.sin(2.0 * np.pi * time_s)
    command_profile = pd.DataFrame(
        {
            "time_s": time_s,
            "target_output": target,
            "target_field_mT": target,
            "expected_output": np.zeros_like(target),
            "expected_field_mT": np.zeros_like(target),
            "request_route": "exact",
            "plot_source": "exact_prediction",
        }
    )
    command_profile.attrs["engine_summary"] = {
        "request_route": "exact",
        "solver_route": "harmonic_surface_inverse_exact",
        "plot_source": "exact_prediction",
    }
    command_profile.attrs["prediction_debug"] = {
        "field_prediction_source_hint": "zero_fill_fallback",
    }

    debug = build_prediction_debug_snapshot(
        command_profile=command_profile,
        validation_frame=None,
        target_output_type="field",
        current_channel="i_sum_signed",
        field_channel="bz_mT",
    )

    assert debug["field_prediction_source"] == "zero_fill_fallback"
    assert debug["field_prediction_status"] == "unavailable"
    assert debug["field_prediction_unavailable_reason"] == "zero_fill_fallback"


def test_execute_validation_retune_payload_includes_prediction_debug_sections() -> None:
    base_profile, validation_frame, candidate = _build_profiles(target_output_type="current", waveform_type="sine")
    base_profile.attrs["engine_summary"] = {
        "request_route": "exact",
        "solver_route": "harmonic_surface_inverse_exact",
        "plot_source": "exact_prediction",
    }
    base_profile.attrs["prediction_debug"] = {
        "field_prediction_source": "current_to_bz_surrogate",
        "expected_current_source": "exact_current_direct",
        "request_route": "exact",
        "solver_route": "harmonic_surface_inverse_exact",
        "plot_source": "exact_prediction",
    }

    result = execute_validation_retune(
        base_profile=base_profile,
        validation_candidate=candidate,
        validation_frame=validation_frame,
        export_file_prefix="current_exact_prediction_debug",
        target_output_type="current",
        current_channel="i_sum_signed",
        field_channel="bz_mT",
        max_daq_voltage_pp=20.0,
        amp_gain_at_100_pct=20.0,
        amp_gain_limit_pct=100.0,
        amp_max_output_pk_v=180.0,
        support_amp_gain_pct=100.0,
        correction_gain=0.75,
        max_iterations=2,
        improvement_threshold=0.0,
        original_recommendation_id="rec_current_exact",
        source_selection={"source_kind": SOURCE_KIND_RECOMMENDATION, "lut_id": "rec_current_exact"},
    )

    assert result is not None
    payload = result.artifact_payload
    assert "baseline_metrics" in payload
    assert "corrected_metrics" in payload
    assert "prediction_debug" in payload
    assert payload["prediction_debug"]["baseline"]["field_prediction_source"] == "current_to_bz_surrogate"
    assert "preferred_output" in payload
    assert "metrics_availability" in payload
