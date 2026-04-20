from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.validation_retune import (  # noqa: E402
    SOURCE_KIND_EXPORT,
    execute_validation_retune,
    save_retune_artifacts,
)


CURRENT_CHANNEL = "i_sum_signed"
FIELD_CHANNEL = "bz_mT"
TRANSIENT_UPLOAD_DIR = REPO_ROOT.parent / "outputs" / "field_analysis_app_state" / "uploads" / "transient"
VALIDATION_FILE_PATH = (
    REPO_ROOT.parent / "outputs" / "field_analysis_app_state" / "uploads" / "validation" / "8d51f7c793c4e3fa_샘플파형테스트결과_1hz_pp20A.csv"
)
OUTPUT_DIR = REPO_ROOT / "artifacts" / "validation_retune_real_example"
SUMMARY_JSON_PATH = REPO_ROOT / "artifacts" / "bz_first_exact_matrix" / "real_validation_e2e_result.json"
SUMMARY_MD_PATH = REPO_ROOT / "artifacts" / "bz_first_exact_matrix" / "real_validation_e2e_result.md"
RETUNE_HISTORY_PATH = REPO_ROOT.parent / "outputs" / "field_analysis_app_state" / "validation_retune_history.json"

TARGET_WAVEFORM = "sine"
TARGET_FREQ_HZ = 1.25
TARGET_CYCLES = 1.25
TARGET_TYPE = "current"
TARGET_LEVEL_PP = 20.0
EXPORT_PREFIX = "real_finite_exact_validation_1p25hz_1p25cycle_20pp"
SUPPORT_GLOB = "*1.25hz_1.25cycle_20pp.csv"


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _safe_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _record_retune_history(retune_summary: dict[str, Any]) -> None:
    history = _load_json(RETUNE_HISTORY_PATH)
    entries = history.get("retunes", [])
    if not isinstance(entries, list):
        entries = []
    retune_id = str(retune_summary.get("corrected_lut_id") or retune_summary.get("retune_id") or "")
    updated = [item for item in entries if str(item.get("corrected_lut_id") or item.get("retune_id") or "") != retune_id]
    updated.append(retune_summary)
    updated.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    _write_json(RETUNE_HISTORY_PATH, {"retunes": updated})


def _peak_to_peak(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.nanmax(finite) - np.nanmin(finite))


def _find_support_file() -> Path:
    matches = sorted(path for path in TRANSIENT_UPLOAD_DIR.rglob(SUPPORT_GLOB) if path.is_file())
    if not matches:
        raise FileNotFoundError(f"could not locate support file matching {SUPPORT_GLOB!r} under {TRANSIENT_UPLOAD_DIR}")
    sine_matches = [path for path in matches if "sinusidal" in {part.lower() for part in path.parts}]
    if sine_matches:
        return sine_matches[-1]
    return matches[-1]


def _read_measurement_csv(path: Path) -> tuple[dict[str, str], pd.DataFrame]:
    metadata: dict[str, str] = {}
    data_lines: list[str] = []
    for line in _decode_text(path.read_bytes()).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            payload = stripped.lstrip("#").strip()
            if "," in payload:
                key, value = payload.split(",", 1)
                metadata[key.strip()] = value.strip()
            continue
        data_lines.append(line)
    if not data_lines:
        raise RuntimeError(f"no tabular rows found in {path}")
    frame = pd.read_csv(io.StringIO("\n".join(data_lines)))
    return metadata, frame


def _measurement_time_seconds(frame: pd.DataFrame) -> np.ndarray:
    if "Timestamp" in frame.columns:
        timestamp = pd.to_datetime(frame["Timestamp"], errors="coerce")
        if timestamp.notna().any():
            base = timestamp.dropna().iloc[0]
            return (timestamp - base).dt.total_seconds().to_numpy(dtype=float)
    return np.arange(len(frame), dtype=float) / 100.0


def _current_signal(frame: pd.DataFrame) -> np.ndarray:
    current1 = pd.to_numeric(frame.get("Current1_A"), errors="coerce").to_numpy(dtype=float)
    current2 = pd.to_numeric(frame.get("Current2_A"), errors="coerce").to_numpy(dtype=float)
    if np.isfinite(current1).any() and np.isfinite(current2).any():
        return current1 + current2
    return np.where(np.isfinite(current2), current2, current1)


def _bz_effective_signal(frame: pd.DataFrame) -> np.ndarray:
    hall_bz_raw = pd.to_numeric(frame.get("HallBz"), errors="coerce").to_numpy(dtype=float)
    return -hall_bz_raw


def _voltage_signal(frame: pd.DataFrame) -> np.ndarray:
    for column in ("Voltage2", "Voltage1"):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return np.zeros(len(frame), dtype=float)


def _ideal_target_output(time_s: np.ndarray, *, freq_hz: float, cycles: float, output_pp: float) -> tuple[np.ndarray, np.ndarray]:
    active_duration = float(cycles) / float(freq_hz)
    phase_total = time_s * float(freq_hz)
    waveform = np.sin(2.0 * np.pi * phase_total)
    active_mask = time_s <= active_duration + 1e-9
    target_output = np.where(active_mask, waveform * float(output_pp) / 2.0, 0.0)
    cycle_progress = np.clip(phase_total / max(float(cycles), 1e-9), 0.0, 1.0)
    return target_output, cycle_progress


def _build_exact_base_profile(path: Path) -> pd.DataFrame:
    _, frame = _read_measurement_csv(path)
    time_s = _measurement_time_seconds(frame)
    current = _current_signal(frame)
    bz_effective = _bz_effective_signal(frame)
    voltage = _voltage_signal(frame)
    target_output, cycle_progress = _ideal_target_output(
        time_s,
        freq_hz=TARGET_FREQ_HZ,
        cycles=TARGET_CYCLES,
        output_pp=TARGET_LEVEL_PP,
    )
    voltage_pp = _peak_to_peak(voltage) or 0.0
    active_duration = float(TARGET_CYCLES) / float(TARGET_FREQ_HZ)
    return pd.DataFrame(
        {
            "time_s": time_s,
            "cycle_progress": cycle_progress,
            "waveform_type": TARGET_WAVEFORM,
            "freq_hz": TARGET_FREQ_HZ,
            "target_cycle_count": TARGET_CYCLES,
            "target_output_type": TARGET_TYPE,
            "target_output_pp": TARGET_LEVEL_PP,
            "finite_cycle_mode": True,
            "preview_tail_cycles": 0.25,
            "is_active_target": time_s <= active_duration + 1e-9,
            "target_output": target_output,
            "used_target_output": target_output,
            "target_current_a": target_output,
            "used_target_current_a": target_output,
            "expected_output": current,
            "expected_current_a": current,
            "expected_field_mT": bz_effective,
            "modeled_output": current,
            "modeled_current_a": current,
            "modeled_field_mT": bz_effective,
            "recommended_voltage_v": voltage,
            "limited_voltage_v": voltage,
            "recommended_voltage_pp": voltage_pp,
            "limited_voltage_pp": voltage_pp,
            "max_daq_voltage_pp": 20.0,
            "max_daq_voltage_pk_v": 10.0,
            "peak_input_limit_margin": 0.25,
            "p95_input_limit_margin": 0.25,
            "required_amp_gain_multiplier": 1.0,
            "support_amp_gain_pct": 100.0,
            "required_amp_gain_pct": 100.0,
            "available_amp_gain_pct": 100.0,
            "amp_gain_limit_pct": 100.0,
            "amp_gain_at_100_pct": 20.0,
            "amp_max_output_pk_v": 180.0,
            "amp_output_pp_at_required": voltage_pp * 20.0,
            "amp_output_pk_at_required": voltage_pp * 10.0,
            "within_daq_limit": True,
            "within_amp_gain_limit": True,
            "within_amp_output_limit": True,
            "within_hardware_limits": True,
            "support_test_id": path.stem,
        }
    )


def _build_validation_frame(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    _, frame = _read_measurement_csv(path)
    time_s = _measurement_time_seconds(frame)
    current = _current_signal(frame)
    bz_effective = _bz_effective_signal(frame)
    voltage = _voltage_signal(frame)
    validation_frame = pd.DataFrame(
        {
            "time_s": time_s,
            "daq_input_v": voltage,
            CURRENT_CHANNEL: current,
            FIELD_CHANNEL: bz_effective,
            "bz_effective_mT": bz_effective,
            "bz_raw_mT": -bz_effective,
        }
    )
    candidate = {
        "label": f"{path.stem} | {path.name}",
        "test_id": path.stem,
        "source_file": path.as_posix(),
        "waveform_type": TARGET_WAVEFORM,
        "freq_hz": TARGET_FREQ_HZ,
        "output_pp": _peak_to_peak(current),
        "score": 0.0,
        "eligible": True,
        "eligibility_reason": "exact validation candidate",
    }
    return validation_frame, candidate


def _write_summary_markdown(payload: dict[str, Any]) -> None:
    lines = [
        "# Real Validation E2E Result",
        "",
        "- finite exact 실측 validation 파일 1건을 source export 기반 corrected LUT까지 end-to-end 재실행한 결과입니다.",
        f"- support_source_file: `{payload['support_source_file']}`",
        f"- validation_source_file: `{payload['validation_source_file']}`",
        f"- support_state: `{payload['support_state']}`",
        f"- quality_label: `{payload['quality_label']}`",
        f"- baseline_nrmse: `{payload['baseline_nrmse']}`",
        f"- corrected_nrmse: `{payload['corrected_nrmse']}`",
        f"- baseline_bz_nrmse: `{payload['baseline_bz_nrmse']}`",
        f"- corrected_bz_nrmse: `{payload['corrected_bz_nrmse']}`",
        f"- corrected_lut_id: `{payload['corrected_lut_id']}`",
        f"- report_path: `{payload['report_path']}`",
    ]
    SUMMARY_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if not VALIDATION_FILE_PATH.exists():
        raise FileNotFoundError(f"validation file not found: {VALIDATION_FILE_PATH}")

    support_path = _find_support_file()
    base_profile = _build_exact_base_profile(support_path)
    validation_frame, validation_candidate = _build_validation_frame(VALIDATION_FILE_PATH)

    retune_result = execute_validation_retune(
        base_profile=base_profile,
        validation_candidate=validation_candidate,
        validation_frame=validation_frame,
        export_file_prefix=EXPORT_PREFIX,
        target_output_type=TARGET_TYPE,
        current_channel=CURRENT_CHANNEL,
        field_channel=FIELD_CHANNEL,
        max_daq_voltage_pp=20.0,
        amp_gain_at_100_pct=20.0,
        amp_gain_limit_pct=100.0,
        amp_max_output_pk_v=180.0,
        support_amp_gain_pct=100.0,
        correction_gain=0.7,
        max_iterations=2,
        improvement_threshold=0.0,
        original_recommendation_id=support_path.stem,
        source_selection={
            "source_kind": SOURCE_KIND_EXPORT,
            "lut_id": support_path.stem,
            "profile_csv_path": support_path.as_posix(),
            "source_lut_filename": support_path.name,
        },
    )
    if retune_result is None:
        raise RuntimeError("validation / retune execution returned None")

    artifact_paths = save_retune_artifacts(retune_result=retune_result, output_dir=OUTPUT_DIR)
    summary_payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "support_source_file": support_path.as_posix(),
        "validation_source_file": VALIDATION_FILE_PATH.as_posix(),
        "support_state": "exact",
        "preview_only": False,
        "quality_label": retune_result.quality_label,
        "quality_tone": retune_result.quality_tone,
        "quality_reasons": retune_result.quality_reasons,
        "acceptance_decision": retune_result.acceptance_decision,
        "preferred_output_id": retune_result.preferred_output_id,
        "candidate_status": retune_result.acceptance_decision.get("decision"),
        "candidate_status_label": retune_result.acceptance_decision.get("label"),
        "rejection_reason": retune_result.acceptance_decision.get("rejection_reason"),
        "original_recommendation_id": retune_result.validation_run.original_recommendation_id,
        "validation_run_id": retune_result.validation_run.validation_run_id,
        "corrected_lut_id": retune_result.validation_run.corrected_lut_id,
        "baseline_nrmse": _safe_float(retune_result.baseline_comparison.nrmse),
        "corrected_nrmse": _safe_float(retune_result.corrected_comparison.nrmse),
        "baseline_bz_nrmse": _safe_float(retune_result.baseline_bz_comparison.nrmse),
        "corrected_bz_nrmse": _safe_float(retune_result.corrected_bz_comparison.nrmse),
        "baseline_shape_corr": _safe_float(retune_result.baseline_comparison.shape_corr),
        "corrected_shape_corr": _safe_float(retune_result.corrected_comparison.shape_corr),
        "baseline_bz_shape_corr": _safe_float(retune_result.baseline_bz_comparison.shape_corr),
        "corrected_bz_shape_corr": _safe_float(retune_result.corrected_bz_comparison.shape_corr),
        "baseline_bz_phase_lag_s": _safe_float(retune_result.baseline_bz_comparison.phase_lag_s),
        "corrected_bz_phase_lag_s": _safe_float(retune_result.corrected_bz_comparison.phase_lag_s),
        "baseline_bz_clipping_detected": bool(retune_result.baseline_bz_comparison.clipping_detected),
        "corrected_bz_clipping_detected": bool(retune_result.corrected_bz_comparison.clipping_detected),
        "baseline_bz_metrics_available": bool(retune_result.baseline_bz_comparison.metrics_available),
        "corrected_bz_metrics_available": bool(retune_result.corrected_bz_comparison.metrics_available),
        "baseline_bz_unavailable_reason": retune_result.baseline_bz_comparison.unavailable_reason,
        "corrected_bz_unavailable_reason": retune_result.corrected_bz_comparison.unavailable_reason,
        "baseline_bz_reason_codes": list(retune_result.baseline_bz_comparison.reason_codes),
        "corrected_bz_reason_codes": list(retune_result.corrected_bz_comparison.reason_codes),
        "report_path": artifact_paths.get("validation_report_json"),
        "artifact_paths": artifact_paths,
        "validation_window": retune_result.validation_run.metadata.get("validation_window", {}),
    }
    _record_retune_history(
        {
            "retune_id": retune_result.validation_run.export_file_prefix,
            "created_at": retune_result.validation_run.created_at,
            "original_recommendation_id": retune_result.validation_run.original_recommendation_id,
            "validation_run_id": retune_result.validation_run.validation_run_id,
            "validation_test_id": retune_result.validation_run.selected_validation_test_id,
            "lut_id": retune_result.validation_run.lut_id,
            "source_kind": retune_result.validation_run.source_kind,
            "source_lut_filename": retune_result.validation_run.source_lut_filename,
            "corrected_lut_id": retune_result.validation_run.corrected_lut_id,
            "iteration_index": retune_result.validation_run.iteration_index,
            "exact_path": retune_result.validation_run.exact_path,
            "correction_rule": retune_result.validation_run.correction_rule,
            "target_output_type": retune_result.validation_run.target_output_type,
            "waveform_type": retune_result.validation_run.waveform_type,
            "freq_hz": retune_result.validation_run.freq_hz,
            "commanded_cycles": retune_result.validation_run.commanded_cycles,
            "target_level_value": retune_result.validation_run.target_level_value,
            "target_level_kind": retune_result.validation_run.target_level_kind,
            "quality_label": retune_result.quality_label,
            "quality_tone": retune_result.quality_tone,
            "quality_reasons": retune_result.quality_reasons,
            "acceptance_decision": retune_result.acceptance_decision,
            "preferred_output_id": retune_result.preferred_output_id,
            "candidate_status": retune_result.acceptance_decision.get("decision"),
            "candidate_status_label": retune_result.acceptance_decision.get("label"),
            "rejection_reason": retune_result.acceptance_decision.get("rejection_reason"),
            "artifact_paths": artifact_paths,
            "before_nrmse": retune_result.baseline_comparison.nrmse,
            "after_nrmse": retune_result.corrected_comparison.nrmse,
            "before_shape_corr": retune_result.baseline_comparison.shape_corr,
            "after_shape_corr": retune_result.corrected_comparison.shape_corr,
            "before_bz_nrmse": retune_result.baseline_bz_comparison.nrmse,
            "after_bz_nrmse": retune_result.corrected_bz_comparison.nrmse,
        }
    )
    SUMMARY_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON_PATH.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_markdown(summary_payload)

    generator_script = REPO_ROOT / "tools" / "generate_bz_first_artifacts.py"
    completed = subprocess.run(
        [sys.executable, str(generator_script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"artifact generator failed after validation run:\n{completed.stderr}")

    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
