from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from field_analysis.compensation import FIELD_ROUTE_NORMALIZED_TARGET_PP, _finite_target_template
from field_analysis.finite_actual_drive_normalization import normalize_peak_to_limit, peak_abs

RESULT_FILENAME_RE = re.compile(
    r"(?P<canonical>finite_recommended_voltage_lut_(?P<waveform>[A-Za-z]+)_(?P<freq>[0-9]+(?:\.[0-9]+)?)Hz_(?P<cycle>[0-9]+(?:\.[0-9]+)?)cycle_result\.csv)$",
    re.IGNORECASE,
)
EXPECTED_ACTUAL_DRIVE_FREQS_HZ, EXPECTED_ACTUAL_DRIVE_CYCLES = (0.5, 1.25, 2.0), (1.0, 1.25, 1.5, 1.75)
PHASE_ONE_NO_SECOND_CORRECTION_CONTRACT = {
    "correction_delta_v_generated": False,
    "second_voltage_v_generated": False,
    "second_lut_generated": False,
}
@dataclass(frozen=True)
class ActualDriveRecord:
    source_file: str
    path: Path
    waveform_type: str
    freq_hz: float
    cycle_count: float
    metadata: dict[str, Any]
    frame: pd.DataFrame

def parse_actual_drive_filename(path: str | Path) -> dict[str, Any]:
    return parse_finite_actual_drive_filename(path)

def parse_finite_actual_drive_filename(path: str | Path) -> dict[str, Any]:
    name = Path(path).name
    match = RESULT_FILENAME_RE.search(name)
    if match is None:
        raise ValueError(f"Unsupported finite actual-drive result filename: {name}")
    prefix = name[: match.start("canonical")]
    if prefix.endswith("_"):
        prefix = prefix[:-1]
    waveform = match.group("waveform").lower()
    return {
        "source_type": "finite_actual_drive_result",
        "source_file": name,
        "canonical_source_filename": match.group("canonical"),
        "upload_internal_id": prefix or None,
        "waveform": waveform,
        "waveform_type": waveform,
        "freq_hz": float(match.group("freq")),
        "cycle_count": float(match.group("cycle")),
    }


def expected_actual_drive_result_filenames() -> list[str]:
    return [
        f"finite_recommended_voltage_lut_sine_{freq:g}Hz_{cycle:g}cycle_result.csv"
        for freq in EXPECTED_ACTUAL_DRIVE_FREQS_HZ
        for cycle in EXPECTED_ACTUAL_DRIVE_CYCLES
    ]


def read_actual_drive_result(path: str | Path) -> ActualDriveRecord:
    source_path = Path(path)
    filename_meta = parse_actual_drive_filename(source_path)
    lines = source_path.read_text(encoding="utf-8-sig").splitlines()
    preamble, header_index = _parse_preamble(lines)
    frame = pd.read_csv(source_path, skiprows=header_index)
    required = {"TimeMs", "Voltage1_V", "HallBz"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing actual-drive result columns in {source_path.name}: {missing}")
    hallbz_raw = pd.to_numeric(frame["HallBz"], errors="coerce")
    voltage1_v = pd.to_numeric(frame["Voltage1_V"], errors="coerce")
    normalized = pd.DataFrame(
        {
            "sample_index": np.arange(len(frame), dtype=int),
            "time_s_abs": pd.to_numeric(frame["TimeMs"], errors="coerce") / 1000.0,
            "first_voltage_v": voltage1_v,
            "command_voltage_v": voltage1_v,
            "actual_drive_voltage_v": voltage1_v,
            "hallbz_raw_mT": hallbz_raw,
            "measured_field_raw": -hallbz_raw,
        }
    )
    if "Current1_A" in frame.columns:
        normalized["current_a"] = pd.to_numeric(frame["Current1_A"], errors="coerce")
    metadata: dict[str, Any] = {
        **filename_meta,
        "source_file": source_path.name,
        "canonical_source_filename": filename_meta["canonical_source_filename"],
        "upload_internal_id": filename_meta["upload_internal_id"],
        "source_type": "finite_actual_drive_result",
        "source_path": str(source_path),
        "pre_delay_s": _numeric_metadata(preamble, "PreDelay(s)"),
        "post_delay_s": _numeric_metadata(preamble, "PostDelay(s)"),
        "auto_sync_hall_lag_ms": _parse_auto_sync_lag_ms(preamble),
        "field_unit": "mT_inferred_from_HallBz",
        "field_units": "mT",
        "hallbz_sign_inverted": True,
        "voltage_unit": "V",
        "command_voltage_source": "Voltage1_V_no_separate_command_reference",
        "actual_drive_voltage_source": "Voltage1_V",
        "time_unit": "ms",
        "raw_preamble": preamble,
    }
    return ActualDriveRecord(
        source_file=source_path.name,
        path=source_path,
        waveform_type=str(filename_meta["waveform_type"]),
        freq_hz=float(filename_meta["freq_hz"]),
        cycle_count=float(filename_meta["cycle_count"]),
        metadata=metadata,
        frame=normalized,
    )


def load_finite_actual_drive_result(path: str | Path) -> dict[str, Any]:
    record = read_actual_drive_result(path)
    review, metadata = build_actual_drive_review_case(record)
    return _case_payload(record, review, metadata)


def build_finite_actual_drive_review_dataset(files: list[str | Path]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for file in files:
        source_file = Path(file).name
        try:
            case = load_finite_actual_drive_result(file)
        except Exception as exc:  # noqa: BLE001 - UI payload must report parse errors, not raise.
            error = {
                "source_file": source_file,
                "canonical_source_filename": None,
                "upload_internal_id": None,
                "parse_status": "error",
                "parse_error": str(exc),
            }
            errors.append(error)
            summary_rows.append(error)
            continue
        cases.append(case)
        summary_rows.append(
            {
                "case_id": case["case_id"],
                "display_label": case["display_label"],
                "source_file": case["source_file"],
                "canonical_source_filename": case["canonical_source_filename"],
                "upload_internal_id": case["upload_internal_id"],
                "waveform": case["waveform"],
                "freq_hz": case["freq_hz"],
                "cycle_count": case["cycle_count"],
                "modeled_cycle_count": case["modeled_cycle_count"],
                "intended_drive_cycle_count": case["intended_drive_cycle_count"],
                "parse_status": case["parse_status"],
                "parse_error": case["parse_error"],
                **case["metrics"],
                **case["status"],
            }
        )
    return {
        "cases": cases,
        "summary": pd.DataFrame(summary_rows),
        "errors": errors,
    }


def build_actual_drive_review_case(
    record: ActualDriveRecord,
    *,
    intended_drive_cycle_count: float | None = None,
    requested_ui_cycle_count: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = record.frame.copy()
    time_abs = frame["time_s_abs"].to_numpy(dtype=float)
    first_voltage = frame["first_voltage_v"].to_numpy(dtype=float)
    raw_field = frame["measured_field_raw"].to_numpy(dtype=float)
    command_start_s, command_end_s = _nonzero_window(time_abs, first_voltage)
    if not np.isfinite(command_start_s):
        command_start_s = float(np.nanmin(time_abs))
    pre_mask = time_abs < command_start_s
    baseline = float(np.nanmedian(raw_field[pre_mask])) if pre_mask.any() else float(np.nanmedian(raw_field))
    measured_field = raw_field - baseline
    field_start_s = _field_motion_start(time_abs, raw_field, baseline)
    relative_time = time_abs - float(command_start_s)
    target_duration_s = float(record.cycle_count / record.freq_hz)
    active_mask = (relative_time >= 0.0) & (relative_time <= target_duration_s + 1e-12)
    physical_target = _finite_target_template(
        relative_time,
        waveform_type=record.waveform_type,
        freq_hz=record.freq_hz,
        target_cycle_count=record.cycle_count,
        target_output_pp=float(FIELD_ROUTE_NORMALIZED_TARGET_PP),
        force_rounded_triangle=True,
    )
    normalized_measured_field, field_norm_meta = normalize_peak_to_limit(
        measured_field,
        active_mask,
        limit=50.0,
        unavailable_status="unavailable_zero_peak",
    )
    normalized_target, target_norm_meta = normalize_peak_to_limit(
        physical_target,
        active_mask,
        limit=50.0,
        unavailable_status="unavailable_zero_peak",
    )
    normalized_first_voltage, voltage_norm_meta = normalize_peak_to_limit(
        first_voltage,
        np.isfinite(first_voltage),
        limit=5.0,
        unavailable_status="unavailable_zero_peak",
    )
    normalized_actual_drive_voltage, _actual_voltage_norm_meta = normalize_peak_to_limit(
        frame["actual_drive_voltage_v"].to_numpy(dtype=float),
        np.isfinite(frame["actual_drive_voltage_v"].to_numpy(dtype=float)),
        limit=5.0,
        unavailable_status="unavailable_zero_peak",
    )
    residual = physical_target - measured_field
    normalized_residual = normalized_target - normalized_measured_field
    corr, nrmse = _shape_corr_and_nrmse(normalized_target[active_mask], normalized_measured_field[active_mask])
    sampled_target_pp = _pp(physical_target[active_mask])
    measured_pp = _pp(measured_field[active_mask])
    measured_peak_error = float(measured_pp - float(FIELD_ROUTE_NORMALIZED_TARGET_PP)) if np.isfinite(measured_pp) else float("nan")
    normalized_peak = peak_abs(normalized_measured_field[active_mask])
    normalized_voltage_peak = peak_abs(normalized_first_voltage)
    normalized_terminal_peak_error = float(
        peak_abs(normalized_target[active_mask]) - peak_abs(normalized_measured_field[active_mask])
    )
    phase_error_s = _estimate_phase_error(relative_time, normalized_target, normalized_measured_field, active_mask)
    terminal_mask = active_mask & (relative_time >= max(target_duration_s * 0.85, 0.0))
    startup_mask = active_mask & (relative_time <= min(target_duration_s * 0.2, 0.25 / max(record.freq_hz, 1e-9)))
    tail_mask = relative_time > target_duration_s
    measured_terminal_error = float(np.nanmean(normalized_residual[terminal_mask])) if terminal_mask.any() else float("nan")
    measured_startup_residual = float(np.nanmean(normalized_residual[startup_mask])) if startup_mask.any() else float("nan")
    measured_tail_residual = (
        float(np.nanmax(np.abs(measured_field[tail_mask]))) / max(abs(measured_pp), 1e-9)
        if tail_mask.any() and np.isfinite(measured_pp)
        else float("nan")
    )
    possible_polarity_flip = bool(np.isfinite(corr) and corr < -0.3)
    intended_cycle = float(record.cycle_count if intended_drive_cycle_count is None else intended_drive_cycle_count)
    requested_cycle = None if requested_ui_cycle_count is None else float(requested_ui_cycle_count)

    review = pd.DataFrame(
        {
            "time_s": relative_time,
            "raw_first_voltage_v": first_voltage,
            "normalized_first_voltage_v": normalized_first_voltage,
            "raw_actual_drive_voltage_v": frame["actual_drive_voltage_v"].to_numpy(dtype=float),
            "normalized_actual_drive_voltage_v": normalized_actual_drive_voltage,
            "raw_measured_field_mT": raw_field,
            "normalized_measured_field_mT": normalized_measured_field,
            "normalized_physical_target_output_mT": normalized_target,
            "measured_residual_normalized_mT": normalized_residual,
            "modeled_cycle_count": float(record.cycle_count),
            "intended_drive_cycle_count": intended_cycle,
            "field_normalization_scale_factor": field_norm_meta["scale_factor"],
            "voltage_normalization_scale_factor": voltage_norm_meta["scale_factor"],
            "first_voltage_v": first_voltage,
            "command_voltage_v": frame["command_voltage_v"].to_numpy(dtype=float),
            "actual_drive_voltage_v": frame["actual_drive_voltage_v"].to_numpy(dtype=float),
            "physical_target_output_mT": physical_target,
            "measured_field_mT": measured_field,
            "measured_residual_mT": residual,
        }
    )
    if "current_a" in frame.columns:
        review["current_a"] = frame["current_a"].to_numpy(dtype=float)
    metadata = {
        **record.metadata,
        "review_packet_type": "finite_actual_drive_phase1",
        "target_active_start_s": 0.0,
        "target_active_end_s": target_duration_s,
        "target_duration_s": target_duration_s,
        "command_start_s": command_start_s,
        "command_end_s": command_end_s,
        "alignment_offset_s": command_start_s,
        "alignment_anchor": "Voltage1_V_command_nonzero_start",
        "alignment_confidence": "medium",
        "voltage_nonzero_start_s": command_start_s,
        "field_motion_start_s": field_start_s,
        "field_baseline_mT": baseline,
        "field_normalization_enabled": True,
        "field_normalization_mode": "peak_to_50mT",
        "field_normalization_status": field_norm_meta["status"],
        "field_normalization_source_peak_mT": field_norm_meta["source_peak"],
        "field_normalization_scale_factor": field_norm_meta["scale_factor"],
        "target_normalization_source_peak_mT": target_norm_meta["source_peak"],
        "target_normalization_scale_factor": target_norm_meta["scale_factor"],
        "voltage_normalization_enabled": True,
        "voltage_normalization_mode": "peak_to_5V",
        "voltage_normalization_status": voltage_norm_meta["status"],
        "voltage_normalization_source_peak_v": voltage_norm_meta["source_peak"],
        "voltage_normalization_scale_factor": voltage_norm_meta["scale_factor"],
        "absolute_gain_evaluation_disabled": True,
        "shape_review_only": True,
        "modeled_cycle_count": float(record.cycle_count),
        "intended_drive_cycle_count": intended_cycle,
        "requested_ui_cycle_count": requested_cycle,
        "source_filename_cycle_count": float(record.cycle_count),
        "cycle_usage_mode": "modeled_cycle_as_drive_candidate",
        "cycle_semantics_note": (
            "Modeled cycle count is not rewritten; user may use this LUT as a drive candidate "
            "for a different intended cycle."
        ),
        "raw_field_peak_mT": peak_abs(raw_field[active_mask]),
        "raw_voltage_peak_v": peak_abs(first_voltage),
        "normalized_peak_mT": normalized_peak,
        "normalized_voltage_peak_v": normalized_voltage_peak,
        "normalized_shape_corr": corr,
        "normalized_nrmse": nrmse,
        "normalized_terminal_peak_time_error_s": phase_error_s,
        "normalized_terminal_peak_shape_error": normalized_terminal_peak_error,
        "terminal_peak_time_error_s": phase_error_s,
        "measured_active_nrmse": nrmse,
        "measured_shape_corr": corr,
        "measured_peak_error_mT": measured_peak_error,
        "measured_phase_error_s": phase_error_s,
        "measured_terminal_error_mT": measured_terminal_error,
        "measured_tail_residual": measured_tail_residual,
        "measured_startup_residual_mT": measured_startup_residual,
        "measured_pp_mT": measured_pp,
        "target_pp_mT": float(FIELD_ROUTE_NORMALIZED_TARGET_PP),
        "target_pp_sampled_mT": sampled_target_pp,
        "possible_polarity_flip_suggested": possible_polarity_flip,
        "baseline_ok": bool(pre_mask.any() and np.isfinite(baseline)),
        "nonzero_start_ok": bool(np.isfinite(command_start_s) and command_start_s >= 0.0),
        "clipping_or_spike_suspected": bool(_clipping_or_spike_suspected(first_voltage, raw_field)),
        "truncation_suspected": bool(np.isfinite(command_end_s) and command_end_s < command_start_s + target_duration_s * 0.75),
        "tail_present": bool(tail_mask.any() and np.isfinite(measured_tail_residual) and measured_tail_residual > 0.05),
        "alignment_status": "ok",
        "correction_delta_generated": False,
        "second_voltage_generated": False,
        "second_lut_generated": False,
        "continuous_touched": False,
    }
    return review, metadata


def _case_payload(record: ActualDriveRecord, review: pd.DataFrame, metadata: dict[str, Any]) -> dict[str, Any]:
    freq = f"{record.freq_hz:g}"
    cycle = f"{record.cycle_count:g}"
    metrics = {
        "measured_active_nrmse": metadata["measured_active_nrmse"],
        "measured_shape_corr": metadata["measured_shape_corr"],
        "measured_peak_error_mT": metadata["measured_peak_error_mT"],
        "measured_phase_error_s": metadata["measured_phase_error_s"],
        "measured_terminal_error_mT": metadata["measured_terminal_error_mT"],
        "measured_tail_residual": metadata["measured_tail_residual"],
        "measured_startup_residual_mT": metadata["measured_startup_residual_mT"],
        "measured_pp": metadata["measured_pp_mT"],
        "target_pp": metadata["target_pp_mT"],
        "raw_field_peak_mT": metadata["raw_field_peak_mT"],
        "field_normalization_scale_factor": metadata["field_normalization_scale_factor"],
        "raw_voltage_peak_v": metadata["raw_voltage_peak_v"],
        "voltage_normalization_scale_factor": metadata["voltage_normalization_scale_factor"],
        "normalized_peak_mT": metadata["normalized_peak_mT"],
        "normalized_voltage_peak_v": metadata["normalized_voltage_peak_v"],
        "normalized_shape_corr": metadata["normalized_shape_corr"],
        "normalized_nrmse": metadata["normalized_nrmse"],
        "terminal_peak_time_error_s": metadata["terminal_peak_time_error_s"],
        "normalized_terminal_peak_shape_error": metadata["normalized_terminal_peak_shape_error"],
        "shape_review_only": metadata["shape_review_only"],
        "alignment_offset_s": metadata["alignment_offset_s"],
        "alignment_confidence": metadata["alignment_confidence"],
        "possible_polarity_flip_suggested": metadata["possible_polarity_flip_suggested"],
    }
    status = {
        "baseline_ok": metadata["baseline_ok"],
        "nonzero_start_ok": metadata["nonzero_start_ok"],
        "clipping_or_spike_suspected": metadata["clipping_or_spike_suspected"],
        "truncation_suspected": metadata["truncation_suspected"],
        "tail_present": metadata["tail_present"],
        "alignment_status": metadata["alignment_status"],
    }
    return {
        "case_id": f"finite_actual_drive_result:{record.waveform_type}:{freq}Hz:{cycle}cycle",
        "display_label": f"{record.waveform_type} {freq}Hz {cycle}cycle actual-drive result",
        "source_file": record.source_file,
        "canonical_source_filename": metadata["canonical_source_filename"],
        "upload_internal_id": metadata["upload_internal_id"],
        "waveform": record.waveform_type,
        "waveform_type": record.waveform_type,
        "freq_hz": record.freq_hz,
        "cycle_count": record.cycle_count,
        "modeled_cycle_count": metadata["modeled_cycle_count"],
        "intended_drive_cycle_count": metadata["intended_drive_cycle_count"],
        "source_type": "finite_actual_drive_result",
        "parse_status": "parsed",
        "parse_error": None,
        "time_series": review,
        "metrics": metrics,
        "status": status,
        "metadata": metadata,
    }


def review_csv_filename(metadata: dict[str, Any]) -> str:
    from field_analysis.finite_actual_drive_review_export import review_csv_filename as _impl

    return _impl(metadata)


def process_actual_drive_review_folder(input_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    from field_analysis.finite_actual_drive_review_export import process_actual_drive_review_folder as _impl

    return _impl(input_dir, output_dir)


def _parse_preamble(lines: list[str]) -> tuple[dict[str, Any], int]:
    metadata: dict[str, Any] = {}
    header_index = -1
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("Row,"):
            header_index = index
            break
        if not stripped.startswith("#"):
            continue
        parts = [part.strip() for part in stripped[1:].split(",")]
        if not parts or not parts[0]:
            continue
        metadata[parts[0]] = parts[1] if len(parts) == 2 else parts[1:]
    if header_index < 0:
        raise ValueError("Could not find actual-drive result table header")
    return metadata, header_index


def _numeric_metadata(metadata: dict[str, Any], key: str) -> float | None:
    value = metadata.get(key)
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return None
    match = re.search(r"[-+]?[0-9]*\.?[0-9]+", str(value))
    return float(match.group(0)) if match else None


def _parse_auto_sync_lag_ms(metadata: dict[str, Any]) -> float | None:
    value = metadata.get("AutoSyncHallLag")
    if value is None:
        return None
    match = re.search(r"applied\s+([-+]?[0-9]*\.?[0-9]+)ms", str(value), flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _nonzero_window(time_s: np.ndarray, values: np.ndarray, threshold_fraction: float = 0.02) -> tuple[float, float]:
    finite = np.isfinite(time_s) & np.isfinite(values)
    if finite.sum() < 2:
        return float("nan"), float("nan")
    signal = values[finite]
    pp = float(np.nanmax(signal) - np.nanmin(signal))
    threshold = max(abs(pp) * threshold_fraction, 1e-4)
    active = finite & (np.abs(values) > threshold)
    if not active.any():
        return float("nan"), float("nan")
    return float(np.nanmin(time_s[active])), float(np.nanmax(time_s[active]))


def _field_motion_start(time_s: np.ndarray, field: np.ndarray, baseline: float) -> float:
    residual = np.asarray(field, dtype=float) - float(baseline)
    start, _ = _nonzero_window(time_s, residual, threshold_fraction=0.02)
    return start


def _clipping_or_spike_suspected(voltage: np.ndarray, field: np.ndarray) -> bool:
    voltage_values = np.asarray(voltage, dtype=float)
    field_values = np.asarray(field, dtype=float)
    finite_voltage = voltage_values[np.isfinite(voltage_values)]
    finite_field = field_values[np.isfinite(field_values)]
    if finite_voltage.size and np.nanmax(np.abs(finite_voltage)) >= 4.99:
        return True
    if finite_field.size < 5:
        return False
    diffs = np.abs(np.diff(finite_field))
    if diffs.size < 2:
        return False
    median_step = float(np.nanmedian(diffs))
    max_step = float(np.nanmax(diffs))
    return bool(median_step > 0.0 and max_step > median_step * 25.0)


def _shape_corr_and_nrmse(target: np.ndarray, measured: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(target) & np.isfinite(measured)
    if finite.sum() < 3:
        return float("nan"), float("nan")
    left = np.asarray(target[finite], dtype=float)
    right = np.asarray(measured[finite], dtype=float)
    left_centered = left - float(np.nanmean(left))
    right_centered = right - float(np.nanmean(right))
    denom = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    corr = float(np.dot(left_centered, right_centered) / denom) if denom > 1e-12 else float("nan")
    rmse = float(np.sqrt(np.nanmean(np.square(left - right))))
    pp = _pp(left)
    nrmse = float(rmse / max(abs(pp) * 0.5, 1e-9)) if np.isfinite(pp) else float("nan")
    return corr, nrmse


def _estimate_phase_error(time_s: np.ndarray, target: np.ndarray, measured: np.ndarray, active_mask: np.ndarray) -> float:
    valid = active_mask & np.isfinite(time_s) & np.isfinite(target) & np.isfinite(measured)
    if valid.sum() < 8:
        return float("nan")
    target_centered = target[valid] - float(np.nanmean(target[valid]))
    measured_centered = measured[valid] - float(np.nanmean(measured[valid]))
    corr = np.correlate(measured_centered, target_centered, mode="full")
    lag_index = int(np.argmax(corr) - (len(target_centered) - 1))
    dt = float(np.nanmedian(np.diff(time_s[valid]))) if valid.sum() > 1 else float("nan")
    return float(lag_index * dt) if np.isfinite(dt) else float("nan")


def _pp(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.nanmax(finite) - np.nanmin(finite))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
