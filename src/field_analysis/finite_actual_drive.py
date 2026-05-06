from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from field_analysis.compensation import FIELD_ROUTE_NORMALIZED_TARGET_PP, _finite_target_template


RESULT_FILENAME_RE = re.compile(
    r"finite_recommended_voltage_lut_(?P<waveform>[A-Za-z]+)_(?P<freq>[0-9]+(?:\.[0-9]+)?)Hz_(?P<cycle>[0-9]+(?:\.[0-9]+)?)cycle_result\.csv$",
    re.IGNORECASE,
)


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
    name = Path(path).name
    match = RESULT_FILENAME_RE.match(name)
    if match is None:
        raise ValueError(f"Unsupported finite actual-drive result filename: {name}")
    return {
        "waveform_type": match.group("waveform").lower(),
        "freq_hz": float(match.group("freq")),
        "cycle_count": float(match.group("cycle")),
    }


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
        key = parts[0]
        if len(parts) == 2:
            metadata[key] = parts[1]
        elif len(parts) > 2:
            metadata[key] = parts[1:]
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
    normalized = pd.DataFrame(
        {
            "sample_index": np.arange(len(frame), dtype=int),
            "time_s_abs": pd.to_numeric(frame["TimeMs"], errors="coerce") / 1000.0,
            "first_voltage_v": pd.to_numeric(frame["Voltage1_V"], errors="coerce"),
            "measured_field_raw": pd.to_numeric(frame["HallBz"], errors="coerce"),
        }
    )
    if "Current1_A" in frame.columns:
        normalized["measured_current_a"] = pd.to_numeric(frame["Current1_A"], errors="coerce")
    metadata: dict[str, Any] = {
        **filename_meta,
        "source_file": source_path.name,
        "source_path": str(source_path),
        "pre_delay_s": _numeric_metadata(preamble, "PreDelay(s)"),
        "post_delay_s": _numeric_metadata(preamble, "PostDelay(s)"),
        "auto_sync_hall_lag_ms": _parse_auto_sync_lag_ms(preamble),
        "field_unit": "mT_inferred_from_HallBz",
        "field_units": "mT",
        "voltage_unit": "V",
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
    pp = float(np.nanmax(left) - np.nanmin(left))
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


def _smooth(values: np.ndarray, window: int = 9) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return np.asarray(values, dtype=float)
    kernel = np.ones(int(window), dtype=float) / float(window)
    return np.convolve(np.asarray(values, dtype=float), kernel, mode="same")


def build_second_correction_case(
    record: ActualDriveRecord,
    *,
    voltage_limit_v: float = 5.0,
    correction_gain: float = 0.25,
    max_delta_fraction_of_limit: float = 0.2,
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
    measured_residual = physical_target - measured_field
    corr, nrmse = _shape_corr_and_nrmse(physical_target[active_mask], measured_field[active_mask])
    target_pp_sampled = float(np.nanmax(physical_target[active_mask]) - np.nanmin(physical_target[active_mask])) if active_mask.any() else float("nan")
    target_pp = float(FIELD_ROUTE_NORMALIZED_TARGET_PP)
    measured_pp = float(np.nanmax(measured_field[active_mask]) - np.nanmin(measured_field[active_mask])) if active_mask.any() else float("nan")
    measured_peak_error = float(measured_pp - target_pp) if np.isfinite(measured_pp) and np.isfinite(target_pp) else float("nan")
    phase_error_s = _estimate_phase_error(relative_time, physical_target, measured_field, active_mask)
    terminal_mask = active_mask & (relative_time >= max(target_duration_s * 0.85, 0.0))
    measured_terminal_error = float(np.nanmean(measured_residual[terminal_mask])) if terminal_mask.any() else float("nan")
    tail_mask = relative_time > target_duration_s
    measured_tail_residual = float(np.nanmax(np.abs(measured_field[tail_mask]))) / max(abs(measured_pp), 1e-9) if tail_mask.any() and np.isfinite(measured_pp) else float("nan")
    startup_mask = active_mask & (relative_time <= min(target_duration_s * 0.2, 0.25 / max(record.freq_hz, 1e-9)))
    measured_startup_residual = float(np.nanmean(measured_residual[startup_mask])) if startup_mask.any() else float("nan")
    polarity_corr, _ = _shape_corr_and_nrmse(physical_target[active_mask], measured_field[active_mask])
    possible_polarity_flip = bool(np.isfinite(polarity_corr) and polarity_corr < -0.3)

    voltage_to_field = float(np.nanmax(np.abs(first_voltage[active_mask])) / max(abs(measured_pp), 1e-9)) if active_mask.any() and np.isfinite(measured_pp) else 0.0
    raw_delta = correction_gain * measured_residual * voltage_to_field
    raw_delta[~active_mask] = 0.0
    correction_delta = _smooth(raw_delta, window=9)
    max_delta = float(voltage_limit_v) * float(max_delta_fraction_of_limit)
    correction_delta = np.clip(correction_delta, -max_delta, max_delta)
    second_voltage = np.clip(first_voltage + correction_delta, -float(voltage_limit_v), float(voltage_limit_v))
    voltage_limit_respected = bool(np.nanmax(np.abs(second_voltage)) <= float(voltage_limit_v) + 1e-9)
    first_smoothness = _smoothness_score(first_voltage)
    second_smoothness = _smoothness_score(second_voltage)
    smoothness_preserved = bool(not np.isfinite(first_smoothness) or second_smoothness <= first_smoothness * 1.5 + 1e-9)
    correction_applied = bool(voltage_limit_respected and smoothness_preserved and np.nanmax(np.abs(correction_delta)) > 1e-9)
    reject_reason = None if correction_applied else "voltage_limit_or_smoothness_guard"
    result = pd.DataFrame(
        {
            "sample_index": frame["sample_index"].to_numpy(dtype=int),
            "time_s": relative_time,
            "first_voltage_v": first_voltage,
            "correction_delta_v": correction_delta,
            "second_voltage_v": second_voltage,
            "physical_target_output": physical_target,
            "measured_field": measured_field,
            "measured_field_raw": raw_field,
            "measured_residual": measured_residual,
            "second_predicted_output": np.nan,
        }
    )
    if "measured_current_a" in frame.columns:
        result["measured_current_a"] = frame["measured_current_a"].to_numpy(dtype=float)
    metadata = {
        **record.metadata,
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
        "measured_active_nrmse": nrmse,
        "measured_shape_corr": corr,
        "measured_peak_error": measured_peak_error,
        "measured_peak_error_mT": measured_peak_error,
        "measured_phase_error_s": phase_error_s,
        "measured_terminal_error": measured_terminal_error,
        "measured_tail_residual": measured_tail_residual,
        "measured_startup_residual": measured_startup_residual,
        "measured_pp": measured_pp,
        "target_pp": target_pp,
        "target_pp_sampled": target_pp_sampled,
        "possible_polarity_flip_suggested": possible_polarity_flip,
        "second_correction_method": "conservative_residual_proportional",
        "second_correction_gain": float(correction_gain),
        "voltage_limit_v": float(voltage_limit_v),
        "voltage_limit_respected": voltage_limit_respected,
        "smoothness_preserved": smoothness_preserved,
        "command_smoothness_first": first_smoothness,
        "command_smoothness_second": second_smoothness,
        "correction_applied": correction_applied,
        "correction_reject_reason": reject_reason,
        "second_prediction_available": False,
        "second_prediction_unavailable_reason": "no_forward_model_from_actual_drive_only",
    }
    return result, metadata


def _smoothness_score(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    if len(finite) < 3 or not np.isfinite(finite).any():
        return float("nan")
    return float(np.sqrt(np.nanmean(np.square(np.diff(finite, n=2)))))


def second_lut_filename(metadata: dict[str, Any]) -> str:
    freq = f"{float(metadata['freq_hz']):g}"
    cycle = f"{float(metadata['cycle_count']):g}"
    return f"finite_second_correction_lut_{metadata['waveform_type']}_{freq}Hz_{cycle}cycle.csv"


def process_actual_drive_folder(input_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    records = [read_actual_drive_result(path) for path in sorted(input_path.glob("finite_recommended_voltage_lut_*_result.csv"))]
    case_rows: list[dict[str, Any]] = []
    for record in records:
        lut_frame, metadata = build_second_correction_case(record)
        lut_name = second_lut_filename(metadata)
        lut_path = output_path / lut_name
        lut_frame.to_csv(lut_path, index=False)
        case_rows.append(
            {
                "source_file": record.source_file,
                "second_lut_file": lut_name,
                "waveform_type": record.waveform_type,
                "freq_hz": record.freq_hz,
                "cycle_count": record.cycle_count,
                "measured_active_nrmse": metadata["measured_active_nrmse"],
                "measured_shape_corr": metadata["measured_shape_corr"],
                "measured_peak_error_mT": metadata["measured_peak_error_mT"],
                "measured_phase_error_s": metadata["measured_phase_error_s"],
                "voltage_limit_respected": metadata["voltage_limit_respected"],
                "smoothness_preserved": metadata["smoothness_preserved"],
                "correction_applied": metadata["correction_applied"],
                "possible_polarity_flip_suggested": metadata["possible_polarity_flip_suggested"],
            }
        )
        (output_path / f"{lut_name.removesuffix('.csv')}_metadata.json").write_text(
            json.dumps(_json_safe(metadata), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    summary = pd.DataFrame(case_rows)
    summary_path = output_path / "finite_second_correction_batch_summary.csv"
    summary.to_csv(summary_path, index=False)
    return {
        "input_dir": str(input_path),
        "output_dir": str(output_path),
        "files_parsed": len(records),
        "summary_path": str(summary_path),
        "summary": summary,
    }


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
