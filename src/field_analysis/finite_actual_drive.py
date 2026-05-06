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
        normalized["current_a"] = pd.to_numeric(frame["Current1_A"], errors="coerce")
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


def build_actual_drive_review_case(record: ActualDriveRecord) -> tuple[pd.DataFrame, dict[str, Any]]:
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
    residual = physical_target - measured_field
    corr, nrmse = _shape_corr_and_nrmse(physical_target[active_mask], measured_field[active_mask])
    sampled_target_pp = _pp(physical_target[active_mask])
    measured_pp = _pp(measured_field[active_mask])
    measured_peak_error = float(measured_pp - float(FIELD_ROUTE_NORMALIZED_TARGET_PP)) if np.isfinite(measured_pp) else float("nan")
    phase_error_s = _estimate_phase_error(relative_time, physical_target, measured_field, active_mask)
    terminal_mask = active_mask & (relative_time >= max(target_duration_s * 0.85, 0.0))
    startup_mask = active_mask & (relative_time <= min(target_duration_s * 0.2, 0.25 / max(record.freq_hz, 1e-9)))
    tail_mask = relative_time > target_duration_s
    measured_terminal_error = float(np.nanmean(residual[terminal_mask])) if terminal_mask.any() else float("nan")
    measured_startup_residual = float(np.nanmean(residual[startup_mask])) if startup_mask.any() else float("nan")
    measured_tail_residual = (
        float(np.nanmax(np.abs(measured_field[tail_mask]))) / max(abs(measured_pp), 1e-9)
        if tail_mask.any() and np.isfinite(measured_pp)
        else float("nan")
    )
    possible_polarity_flip = bool(np.isfinite(corr) and corr < -0.3)

    review = pd.DataFrame(
        {
            "time_s": relative_time,
            "first_voltage_v": first_voltage,
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
        "correction_delta_generated": False,
        "second_voltage_generated": False,
        "second_lut_generated": False,
        "continuous_touched": False,
    }
    return review, metadata


def review_csv_filename(metadata: dict[str, Any]) -> str:
    freq = f"{float(metadata['freq_hz']):g}"
    cycle = f"{float(metadata['cycle_count']):g}"
    return f"finite_actual_drive_review_{metadata['waveform_type']}_{freq}Hz_{cycle}cycle.csv"


def process_actual_drive_review_folder(input_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    records = [read_actual_drive_result(path) for path in sorted(input_path.glob("finite_recommended_voltage_lut_*_result.csv"))]
    case_rows: list[dict[str, Any]] = []
    for record in records:
        review_frame, metadata = build_actual_drive_review_case(record)
        review_name = review_csv_filename(metadata)
        review_path = output_path / review_name
        review_frame.to_csv(review_path, index=False)
        case_rows.append(
            {
                "source_file": record.source_file,
                "review_csv_file": review_name,
                "waveform_type": record.waveform_type,
                "freq_hz": record.freq_hz,
                "cycle_count": record.cycle_count,
                "measured_active_nrmse": metadata["measured_active_nrmse"],
                "measured_shape_corr": metadata["measured_shape_corr"],
                "measured_peak_error_mT": metadata["measured_peak_error_mT"],
                "measured_phase_error_s": metadata["measured_phase_error_s"],
                "measured_terminal_error_mT": metadata["measured_terminal_error_mT"],
                "measured_tail_residual": metadata["measured_tail_residual"],
                "measured_startup_residual_mT": metadata["measured_startup_residual_mT"],
                "possible_polarity_flip_suggested": metadata["possible_polarity_flip_suggested"],
            }
        )
        (output_path / f"{review_name.removesuffix('.csv')}_metadata.json").write_text(
            json.dumps(_json_safe(metadata), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    summary = pd.DataFrame(case_rows)
    summary_path = output_path / "finite_actual_drive_review_summary.csv"
    summary.to_csv(summary_path, index=False)
    return {
        "input_dir": str(input_path),
        "output_dir": str(output_path),
        "files_parsed": len(records),
        "summary_path": str(summary_path),
        "summary": summary,
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
