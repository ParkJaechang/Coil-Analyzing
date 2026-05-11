from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from field_analysis.finite_actual_drive import build_actual_drive_review_case
from field_analysis.finite_actual_drive import expected_actual_drive_result_filenames
from field_analysis.finite_actual_drive import parse_actual_drive_filename
from field_analysis.finite_actual_drive import parse_finite_actual_drive_filename
from field_analysis.finite_actual_drive import read_actual_drive_result


def review_csv_filename(metadata: dict[str, Any]) -> str:
    freq = f"{float(metadata['freq_hz']):g}"
    cycle = f"{float(metadata['cycle_count']):g}"
    return f"finite_actual_drive_review_{metadata['waveform_type']}_{freq}Hz_{cycle}cycle.csv"


def process_actual_drive_review_folder(input_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    expected_files = expected_actual_drive_result_filenames()
    existing_paths = {
        parse_finite_actual_drive_filename(path)["canonical_source_filename"]: path
        for path in sorted(input_path.glob("*finite_recommended_voltage_lut_*_result.csv"))
    }
    records = [read_actual_drive_result(existing_paths[name]) for name in expected_files if name in existing_paths]
    missing_files = [name for name in expected_files if name not in existing_paths]
    case_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    missing_rows = [_missing_case_row(name, input_path) for name in missing_files]
    for record in records:
        review_frame, metadata = build_actual_drive_review_case(record)
        review_name = review_csv_filename(metadata)
        review_path = output_path / review_name
        review_frame.to_csv(review_path, index=False)
        plot_paths = _write_review_plots(review_frame, metadata, output_path / "plots")
        parsed_row = _parsed_case_row(record, metadata, review_name, review_path, plot_paths)
        case_rows.append(parsed_row)
        metrics_rows.append(parsed_row)
        (output_path / f"{review_name.removesuffix('.csv')}_metadata.json").write_text(
            json.dumps(_json_safe({**metadata, "plot_paths": plot_paths}), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    summary = pd.DataFrame([*case_rows, *missing_rows])
    summary_path = output_path / "finite_actual_drive_review_summary.csv"
    summary.to_csv(summary_path, index=False)
    metrics_path = output_path / "finite_actual_drive_case_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    missing_path = output_path / "finite_actual_drive_missing_cases.csv"
    pd.DataFrame(missing_rows).to_csv(missing_path, index=False)
    manifest_path = output_path / "finite_actual_drive_review_manifest.json"
    packet_complete = len(missing_files) == 0
    manifest = {
        "packet_type": "finite_actual_drive_phase1_review",
        "input_dir": str(input_path),
        "output_dir": str(output_path),
        "expected_files_count": len(expected_files),
        "parsed_files_count": len(records),
        "missing_files": missing_files,
        "review_packet_complete": packet_complete,
        "review_packet_status": "complete" if packet_complete else "partial",
        "summary_csv": str(summary_path),
        "metrics_csv": str(metrics_path),
        "missing_cases_csv": str(missing_path),
        "plots_dir": str(output_path / "plots"),
        "correction_delta_v_generated": False,
        "second_voltage_v_generated": False,
        "second_lut_generated": False,
        "continuous_touched": False,
        "stale_second_correction_artifacts_ignored": True,
    }
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "input_dir": str(input_path),
        "output_dir": str(output_path),
        "files_parsed": len(records),
        "parsed_files_count": len(records),
        "expected_files_count": len(expected_files),
        "missing_files": missing_files,
        "review_packet_complete": packet_complete,
        "summary_path": str(summary_path),
        "metrics_path": str(metrics_path),
        "missing_cases_path": str(missing_path),
        "manifest_path": str(manifest_path),
        "summary": summary,
    }


def _parsed_case_row(
    record: Any,
    metadata: dict[str, Any],
    review_name: str,
    review_path: Path,
    plot_paths: list[str],
) -> dict[str, Any]:
    return {
        "source_file": record.source_file,
        "file_path": str(record.path),
        "parse_status": "parsed",
        "alignment_status": "ok",
        "review_csv_file": review_name,
        "review_csv_path": str(review_path),
        "plot_paths": "|".join(plot_paths),
        "waveform_type": record.waveform_type,
        "freq_hz": record.freq_hz,
        "cycle_count": record.cycle_count,
        "modeled_cycle_count": metadata["modeled_cycle_count"],
        "intended_drive_cycle_count": metadata["intended_drive_cycle_count"],
        "measured_active_nrmse": metadata["measured_active_nrmse"],
        "measured_shape_corr": metadata["measured_shape_corr"],
        "measured_peak_error_mT": metadata["measured_peak_error_mT"],
        "measured_phase_error_s": metadata["measured_phase_error_s"],
        "measured_terminal_error_mT": metadata["measured_terminal_error_mT"],
        "measured_tail_residual": metadata["measured_tail_residual"],
        "measured_startup_residual_mT": metadata["measured_startup_residual_mT"],
        "raw_field_peak_mT": metadata["raw_field_peak_mT"],
        "field_normalization_scale_factor": metadata["field_normalization_scale_factor"],
        "raw_voltage_peak_v": metadata["raw_voltage_peak_v"],
        "voltage_normalization_scale_factor": metadata["voltage_normalization_scale_factor"],
        "normalized_peak_mT": metadata["normalized_peak_mT"],
        "normalized_voltage_peak_v": metadata["normalized_voltage_peak_v"],
        "normalized_shape_corr": metadata["normalized_shape_corr"],
        "normalized_nrmse": metadata["normalized_nrmse"],
        "terminal_peak_time_error_s": metadata["terminal_peak_time_error_s"],
        "shape_review_only": metadata["shape_review_only"],
        "possible_polarity_flip_suggested": metadata["possible_polarity_flip_suggested"],
    }


def _missing_case_row(source_file: str, input_path: Path) -> dict[str, Any]:
    parsed = parse_actual_drive_filename(source_file)
    return {
        "source_file": source_file,
        "file_path": str(input_path / source_file),
        "parse_status": "missing",
        "alignment_status": "unavailable_missing_file",
        "review_csv_file": "",
        "review_csv_path": "",
        "plot_paths": "",
        "waveform_type": parsed["waveform_type"],
        "freq_hz": parsed["freq_hz"],
        "cycle_count": parsed["cycle_count"],
        "modeled_cycle_count": parsed["cycle_count"],
        "intended_drive_cycle_count": parsed["cycle_count"],
        "measured_active_nrmse": np.nan,
        "measured_shape_corr": np.nan,
        "measured_peak_error_mT": np.nan,
        "measured_phase_error_s": np.nan,
        "measured_terminal_error_mT": np.nan,
        "measured_tail_residual": np.nan,
        "measured_startup_residual_mT": np.nan,
        "raw_field_peak_mT": np.nan,
        "field_normalization_scale_factor": np.nan,
        "raw_voltage_peak_v": np.nan,
        "voltage_normalization_scale_factor": np.nan,
        "normalized_peak_mT": np.nan,
        "normalized_voltage_peak_v": np.nan,
        "normalized_shape_corr": np.nan,
        "normalized_nrmse": np.nan,
        "terminal_peak_time_error_s": np.nan,
        "shape_review_only": True,
        "possible_polarity_flip_suggested": False,
    }


def _write_review_plots(review: pd.DataFrame, metadata: dict[str, Any], plot_dir: Path) -> list[str]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    stem = f"actual_drive_review_{metadata['waveform_type']}_{float(metadata['freq_hz']):g}Hz_{float(metadata['cycle_count']):g}cycle"
    plot_specs: list[tuple[str, list[tuple[str, str]], str]] = [
        (
            "target_vs_measured",
            [
                ("normalized_physical_target_output_mT", "Physical Target normalized"),
                ("normalized_measured_field_mT", "Measured HallBz normalized"),
            ],
            "review-normalized field (mT)",
        ),
        ("voltage", [("normalized_first_voltage_v", "Voltage1_V normalized")], "review-normalized voltage (V)"),
        ("residual", [("measured_residual_normalized_mT", "Target - measured normalized")], "normalized residual (mT)"),
    ]
    if "current_a" in review.columns:
        plot_specs.append(("current", [("current_a", "Current1_A")], "current (A)"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        paths = []
        for suffix, _, _ in plot_specs:
            path = plot_dir / f"{stem}_{suffix}.png"
            _write_placeholder_png(path)
            paths.append(str(path))
        return paths
    paths: list[str] = []
    for suffix, columns, ylabel in plot_specs:
        fig, ax = plt.subplots(figsize=(9, 4))
        for column, label in columns:
            ax.plot(review["time_s"], review[column], label=label, linewidth=1.2)
        ax.axvline(metadata["target_active_end_s"], color="k", linestyle="--", linewidth=0.8, label="target end")
        ax.set_title(f"{metadata['waveform_type']} {float(metadata['freq_hz']):g}Hz {float(metadata['cycle_count']):g}cycle - {suffix}")
        ax.set_xlabel("time_s aligned to Voltage1_V command start")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        path = plot_dir / f"{stem}_{suffix}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))
    return paths


def _write_placeholder_png(path: Path) -> None:
    width, height = 2, 2
    raw_rows = b"".join(b"\x00" + b"\xff\xff\xff" * width for _ in range(height))
    payload = zlib.compress(raw_rows)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", payload)
        + chunk(b"IEND", b"")
    )


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
