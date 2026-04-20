from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.analysis import analyze_measurements, combine_analysis_frames
from field_analysis.canonicalize import CanonicalizeConfig, canonicalize_batch
from field_analysis.control_formula import build_control_formula, build_control_lut
from field_analysis.models import CycleDetectionConfig, PreprocessConfig
from field_analysis.parser import parse_measurement_file
from field_analysis.recommendation_service import (
    DEFAULT_RECOMMENDATION_POLICY_CONFIG,
    LegacyRecommendationContext,
    RecommendationOptions,
    TargetRequest,
    recommend,
)
from field_analysis.schema_config import load_schema_config


CURRENT_CHANNEL = "i_sum_signed"
FIELD_CHANNEL = "bz_mT"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "policy_eval" / "export_validation"
CONTINUOUS_DIR = REPO_ROOT.parent / "outputs" / "field_analysis_app_state" / "uploads" / "continuous"
TRANSIENT_DIR = REPO_ROOT.parent / "outputs" / "field_analysis_app_state" / "uploads" / "transient"


def _load_dataset(directory: Path, *, regime: str, expected_cycles: int = 10) -> dict[str, Any]:
    schema = load_schema_config(None)
    parsed_measurements = []
    for path in sorted(directory.rglob("*.csv")):
        source_name = path.relative_to(directory).as_posix()
        parsed_measurements.extend(
            parse_measurement_file(
                source_name,
                path.read_bytes(),
                schema=schema,
                expected_cycles=expected_cycles,
                target_current_mode="auto",
            )
        )
    canonical_runs = canonicalize_batch(
        parsed_measurements,
        regime=regime,
        role="train",
        config=CanonicalizeConfig(preferred_field_axis=FIELD_CHANNEL),
    )
    analyses = analyze_measurements(
        parsed_measurements,
        canonical_runs=canonical_runs,
        preprocess_config=PreprocessConfig(),
        cycle_config=CycleDetectionConfig(expected_cycles=expected_cycles, reference_channel=CURRENT_CHANNEL),
        current_channel=CURRENT_CHANNEL,
        main_field_axis=FIELD_CHANNEL,
    )
    _, per_test_summary, _ = combine_analysis_frames(analyses, field_axis=FIELD_CHANNEL)
    analysis_lookup = {}
    for analysis in analyses:
        if analysis.per_test_summary.empty:
            continue
        test_id = str(analysis.per_test_summary.iloc[0]["test_id"])
        analysis_lookup[test_id] = analysis
    return {
        "parsed": parsed_measurements,
        "canonical_runs": canonical_runs,
        "analyses": analyses,
        "per_test_summary": per_test_summary,
        "analysis_lookup": analysis_lookup,
    }


def _pick_exact_field_level(per_test_summary: pd.DataFrame, *, waveform: str, freq_hz: float) -> float:
    subset = per_test_summary[
        (per_test_summary["waveform_type"].astype(str).str.lower() == waveform.lower())
        & np.isclose(pd.to_numeric(per_test_summary["freq_hz"], errors="coerce"), float(freq_hz), atol=1e-6)
    ].copy()
    subset = subset.dropna(subset=["achieved_bz_mT_pp_mean"])
    if subset.empty:
        raise RuntimeError(f"No exact field support row found for {waveform} {freq_hz} Hz")
    return float(pd.to_numeric(subset["achieved_bz_mT_pp_mean"], errors="coerce").iloc[0])


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _validate_command_profile(
    command_profile: pd.DataFrame,
    *,
    case_name: str,
    target_waveform: str,
    target_type: str,
    freq_hz: float,
    finite_cycle_mode: bool,
    target_cycle_count: float | None,
) -> dict[str, Any]:
    required_columns = ["time_s", "limited_voltage_v", "waveform_type", "freq_hz", "finite_cycle_mode"]
    missing_columns = [column for column in required_columns if column not in command_profile.columns]
    if missing_columns:
        raise AssertionError(f"{case_name}: missing command profile columns {missing_columns}")

    line_count = int(len(command_profile) + 1)
    time_s = pd.to_numeric(command_profile["time_s"], errors="coerce").to_numpy(dtype=float)
    if len(time_s) >= 2 and not np.all(np.diff(time_s) >= 0):
        raise AssertionError(f"{case_name}: time_s is not monotonic")
    limited_voltage = pd.to_numeric(command_profile["limited_voltage_v"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(limited_voltage).any():
        raise AssertionError(f"{case_name}: limited_voltage_v has no finite samples")
    if float(np.nanmax(np.abs(limited_voltage))) <= 0:
        raise AssertionError(f"{case_name}: limited_voltage_v is all zeros")

    profile_freq = float(pd.to_numeric(command_profile["freq_hz"], errors="coerce").iloc[0])
    if not np.isclose(profile_freq, float(freq_hz), atol=1e-6):
        raise AssertionError(f"{case_name}: freq_hz metadata mismatch {profile_freq} != {freq_hz}")

    waveform_values = command_profile["waveform_type"].astype(str).str.lower().dropna().unique().tolist()
    if waveform_values != [target_waveform.lower()]:
        raise AssertionError(f"{case_name}: waveform_type metadata mismatch {waveform_values} != {[target_waveform.lower()]}")

    finite_flag = bool(command_profile["finite_cycle_mode"].iloc[0])
    if finite_flag != bool(finite_cycle_mode):
        raise AssertionError(f"{case_name}: finite_cycle_mode metadata mismatch")

    if finite_cycle_mode:
        cycle_value = float(pd.to_numeric(command_profile["target_cycle_count"], errors="coerce").iloc[0])
        if not np.isclose(cycle_value, float(target_cycle_count), atol=1e-6):
            raise AssertionError(f"{case_name}: target_cycle_count metadata mismatch")

    return {
        "line_count": line_count,
        "sample_count": int(len(command_profile)),
        "limited_voltage_pk_v": float(np.nanmax(np.abs(limited_voltage))),
        "limited_voltage_pp_v": float(np.nanmax(limited_voltage) - np.nanmin(limited_voltage)),
        "profile_freq_hz": profile_freq,
        "target_waveform": target_waveform,
        "target_type": target_type,
        "finite_cycle_mode": finite_flag,
        "time_start_s": float(np.nanmin(time_s)),
        "time_end_s": float(np.nanmax(time_s)),
    }


def _validate_case(
    *,
    case_name: str,
    result: Any,
    target_waveform: str,
    target_type: str,
    freq_hz: float,
    finite_cycle_mode: bool,
    target_cycle_count: float | None,
    output_dir: Path,
) -> dict[str, Any]:
    command_profile = getattr(result, "command_profile", None)
    if command_profile is None or command_profile.empty:
        raise AssertionError(f"{case_name}: empty command profile")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{case_name}.csv"
    csv_bytes = command_profile.to_csv(index=False).encode("utf-8-sig")
    csv_path.write_bytes(csv_bytes)
    profile_validation = _validate_command_profile(
        command_profile,
        case_name=case_name,
        target_waveform=target_waveform,
        target_type=target_type,
        freq_hz=freq_hz,
        finite_cycle_mode=finite_cycle_mode,
        target_cycle_count=target_cycle_count,
    )

    formula = build_control_formula(command_profile, value_column="limited_voltage_v")
    if formula is None:
        raise AssertionError(f"{case_name}: control formula could not be built")
    formula_text = str(formula.get("formula_text", "") or "")
    coefficients = formula.get("coefficient_table", pd.DataFrame())
    if not formula_text.strip():
        raise AssertionError(f"{case_name}: empty control formula text")
    if not isinstance(coefficients, pd.DataFrame) or coefficients.empty:
        raise AssertionError(f"{case_name}: empty control formula coefficients")
    control_lut = build_control_lut(command_profile, value_column="limited_voltage_v")
    if control_lut is None or control_lut.empty:
        raise AssertionError(f"{case_name}: control LUT could not be built")

    formula_path = output_dir / f"{case_name}_formula.txt"
    coeff_path = output_dir / f"{case_name}_coefficients.csv"
    lut_path = output_dir / f"{case_name}_control_lut.csv"
    _write_text(formula_path, formula_text)
    coeff_bytes = coefficients.to_csv(index=False).encode("utf-8-sig")
    coeff_path.write_bytes(coeff_bytes)
    lut_bytes = control_lut.to_csv(index=False).encode("utf-8-sig")
    lut_path.write_bytes(lut_bytes)

    progress_column = "cycle_progress" if "cycle_progress" in control_lut.columns else "time_fraction"
    required_lut_columns = {"lut_index", "time_s", "command_voltage_v", "finite_cycle_mode", progress_column}
    missing_lut_columns = required_lut_columns - set(control_lut.columns)
    if missing_lut_columns:
        raise AssertionError(f"{case_name}: missing control LUT columns {sorted(missing_lut_columns)}")

    lut_time = pd.to_numeric(control_lut["time_s"], errors="coerce").to_numpy(dtype=float)
    if len(lut_time) >= 2 and not np.all(np.diff(lut_time) >= 0):
        raise AssertionError(f"{case_name}: control LUT time_s is not monotonic")
    lut_voltage = pd.to_numeric(control_lut["command_voltage_v"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(lut_voltage).any():
        raise AssertionError(f"{case_name}: control LUT command_voltage_v has no finite samples")

    waveform_voltage = pd.to_numeric(command_profile["limited_voltage_v"], errors="coerce").to_numpy(dtype=float)
    waveform_pk = float(np.nanmax(np.abs(waveform_voltage)))
    lut_pk = float(np.nanmax(np.abs(lut_voltage)))
    if lut_pk > waveform_pk + 1e-6:
        raise AssertionError(f"{case_name}: control LUT voltage exceeds waveform envelope {lut_pk} > {waveform_pk}")

    lut_finite_flags = control_lut["finite_cycle_mode"].astype(bool).unique().tolist()
    if lut_finite_flags != [bool(finite_cycle_mode)]:
        raise AssertionError(f"{case_name}: control LUT finite_cycle_mode mismatch {lut_finite_flags}")

    cycle_progress = pd.to_numeric(control_lut[progress_column], errors="coerce").to_numpy(dtype=float)
    finite_preview_window_ok = bool(np.nanmin(cycle_progress) >= -1e-6 and np.nanmax(cycle_progress) <= 1.0 + 1e-6)

    return {
        "preview_only": bool(getattr(result, "preview_only", False)),
        "allow_auto_download": bool(getattr(result, "allow_auto_download", False)),
        "support_state": str(getattr(result, "engine_summary", {}).get("support_state", "unknown")),
        "policy_version": str(getattr(result, "debug_info", {}).get("policy_version", "")),
        "csv_file": str(csv_path),
        "csv_size_bytes": int(csv_path.stat().st_size),
        "formula_file": str(formula_path),
        "formula_size_bytes": int(formula_path.stat().st_size),
        "coeff_file": str(coeff_path),
        "coeff_size_bytes": int(coeff_path.stat().st_size),
        "lut_file": str(lut_path),
        "lut_size_bytes": int(lut_path.stat().st_size),
        "formula_text_nonempty": bool(formula_text.strip()),
        "coeff_row_count": int(len(coefficients)),
        "lut_row_count": int(len(control_lut)),
        "lut_progress_column": progress_column,
        "lut_voltage_pk_v": lut_pk,
        "lut_voltage_within_waveform_envelope": bool(lut_pk <= waveform_pk + 1e-6),
        "lut_cycle_progress_in_unit_interval": finite_preview_window_ok,
        **profile_validation,
    }


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    continuous = _load_dataset(CONTINUOUS_DIR, regime="continuous", expected_cycles=10)
    transient = _load_dataset(TRANSIENT_DIR, regime="transient", expected_cycles=10)
    exact_field_level = _pick_exact_field_level(continuous["per_test_summary"], waveform="sine", freq_hz=0.25)

    legacy_context = LegacyRecommendationContext(
        per_test_summary=continuous["per_test_summary"],
        analysis_lookup=continuous["analysis_lookup"],
        transient_measurements=transient["parsed"],
        transient_preprocess_results=[analysis.preprocess for analysis in transient["analyses"]],
        transient_canonical_runs=transient["canonical_runs"],
        validation_measurements=[],
        validation_preprocess_results=[],
    )
    options = RecommendationOptions(current_channel=CURRENT_CHANNEL, field_channel=FIELD_CHANNEL)
    policy_config = DEFAULT_RECOMMENDATION_POLICY_CONFIG

    cases = [
        {
            "name": "continuous_exact_current",
            "target": TargetRequest(
                regime="continuous",
                target_waveform="sine",
                command_waveform="sine",
                freq_hz=0.5,
                target_type="current",
                target_level_value=20.0,
                target_level_kind="pp",
                context={"request_kind": "waveform_compensation", "frequency_mode": "exact", "finite_cycle_mode": False, "preview_tail_cycles": 0.25},
            ),
            "finite_cycle_mode": False,
            "target_cycle_count": None,
        },
        {
            "name": "continuous_exact_field",
            "target": TargetRequest(
                regime="continuous",
                target_waveform="sine",
                command_waveform="sine",
                freq_hz=0.25,
                target_type="field",
                target_level_value=exact_field_level,
                target_level_kind="pp",
                context={"request_kind": "waveform_compensation", "frequency_mode": "exact", "finite_cycle_mode": False, "preview_tail_cycles": 0.25},
            ),
            "finite_cycle_mode": False,
            "target_cycle_count": None,
        },
        {
            "name": "finite_exact_sine",
            "target": TargetRequest(
                regime="transient",
                target_waveform="sine",
                command_waveform="sine",
                freq_hz=0.5,
                commanded_cycles=1.0,
                target_type="current",
                target_level_value=20.0,
                target_level_kind="pp",
                context={"request_kind": "waveform_compensation", "frequency_mode": "exact", "finite_cycle_mode": True, "preview_tail_cycles": 0.25},
            ),
            "finite_cycle_mode": True,
            "target_cycle_count": 1.0,
        },
        {
            "name": "finite_exact_triangle_0p5hz_1p0cycle_20pp",
            "target": TargetRequest(
                regime="transient",
                target_waveform="triangle",
                command_waveform="triangle",
                freq_hz=0.5,
                commanded_cycles=1.0,
                target_type="current",
                target_level_value=20.0,
                target_level_kind="pp",
                context={"request_kind": "waveform_compensation", "frequency_mode": "exact", "finite_cycle_mode": True, "preview_tail_cycles": 0.25},
            ),
            "finite_cycle_mode": True,
            "target_cycle_count": 1.0,
        },
        {
            "name": "finite_exact_triangle_1p25hz_1p25cycle_10pp",
            "target": TargetRequest(
                regime="transient",
                target_waveform="triangle",
                command_waveform="triangle",
                freq_hz=1.25,
                commanded_cycles=1.25,
                target_type="current",
                target_level_value=10.0,
                target_level_kind="pp",
                context={"request_kind": "waveform_compensation", "frequency_mode": "exact", "finite_cycle_mode": True, "preview_tail_cycles": 0.25},
            ),
            "finite_cycle_mode": True,
            "target_cycle_count": 1.25,
        },
        {
            "name": "finite_exact_triangle_3hz_1p5cycle_20pp",
            "target": TargetRequest(
                regime="transient",
                target_waveform="triangle",
                command_waveform="triangle",
                freq_hz=3.0,
                commanded_cycles=1.5,
                target_type="current",
                target_level_value=20.0,
                target_level_kind="pp",
                context={"request_kind": "waveform_compensation", "frequency_mode": "exact", "finite_cycle_mode": True, "preview_tail_cycles": 0.25},
            ),
            "finite_cycle_mode": True,
            "target_cycle_count": 1.5,
        },
    ]

    report: dict[str, Any] = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "policy_version": policy_config.version,
        "cases": {},
    }
    for case in cases:
        result = recommend(
            continuous_runs=continuous["canonical_runs"],
            transient_runs=transient["canonical_runs"],
            validation_runs=[],
            target=case["target"],
            options=options,
            legacy_context=legacy_context,
            policy_config=policy_config,
        )
        report["cases"][case["name"]] = _validate_case(
            case_name=case["name"],
            result=result,
            target_waveform=str(case["target"].target_waveform),
            target_type=str(case["target"].target_type),
            freq_hz=float(case["target"].freq_hz),
            finite_cycle_mode=bool(case["finite_cycle_mode"]),
            target_cycle_count=case["target_cycle_count"],
            output_dir=ARTIFACT_DIR,
        )

    json_path = REPO_ROOT / "artifacts" / "policy_eval" / "exact_export_validation.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = REPO_ROOT / "artifacts" / "policy_eval" / "exact_export_validation.md"
    lines = [
        "# Exact Export Validation",
        "",
        f"- policy version: `{policy_config.version}`",
        "",
        "| case | support_state | preview_only | auto_download | csv lines | lut rows | voltage envelope | cycle_progress ok |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case_name, case_report in report["cases"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    case_name,
                    str(case_report["support_state"]),
                    str(case_report["preview_only"]),
                    str(case_report["allow_auto_download"]),
                    str(case_report["line_count"]),
                    str(case_report["lut_row_count"]),
                    str(case_report["lut_voltage_within_waveform_envelope"]),
                    str(case_report["lut_cycle_progress_in_unit_interval"]),
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
