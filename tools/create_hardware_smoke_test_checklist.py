from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


OUTPUT_MD = Path("artifacts/policy_eval/hardware_smoke_test_checklist.md")
OUTPUT_JSON = Path("artifacts/policy_eval/hardware_smoke_test_manifest.json")
EXPORT_DIR = Path("artifacts/policy_eval/export_validation")


def main() -> int:
    cases = [
        {
            "case_id": "cont_exact_current_01",
            "regime": "continuous",
            "target_type": "current",
            "waveform": "sine",
            "freq_hz": 0.5,
            "level_or_field": 20,
            "cycles": None,
            "expected_mode": "exact auto",
            "waveform_file": str((EXPORT_DIR / "continuous_exact_current.csv").resolve()),
            "formula_file": str((EXPORT_DIR / "continuous_exact_current_formula.txt").resolve()),
            "lut_file": str((EXPORT_DIR / "continuous_exact_current_control_lut.csv").resolve()),
        },
        {
            "case_id": "cont_exact_field_01",
            "regime": "continuous",
            "target_type": "field",
            "waveform": "sine",
            "freq_hz": 0.25,
            "level_or_field": 20,
            "cycles": None,
            "expected_mode": "exact auto",
            "waveform_file": str((EXPORT_DIR / "continuous_exact_field.csv").resolve()),
            "formula_file": str((EXPORT_DIR / "continuous_exact_field_formula.txt").resolve()),
            "lut_file": str((EXPORT_DIR / "continuous_exact_field_control_lut.csv").resolve()),
        },
        {
            "case_id": "finite_exact_sine_01",
            "regime": "transient",
            "target_type": "current",
            "waveform": "sine",
            "freq_hz": 0.5,
            "level_or_field": 20,
            "cycles": 1.0,
            "expected_mode": "exact recipe",
            "waveform_file": str((EXPORT_DIR / "finite_exact_sine.csv").resolve()),
            "formula_file": str((EXPORT_DIR / "finite_exact_sine_formula.txt").resolve()),
            "lut_file": str((EXPORT_DIR / "finite_exact_sine_control_lut.csv").resolve()),
        },
    ]

    lines: list[str] = []
    lines.append("# Hardware Smoke Test Checklist")
    lines.append("")
    lines.append(f"- generated_at_utc: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append("- purpose: `exact-supported operational path confirmation only`")
    lines.append("- note: `not executed in current coding environment; use on attached hardware bench`")
    lines.append("")
    lines.append("## Continuous Exact Current")
    lines.append("")
    lines.append("| case_id | waveform | freq_hz | target_current_pp_a | waveform_file | lut_file | shape_corr | nrmse | phase_lag_deg | clipping | pass_fail | notes |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for case in [row for row in cases if row["regime"] == "continuous" and row["target_type"] == "current"]:
        lines.append(
            f"| {case['case_id']} | {case['waveform']} | {case['freq_hz']} | {case['level_or_field']} | "
            f"{Path(case['waveform_file']).name} | {Path(case['lut_file']).name} |  |  |  |  |  |  |"
        )
    lines.append("")
    lines.append("## Continuous Exact Field")
    lines.append("")
    lines.append("| case_id | waveform | freq_hz | target_field_pp_mT | waveform_file | lut_file | shape_corr | nrmse | phase_lag_deg | clipping | pass_fail | notes |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for case in [row for row in cases if row["regime"] == "continuous" and row["target_type"] == "field"]:
        lines.append(
            f"| {case['case_id']} | {case['waveform']} | {case['freq_hz']} | {case['level_or_field']} | "
            f"{Path(case['waveform_file']).name} | {Path(case['lut_file']).name} |  |  |  |  |  |  |"
        )
    lines.append("")
    lines.append("## Finite Exact Recipe")
    lines.append("")
    lines.append("| case_id | waveform | freq_hz | cycles | level_pp_a | waveform_file | lut_file | shape_corr | nrmse | phase_lag_deg | clipping | pass_fail | notes |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for case in [row for row in cases if row["regime"] == "transient"]:
        lines.append(
            f"| {case['case_id']} | {case['waveform']} | {case['freq_hz']} | {case['cycles']} | {case['level_or_field']} | "
            f"{Path(case['waveform_file']).name} | {Path(case['lut_file']).name} |  |  |  |  |  |  |"
        )
    lines.append("")
    lines.append("## Acceptance")
    lines.append("")
    lines.append("- continuous exact current: `shape corr >= 0.95`, `NRMSE <= 0.15`, no clipping")
    lines.append("- continuous exact field: `shape corr >= 0.95`, `NRMSE <= 0.15`, no clipping")
    lines.append("- finite exact recipe: `shape corr >= 0.90`, `NRMSE <= 0.20`, no clipping")
    lines.append("")
    lines.append("## Artifact Files")
    lines.append("")
    for case in cases:
        lines.append(f"- `{case['case_id']}`")
        lines.append(f"  waveform: `{case['waveform_file']}`")
        lines.append(f"  formula: `{case['formula_file']}`")
        lines.append(f"  lut: `{case['lut_file']}`")
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "note": "not executed in current coding environment; use on attached hardware bench",
                "cases": cases,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
